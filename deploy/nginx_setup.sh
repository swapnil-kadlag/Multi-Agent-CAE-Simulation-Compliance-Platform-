#!/bin/bash
# deploy/nginx_setup.sh
# ─────────────────────────────────────────────────────────────────────────────
# Configures Nginx as a reverse proxy in front of the FastAPI container
# and provisions a free SSL certificate via Let's Encrypt (Certbot).
#
# Usage:
#   bash deploy/nginx_setup.sh your-domain.com  your@email.com
#
# Requirements:
#   • setup_ec2.sh must have been run first
#   • DNS A record for your-domain.com must point to this EC2 IP
#     (check: dig your-domain.com +short  →  should show EC2 public IP)
#   • Port 80 + 443 open in EC2 Security Group
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

DOMAIN="${1:-}"
EMAIL="${2:-admin@example.com}"

if [[ -z "$DOMAIN" ]]; then
  echo "Usage: bash deploy/nginx_setup.sh your-domain.com your@email.com"
  exit 1
fi

APP_DIR="/opt/cae-platform"
NGINX_CONF="/etc/nginx/sites-available/cae-platform"

echo "================================================================"
echo "  Nginx + SSL Setup for: $DOMAIN"
echo "================================================================"

# ── Step 1: Write initial HTTP-only Nginx config ──────────────────────────────
# Certbot needs the /.well-known/acme-challenge route to verify domain ownership
# before it can issue the SSL cert — so we start with HTTP only.

echo ""
echo "[1/3] Writing initial Nginx config (HTTP only for cert verification)..."

sudo tee "$NGINX_CONF" > /dev/null <<EOF
server {
    listen 80;
    server_name ${DOMAIN};

    # Let's Encrypt domain verification
    location /.well-known/acme-challenge/ {
        root /var/www/certbot;
    }

    # Proxy all other traffic to FastAPI
    location / {
        proxy_pass         http://127.0.0.1:8000;
        proxy_set_header   Host              \$host;
        proxy_set_header   X-Real-IP         \$remote_addr;
        proxy_set_header   X-Forwarded-For   \$proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto \$scheme;

        # SSE support — required for MCP /mcp/sse endpoint
        proxy_read_timeout          3600;
        proxy_send_timeout          3600;
        proxy_buffering             off;
        proxy_cache                 off;
        proxy_http_version          1.1;
        proxy_set_header Connection "";
        chunked_transfer_encoding   on;
    }
}
EOF

sudo mkdir -p /var/www/certbot
sudo ln -sf "$NGINX_CONF" /etc/nginx/sites-enabled/cae-platform
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl reload nginx
echo "   HTTP config active"

# ── Step 2: Obtain SSL certificate via Certbot ────────────────────────────────
echo ""
echo "[2/3] Obtaining Let's Encrypt SSL certificate..."
sudo certbot --nginx \
  -d "$DOMAIN" \
  --non-interactive \
  --agree-tos \
  --email "$EMAIL" \
  --redirect        # auto-redirect HTTP → HTTPS

echo "   SSL certificate issued for $DOMAIN"

# ── Step 3: Overwrite Nginx config with full HTTPS version ────────────────────
echo ""
echo "[3/3] Writing final HTTPS Nginx config..."

sudo tee "$NGINX_CONF" > /dev/null <<EOF
# Redirect HTTP → HTTPS
server {
    listen 80;
    server_name ${DOMAIN};
    return 301 https://\$host\$request_uri;
}

server {
    listen 443 ssl;
    server_name ${DOMAIN};

    # Let's Encrypt certificates (auto-renewed by certbot timer)
    ssl_certificate     /etc/letsencrypt/live/${DOMAIN}/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/${DOMAIN}/privkey.pem;

    # Modern TLS — disable old insecure protocols
    ssl_protocols             TLSv1.2 TLSv1.3;
    ssl_ciphers               ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache         shared:SSL:10m;
    ssl_session_timeout       1d;

    # Security headers
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;
    add_header X-Content-Type-Options    nosniff                                         always;
    add_header X-Frame-Options           DENY                                            always;

    # ── /health — open, no auth (Kubernetes / AWS ALB health checks) ──────────
    location /health {
        proxy_pass       http://127.0.0.1:8000/health;
        proxy_set_header Host \$host;
        access_log       off;
    }

    # ── /docs and /redoc — Swagger UI (consider restricting in production) ────
    location ~ ^/(docs|redoc|openapi.json) {
        proxy_pass         http://127.0.0.1:8000;
        proxy_set_header   Host            \$host;
        proxy_set_header   X-Real-IP       \$remote_addr;
        proxy_set_header   X-Forwarded-For \$proxy_add_x_forwarded_for;
    }

    # ── /mcp/sse — MCP Server-Sent Events endpoint ────────────────────────────
    # SSE requires: no buffering, long timeout, HTTP/1.1 keep-alive
    location /mcp {
        proxy_pass              http://127.0.0.1:8000/mcp;
        proxy_set_header        Host              \$host;
        proxy_set_header        X-Real-IP         \$remote_addr;
        proxy_set_header        X-Forwarded-For   \$proxy_add_x_forwarded_for;
        proxy_set_header        X-Forwarded-Proto https;
        proxy_set_header        Connection        "";
        proxy_http_version      1.1;
        proxy_read_timeout      3600;
        proxy_send_timeout      3600;
        proxy_buffering         off;
        proxy_cache             off;
        chunked_transfer_encoding on;
    }

    # ── All other API endpoints ───────────────────────────────────────────────
    location / {
        proxy_pass         http://127.0.0.1:8000;
        proxy_set_header   Host              \$host;
        proxy_set_header   X-Real-IP         \$remote_addr;
        proxy_set_header   X-Forwarded-For   \$proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto https;
        proxy_connect_timeout 60s;
        proxy_read_timeout    120s;

        # Rate limiting — prevents abuse of the LLM-backed endpoints
        limit_req zone=api burst=20 nodelay;
    }
}
EOF

# Rate limiting zone (defined at http level)
sudo tee /etc/nginx/conf.d/rate_limit.conf > /dev/null <<EOF
# 10 requests/second per IP for API endpoints
limit_req_zone \$binary_remote_addr zone=api:10m rate=10r/s;
EOF

sudo nginx -t
sudo systemctl reload nginx

# Certbot auto-renewal is enabled by default (systemd timer)
echo "   SSL auto-renewal: $(systemctl is-active certbot.timer 2>/dev/null || echo 'check: sudo certbot renew --dry-run')"

echo ""
echo "================================================================"
echo "  Nginx + SSL setup complete!"
echo "================================================================"
echo ""
echo "  API:          https://${DOMAIN}"
echo "  Swagger UI:   https://${DOMAIN}/docs"
echo "  MCP SSE:      https://${DOMAIN}/mcp/sse"
echo ""
echo "  Claude Desktop config:"
echo "  {"
echo "    \"mcpServers\": {"
echo "      \"cae-platform\": {"
echo "        \"transport\": \"sse\","
echo "        \"url\": \"https://${DOMAIN}/mcp/sse\","
echo "        \"headers\": { \"X-API-Key\": \"your-cae-api-key\" }"
echo "      }"
echo "    }"
echo "  }"
echo "================================================================"
