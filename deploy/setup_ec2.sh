#!/bin/bash
# deploy/setup_ec2.sh
# ─────────────────────────────────────────────────────────────────────────────
# One-time bootstrap script for a fresh AWS EC2 instance.
# Tested on: Ubuntu 22.04 LTS (ami-0c7217cdde317cfec in us-east-1)
#
# Recommended instance:  t3.medium (2 vCPU, 4 GB RAM) — handles FAISS + model
# Storage:               20 GB gp3 EBS
# Security Group ports:  22 (SSH), 80 (HTTP), 443 (HTTPS)
#
# Run ONCE after launching your EC2 instance:
#   chmod +x deploy/setup_ec2.sh
#   scp -i your-key.pem deploy/setup_ec2.sh ubuntu@<EC2-IP>:~/
#   ssh -i your-key.pem ubuntu@<EC2-IP>
#   bash setup_ec2.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail   # exit on any error, undefined var, or pipe failure

echo "================================================================"
echo "  CAE NVH Platform — EC2 Bootstrap"
echo "================================================================"

# ── 1. System update ─────────────────────────────────────────────────────────
echo ""
echo "[1/7] Updating system packages..."
sudo apt-get update -qq
sudo apt-get upgrade -y -qq

# ── 2. Install Docker ────────────────────────────────────────────────────────
echo ""
echo "[2/7] Installing Docker..."
sudo apt-get install -y -qq ca-certificates curl gnupg lsb-release

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update -qq
sudo apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Let ubuntu user run docker without sudo
sudo usermod -aG docker ubuntu
echo "   Docker $(docker --version) installed"

# ── 3. Install Nginx ─────────────────────────────────────────────────────────
echo ""
echo "[3/7] Installing Nginx..."
sudo apt-get install -y -qq nginx
sudo systemctl enable nginx
echo "   Nginx $(nginx -v 2>&1) installed"

# ── 4. Install Certbot (Let's Encrypt SSL) ───────────────────────────────────
echo ""
echo "[4/7] Installing Certbot..."
sudo apt-get install -y -qq certbot python3-certbot-nginx
echo "   Certbot $(certbot --version) installed"

# ── 5. Install Git ───────────────────────────────────────────────────────────
echo ""
echo "[5/7] Installing Git..."
sudo apt-get install -y -qq git
echo "   Git $(git --version) installed"

# ── 6. Configure firewall (ufw) ──────────────────────────────────────────────
echo ""
echo "[6/7] Configuring firewall..."
sudo ufw allow OpenSSH
sudo ufw allow 'Nginx Full'   # opens 80 (HTTP) and 443 (HTTPS)
sudo ufw --force enable
echo "   Firewall rules: SSH + HTTP(80) + HTTPS(443) allowed"

# ── 7. Create app directory ──────────────────────────────────────────────────
echo ""
echo "[7/7] Creating app directory..."
sudo mkdir -p /opt/cae-platform
sudo chown ubuntu:ubuntu /opt/cae-platform
echo "   App directory: /opt/cae-platform"

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Bootstrap complete!"
echo "================================================================"
echo ""
echo "NEXT STEPS:"
echo ""
echo "  1. Clone your repo:"
echo "     cd /opt/cae-platform"
echo "     git clone <your-repo-url> ."
echo ""
echo "  2. Copy environment file:"
echo "     cp .env.example .env"
echo "     nano .env   # fill in your API keys + CAE_API_KEY"
echo ""
echo "  3. Set up Nginx (run deploy/nginx_setup.sh):"
echo "     bash deploy/nginx_setup.sh your-domain.com"
echo ""
echo "  4. Start the platform:"
echo "     bash deploy/deploy.sh"
echo ""
echo "  Note: Log out and back in for docker group to take effect."
echo "================================================================"
