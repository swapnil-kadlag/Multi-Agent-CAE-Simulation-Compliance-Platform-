#!/bin/bash
# deploy/setup_ec2.sh
# ─────────────────────────────────────────────────────────────────────────────
# One-time bootstrap script for a fresh AWS EC2 instance.
# Tested on: Ubuntu 22.04 LTS (ami-0c7217cdde317cfec in us-east-1)
#
# Instance:   t2.micro (1 vCPU, 1 GB RAM) — AWS Free Tier eligible
# Storage:    20 GB gp3 EBS
# Ports:      22 (SSH your IP only), 80 (HTTP), 443 (HTTPS)
#
# NOTE: t2.micro has only 1 GB RAM. This script adds a 2 GB swap file so
# Docker image builds (pip install of heavy ML packages) don't OOM-kill.
# At runtime the app uses ~600-700 MB and fits comfortably in 1 GB.
#
# Run ONCE after launching your EC2 instance:
#   scp -i your-key.pem deploy/setup_ec2.sh ubuntu@<EC2-IP>:~/
#   ssh -i your-key.pem ubuntu@<EC2-IP>
#   bash setup_ec2.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail   # exit on any error, undefined var, or pipe failure

echo "================================================================"
echo "  CAE NVH Platform — EC2 Bootstrap (Free Tier / t2.micro)"
echo "================================================================"

# ── 0. Swap file — CRITICAL for t2.micro ─────────────────────────────────────
# Docker's pip install of faiss-cpu + langchain + scikit-learn needs ~1.5 GB
# temporarily. Without swap the OOM killer terminates the build mid-way.
# 2 GB swap on gp3 SSD is fast enough; it's only hit during build, not runtime.
echo ""
echo "[0/7] Creating 2 GB swap file (required for Docker build on 1 GB RAM)..."
if [[ ! -f /swapfile ]]; then
  sudo fallocate -l 2G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
  echo "   Swap enabled: $(free -h | grep Swap)"
else
  echo "   Swap already exists — skipping"
fi

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
