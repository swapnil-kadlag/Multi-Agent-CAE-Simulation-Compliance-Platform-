#!/bin/bash
# deploy/deploy.sh
# ─────────────────────────────────────────────────────────────────────────────
# Deployment script — run this every time you push a code update.
# Safe to run repeatedly: it pulls latest, rebuilds only changed layers,
# and does a zero-downtime restart via Docker Compose.
#
# Usage (on the EC2 server):
#   cd /opt/cae-platform
#   bash deploy/deploy.sh
#
# First-time deploy only — also rebuild ML artifacts:
#   bash deploy/deploy.sh --rebuild-models
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

APP_DIR="/opt/cae-platform"
REBUILD_MODELS=false

for arg in "$@"; do
  [[ "$arg" == "--rebuild-models" ]] && REBUILD_MODELS=true
done

echo "================================================================"
echo "  CAE NVH Platform — Deploy"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"

cd "$APP_DIR"

# ── 1. Validate .env exists ───────────────────────────────────────────────────
if [[ ! -f ".env" ]]; then
  echo ""
  echo "ERROR: .env file not found."
  echo "  cp .env.example .env && nano .env"
  exit 1
fi

# Check CAE_API_KEY is set
source .env 2>/dev/null || true
if [[ -z "${CAE_API_KEY:-}" ]]; then
  echo ""
  echo "WARNING: CAE_API_KEY is not set in .env"
  echo "  Generate one: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
  echo "  Add to .env:  CAE_API_KEY=<generated-key>"
  echo ""
  read -rp "Continue without API key auth? [y/N] " confirm
  [[ "$confirm" == "y" || "$confirm" == "Y" ]] || exit 1
fi

# ── 2. Pull latest code ───────────────────────────────────────────────────────
echo ""
echo "[1/5] Pulling latest code..."
git pull origin main
echo "   HEAD: $(git log --oneline -1)"

# ── 3. Rebuild ML artifacts if requested or missing ───────────────────────────
echo ""
echo "[2/5] Checking ML artifacts..."

RETRIEVER_OK=false
MODEL_OK=false

[[ -f "data/retriever/faiss.index" ]] && [[ -f "data/retriever/retriever_state.pkl" ]] && RETRIEVER_OK=true
[[ -f "data/surrogate_model.pkl" ]] && MODEL_OK=true

if [[ "$REBUILD_MODELS" == "true" ]] || [[ "$RETRIEVER_OK" == "false" ]] || [[ "$MODEL_OK" == "false" ]]; then
  echo "   Building ML artifacts (this runs inside a temp container)..."
  docker compose run --rm --no-deps cae-api bash -c \
    "python knowledge_base/build_retriever.py && python tools/surrogate_model.py"
  echo "   Artifacts built."
else
  echo "   Artifacts OK (use --rebuild-models to force rebuild)"
fi

# ── 4. Build and start containers ─────────────────────────────────────────────
echo ""
echo "[3/5] Building Docker image..."
docker compose build --no-cache

echo ""
echo "[4/5] Starting containers..."
docker compose up -d --remove-orphans

# ── 5. Health check ───────────────────────────────────────────────────────────
echo ""
echo "[5/5] Waiting for health check..."
sleep 5

MAX_RETRIES=12
for i in $(seq 1 $MAX_RETRIES); do
  STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health || echo "000")
  if [[ "$STATUS" == "200" ]]; then
    echo "   Health check passed (HTTP $STATUS)"
    break
  fi
  echo "   Attempt $i/$MAX_RETRIES — HTTP $STATUS, retrying in 5s..."
  sleep 5
  if [[ $i -eq $MAX_RETRIES ]]; then
    echo "   ERROR: Health check failed after $MAX_RETRIES attempts"
    echo "   Check logs: docker compose logs cae-api --tail 50"
    exit 1
  fi
done

echo ""
echo "================================================================"
echo "  Deploy complete!"
echo "================================================================"
echo ""
echo "  Container: $(docker compose ps --format 'table {{.Name}}\t{{.Status}}' | tail -1)"
echo "  Local:     http://localhost:8000/health"
echo ""
echo "  Useful commands:"
echo "    docker compose logs cae-api -f          # live logs"
echo "    docker compose ps                       # container status"
echo "    docker compose down                     # stop"
echo "    docker stats cae-nvh-api                # resource usage"
echo "================================================================"
