# Deployment Status

## Live EC2 Instance

| Item | Value |
|---|---|
| Provider | AWS EC2 t2.micro (Free Tier) |
| Region | eu-north-1 (Stockholm) |
| Public IP | 13.60.43.34 |
| OS | Ubuntu 22.04 LTS |
| Deployed | 2026-06-13 |
| Last updated | 2026-06-14 |

## Access

| Endpoint | URL |
|---|---|
| Health check | https://cae-platform.duckdns.org/health |
| Swagger UI | https://cae-platform.duckdns.org/docs |
| MCP SSE | https://cae-platform.duckdns.org/mcp/sse |

> HTTP (port 80) auto-redirects to HTTPS. SSL certificate issued by Let's Encrypt, expires 2026-09-11, auto-renewed via certbot systemd timer.

**API Key:** stored in `/opt/cae-platform/.env` as `CAE_API_KEY`  
**SSH key:** `cae-key.pem` — keep this file safe, do not commit it

## SSH Access

```bash
ssh -i "path/to/cae-key.pem" ubuntu@13.60.43.34
```

Use tmux to prevent disconnection during long operations:
```bash
tmux new -s work        # start session
tmux attach -t work     # re-attach after disconnect
```

## Stack

| Component | Detail |
|---|---|
| Container runtime | Docker 29.5.3 |
| Orchestration | Docker Compose (`docker-compose.yml`) |
| Reverse proxy | Nginx 1.28.3 |
| App server | Uvicorn (single worker, port 8000) |
| SSL | Not configured (add via `deploy/nginx_setup.sh`) |

## How to Update After a Code Push

```bash
ssh -i "cae-key.pem" ubuntu@13.60.43.34
cd /opt/cae-platform
bash deploy/deploy.sh
```

To also rebuild ML artifacts (retriever + surrogate model):
```bash
bash deploy/deploy.sh --rebuild-models
```

## How to Check Status

```bash
# Container health
docker compose ps
docker stats cae-nvh-api

# Live logs
docker compose logs cae-api -f

# API health
curl http://localhost:8000/health
```

## Repository

GitHub: https://github.com/swapnil-kadlag/Multi-Agent-CAE-Simulation-Compliance-Platform-

The EC2 instance is linked to the repo via git remote:
```
remote: origin → https://github.com/swapnil-kadlag/Multi-Agent-CAE-Simulation-Compliance-Platform-
branch: main
app dir: /opt/cae-platform
```

## Important Notes

- **No swap needed at runtime** — app uses ~600-700 MB, fits in 1 GB RAM. Swap (2 GB) is only hit during Docker image builds.
- **Lazy loading** — `/health` shows `not_loaded` for retriever/surrogate on cold start; they load on first request.
- **Artifacts on host** — ML artifacts are volume-mounted from `/opt/cae-platform/data/` into the container (not baked into the image). Safe across image rebuilds.
- **Never run `build_retriever.py` directly** — always use `python build_all.py` to avoid pickle `__main__` class path issues.

## Optional Next Steps

1. **SSL certificate** — ✅ Done. Domain: `cae-platform.duckdns.org`, cert expires 2026-09-11 (auto-renewed).

2. **Real API keys** — edit `/opt/cae-platform/.env` and add:
   - `OPENAI_API_KEY` — for LLM-as-Judge evaluation mode
   - `LANGCHAIN_API_KEY` — for LangSmith tracing

3. **GitHub Actions CI** — already configured in `.github/workflows/ci.yml`. Runs 100 tests + eval on every push to `main`.
