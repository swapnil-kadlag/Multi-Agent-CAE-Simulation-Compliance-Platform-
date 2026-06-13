# ── Stage 1: builder — install deps into a clean venv ─────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# System libs needed by faiss-cpu and WeasyPrint (if used later)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime — copy only what's needed ────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# libgomp1 is required at runtime by faiss-cpu (OpenMP threading)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY build_all.py build_all.py
COPY agents/     agents/
COPY api/        api/
COPY data/       data/
COPY eval/       eval/
COPY knowledge_base/ knowledge_base/
COPY tools/      tools/

# Copy env template (actual secrets injected at runtime via env vars or secrets manager)
COPY .env.example .env.example

# Pre-built indexes must exist — rebuild them if not present
# Run:  python knowledge_base/build_retriever.py
#       python tools/surrogate_model.py
# before building this image, OR mount a volume with data/retriever/ and data/surrogate_model.pkl

EXPOSE 8000

# Uvicorn with:
#   --workers 1        single worker (stateful lazy-loaded models are not fork-safe)
#   --host 0.0.0.0     bind to all interfaces (required in Docker)
#   --port 8000
#   --log-level info
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
