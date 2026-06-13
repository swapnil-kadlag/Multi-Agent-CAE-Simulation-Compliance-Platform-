# Multi-Agent CAE Simulation & Compliance Platform
## Comprehensive Project Report

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Technology Stack](#3-technology-stack)
4. [Component Deep Dive](#4-component-deep-dive)
5. [API Reference](#5-api-reference)
6. [Evaluation Results](#6-evaluation-results)
7. [Test Suite](#7-test-suite)
8. [CI/CD Pipeline](#8-cicd-pipeline)
9. [Step-by-Step Setup Guide](#9-step-by-step-setup-guide)
10. [Deployment Guide (EC2)](#10-deployment-guide-ec2)
11. [Live Demo](#11-live-demo)

---

## 1. Project Overview

A production-grade AI platform for **NVH (Noise-Vibration-Harshness) engineering** that combines:

- **Hybrid RAG** — retrieves from a 50-case NVH knowledge base using FAISS + BM25 + Reciprocal Rank Fusion
- **ML Surrogate Model** — replaces 2-hour FEA simulations with <1ms GradientBoosting predictions
- **LangGraph Multi-Agent System** — 4 specialist agents (RAG, Surrogate, Compliance, Sensor) orchestrated by a smart router
- **FastAPI + MCP Server** — production REST API + Model Context Protocol for AI-to-AI tool calling
- **AWS EC2 Deployment** — Docker + Nginx + Let's Encrypt SSL on Free Tier t2.micro

### Key Metrics

| Metric | Value |
|---|---|
| Retrieval Hit Rate | 95.0% |
| Mean Reciprocal Rank | 95.0% |
| Context Coverage | 91.5% |
| Answer Coverage | 73.0% |
| Avg Retrieval Latency | 9 ms |
| Surrogate vs FEA speedup | ~7,200,000× (2h → <1ms) |
| Test coverage | 100 tests across 4 modules |

---

## 2. Architecture

```
User Query (natural language)
        │
        ▼
┌───────────────────────────────────────────────────────┐
│                   FastAPI + MCP Server                │
│  POST /invoke  POST /diagnose  POST /predict  GET /cases │
└───────────────────────────┬───────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────┐
│              LangGraph StateGraph                     │
│                                                       │
│   ┌─────────────────────────────────────────────┐    │
│   │                  ROUTER                     │    │
│   │  (keyword-based, no LLM, deterministic)     │    │
│   └──────┬──────────┬──────────┬──────────┬────┘    │
│          │          │          │          │          │
│      nvh_query  surrogate  compliance  sensor        │
│          │          │          │          │          │
│          ▼          ▼          ▼          ▼          │
│      ┌───────┐ ┌────────┐ ┌──────────┐ ┌──────┐    │
│      │  RAG  │ │Surrogt.│ │Compliance│ │Sensor│    │
│      │ Agent │ │ Agent  │ │  Agent   │ │Agent │    │
│      └───┬───┘ └───┬────┘ └────┬─────┘ └──┬───┘    │
│          └─────────┴───────────┴───────────┘        │
│                            │                         │
│                      ANSWER NODE                     │
└───────────────────────────────────────────────────────┘
                            │
                            ▼
                     Markdown Response
```

### Data Flow (4 Progressive Steps)

```
Step 1 ──► Synthetic Data Generation
           50 NVH cases + sensor readings + motor simulation CSV

Step 2 ──► Hybrid RAG Retriever
           FAISS (dense) + BM25 (sparse) + RRF fusion
           Parent-child chunking: 50 parents → 200 children

Step 3 ──► LangGraph Multi-Agent Graph
           Router → 4 specialist agents → unified answer

Step 4 ──► FastAPI + MCP Production API
           REST endpoints + Model Context Protocol SSE
```

---

## 3. Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Orchestration** | LangGraph 0.2 | Multi-agent StateGraph |
| **RAG — Dense** | FAISS + TF-IDF | Semantic vector search |
| **RAG — Sparse** | BM25 (rank-bm25) | Keyword/exact-term search |
| **RAG — Fusion** | Reciprocal Rank Fusion | Combine dense + sparse rankings |
| **Surrogate Model** | GradientBoostingRegressor | NVH level prediction (<1ms) |
| **ML Pipeline** | scikit-learn Pipeline | StandardScaler + GBR |
| **API Framework** | FastAPI | Async REST API |
| **MCP Server** | fastapi-mcp | Model Context Protocol SSE |
| **Tracing** | LangSmith | Agent execution traces |
| **Experiment tracking** | Weights & Biases | Model training metrics |
| **Testing** | pytest | 100-test suite |
| **Evaluation** | RAGAS | LLM-as-Judge + offline metrics |
| **Containerisation** | Docker + Docker Compose | Reproducible deployment |
| **Reverse proxy** | Nginx | SSL termination + rate limiting |
| **SSL** | Let's Encrypt (Certbot) | Auto-renewing HTTPS |
| **CI/CD** | GitHub Actions | Build → Test → Eval on push |
| **Cloud** | AWS EC2 t2.micro | Free Tier production deployment |
| **Language** | Python 3.12 | All components |

---

## 4. Component Deep Dive

### 4.1 Hybrid RAG Retriever (`knowledge_base/build_retriever.py`)

**Problem:** Standard single-retriever RAG misses exact technical terms like `BPFI`, `ISO 15243`, or `847 Hz`.

**Solution:** Hybrid retrieval combining two complementary approaches:

```
Query: "bearing BPFI inner race defect at 2400 Hz"
         │
         ├──► FAISS (TF-IDF + cosine)  → finds semantically similar docs
         │         ranked list [A, C, B, D, E]
         │
         └──► BM25 (Okapi BM25)        → finds exact keyword matches
                   ranked list [C, A, E, B, D]
                         │
                         ▼
               Reciprocal Rank Fusion
               score(doc) = Σ 1/(k + rank_i)  where k=60
                         │
                         ▼
               Final ranked list [C, A, B, E, D]
```

**Parent-Child Chunking:**
- Each of the 50 NVH cases is split into 4 child chunks:
  - `context` — title + description (broad)
  - `root_cause` — diagnostic precision
  - `action` — remediation steps
  - `compliance` — standards + component
- Children are indexed (smaller = more precise retrieval)
- On retrieval: child chunk matched → parent document returned (full context)
- Result: 50 parents × 4 children = **200 indexed chunks**

**Performance:**
- Hit Rate: **95.0%** (known case appears in top-5 results)
- MRR: **95.0%**
- Avg latency: **9ms** per query

---

### 4.2 GradientBoosting Surrogate Model (`tools/surrogate_model.py`)

**Problem:** Real FEA (Finite Element Analysis) simulations take 2 hours per run. Too slow for interactive agent queries.

**Solution:** Train a GradientBoosting ML model on synthetic simulation data to predict NVH level in <1ms.

**Input features:**

| Feature | Unit | Range |
|---|---|---|
| `rpm` | rev/min | 500 – 15,000 |
| `load_nm` | Newton-metres | 5 – 300 |
| `temperature_c` | °C | 0 – 120 |
| `stator_slots` | count | 24 – 72 |
| `rotor_poles` | count | 4 – 12 |
| `air_gap_mm` | mm | 0.3 – 2.0 |

**Output:**

```json
{
  "predicted_nvh_db": 89.4,
  "severity": 4,
  "assessment": "High — design review recommended",
  "bpf_hz": 300.0,
  "operating_point": {"rpm": 3000, "load_nm": 80, ...},
  "model_metrics": {"rmse": 2.1, "r2": 0.94, "cv_r2": 0.93}
}
```

**BPF (Blade Pass Frequency) formula:**
```
BPF (Hz) = (rpm / 60) × stator_slots / rotor_poles
```

**Severity scale:**

| Level | dB Range | Assessment |
|---|---|---|
| 1 | < 60 | Acceptable |
| 2 | 60–70 | Monitor |
| 3 | 70–80 | Investigation needed |
| 4 | 80–90 | High — design review |
| 5 | > 90 | Critical — immediate action |

---

### 4.3 LangGraph Multi-Agent System (`agents/cae_graph.py`)

**State (shared whiteboard):**
```python
class CAEState(TypedDict):
    query:    str        # original user question
    route:    str        # which agent was selected
    context:  list       # retrieved documents (RAG only)
    answer:   str        # final markdown response
```

**Router (keyword-based, no LLM cost):**

| Keywords detected | Route | Agent |
|---|---|---|
| `rpm`, `predict`, `noise level`, `db` | `surrogate` | Surrogate Agent |
| `compliance`, `ISO`, `standard`, `limit` | `compliance` | Compliance Agent |
| `sensor`, `vibration`, `reading`, `bearing fault` | `sensor` | Sensor Agent |
| everything else | `nvh_query` | RAG Agent |

**Agents:**

- **RAG Agent** — retrieves top-5 documents from hybrid retriever, formats as structured markdown with case ID, score, root cause, recommended action
- **Surrogate Agent** — extracts `rpm` and `load_nm` from query text using regex, calls `predict_nvh_level()`, returns prediction with severity and BPF
- **Compliance Agent** — inline knowledge base for ISO 362, IEC 60704, ISO 2631, ISO 10816, ISO 15243
- **Sensor Agent** — parses sensor readings (frequency, amplitude, bearing ID) and generates diagnostic report

**LangSmith Tracing:** Each agent node is decorated with `@traceable`, creating child spans visible at smith.langchain.com under project `cae-nvh-platform`.

---

### 4.4 FastAPI + MCP Server (`api/main.py`)

- **CORS middleware** — configurable via `CORS_ORIGINS` env var (default: `*`)
- **Auth middleware** — `X-API-Key` header required on `/invoke`, `/diagnose`, `/predict` when `CAE_API_KEY` is set
- **Lazy loading** — retriever and graph load on first request (saves ~400MB RAM during cold start on t2.micro)
- **MCP SSE** — `/mcp/sse` endpoint exposes all tools to AI agents via Model Context Protocol

---

## 5. API Reference

**Base URL:** `https://cae-platform.duckdns.org`  
**Auth:** `X-API-Key: <your-key>` header on protected endpoints

### Endpoints

#### `GET /health`
Returns system status. No auth required.
```json
{
  "status": "healthy",
  "uptime_seconds": 3600,
  "components": {
    "langgraph": "ready",
    "retriever": "ready",
    "surrogate": "not_loaded",
    "knowledge_base": "50 cases",
    "api": "ready"
  },
  "version": "1.0.0"
}
```

#### `POST /invoke` 🔒
Runs the full multi-agent LangGraph pipeline.
```json
// Request
{ "query": "electric motor whine at 3000 rpm" }

// Response
{
  "query": "electric motor whine at 3000 rpm",
  "route": "surrogate",
  "answer": "## NVH Prediction...",
  "latency_ms": 12
}
```

#### `POST /diagnose` 🔒
Diagnose from a sensor reading object.
```json
// Request
{
  "frequency_hz": 2400,
  "amplitude_db": 85,
  "component": "bearing",
  "bearing_id": "SKF-6205"
}
```

#### `POST /predict` 🔒
Direct surrogate model prediction.
```json
// Request
{ "rpm": 3000, "load_nm": 80, "temperature_c": 25 }

// Response
{ "predicted_nvh_db": 89.4, "severity": 4, "bpf_hz": 300.0, ... }
```

#### `GET /cases`
Browse NVH knowledge base with optional filters.
```
GET /cases?component=bearing&freq_range=high&limit=10
```

#### `GET /mcp/sse`
Model Context Protocol SSE endpoint — exposes all tools to AI agents (Claude Desktop, LangChain, etc.).

---

## 6. Evaluation Results

Evaluated on 20 golden Q&A pairs from `data/synthetic/golden_qa.json`.

### Offline Mode (keyword-overlap metrics)

| Metric | Score | Description |
|---|---|---|
| **Hit Rate** | **0.950** | Known case in top-5 results |
| **MRR** | **0.950** | Mean Reciprocal Rank |
| **Context Coverage** | **0.915** | Query terms covered by retrieved context |
| **Answer Coverage** | **0.730** | Answer terms present in context |
| **Avg Latency** | **9 ms** | End-to-end retrieval + answer |

### LLM-as-Judge Mode (RAGAS with GPT-4o)

Run with:
```bash
python eval/run_eval.py   # requires OPENAI_API_KEY in .env
```

Target: `faithfulness ≥ 0.85`

Metrics evaluated:
- `context_precision` — are retrieved chunks actually relevant?
- `context_recall` — are all relevant facts covered?
- `faithfulness` — does the answer only assert things in the context?
- `answer_relevancy` — does the answer address the question?

Results saved to `eval/eval_results.json`.

---

## 7. Test Suite

100 tests across 4 modules. Run with:
```bash
pytest tests/ -v
```

| File | Tests | Covers |
|---|---|---|
| `tests/test_retriever.py` | 17 | Load, retrieve, scores, filters, hit-rate |
| `tests/test_surrogate.py` | 22 | Load, predictions, BPF formula, severity, edge cases |
| `tests/test_graph.py` | 28 | Graph compile, router (9 queries), each agent |
| `tests/test_api.py` | 33 | Health, invoke, diagnose, predict, cases, CORS |

### Key test cases

```python
# BPF formula verification
expected_bpf = round((3000 / 60) * 36 / 6, 1)  # = 300.0 Hz
assert abs(result["bpf_hz"] - expected_bpf) < 1.0

# Hit-rate smoke test
results = retriever.retrieve("bearing BPFI inner race defect 2400 Hz", top_k=5)
assert "bearing" in [r.metadata["component"] for r in results]

# Router correctness
assert graph.invoke({"query": "predict NVH at 3000 rpm 80 Nm"})["route"] == "surrogate"
assert graph.invoke({"query": "ISO 362 noise limit"})["route"] == "compliance"
```

---

## 8. CI/CD Pipeline

**File:** `.github/workflows/ci.yml`  
**Trigger:** every push to `main` or `develop`, every PR to `main`

```
Push to GitHub
      │
      ▼
GitHub Actions (ubuntu-latest, Python 3.12)
      │
      ├── pip install -r requirements.txt
      ├── python data/generate_synthetic_data.py
      ├── python build_all.py          ← builds retriever + surrogate
      ├── pytest tests/ -v --tb=short  ← 100 tests
      ├── python eval/run_eval.py      ← eval metrics
      └── Upload eval_results.json as artifact
```

---

## 9. Step-by-Step Setup Guide

### Prerequisites

- Python 3.12
- Git

### Step 1 — Clone the repository

```bash
git clone https://github.com/swapnil-kadlag/Multi-Agent-CAE-Simulation-Compliance-Platform-.git
cd Multi-Agent-CAE-Simulation-Compliance-Platform-
```

### Step 2 — Create virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> On Windows, use `$env:PYTHONIOENCODING="utf-8"; python -X utf8` prefix for all Python commands to avoid Unicode encoding errors.

### Step 4 — Configure environment

```bash
cp .env.example .env
# Edit .env — minimum required: leave CAE_API_KEY empty for local dev (disables auth)
```

### Step 5 — Generate synthetic data

```bash
python data/generate_synthetic_data.py
```

This creates:
- `data/synthetic/nvh_knowledge_base.json` — 50 NVH engineering cases
- `data/synthetic/sensor_readings.json` — simulated sensor data
- `data/synthetic/motor_simulation.csv` — training data for surrogate model
- `data/synthetic/golden_qa.json` — 20 Q&A pairs for evaluation

### Step 6 — Build ML artifacts

```bash
python build_all.py
```

> **Important:** Always use `build_all.py`, never run `knowledge_base/build_retriever.py` or `tools/surrogate_model.py` directly. Running them as `__main__` causes pickle to serialize class names as `__main__.NVHDocument` which breaks loading from other modules.

This builds:
- `data/retriever/faiss.index` — FAISS vector index
- `data/retriever/retriever_state.pkl` — BM25 + metadata
- `data/surrogate_model.pkl` — trained GradientBoosting pipeline

### Step 7 — Run tests

```bash
pytest tests/ -v
```

All 100 tests should pass.

### Step 8 — Start the API server

```bash
uvicorn api.main:app --reload --port 8000
```

### Step 9 — Open Swagger UI

Go to: http://localhost:8000/docs

- Click **Authorize** (top right)
- Enter your `CAE_API_KEY` (or leave blank if not set)
- Try `/health`, `/invoke`, `/predict`

### Step 10 — Run evaluation

```bash
# Offline mode (no API key needed)
python eval/run_eval.py

# LLM-as-Judge mode
# Add OPENAI_API_KEY to .env first, then:
python eval/run_eval.py
```

Results saved to `eval/eval_results.json`.

---

## 10. Deployment Guide (EC2)

### Infrastructure

| Item | Detail |
|---|---|
| Instance | AWS EC2 t2.micro (Free Tier) |
| OS | Ubuntu 22.04 LTS |
| RAM | 1 GB + 2 GB swap (for Docker builds) |
| Storage | 20 GB gp3 EBS |
| Security groups | 22 (SSH), 80 (HTTP), 443 (HTTPS) |

### First-time EC2 setup

```bash
# 1. Upload setup script
scp -i your-key.pem deploy/setup_ec2.sh ubuntu@<EC2-IP>:~/

# 2. SSH and run
ssh -i your-key.pem ubuntu@<EC2-IP>
tmux new -s deploy          # use tmux to survive disconnects
bash setup_ec2.sh

# 3. Log out and back in (docker group needs fresh session)
exit
ssh -i your-key.pem ubuntu@<EC2-IP>
```

This installs: Docker, Docker Compose, Nginx, Certbot, Git, ufw firewall rules.

### Clone and configure

```bash
cd /opt/cae-platform
git clone https://github.com/swapnil-kadlag/Multi-Agent-CAE-Simulation-Compliance-Platform- .
cp .env.example .env
nano .env    # set CAE_API_KEY to a secure random value
```

Generate a secure API key:
```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

### First deploy (builds artifacts + Docker image)

```bash
bash deploy/deploy.sh --rebuild-models
```

### Subsequent deploys (after a code push)

```bash
bash deploy/deploy.sh
```

### SSL setup (requires a domain)

```bash
# Point your domain's A record to the EC2 IP first, then:
bash deploy/nginx_setup.sh your-domain.com your@email.com
```

SSL certificate is issued by Let's Encrypt and auto-renews via systemd timer.

### Useful operations

```bash
# Check container status
docker compose ps

# Live logs
docker compose logs cae-api -f

# Resource usage (critical on t2.micro)
docker stats cae-nvh-api

# Restart
docker compose restart

# Stop
docker compose down
```

---

## 11. Live Demo

| | |
|---|---|
| **Health** | https://cae-platform.duckdns.org/health |
| **Swagger UI** | https://cae-platform.duckdns.org/docs |
| **MCP SSE** | https://cae-platform.duckdns.org/mcp/sse |
| **GitHub** | https://github.com/swapnil-kadlag/Multi-Agent-CAE-Simulation-Compliance-Platform- |

### Quick test via curl

```bash
# Health check
curl https://cae-platform.duckdns.org/health

# NVH prediction
curl -X POST https://cae-platform.duckdns.org/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <your-api-key>" \
  -d '{"rpm": 3000, "load_nm": 80}'

# Multi-agent query
curl -X POST https://cae-platform.duckdns.org/invoke \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <your-api-key>" \
  -d '{"query": "bearing defect noise at 2400 Hz"}'
```

### Claude Desktop MCP Integration

Add to Claude Desktop `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "cae-platform": {
      "transport": "sse",
      "url": "https://cae-platform.duckdns.org/mcp/sse",
      "headers": { "X-API-Key": "<your-api-key>" }
    }
  }
}
```

Claude can then call NVH diagnosis, surrogate prediction, and compliance lookup directly as tools.

---

*Generated: 2026-06-14*
