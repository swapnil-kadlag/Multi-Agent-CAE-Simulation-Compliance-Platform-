# CAE NVH Multi-Agent Platform — Step 4: FastAPI + MCP Server

## Project Structure

```
cae_platform/
├── api/
│   └── main.py              ← STEP 4 — FastAPI + MCP server
├── agents/
│   └── cae_graph.py         ← Step 3 — LangGraph multi-agent graph
├── tools/
│   └── surrogate_model.py   ← Step 3 — ML surrogate model
├── knowledge_base/
│   └── build_retriever.py   ← Step 2 — Hybrid FAISS+BM25 retriever
├── data/
│   ├── generate_synthetic_data.py
│   ├── surrogate_model.pkl  (pre-trained)
│   ├── retriever/           (pre-built FAISS + BM25 indexes)
│   └── synthetic/           (all 4 datasets)
└── requirements.txt
```

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Run tests (no server needed)
python api/main.py

# 3. Start live server
uvicorn api.main:app --reload --port 8000

# 4. Open Swagger UI
open http://localhost:8000/docs
```

## API Endpoints

| Method | Endpoint    | Description                          |
|--------|-------------|--------------------------------------|
| GET    | /health     | System status + component check      |
| POST   | /invoke     | Run full multi-agent LangGraph       |
| POST   | /diagnose   | Diagnose from sensor reading         |
| POST   | /predict    | ML surrogate NVH prediction          |
| GET    | /cases      | Browse NVH knowledge base            |
| GET    | /mcp        | MCP server schema                    |
| GET    | /docs       | Swagger UI                           |

## MCP Server

The /mcp endpoint exposes your tools to other AI agents via the Model Context Protocol.
For full MCP protocol: pip install fastapi-mcp

## Interview Question

"What is MCP and why does this platform use it?"

Answer: Model Context Protocol is a standard for exposing AI tools to
other AI agents. Like REST is for humans calling APIs, MCP is for AI
agents calling other AI agents' tools. When an engineering assistant
needs NVH diagnosis, it calls /mcp instead of a custom integration.
One standard — any AI agent can discover and call any MCP server.
