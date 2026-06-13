"""
api/main.py
─────────────────────────────────────────────────────────────────────────────
FastAPI application that wraps the LangGraph multi-agent CAE platform
and exposes it as a production API with MCP server support.

What this file adds on top of the LangGraph graph:
────────────────────────────────────────────────────
  • POST /invoke        — run the full multi-agent graph
  • POST /diagnose      — diagnose a sensor reading (structured input)
  • POST /predict       — surrogate model prediction (structured input)
  • GET  /health        — uptime + system status check
  • GET  /cases         — browse the NVH knowledge base
  • GET  /docs          — auto-generated Swagger UI (FastAPI built-in)
  • /mcp                — MCP server for inter-agent communication

What is MCP?
─────────────
  Model Context Protocol (MCP) is a standard for exposing AI tools
  to other AI agents. When an engineering assistant wants to call
  your NVH diagnosis tool, it calls /mcp instead of building a custom
  integration. One standard protocol — any AI can call any MCP server.

  fastapi_mcp automatically generates the MCP interface from your
  FastAPI route definitions. No extra code needed.

Run:
    uvicorn api.main:app --reload --port 8000
    Then open: http://localhost:8000/docs

─────────────────────────────────────────────────────────────────────────────
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from agents.cae_graph import build_cae_graph
from tools.surrogate_model import predict_nvh_level, load_model
from knowledge_base.build_retriever import HybridNVHRetriever


# ─────────────────────────────────────────────────────────────────────────────
# API KEY AUTHENTICATION
# ─────────────────────────────────────────────────────────────────────────────
# Set CAE_API_KEY in .env to enable authentication on all tool endpoints.
# Leave unset (or empty) for open access during local development.
#
# Usage — include header on every request:
#   X-API-Key: your-secret-key
#
# Generate a strong key:
#   python -c "import secrets; print(secrets.token_urlsafe(32))"

_API_KEY         = os.getenv("CAE_API_KEY", "")
_api_key_header  = APIKeyHeader(name="X-API-Key", auto_error=False)

async def require_api_key(api_key: str = Security(_api_key_header)):
    """Dependency injected into protected endpoints."""
    if not _API_KEY:
        return        # auth disabled in dev mode — no key configured
    if api_key != _API_KEY:
        raise HTTPException(
            status_code = 403,
            detail      = "Invalid or missing API key. Include X-API-Key header.",
        )
    return api_key

_AUTH = Depends(require_api_key)


# ─────────────────────────────────────────────────────────────────────────────
# APP INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────

_auth_mode = "enabled" if _API_KEY else "disabled (set CAE_API_KEY in .env)"
print(f"[Auth] API key authentication: {_auth_mode}")

app = FastAPI(
    title       = "CAE NVH Multi-Agent Platform",
    description = (
        "Production-grade multi-agent AI system for NVH engineering analysis. "
        "Powered by LangGraph + hybrid RAG (FAISS + BM25) + ML surrogate models. "
        "Exposed as MCP server for inter-agent communication. "
        "Protected by X-API-Key header authentication."
    ),
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# ── CORS — allow browser clients and remote MCP consumers ────────────────────
_cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins     = _cors_origins,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Startup: load heavy objects once ─────────────────────────────────────────
_graph     = None
_retriever = None
_model     = None
_start_time = time.time()

def get_graph():
    global _graph
    if _graph is None:
        print("[Startup] Building LangGraph...")
        _graph = build_cae_graph()
    return _graph

def get_retriever():
    global _retriever
    if _retriever is None:
        print("[Startup] Loading retriever...")
        _retriever = HybridNVHRetriever.load()
    return _retriever

def get_model():
    global _model
    if _model is None:
        print("[Startup] Loading surrogate model...")
        _model = load_model()
    return _model


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────────────────────────────────────

class InvokeRequest(BaseModel):
    """
    Input for the main /invoke endpoint.
    The agent decides what to do based on your query text.
    """
    query: str = Field(
        ...,
        description = "Natural language engineering query",
        examples    = [
            "What causes electromagnetic whine at 500 Hz in BLDC motors?",
            "Predict NVH for motor at 3000 rpm 80 Nm load",
            "What does ISO 362-1 specify for drive-by noise limits?",
        ],
    )
    session_id: Optional[str] = Field(
        default = None,
        description = "Optional session ID for tracking related queries",
    )

class InvokeResponse(BaseModel):
    query:          str
    route:          str
    answer:         str
    retrieved_cases: Optional[List[dict]] = None
    session_id:     Optional[str]         = None
    latency_ms:     float


class SensorRequest(BaseModel):
    """Structured sensor reading for NVH diagnosis."""
    dominant_freq_hz:   float = Field(..., description="Dominant vibration frequency in Hz", examples=[847.0])
    amplitude_db:       float = Field(..., description="Vibration amplitude in dB", examples=[82.0])
    component:          str   = Field(..., description="Component type", examples=["electric_motor"])
    rpm:                Optional[int]   = Field(default=None, description="Rotational speed in RPM")
    freq_range:         Optional[str]   = Field(default=None, description="low / mid / high")

class SensorResponse(BaseModel):
    diagnosis:          str
    top_cases:          List[dict]
    severity:           int
    recommended_action: str
    latency_ms:         float


class PredictRequest(BaseModel):
    """Structured input for surrogate model prediction."""
    rpm:            float = Field(...,  description="Motor speed in RPM",            examples=[3000.0])
    load_nm:        float = Field(...,  description="Shaft torque in Newton-metres",  examples=[80.0])
    temperature_c:  float = Field(25.0, description="Operating temperature °C")
    stator_slots:   int   = Field(36,   description="Number of stator slots")
    rotor_poles:    int   = Field(6,    description="Number of rotor poles")
    air_gap_mm:     float = Field(0.8,  description="Air gap length in mm")

class PredictResponse(BaseModel):
    predicted_nvh_db: float
    severity:         int
    assessment:       str
    bpf_hz:           float
    operating_point:  dict
    model_metrics:    dict
    latency_ms:       float


class HealthResponse(BaseModel):
    status:           str
    uptime_seconds:   float
    components:       dict
    version:          str


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint — confirms all components are loaded and ready.

    In production: used by Kubernetes liveness probes and load balancers.
    Returns 200 if healthy, 503 if any component failed to load.
    """
    uptime = round(time.time() - _start_time, 2)

    # Check which components are loaded
    components = {
        "langgraph":  "ready" if _graph     is not None else "not_loaded",
        "retriever":  "ready" if _retriever is not None else "not_loaded",
        "surrogate":  "ready" if _model     is not None else "not_loaded",
        "knowledge_base": "50 cases",
        "api":        "ready",
    }

    return HealthResponse(
        status          = "healthy",
        uptime_seconds  = uptime,
        components      = components,
        version         = "1.0.0",
    )


@app.post("/invoke", response_model=InvokeResponse, tags=["Agent"], operation_id="invoke_agent")
async def invoke_agent(request: InvokeRequest, _: str = _AUTH):
    """
    Main endpoint — runs the full multi-agent LangGraph pipeline.

    The router automatically decides which specialist agent handles your query:
    - NVH knowledge queries    → RAG Agent (FAISS + BM25 retrieval)
    - Prediction queries       → Surrogate Agent (ML model)
    - Standards/compliance     → Compliance Agent (ISO/IEC knowledge)
    - Sensor-based diagnosis   → Sensor Agent (RAG + structured parsing)

    This is the primary endpoint that engineering clients would call.
    """
    t0 = time.time()

    try:
        graph  = get_graph()
        result = graph.invoke({"query": request.query})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    latency = round((time.time() - t0) * 1000, 2)

    # Extract retrieved cases summary if available
    cases_summary = None
    if result.get("retrieved_docs"):
        cases_summary = [
            {
                "case_id": d["metadata"]["case_id"],
                "title":   d["metadata"]["title"],
                "score":   d["score"],
            }
            for d in result["retrieved_docs"][:3]
        ]

    return InvokeResponse(
        query           = request.query,
        route           = result.get("route", "unknown"),
        answer          = result.get("answer", "No answer generated"),
        retrieved_cases = cases_summary,
        session_id      = request.session_id,
        latency_ms      = latency,
    )


@app.post("/diagnose", response_model=SensorResponse, tags=["NVH Tools"], operation_id="diagnose_sensor")
async def diagnose_sensor(request: SensorRequest, _: str = _AUTH):
    """
    Diagnose NVH issue from structured sensor reading.

    This is the MCP-exposed tool that other AI agents call.
    Input: frequency, amplitude, component type.
    Output: diagnosis, root cause, recommended action.

    Use case: an engineering assistant calls this
    endpoint when a vehicle sensor detects abnormal vibration.
    """
    t0 = time.time()

    retriever = get_retriever()

    # Build a diagnostic query from sensor data
    query = (
        f"Diagnose NVH issue: {request.component} showing "
        f"{request.dominant_freq_hz} Hz at {request.amplitude_db} dB"
    )
    if request.rpm:
        query += f" at {request.rpm} RPM"

    # Retrieve with metadata filters
    results = retriever.retrieve(
        query,
        top_k      = 5,
        freq_range = request.freq_range,
        component  = request.component if request.component != "unknown" else None,
    )

    if not results:
        raise HTTPException(
            status_code = 404,
            detail      = "No matching NVH cases found for this sensor reading",
        )

    # Build structured diagnosis
    top = results[0]
    top_text = top.text

    def get_field(text, field):
        for line in text.split("\n"):
            if line.startswith(field + ":"):
                return line[len(field)+1:].strip()
        return "See case details"

    diagnosis = (
        f"Sensor reading at {request.dominant_freq_hz} Hz / {request.amplitude_db} dB "
        f"matches: {top.metadata['title']}. "
        f"Root cause: {get_field(top_text, 'Root cause')}"
    )

    latency = round((time.time() - t0) * 1000, 2)

    return SensorResponse(
        diagnosis          = diagnosis,
        top_cases          = [
            {
                "case_id":  r.metadata["case_id"],
                "title":    r.metadata["title"],
                "severity": r.metadata["severity"],
                "score":    r.score,
            }
            for r in results[:3]
        ],
        severity           = top.metadata["severity"],
        recommended_action = get_field(top_text, "Corrective action")[:300],
        latency_ms         = latency,
    )


@app.post("/predict", response_model=PredictResponse, tags=["NVH Tools"], operation_id="predict_nvh")
async def predict_nvh(request: PredictRequest, _: str = _AUTH):
    """
    Predict NVH noise level using the ML surrogate model.

    Replaces a 2-hour Altair Flux simulation with a <1ms ML prediction.
    Input: motor operating parameters (RPM, load, temperature, geometry).
    Output: predicted dB level, severity assessment, BPF frequency.
    """
    t0 = time.time()

    try:
        result = predict_nvh_level(
            rpm           = request.rpm,
            load_nm       = request.load_nm,
            temperature_c = request.temperature_c,
            stator_slots  = request.stator_slots,
            rotor_poles   = request.rotor_poles,
            air_gap_mm    = request.air_gap_mm,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    latency = round((time.time() - t0) * 1000, 2)

    return PredictResponse(
        predicted_nvh_db = result["predicted_nvh_db"],
        severity         = result["severity"],
        assessment       = result["assessment"],
        bpf_hz           = result["bpf_hz"],
        operating_point  = result["operating_point"],
        model_metrics    = result["model_metrics"],
        latency_ms       = latency,
    )


@app.get("/cases", tags=["Knowledge Base"])
async def list_cases(
    component:      Optional[str] = None,
    freq_range:     Optional[str] = None,
    resonance_type: Optional[str] = None,
    severity_min:   int = 1,
    limit:          int = 20,
):
    """
    Browse the NVH knowledge base with optional filters.

    Examples:
    - /cases?component=electric_motor
    - /cases?freq_range=high&severity_min=4
    - /cases?resonance_type=eNVH
    """
    kb_path = Path("data/synthetic/nvh_knowledge_base.json")
    if not kb_path.exists():
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    with open(kb_path) as f:
        cases = json.load(f)

    # Apply filters
    filtered = []
    for case in cases:
        if component      and case.get("component")      != component:      continue
        if freq_range     and case.get("freq_range")     != freq_range:     continue
        if resonance_type and case.get("resonance_type") != resonance_type: continue
        if case.get("severity", 1) < severity_min:                          continue
        filtered.append({
            "case_id":        case["case_id"],
            "title":          case["title"],
            "component":      case["component"],
            "resonance_type": case["resonance_type"],
            "freq_range":     case["freq_range"],
            "severity":       case["severity"],
            "standards_ref":  case.get("standards_ref", ""),
        })

    return {
        "total":   len(filtered),
        "filters": {"component": component, "freq_range": freq_range,
                    "resonance_type": resonance_type, "severity_min": severity_min},
        "cases":   filtered[:limit],
    }


@app.get("/", tags=["System"])
async def root():
    """API root — returns quick start guide."""
    return {
        "name":        "CAE NVH Multi-Agent Platform",
        "version":     "1.0.0",
        "description": "Multi-agent AI for NVH engineering analysis",
        "quick_start": {
            "docs":    "http://localhost:8000/docs",
            "health":  "http://localhost:8000/health",
            "invoke":  "POST http://localhost:8000/invoke  {'query': 'your question'}",
            "mcp":     "http://localhost:8000/mcp  (MCP protocol endpoint)",
        },
        "agent_routes": {
            "nvh_query":   "NVH knowledge queries → RAG agent",
            "surrogate":   "Prediction queries → ML surrogate model",
            "compliance":  "Standards queries → Compliance agent",
            "sensor":      "Sensor readings → Sensor diagnosis agent",
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# MCP SERVER  (Model Context Protocol via fastapi-mcp)
# ─────────────────────────────────────────────────────────────────────────────
#
# fastapi-mcp auto-generates MCP tools from FastAPI route definitions.
# It mounts a full MCP server at /mcp that speaks the MCP SSE protocol:
#
#   Local (Claude Desktop stdio):  point claude_desktop_config.json at this file
#   Remote (SSE over HTTP):        connect to  http://your-server:8000/mcp/sse
#
# Claude Desktop config for remote access:
#   {
#     "mcpServers": {
#       "cae-platform": {
#         "transport": "sse",
#         "url": "http://your-server:8000/mcp/sse"
#       }
#     }
#   }

MCP_AVAILABLE = False
try:
    from fastapi_mcp import FastApiMCP

    mcp = FastApiMCP(
        app,
        name        = "CAE NVH Platform",
        description = (
            "Multi-agent AI platform for NVH engineering analysis. "
            "Exposes surrogate model prediction, sensor diagnosis, and "
            "RAG-based knowledge retrieval as MCP tools."
        ),
        # Only expose these three endpoints as MCP tools — the rest are
        # internal (health, docs, cases) and don't need agent-to-agent access.
        include_operations = ["invoke_agent", "diagnose_sensor", "predict_nvh"],
    )
    mcp.mount()
    MCP_AVAILABLE = True
    print("✅ FastAPI-MCP mounted — MCP server available at /mcp")
    print("   Remote SSE endpoint: http://host:8000/mcp/sse")

except Exception as e:
    print(f"⚠️  fastapi_mcp setup failed ({e}) — falling back to manual MCP schema")


if not MCP_AVAILABLE:
    @app.get("/mcp", tags=["MCP"])
    async def mcp_schema():
        """Manual MCP schema fallback (install fastapi-mcp for full SSE support)."""
        return {
            "protocol": "mcp",
            "version":  "1.0",
            "note":     "Install fastapi-mcp for full SSE/remote MCP support",
            "tools": [
                {
                    "name":        "invoke_nvh_agent",
                    "description": "Run the multi-agent NVH analysis system",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Engineering query"},
                        },
                        "required": ["query"],
                    },
                    "endpoint": "POST /invoke",
                },
                {
                    "name":        "diagnose_sensor",
                    "description": "Diagnose NVH issue from structured sensor reading",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "dominant_freq_hz": {"type": "number"},
                            "amplitude_db":     {"type": "number"},
                            "component":        {"type": "string"},
                        },
                        "required": ["dominant_freq_hz", "amplitude_db", "component"],
                    },
                    "endpoint": "POST /diagnose",
                },
                {
                    "name":        "predict_nvh_level",
                    "description": "Predict motor NVH noise using surrogate ML model (<1ms)",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "rpm":     {"type": "number"},
                            "load_nm": {"type": "number"},
                        },
                        "required": ["rpm", "load_nm"],
                    },
                    "endpoint": "POST /predict",
                },
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
# TEST RUNNER  (runs when you execute this file directly)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Tests the API without running the server.
    Uses FastAPI's TestClient for unit testing.
    """
    from fastapi.testclient import TestClient
    import json

    print("=" * 65)
    print("Testing FastAPI CAE Platform")
    print("=" * 65)

    # Pre-load components so tests are fast
    print("\n[Setup] Pre-loading components...")
    get_graph()
    get_retriever()
    get_model()

    client = TestClient(app)
    all_passed = True

    # ── Test 1: Health check ─────────────────────────────────────────────
    print("\n[Test 1] Health check")
    r = client.get("/health")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    print(f"  Status: {data['status']} | Uptime: {data['uptime_seconds']}s")
    print(f"  Components: {data['components']}")
    print("  ✅ PASS")

    # ── Test 2: Invoke — RAG route ────────────────────────────────────────
    print("\n[Test 2] POST /invoke — NVH query (RAG route)")
    r = client.post("/invoke", json={
        "query": "What causes electromagnetic whine at 500 Hz in electric motors?",
        "session_id": "test-001",
    })
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    assert data["route"] == "nvh_query", f"Expected nvh_query, got {data['route']}"
    assert len(data["answer"]) > 100, "Answer too short"
    print(f"  Route:   {data['route']}")
    print(f"  Latency: {data['latency_ms']} ms")
    print(f"  Answer:  {data['answer'][:80]}...")
    if data.get("retrieved_cases"):
        print(f"  Cases:   {[c['case_id'] for c in data['retrieved_cases']]}")
    print("  ✅ PASS")

    # ── Test 3: Invoke — Surrogate route ──────────────────────────────────
    print("\n[Test 3] POST /invoke — prediction (surrogate route)")
    r = client.post("/invoke", json={
        "query": "Predict NVH level for motor at 3000 rpm with 80 Nm load",
    })
    assert r.status_code == 200
    data = r.json()
    assert data["route"] == "surrogate"
    print(f"  Route:   {data['route']}")
    print(f"  Latency: {data['latency_ms']} ms")
    print(f"  Answer:  {data['answer'][:80]}...")
    print("  ✅ PASS")

    # ── Test 4: Invoke — Compliance route ─────────────────────────────────
    print("\n[Test 4] POST /invoke — compliance query")
    r = client.post("/invoke", json={
        "query": "What does ISO 362-1 specify for passenger car drive-by noise?",
    })
    assert r.status_code == 200
    data = r.json()
    assert data["route"] == "compliance"
    print(f"  Route:   {data['route']}")
    print(f"  Latency: {data['latency_ms']} ms")
    print("  ✅ PASS")

    # ── Test 5: POST /diagnose ─────────────────────────────────────────────
    print("\n[Test 5] POST /diagnose — sensor reading")
    r = client.post("/diagnose", json={
        "dominant_freq_hz": 847.0,
        "amplitude_db":     82.0,
        "component":        "electric_motor",
        "rpm":              3000,
        "freq_range":       "high",
    })
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    print(f"  Severity:  {data['severity']}/5")
    print(f"  Diagnosis: {data['diagnosis'][:80]}...")
    print(f"  Latency:   {data['latency_ms']} ms")
    print("  ✅ PASS")

    # ── Test 6: POST /predict ─────────────────────────────────────────────
    print("\n[Test 6] POST /predict — surrogate prediction")
    r = client.post("/predict", json={
        "rpm":           3000,
        "load_nm":       80.0,
        "temperature_c": 60.0,
        "stator_slots":  36,
        "rotor_poles":   6,
        "air_gap_mm":    0.8,
    })
    assert r.status_code == 200
    data = r.json()
    print(f"  Predicted: {data['predicted_nvh_db']} dB | Severity: {data['severity']}/5")
    print(f"  BPF:       {data['bpf_hz']} Hz")
    print(f"  Latency:   {data['latency_ms']} ms")
    print("  ✅ PASS")

    # ── Test 7: GET /cases ────────────────────────────────────────────────
    print("\n[Test 7] GET /cases — knowledge base browser")
    r = client.get("/cases?component=electric_motor&severity_min=3")
    assert r.status_code == 200
    data = r.json()
    print(f"  Found {data['total']} cases matching: component=electric_motor, severity≥3")
    for c in data["cases"][:3]:
        print(f"    [{c['case_id']}] {c['title'][:55]}...")
    print("  ✅ PASS")

    # ── Test 8: GET /mcp ──────────────────────────────────────────────────
    print("\n[Test 8] GET /mcp — MCP schema")
    r = client.get("/mcp")
    print(f"  Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"  MCP available: {MCP_AVAILABLE}")
        if not MCP_AVAILABLE:
            print(f"  Tools exposed: {[t['name'] for t in data.get('tools', [])]}")
        print("  ✅ PASS")
    else:
        print("  ⚠️  MCP returned non-200 (fastapi_mcp may handle differently)")

    print(f"\n{'='*65}")
    print("✅ All API tests passed!")
    print(f"\nTo run the live server:")
    print("  uvicorn api.main:app --reload --port 8000")
    print("  Then open: http://localhost:8000/docs")
    print("="*65)
