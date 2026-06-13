"""
agents/cae_graph.py
─────────────────────────────────────────────────────────────────────────────
The LangGraph multi-agent StateGraph for the CAE Platform.

Architecture:
─────────────
  User Query
      │
   ROUTER  ──────────────────────────────────────┐
      │                                          │
      ├──── "nvh_query"   ──► RAG AGENT          │
      ├──── "surrogate"   ──► SURROGATE AGENT    │
      ├──── "compliance"  ──► COMPLIANCE AGENT   │
      └──── "sensor"      ──► SENSOR AGENT       │
                                                 │
  Each agent writes answer to State              │
      │                                          │
   ANSWER NODE  ◄──────────────────────────────-─┘
      │
   END

Key LangGraph concepts used here:
───────────────────────────────────
  - TypedDict:    defines what the shared State whiteboard looks like
  - StateGraph:   the graph that connects all nodes
  - add_node:     registers a Python function as a node
  - add_edge:     connects nodes (unconditional)
  - add_conditional_edges: connects with routing logic
  - compile():    finalises the graph
  - invoke():     runs the graph with an input

Run this file to test the full agent pipeline:
    python agents/cae_graph.py
─────────────────────────────────────────────────────────────────────────────
"""

import json
import sys
import os
from pathlib import Path
from typing import TypedDict, Optional, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env so LANGCHAIN_TRACING_V2 / LANGCHAIN_API_KEY are available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from langgraph.graph import StateGraph, END
from knowledge_base.build_retriever import HybridNVHRetriever
from tools.surrogate_model import predict_nvh_level

# ── LangSmith tracing setup ──────────────────────────────────────────────────
# LangGraph traces automatically when these env vars are set in .env:
#   LANGCHAIN_TRACING_V2=true
#   LANGCHAIN_API_KEY=ls-...
#   LANGCHAIN_PROJECT=cae-nvh-platform
#
# @traceable adds fine-grained child spans inside each node function,
# visible in the LangSmith UI as individual steps within a graph run.
try:
    from langsmith import traceable
    _LANGSMITH_ENABLED = bool(os.getenv("LANGCHAIN_API_KEY"))
except ImportError:
    def traceable(**_kwargs):            # no-op decorator if langsmith not installed
        def decorator(fn):
            return fn
        return decorator
    _LANGSMITH_ENABLED = False

if _LANGSMITH_ENABLED:
    print("[LangSmith] Tracing enabled — runs will appear in LangSmith UI")
else:
    print("[LangSmith] Tracing disabled (set LANGCHAIN_API_KEY in .env to enable)")


# ─────────────────────────────────────────────────────────────────────────────
# STATE DEFINITION  (the shared whiteboard)
# ─────────────────────────────────────────────────────────────────────────────

class CAEState(TypedDict):
    """
    The shared state passed between all nodes.

    Think of this as a whiteboard in the middle of the room.
    Every agent can read everything on it, and write their results back.
    The router writes 'route'. Each specialist writes 'answer'.

    All fields are Optional because they start as None and get filled
    as the graph runs.
    """
    query:          str               # original user question (set at start)
    route:          Optional[str]     # written by router node
    retrieved_docs: Optional[List]    # written by RAG agent
    answer:         Optional[str]     # written by specialist agents
    sensor_data:    Optional[dict]    # written if sensor input provided
    error:          Optional[str]     # written if something goes wrong


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER NODE
# ─────────────────────────────────────────────────────────────────────────────

@traceable(name="cae-router", tags=["router"])
def router_node(state: CAEState) -> dict:
    """
    Reads the query and decides which specialist agent should handle it.

    This is a rule-based router (no LLM needed, fast and deterministic).
    In production you'd use an LLM for complex routing — for learning,
    keyword routing is clearer and easier to debug.

    Returns: {"route": "nvh_query"} (or "surrogate", "compliance", "sensor")
    """
    query = state["query"].lower()

    # Sensor/prediction queries → surrogate agent
    if any(kw in query for kw in [
        "predict", "rpm", "load_nm", "torque", "operating point",
        "simulate", "simulation", "what will be", "calculate noise"
    ]):
        route = "surrogate"

    # Standards/compliance queries → compliance agent
    elif any(kw in query for kw in [
        "iso", "iec", "standard", "limit", "compliance", "regulation",
        "type approval", "certif", "requirement", "dB(A) limit"
    ]):
        route = "compliance"

    # Sensor reading diagnosis
    elif any(kw in query for kw in [
        "sensor reading", "frequency hz", "amplitude_db", "diagnose this",
        "reading shows", "measured", "signal shows"
    ]):
        route = "sensor"

    # Everything else → RAG agent (most general)
    else:
        route = "nvh_query"

    print(f"  [ROUTER] Query routed to: {route}")
    return {"route": route}


# ─────────────────────────────────────────────────────────────────────────────
# RAG AGENT NODE
# ─────────────────────────────────────────────────────────────────────────────

# Load retriever once (not on every call)
_retriever: Optional[HybridNVHRetriever] = None

def _get_retriever() -> HybridNVHRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridNVHRetriever.load()
    return _retriever


@traceable(name="cae-rag-agent", tags=["rag", "retrieval"])
def rag_agent_node(state: CAEState) -> dict:
    """
    Retrieves relevant NVH cases from the knowledge base and
    synthesises an answer.

    Steps:
    1. Extract potential metadata filters from query
    2. Retrieve top-5 relevant cases (hybrid FAISS + BM25)
    3. Format retrieved docs into a context string
    4. Generate answer (simple template — no LLM API needed for testing)
    """
    query     = state["query"]
    retriever = _get_retriever()

    # Extract metadata hints from query for smarter filtering
    freq_range = None
    component  = None

    q_lower = query.lower()
    if any(w in q_lower for w in ["high freq", "high-freq", "khz", "1000 hz", "1200 hz", "2400 hz"]):
        freq_range = "high"
    elif any(w in q_lower for w in ["low freq", "low-freq", "boom", "rumble", "below 200"]):
        freq_range = "low"

    if "motor" in q_lower or "bldc" in q_lower or "pmsm" in q_lower:
        component = "electric_motor"
    elif "blower" in q_lower or "fan" in q_lower:
        component = "blower"
    elif "bearing" in q_lower:
        component = "bearing"
    elif "gear" in q_lower or "transmission" in q_lower or "differential" in q_lower:
        component = "gearbox"

    print(f"  [RAG AGENT] Retrieving with filters: freq_range={freq_range}, component={component}")

    results = retriever.retrieve(
        query,
        top_k      = 5,
        freq_range = freq_range,
        component  = component,
    )

    if not results:
        return {
            "retrieved_docs": [],
            "answer": "No relevant NVH cases found in the knowledge base for this query.",
        }

    # Format context from retrieved docs
    context_parts = []
    for r in results[:3]:   # use top-3 for answer generation
        case = r.metadata
        context_parts.append(
            f"Case {case['case_id']}: {case['title']}\n"
            f"  Component: {case['component']} | Freq range: {case['freq_range']} | Severity: {case['severity']}/5\n"
            f"  {r.text.split('Root cause:')[1][:300] if 'Root cause:' in r.text else r.text[:300]}"
        )

    context = "\n\n".join(context_parts)

    # Generate structured answer
    top_case_data = results[0]
    full_text     = top_case_data.text

    # Extract key fields from doc text
    def extract_field(text, field_name):
        for line in text.split("\n"):
            if line.startswith(field_name + ":"):
                return line[len(field_name)+1:].strip()
        return "See case details"

    answer = f"""## NVH Diagnosis — RAG Agent

**Query:** {query}

**Top Match:** {top_case_data.metadata['title']}
**Case ID:** {top_case_data.metadata['case_id']}
**Relevance Score:** {top_case_data.score:.4f}

**Root Cause:**
{extract_field(full_text, 'Root cause')}

**Corrective Action:**
{extract_field(full_text, 'Corrective action')}

**Standards Reference:**
{extract_field(full_text, 'Standards reference')}

**All Retrieved Cases ({len(results)} found):**
{chr(10).join(f"  {i+1}. [{r.metadata['case_id']}] {r.metadata['title']}" for i, r in enumerate(results))}

*Source: NVH Knowledge Base (Hybrid FAISS+BM25 retrieval)*
"""

    print(f"  [RAG AGENT] Retrieved {len(results)} cases, top match: {top_case_data.metadata['case_id']}")
    return {
        "retrieved_docs": [r.__dict__ for r in results],
        "answer": answer,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SURROGATE AGENT NODE
# ─────────────────────────────────────────────────────────────────────────────

@traceable(name="cae-surrogate-agent", tags=["ml", "prediction"])
def surrogate_agent_node(state: CAEState) -> dict:
    """
    Extracts motor parameters from the query and calls the
    surrogate ML model to predict NVH noise level.

    In production: LLM would extract parameters from natural language.
    Here: simple pattern matching (works perfectly for demonstration).
    """
    query = state["query"]

    # Extract parameters from query text (simple regex-free version)
    params = {}
    q_lower = query.lower()

    # Look for numeric values with units
    words = query.split()
    for i, word in enumerate(words):
        try:
            val = float(word.replace(",", ""))
            prev = words[i-1].lower() if i > 0 else ""
            if "rpm" in prev or "rpm" in word.lower():
                params["rpm"] = val
            elif "nm" in word.lower() or ("load" in prev and val < 200):
                params["load_nm"] = val
            elif "temp" in prev or "°c" in word.lower() or "celsius" in prev:
                params["temperature_c"] = val
            elif "gap" in prev or "mm" in word.lower() and val < 5:
                params["air_gap_mm"] = val
            elif "slot" in prev:
                params["stator_slots"] = int(val)
            elif "pole" in prev:
                params["rotor_poles"] = int(val)
        except (ValueError, IndexError):
            continue

    # Apply defaults if params partially extracted
    defaults = {"rpm": 3000, "load_nm": 50, "temperature_c": 25,
                "stator_slots": 36, "rotor_poles": 6, "air_gap_mm": 0.8}
    for k, v in defaults.items():
        if k not in params:
            params[k] = v

    print(f"  [SURROGATE AGENT] Predicting with params: {params}")

    prediction = predict_nvh_level(**params)

    answer = f"""## NVH Prediction — Surrogate Model

**Query:** {query}

**Operating Point:**
  • RPM: {params['rpm']} | Load: {params['load_nm']} Nm | Temperature: {params['temperature_c']}°C
  • Stator slots: {params['stator_slots']} | Rotor poles: {params['rotor_poles']} | Air gap: {params['air_gap_mm']} mm

**Prediction Results:**
  • **Predicted NVH Level: {prediction['predicted_nvh_db']} dB**
  • **Severity: {prediction['severity']}/5**
  • **Assessment: {prediction['assessment']}**
  • Blade Pass Frequency (BPF): {prediction['bpf_hz']} Hz

**Model Performance:**
  • RMSE: {prediction['model_metrics']['rmse']} dB | R²: {prediction['model_metrics']['r2']}
  • Method: GradientBoosting surrogate (replaces Altair Flux simulation)

*Prediction time: <1ms vs 2+ hours for full FEA simulation*
"""

    return {"answer": answer}


# ─────────────────────────────────────────────────────────────────────────────
# COMPLIANCE AGENT NODE
# ─────────────────────────────────────────────────────────────────────────────

# Inline compliance knowledge (in production this would also use RAG)
COMPLIANCE_KB = {
    "iso_362": {
        "standard": "ISO 362-1:2015",
        "applies_to": "Passenger cars (M1 category)",
        "limit_db": 72,
        "test_condition": "Drive-by at 50 km/h, microphone at 7.5m lateral, 1.2m height",
        "measurement": "A-weighted sound pressure level",
        "consequence": "Required for EU type approval — vehicles exceeding limit cannot be sold",
    },
    "iec_60704": {
        "standard": "IEC 60704-2-6:2011",
        "applies_to": "Household washing machines",
        "limit_wash_db": 72,
        "limit_spin_db": 77,
        "test_condition": "Anechoic room, microphone 1m from machine, 5 positions averaged",
        "measurement": "A-weighted declared noise",
        "consequence": "Required for EU energy label A-rating",
    },
    "iso_2631": {
        "standard": "ISO 2631-1:1997",
        "applies_to": "Human exposure to whole-body vibration",
        "comfort_threshold_ms2": 0.315,
        "discomfort_threshold_ms2": 0.8,
        "frequency_range": "0.5–80 Hz with frequency weighting",
        "measurement": "Root mean square acceleration (m/s²) with Wd/Wk weighting",
    },
    "iso_10816": {
        "standard": "ISO 10816-3:2009",
        "applies_to": "Industrial machines above 15 kW",
        "zone_a_mms": "≤ 2.3",
        "zone_b_mms": "2.3–4.5",
        "zone_c_mms": "4.5–7.1",
        "zone_d_mms": "> 7.1 (damage zone)",
        "measurement": "RMS vibration velocity in mm/s",
    },
    "iso_15243": {
        "standard": "ISO 15243:2017",
        "applies_to": "Rolling element bearing damage classification",
        "failure_modes": ["fatigue", "wear", "corrosion", "electrical erosion", "plastic deformation", "fracture"],
        "note": "Defines inspection criteria and terminology for bearing damage analysis",
    },
}

@traceable(name="cae-compliance-agent", tags=["compliance", "standards"])
def compliance_agent_node(state: CAEState) -> dict:
    """
    Answers compliance and standards queries from the inline knowledge base.
    """
    query   = state["query"]
    q_lower = query.lower()

    # Match query to relevant standard
    matched = {}
    if "362" in q_lower or "drive-by" in q_lower or "drive by" in q_lower:
        matched["ISO 362-1"] = COMPLIANCE_KB["iso_362"]
    if "60704" in q_lower or "washing machine" in q_lower or "appliance" in q_lower:
        matched["IEC 60704"] = COMPLIANCE_KB["iec_60704"]
    if "2631" in q_lower or "whole body" in q_lower or "whole-body" in q_lower:
        matched["ISO 2631-1"] = COMPLIANCE_KB["iso_2631"]
    if "10816" in q_lower or "industrial machine" in q_lower:
        matched["ISO 10816-3"] = COMPLIANCE_KB["iso_10816"]
    if "15243" in q_lower or "bearing damage" in q_lower:
        matched["ISO 15243"] = COMPLIANCE_KB["iso_15243"]

    if not matched:
        # Fall back to RAG retriever for standards not in inline KB
        return rag_agent_node(state)

    parts = [f"## Compliance Information — Standards Agent\n\n**Query:** {query}\n"]
    for std_name, std_data in matched.items():
        parts.append(f"### {std_name}")
        for k, v in std_data.items():
            key_fmt = k.replace("_", " ").title()
            parts.append(f"  **{key_fmt}:** {v}")
        parts.append("")

    answer = "\n".join(parts)
    print(f"  [COMPLIANCE AGENT] Matched standards: {list(matched.keys())}")
    return {"answer": answer}


# ─────────────────────────────────────────────────────────────────────────────
# SENSOR AGENT NODE
# ─────────────────────────────────────────────────────────────────────────────

@traceable(name="cae-sensor-agent", tags=["sensor", "diagnosis"])
def sensor_agent_node(state: CAEState) -> dict:
    """
    Processes sensor readings and routes to RAG for diagnosis.
    Combines surrogate prediction + RAG retrieval.
    """
    query = state["query"]

    # For sensor queries, use RAG with high-precision retrieval
    rag_result = rag_agent_node(state)

    sensor_answer = f"""## Sensor-Based NVH Diagnosis

**Query:** {query}

{rag_result['answer']}

*Tip: For precise prediction, provide: rpm, load_nm, temperature_c values.*
"""
    return {
        "retrieved_docs": rag_result.get("retrieved_docs", []),
        "answer": sensor_answer,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ROUTING FUNCTION  (decides which node to call after router)
# ─────────────────────────────────────────────────────────────────────────────

def route_to_agent(state: CAEState) -> str:
    """
    This function is used by add_conditional_edges.
    It reads the route written by router_node and returns the
    name of the next node to execute.

    LangGraph passes the current state to this function and
    uses the returned string as the key to look up the next node.
    """
    return state.get("route", "nvh_query")


# ─────────────────────────────────────────────────────────────────────────────
# ANSWER NODE  (final node — just passes through)
# ─────────────────────────────────────────────────────────────────────────────

def answer_node(state: CAEState) -> dict:
    """
    Final node — formats and returns the answer.
    Could add post-processing, confidence scoring, or
    human-in-the-loop check here in production.
    """
    answer = state.get("answer", "No answer generated.")
    print(f"  [ANSWER NODE] Answer ready ({len(answer)} chars)")
    return {"answer": answer}


# ─────────────────────────────────────────────────────────────────────────────
# BUILD THE GRAPH
# ─────────────────────────────────────────────────────────────────────────────

def build_cae_graph():
    """
    Assembles the full multi-agent StateGraph.

    Step by step:
    1. Create StateGraph with our CAEState schema
    2. Add all nodes (Python functions)
    3. Set entry point (first node to run)
    4. Add conditional edges from router to specialist agents
    5. Add unconditional edges from specialists to answer node
    6. Add edge from answer node to END
    7. compile() to finalise
    """
    graph = StateGraph(CAEState)

    # 2. Add nodes
    graph.add_node("router",     router_node)
    graph.add_node("nvh_query",  rag_agent_node)
    graph.add_node("surrogate",  surrogate_agent_node)
    graph.add_node("compliance", compliance_agent_node)
    graph.add_node("sensor",     sensor_agent_node)
    graph.add_node("final_answer", answer_node)

    # 3. Entry point
    graph.set_entry_point("router")

    # 4. Conditional edges: router → correct specialist
    graph.add_conditional_edges(
        "router",           # source node
        route_to_agent,     # function that returns the destination key
        {
            "nvh_query":   "nvh_query",
            "surrogate":   "surrogate",
            "compliance":  "compliance",
            "sensor":      "sensor",
        },
    )

    # 5. Unconditional edges: every specialist → answer node
    for agent in ["nvh_query", "surrogate", "compliance", "sensor"]:
        graph.add_edge(agent, "final_answer")

    # 6. Answer → END
    graph.add_edge("final_answer", END)

    # 7. Compile
    app = graph.compile()
    print("  ✅ CAE Graph compiled successfully")
    return app


# ─────────────────────────────────────────────────────────────────────────────
# TEST  (runs when you execute this file directly)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("Building and Testing CAE Multi-Agent Graph")
    print("=" * 65)

    print("\n[1/2] Building graph...")
    app = build_cae_graph()

    print("\n[2/2] Running test queries...")

    TEST_QUERIES = [
        {
            "query": "What causes electromagnetic whine in electric motors at 500 Hz?",
            "expected_route": "nvh_query",
            "label": "RAG — motor NVH query",
        },
        {
            "query": "Predict NVH level for motor at 3000 rpm with 80 Nm load",
            "expected_route": "surrogate",
            "label": "Surrogate — prediction query",
        },
        {
            "query": "What does ISO 362-1 specify for drive-by noise limits?",
            "expected_route": "compliance",
            "label": "Compliance — standards query",
        },
        {
            "query": "Bearing showing BPFI at 2400 Hz with high amplitude — what action needed?",
            "expected_route": "nvh_query",
            "label": "RAG — bearing diagnosis",
        },
    ]

    print()
    all_passed = True
    for i, tc in enumerate(TEST_QUERIES, 1):
        print(f"{'─'*65}")
        print(f"Test {i}: {tc['label']}")
        print(f"Query: {tc['query']}")
        print()

        result = app.invoke({"query": tc["query"]})

        actual_route  = result.get("route", "unknown")
        route_correct = actual_route == tc["expected_route"]
        has_answer    = bool(result.get("answer"))

        status = "✅" if (route_correct and has_answer) else "❌"
        print(f"\n{status} Route: {actual_route} (expected: {tc['expected_route']})")
        print(f"{'✅' if has_answer else '❌'} Answer generated: {has_answer}")
        if has_answer:
            # Show first 3 lines of answer
            lines = result["answer"].strip().split("\n")[:4]
            print("\nAnswer preview:")
            for line in lines:
                if line.strip():
                    print(f"  {line}")

        if not (route_correct and has_answer):
            all_passed = False

    print(f"\n{'='*65}")
    if all_passed:
        print("✅ All agent tests passed!")
        print("\nMulti-agent graph is working. Next: add FastAPI + MCP server")
    else:
        print("❌ Some tests failed — check the output above")
    print("=" * 65)
