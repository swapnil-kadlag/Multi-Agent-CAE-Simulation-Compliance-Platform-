"""
tests/test_graph.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for the LangGraph multi-agent state machine.

Covers:
  - Graph compiles without error
  - Router correctly classifies each query type
  - RAG agent returns an answer with retrieved docs
  - Surrogate agent returns a numeric prediction answer
  - Compliance agent returns ISO/IEC standard content
  - Sensor agent delegates to RAG and wraps the result
  - Answer node always produces a non-empty string
  - State fields are populated correctly after invocation

Run:
    pytest tests/test_graph.py -v
─────────────────────────────────────────────────────────────────────────────
"""

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Graph compilation
# ─────────────────────────────────────────────────────────────────────────────

class TestGraphCompile:
    def test_graph_is_not_none(self, graph):
        assert graph is not None

    def test_graph_has_invoke(self, graph):
        assert hasattr(graph, "invoke")


# ─────────────────────────────────────────────────────────────────────────────
# Router — keyword-based routing logic
# ─────────────────────────────────────────────────────────────────────────────

class TestRouter:
    @pytest.mark.parametrize("query,expected_route", [
        # Surrogate triggers
        ("Predict NVH level for motor at 3000 rpm with 80 Nm load", "surrogate"),
        ("What will be the noise at 5000 rpm?", "surrogate"),
        ("Calculate noise for 6000 rpm 100 Nm torque", "surrogate"),
        # Compliance triggers
        ("What does ISO 362-1 specify for drive-by noise?", "compliance"),
        ("What is the IEC 60704 standard for washing machines?", "compliance"),
        ("What are the compliance limits for passenger car noise?", "compliance"),
        # NVH query (default RAG)
        ("What causes electromagnetic whine in electric motors?", "nvh_query"),
        ("Tell me about bearing noise diagnosis", "nvh_query"),
        ("Why does my blower make noise at 300 Hz?", "nvh_query"),
    ])
    def test_route(self, graph, query, expected_route):
        result = graph.invoke({"query": query})
        assert result["route"] == expected_route, (
            f"Query: {query!r}\n"
            f"Expected route: {expected_route}\n"
            f"Got route: {result['route']}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# RAG agent
# ─────────────────────────────────────────────────────────────────────────────

class TestRAGAgent:
    @pytest.fixture(scope="class")
    def rag_result(self, graph):
        return graph.invoke({
            "query": "What causes electromagnetic whine in electric motors at 500 Hz?"
        })

    def test_route_is_nvh_query(self, rag_result):
        assert rag_result["route"] == "nvh_query"

    def test_answer_not_empty(self, rag_result):
        assert rag_result.get("answer")
        assert len(rag_result["answer"]) > 50

    def test_retrieved_docs_present(self, rag_result):
        assert rag_result.get("retrieved_docs")
        assert len(rag_result["retrieved_docs"]) > 0

    def test_retrieved_docs_have_metadata(self, rag_result):
        for doc in rag_result["retrieved_docs"]:
            assert "metadata" in doc
            assert "case_id" in doc["metadata"]


# ─────────────────────────────────────────────────────────────────────────────
# Surrogate agent
# ─────────────────────────────────────────────────────────────────────────────

class TestSurrogateAgent:
    @pytest.fixture(scope="class")
    def surrogate_result(self, graph):
        return graph.invoke({
            "query": "Predict NVH level for motor at 3000 rpm with 80 Nm load"
        })

    def test_route_is_surrogate(self, surrogate_result):
        assert surrogate_result["route"] == "surrogate"

    def test_answer_contains_db(self, surrogate_result):
        assert "dB" in surrogate_result.get("answer", "")

    def test_answer_contains_severity(self, surrogate_result):
        assert "Severity" in surrogate_result.get("answer", "") or \
               "severity" in surrogate_result.get("answer", "")

    def test_answer_not_empty(self, surrogate_result):
        assert len(surrogate_result.get("answer", "")) > 50


# ─────────────────────────────────────────────────────────────────────────────
# Compliance agent
# ─────────────────────────────────────────────────────────────────────────────

class TestComplianceAgent:
    @pytest.fixture(scope="class")
    def compliance_result(self, graph):
        return graph.invoke({
            "query": "What does ISO 362-1 specify for drive-by noise limits?"
        })

    def test_route_is_compliance(self, compliance_result):
        assert compliance_result["route"] == "compliance"

    def test_answer_contains_iso_362(self, compliance_result):
        answer = compliance_result.get("answer", "")
        assert "362" in answer or "ISO" in answer

    def test_answer_not_empty(self, compliance_result):
        assert len(compliance_result.get("answer", "")) > 50

    def test_iso_2631_query(self, graph):
        result = graph.invoke({
            "query": "What is ISO 2631 whole-body vibration standard?"
        })
        assert result["route"] == "compliance"
        assert "2631" in result.get("answer", "") or "vibration" in result.get("answer", "").lower()


# ─────────────────────────────────────────────────────────────────────────────
# State completeness after invocation
# ─────────────────────────────────────────────────────────────────────────────

class TestStateCompleteness:
    @pytest.mark.parametrize("query", [
        "What causes bearing noise?",
        "Predict NVH at 4000 rpm 60 Nm",
        "What does ISO 362 specify?",
    ])
    def test_answer_always_present(self, graph, query):
        result = graph.invoke({"query": query})
        assert "answer" in result
        assert result["answer"]

    @pytest.mark.parametrize("query", [
        "What causes bearing noise?",
        "Predict NVH at 4000 rpm 60 Nm",
    ])
    def test_route_always_set(self, graph, query):
        result = graph.invoke({"query": query})
        assert "route" in result
        assert result["route"] in ("nvh_query", "surrogate", "compliance", "sensor")
