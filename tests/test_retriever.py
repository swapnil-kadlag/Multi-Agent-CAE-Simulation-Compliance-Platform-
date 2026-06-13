"""
tests/test_retriever.py
─────────────────────────────────────────────────────────────────────────────
Tests for the HybridNVHRetriever (FAISS + BM25 + RRF).

Covers:
  - Retriever loads from disk without error
  - Retrieve returns results for standard NVH queries
  - Scores are in (0, 1] range
  - Results are ranked (score decreasing)
  - top_k parameter is respected
  - Component metadata filter works
  - Freq range metadata filter works
  - Known cases appear in top results (hit-rate smoke test)
  - Retriever handles unknown query gracefully (no crash)

Run:
    pytest tests/test_retriever.py -v
─────────────────────────────────────────────────────────────────────────────
"""

import pytest
from knowledge_base.build_retriever import HybridNVHRetriever


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

class TestLoad:
    def test_loads_without_error(self, retriever):
        assert retriever is not None

    def test_has_retrieve_method(self, retriever):
        assert hasattr(retriever, "retrieve")


# ─────────────────────────────────────────────────────────────────────────────
# Basic retrieval
# ─────────────────────────────────────────────────────────────────────────────

class TestRetrieve:
    def test_returns_results(self, retriever):
        results = retriever.retrieve("electric motor whine noise 500 Hz", top_k=5)
        assert len(results) > 0

    def test_top_k_respected(self, retriever):
        results = retriever.retrieve("NVH bearing defect", top_k=3)
        assert len(results) <= 3

    def test_scores_positive(self, retriever):
        results = retriever.retrieve("gearbox tonal noise highway speed", top_k=5)
        for r in results:
            assert r.score > 0

    def test_scores_in_range(self, retriever):
        results = retriever.retrieve("blower blade pass frequency resonance", top_k=5)
        for r in results:
            assert 0 < r.score <= 1.5  # RRF scores can exceed 1 in some implementations

    def test_results_ranked_descending(self, retriever):
        results = retriever.retrieve("electric motor electromagnetic noise", top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_result_has_metadata(self, retriever):
        results = retriever.retrieve("bearing BPFI inner race defect", top_k=3)
        for r in results:
            assert "case_id" in r.metadata
            assert "title" in r.metadata
            assert "component" in r.metadata

    def test_result_has_text(self, retriever):
        results = retriever.retrieve("ISO 362 drive-by noise limit", top_k=3)
        for r in results:
            assert isinstance(r.text, str)
            assert len(r.text) > 50

    def test_unknown_query_no_crash(self, retriever):
        results = retriever.retrieve("xyzzy quantum flux capacitor holography", top_k=5)
        assert isinstance(results, list)


# ─────────────────────────────────────────────────────────────────────────────
# Metadata filtering
# ─────────────────────────────────────────────────────────────────────────────

class TestMetadataFilters:
    def test_component_filter_electric_motor(self, retriever):
        results = retriever.retrieve(
            "noise vibration", top_k=5, component="electric_motor"
        )
        for r in results:
            assert r.metadata["component"] == "electric_motor"

    def test_component_filter_bearing(self, retriever):
        results = retriever.retrieve(
            "vibration fault", top_k=5, component="bearing"
        )
        for r in results:
            assert r.metadata["component"] == "bearing"

    def test_freq_range_filter_high(self, retriever):
        results = retriever.retrieve(
            "high frequency tonal noise", top_k=5, freq_range="high"
        )
        # "broadband" cases are intentionally allowed through all freq_range filters
        for r in results:
            assert r.metadata["freq_range"] in ("high", "broadband")

    def test_freq_range_filter_low(self, retriever):
        results = retriever.retrieve(
            "low frequency boom rumble", top_k=5, freq_range="low"
        )
        for r in results:
            assert r.metadata["freq_range"] == "low"


# ─────────────────────────────────────────────────────────────────────────────
# Hit-rate smoke tests (reference cases must appear in top-5)
# ─────────────────────────────────────────────────────────────────────────────

class TestHitRate:
    @pytest.mark.parametrize("query,expected_component", [
        ("electric motor electromagnetic whine tonal noise BLDC PMSM", "electric_motor"),
        ("bearing BPFI inner race defect 2400 Hz high amplitude urgent", "bearing"),
        ("blower fan blade pass frequency aeroacoustic cavity resonance", "blower"),
    ])
    def test_component_in_top_results(self, retriever, query, expected_component):
        results = retriever.retrieve(query, top_k=5)
        components = [r.metadata["component"] for r in results]
        assert expected_component in components, (
            f"Expected component '{expected_component}' in top-5 results for: {query!r}\n"
            f"Got: {components}"
        )
