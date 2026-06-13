"""
tests/conftest.py
─────────────────────────────────────────────────────────────────────────────
Shared pytest fixtures for the CAE NVH Platform test suite.

Fixtures:
  api_client   — FastAPI TestClient with models pre-loaded
  retriever    — pre-loaded HybridNVHRetriever (loaded once per session)
  graph        — compiled LangGraph multi-agent app (loaded once per session)
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import pytest
from pathlib import Path

# Add project root to path so imports work from the tests/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def retriever():
    """Load the hybrid retriever once for the entire test session."""
    from knowledge_base.build_retriever import HybridNVHRetriever
    return HybridNVHRetriever.load()


@pytest.fixture(scope="session")
def graph():
    """Compile the LangGraph multi-agent app once for the entire test session."""
    from agents.cae_graph import build_cae_graph
    return build_cae_graph()


@pytest.fixture(scope="session")
def api_client(retriever, graph):
    """
    Return a FastAPI TestClient with all heavy components pre-loaded.
    Pre-loading avoids cold-start latency on the first request.
    """
    from fastapi.testclient import TestClient
    import api.main as main_module

    # Inject pre-loaded components into the module globals so
    # lazy loaders don't reload them during tests.
    main_module._retriever = retriever
    main_module._graph = graph

    return TestClient(main_module.app)
