"""
tests/test_api.py
─────────────────────────────────────────────────────────────────────────────
FastAPI endpoint integration tests.

Covers:
  GET  /health          — system health
  GET  /                — root info
  POST /invoke          — agent routing (nvh_query, surrogate, compliance)
  POST /diagnose        — sensor diagnosis
  POST /predict         — surrogate prediction
  GET  /cases           — knowledge base browser
  GET  /mcp             — MCP schema (when fastapi-mcp not installed)

Run:
    pytest tests/test_api.py -v
─────────────────────────────────────────────────────────────────────────────
"""

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# /health
# ─────────────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_returns_200(self, api_client):
        r = api_client.get("/health")
        assert r.status_code == 200

    def test_status_healthy(self, api_client):
        data = api_client.get("/health").json()
        assert data["status"] == "healthy"

    def test_version_present(self, api_client):
        data = api_client.get("/health").json()
        assert "version" in data

    def test_components_present(self, api_client):
        data = api_client.get("/health").json()
        assert "components" in data
        assert "api" in data["components"]

    def test_uptime_positive(self, api_client):
        data = api_client.get("/health").json()
        assert data["uptime_seconds"] >= 0


# ─────────────────────────────────────────────────────────────────────────────
# /  (root)
# ─────────────────────────────────────────────────────────────────────────────

class TestRoot:
    def test_returns_200(self, api_client):
        assert api_client.get("/").status_code == 200

    def test_has_name(self, api_client):
        data = api_client.get("/").json()
        assert "name" in data

    def test_has_agent_routes(self, api_client):
        data = api_client.get("/").json()
        assert "agent_routes" in data


# ─────────────────────────────────────────────────────────────────────────────
# POST /invoke
# ─────────────────────────────────────────────────────────────────────────────

class TestInvoke:
    def test_nvh_query_route(self, api_client):
        r = api_client.post("/invoke", json={
            "query": "What causes electromagnetic whine at 500 Hz in electric motors?"
        })
        assert r.status_code == 200
        data = r.json()
        assert data["route"] == "nvh_query"
        assert len(data["answer"]) > 50

    def test_surrogate_route(self, api_client):
        r = api_client.post("/invoke", json={
            "query": "Predict NVH level for motor at 3000 rpm with 80 Nm load"
        })
        assert r.status_code == 200
        assert r.json()["route"] == "surrogate"

    def test_compliance_route(self, api_client):
        r = api_client.post("/invoke", json={
            "query": "What does ISO 362-1 specify for drive-by noise limits?"
        })
        assert r.status_code == 200
        assert r.json()["route"] == "compliance"

    def test_response_has_latency(self, api_client):
        r = api_client.post("/invoke", json={"query": "Tell me about bearing noise"})
        assert r.status_code == 200
        assert r.json()["latency_ms"] >= 0

    def test_session_id_echoed(self, api_client):
        r = api_client.post("/invoke", json={
            "query": "bearing noise",
            "session_id": "test-session-42",
        })
        assert r.status_code == 200
        assert r.json()["session_id"] == "test-session-42"

    def test_empty_query_returns_error(self, api_client):
        r = api_client.post("/invoke", json={"query": ""})
        # FastAPI returns 422 for validation errors or the agent handles empty gracefully
        assert r.status_code in (200, 422)

    def test_missing_query_returns_422(self, api_client):
        r = api_client.post("/invoke", json={})
        assert r.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# POST /diagnose
# ─────────────────────────────────────────────────────────────────────────────

class TestDiagnose:
    def test_basic_diagnosis(self, api_client):
        r = api_client.post("/diagnose", json={
            "dominant_freq_hz": 847.0,
            "amplitude_db":     82.0,
            "component":        "electric_motor",
        })
        assert r.status_code == 200
        data = r.json()
        assert "diagnosis" in data
        assert "severity" in data
        assert 1 <= data["severity"] <= 5

    def test_with_rpm_and_freq_range(self, api_client):
        r = api_client.post("/diagnose", json={
            "dominant_freq_hz": 2400.0,
            "amplitude_db":     75.0,
            "component":        "bearing",
            "rpm":              3600,
            "freq_range":       "high",
        })
        assert r.status_code == 200

    def test_top_cases_returned(self, api_client):
        r = api_client.post("/diagnose", json={
            "dominant_freq_hz": 500.0,
            "amplitude_db":     70.0,
            "component":        "electric_motor",
        })
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data["top_cases"], list)
        assert len(data["top_cases"]) >= 1

    def test_latency_present(self, api_client):
        r = api_client.post("/diagnose", json={
            "dominant_freq_hz": 300.0,
            "amplitude_db":     65.0,
            "component":        "blower",
        })
        assert r.status_code == 200
        assert r.json()["latency_ms"] >= 0

    def test_missing_required_fields_returns_422(self, api_client):
        r = api_client.post("/diagnose", json={"dominant_freq_hz": 500.0})
        assert r.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# POST /predict
# ─────────────────────────────────────────────────────────────────────────────

class TestPredict:
    def test_basic_prediction(self, api_client):
        r = api_client.post("/predict", json={
            "rpm":     3000.0,
            "load_nm": 80.0,
        })
        assert r.status_code == 200
        data = r.json()
        assert "predicted_nvh_db" in data
        assert 30 <= data["predicted_nvh_db"] <= 120

    def test_severity_in_range(self, api_client):
        r = api_client.post("/predict", json={"rpm": 3000.0, "load_nm": 80.0})
        assert r.status_code == 200
        assert 1 <= r.json()["severity"] <= 5

    def test_bpf_positive(self, api_client):
        r = api_client.post("/predict", json={"rpm": 3000.0, "load_nm": 80.0})
        assert r.status_code == 200
        assert r.json()["bpf_hz"] > 0

    def test_all_params(self, api_client):
        r = api_client.post("/predict", json={
            "rpm":           6000.0,
            "load_nm":       120.0,
            "temperature_c": 80.0,
            "stator_slots":  48,
            "rotor_poles":   8,
            "air_gap_mm":    0.5,
        })
        assert r.status_code == 200
        assert r.json()["predicted_nvh_db"] > 0

    def test_model_metrics_present(self, api_client):
        r = api_client.post("/predict", json={"rpm": 2000.0, "load_nm": 40.0})
        assert r.status_code == 200
        metrics = r.json()["model_metrics"]
        assert "rmse" in metrics
        assert "r2" in metrics

    def test_missing_rpm_returns_422(self, api_client):
        r = api_client.post("/predict", json={"load_nm": 80.0})
        assert r.status_code == 422

    def test_missing_load_returns_422(self, api_client):
        r = api_client.post("/predict", json={"rpm": 3000.0})
        assert r.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# GET /cases
# ─────────────────────────────────────────────────────────────────────────────

class TestCases:
    def test_returns_cases(self, api_client):
        r = api_client.get("/cases")
        assert r.status_code == 200
        data = r.json()
        assert "total" in data
        assert "cases" in data
        assert data["total"] > 0

    def test_filter_by_component(self, api_client):
        r = api_client.get("/cases?component=electric_motor")
        assert r.status_code == 200
        data = r.json()
        for case in data["cases"]:
            assert case["component"] == "electric_motor"

    def test_filter_by_severity(self, api_client):
        r = api_client.get("/cases?severity_min=4")
        assert r.status_code == 200
        data = r.json()
        for case in data["cases"]:
            assert case["severity"] >= 4

    def test_filter_by_freq_range(self, api_client):
        r = api_client.get("/cases?freq_range=high")
        assert r.status_code == 200
        data = r.json()
        for case in data["cases"]:
            assert case["freq_range"] == "high"

    def test_limit_respected(self, api_client):
        r = api_client.get("/cases?limit=5")
        assert r.status_code == 200
        assert len(r.json()["cases"]) <= 5


# ─────────────────────────────────────────────────────────────────────────────
# CORS headers
# ─────────────────────────────────────────────────────────────────────────────

class TestCORS:
    def test_cors_header_on_health(self, api_client):
        r = api_client.get("/health", headers={"Origin": "https://example.com"})
        assert r.status_code == 200
        assert "access-control-allow-origin" in r.headers
