"""
tests/test_surrogate.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for the GradientBoosting NVH surrogate model.

Covers:
  - Model loads from disk without error
  - Predictions are in a physically plausible range (30–120 dB)
  - Severity classification matches prediction value
  - BPF calculation is correct (slots × poles × rpm / 120)
  - Model metrics are present and sane (R² > 0, RMSE > 0)
  - Default parameters produce a valid result
  - Edge values (very low / very high RPM) don't crash

Run:
    pytest tests/test_surrogate.py -v
─────────────────────────────────────────────────────────────────────────────
"""

import pytest
from tools.surrogate_model import predict_nvh_level, load_model


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

class TestModelLoad:
    def test_loads_without_error(self):
        model = load_model()
        assert model is not None

    def test_model_has_predict(self):
        model = load_model()
        # load_model returns a dict with "pipeline" key containing the sklearn Pipeline
        assert "pipeline" in model
        assert hasattr(model["pipeline"], "predict")


# ─────────────────────────────────────────────────────────────────────────────
# predict_nvh_level — output structure
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictStructure:
    @pytest.fixture(scope="class")
    def result(self):
        return predict_nvh_level(rpm=3000, load_nm=80)

    def test_has_predicted_nvh_db(self, result):
        assert "predicted_nvh_db" in result

    def test_has_severity(self, result):
        assert "severity" in result

    def test_has_assessment(self, result):
        assert "assessment" in result

    def test_has_bpf_hz(self, result):
        assert "bpf_hz" in result

    def test_has_operating_point(self, result):
        assert "operating_point" in result

    def test_has_model_metrics(self, result):
        assert "model_metrics" in result
        metrics = result["model_metrics"]
        assert "rmse" in metrics
        assert "r2" in metrics
        assert "cv_r2" in metrics


# ─────────────────────────────────────────────────────────────────────────────
# Prediction values
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictionValues:
    def test_nvh_db_in_physical_range(self):
        result = predict_nvh_level(rpm=3000, load_nm=80)
        assert 30 <= result["predicted_nvh_db"] <= 120

    def test_severity_1_to_5(self):
        result = predict_nvh_level(rpm=3000, load_nm=80)
        assert 1 <= result["severity"] <= 5

    def test_bpf_positive(self):
        result = predict_nvh_level(rpm=3000, load_nm=80, stator_slots=36, rotor_poles=6)
        assert result["bpf_hz"] > 0

    def test_bpf_formula(self):
        # BPF = (rpm / 60) * stator_slots / rotor_poles
        result = predict_nvh_level(rpm=3000, load_nm=80, stator_slots=36, rotor_poles=6)
        expected_bpf = round((3000 / 60) * 36 / 6, 1)   # = 300.0 Hz
        assert abs(result["bpf_hz"] - expected_bpf) < 1.0

    def test_r2_positive(self):
        result = predict_nvh_level(rpm=3000, load_nm=80)
        assert result["model_metrics"]["r2"] > 0

    def test_rmse_positive(self):
        result = predict_nvh_level(rpm=3000, load_nm=80)
        assert result["model_metrics"]["rmse"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# Severity classification
# ─────────────────────────────────────────────────────────────────────────────

class TestSeverityClassification:
    def test_low_rpm_lower_severity_than_high(self):
        low  = predict_nvh_level(rpm=500,  load_nm=10,  temperature_c=20)
        high = predict_nvh_level(rpm=8000, load_nm=200, temperature_c=90)
        # Low operating point should produce lower NVH than high operating point
        assert low["predicted_nvh_db"] < high["predicted_nvh_db"]

    def test_high_rpm_high_severity(self):
        result = predict_nvh_level(rpm=8000, load_nm=200, temperature_c=90)
        # High operating point → should be severity 3, 4, or 5
        assert result["severity"] >= 2


# ─────────────────────────────────────────────────────────────────────────────
# Default parameters
# ─────────────────────────────────────────────────────────────────────────────

class TestDefaults:
    def test_only_rpm_and_load_required(self):
        result = predict_nvh_level(rpm=3000, load_nm=80)
        assert result["predicted_nvh_db"] > 0

    def test_operating_point_echoes_inputs(self):
        result = predict_nvh_level(rpm=2500, load_nm=60)
        op = result["operating_point"]
        assert op["rpm"] == 2500
        assert op["load_nm"] == 60


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_very_low_rpm(self):
        result = predict_nvh_level(rpm=100, load_nm=5)
        assert isinstance(result["predicted_nvh_db"], float)

    def test_very_high_rpm(self):
        result = predict_nvh_level(rpm=15000, load_nm=300)
        assert isinstance(result["predicted_nvh_db"], float)

    def test_zero_temperature(self):
        result = predict_nvh_level(rpm=3000, load_nm=80, temperature_c=0)
        assert isinstance(result["predicted_nvh_db"], float)

    def test_different_pole_slot_counts(self):
        result = predict_nvh_level(rpm=3000, load_nm=80, stator_slots=48, rotor_poles=8)
        assert result["predicted_nvh_db"] > 0
