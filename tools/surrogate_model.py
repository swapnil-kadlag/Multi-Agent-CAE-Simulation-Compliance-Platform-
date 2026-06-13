"""
tools/surrogate_model.py
─────────────────────────────────────────────────────────────────────────────
Trains a scikit-learn surrogate model on synthetic motor simulation data
and exposes it as a LangChain @tool so the agent can call it.

What is a surrogate model?
───────────────────────────
A surrogate model is a fast ML approximation of an expensive physics solver.
Instead of running a 2-hour Altair Flux simulation to predict NVH noise,
the agent calls predict_nvh_level() which returns a prediction in <1ms.

The model learns the relationship:
    [rpm, load_nm, temperature_c, stator_slots, rotor_poles, air_gap_mm]
                            ↓
                       nvh_db  (dB noise level)

Run this file to train and test:
    python tools/surrogate_model.py
─────────────────────────────────────────────────────────────────────────────
"""

import csv
import json
import os
import pickle
import numpy as np
from pathlib import Path
from typing import Optional

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# Load .env for WANDB_API_KEY / WANDB_PROJECT
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# W&B tracking — graceful no-op if not installed or not logged in
try:
    import wandb
    _WANDB_ENABLED = bool(os.getenv("WANDB_API_KEY"))
except ImportError:
    wandb = None
    _WANDB_ENABLED = False


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "rpm", "load_nm", "temperature_c",
    "stator_slots", "rotor_poles", "air_gap_mm",
]
TARGET_NAME = "nvh_db"
MODEL_PATH  = "data/surrogate_model.pkl"


def load_training_data(csv_path: str = "data/synthetic/motor_simulation.csv"):
    """Load synthetic motor simulation dataset."""
    rows = list(csv.DictReader(open(csv_path)))
    X = np.array([[float(r[f]) for f in FEATURE_NAMES] for r in rows])
    y = np.array([float(r[TARGET_NAME]) for r in rows])
    print(f"  Loaded {len(rows)} samples | Features: {len(FEATURE_NAMES)} | Target: {TARGET_NAME}")
    print(f"  NVH range: {y.min():.1f} – {y.max():.1f} dB | Mean: {y.mean():.1f} dB")
    return X, y


def train_surrogate(X, y) -> dict:
    """
    Train two models and return the best one.

    We use GradientBoosting as champion — it handles non-linear
    interactions (RPM × load cross-terms) better than RandomForest
    on tabular engineering data.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.05,
                max_depth=4, subsample=0.8, random_state=42,
            )),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  RandomForestRegressor(
                n_estimators=200, max_depth=8,
                min_samples_leaf=3, random_state=42,
            )),
        ]),
    }

    results = {}
    print("\n  Training models:")

    wandb_project = os.getenv("WANDB_PROJECT", "cae-surrogate-models")
    wandb_entity  = os.getenv("WANDB_ENTITY", None)

    for name, pipeline in models.items():
        # ── W&B run per model ────────────────────────────────────────────
        run = None
        if _WANDB_ENABLED and wandb:
            run = wandb.init(
                project = wandb_project,
                entity  = wandb_entity,
                name    = f"{name}-{len(X)}-samples",
                tags    = ["surrogate", "nvh", name.lower()],
                config  = {
                    "model":         name,
                    "n_samples":     len(X),
                    "n_features":    len(FEATURE_NAMES),
                    "features":      FEATURE_NAMES,
                    "target":        TARGET_NAME,
                    "test_size":     0.2,
                    "cv_folds":      5,
                },
                reinit  = True,
            )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
        r2     = r2_score(y_test, y_pred)
        cv     = cross_val_score(pipeline, X, y, cv=5, scoring="r2")

        results[name] = {
            "pipeline": pipeline,
            "rmse":     round(rmse, 3),
            "r2":       round(r2,   4),
            "cv_r2":    round(cv.mean(), 4),
        }
        print(f"    {name}: RMSE={rmse:.2f} dB | R²={r2:.4f} | CV R²={cv.mean():.4f}")

        if run:
            run.log({
                "rmse":        rmse,
                "r2":          r2,
                "cv_r2_mean":  cv.mean(),
                "cv_r2_std":   cv.std(),
                "train_size":  len(X_train),
                "test_size":   len(X_test),
            })
            # Log feature importance if available on the inner estimator
            inner = pipeline.named_steps["model"]
            if hasattr(inner, "feature_importances_"):
                importance_table = wandb.Table(
                    columns = ["feature", "importance"],
                    data    = [
                        [f, float(imp)]
                        for f, imp in zip(FEATURE_NAMES, inner.feature_importances_)
                    ],
                )
                run.log({"feature_importance": importance_table})
            run.finish()

    # Pick best model by CV R²
    best_name = max(results, key=lambda k: results[k]["cv_r2"])
    print(f"\n  ✅ Champion model: {best_name} (CV R²={results[best_name]['cv_r2']})")

    if _WANDB_ENABLED and wandb:
        print(f"  W&B: runs logged to project '{wandb_project}'")

    return results[best_name]


def save_model(result: dict, path: str = MODEL_PATH):
    """Save trained pipeline to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pipeline":     result["pipeline"],
        "feature_names": FEATURE_NAMES,
        "metrics":      {"rmse": result["rmse"], "r2": result["r2"], "cv_r2": result["cv_r2"]},
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"  Model saved → {path}")


def load_model(path: str = MODEL_PATH) -> dict:
    """Load trained model from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# LANGCHAIN TOOL WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

# Load model once at module level (so it isn't reloaded on every tool call)
_model_cache: Optional[dict] = None

def _get_model() -> dict:
    global _model_cache
    if _model_cache is None:
        _model_cache = load_model()
    return _model_cache


def predict_nvh_level(
    rpm:          float,
    load_nm:      float,
    temperature_c: float  = 25.0,
    stator_slots: int     = 36,
    rotor_poles:  int     = 6,
    air_gap_mm:   float   = 0.8,
) -> dict:
    """
    Predict NVH noise level for given motor operating parameters.

    This function is the surrogate model — it replaces a full
    Altair Flux electromagnetic simulation.

    Args:
        rpm:           Motor speed in RPM
        load_nm:       Shaft torque in Newton-metres
        temperature_c: Operating temperature in Celsius
        stator_slots:  Number of stator slots (default 36)
        rotor_poles:   Number of rotor poles  (default 6)
        air_gap_mm:    Air gap length in mm   (default 0.8)

    Returns:
        dict with predicted_nvh_db, severity, recommendation
    """
    model_data = _get_model()
    pipeline   = model_data["pipeline"]

    features = np.array([[
        rpm, load_nm, temperature_c,
        stator_slots, rotor_poles, air_gap_mm,
    ]])

    predicted_db = float(pipeline.predict(features)[0])
    predicted_db = round(max(40.0, min(predicted_db, 100.0)), 2)

    # Severity classification
    if predicted_db < 55:
        severity = 1; assessment = "Acceptable — within typical design targets"
    elif predicted_db < 65:
        severity = 2; assessment = "Marginal — monitor during validation testing"
    elif predicted_db < 75:
        severity = 3; assessment = "Elevated — engineering review recommended"
    elif predicted_db < 85:
        severity = 4; assessment = "High — corrective action required before production"
    else:
        severity = 5; assessment = "Critical — immediate design change required"

    # BPF calculation for context
    bpf_hz = round((rpm / 60) * stator_slots / rotor_poles, 1)

    return {
        "predicted_nvh_db": predicted_db,
        "severity":         severity,
        "assessment":       assessment,
        "bpf_hz":           bpf_hz,
        "operating_point": {
            "rpm": rpm, "load_nm": load_nm,
            "temperature_c": temperature_c,
            "air_gap_mm": air_gap_mm,
        },
        "model_metrics": model_data["metrics"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# LANGCHAIN TOOL DECORATOR  (used by the agent in Step 3)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from langchain.tools import tool

    @tool
    def nvh_surrogate_tool(input_json: str) -> str:
        """
        Predict NVH noise level for a motor operating point using the
        surrogate ML model (replaces expensive Altair Flux simulation).

        Input: JSON string with keys: rpm, load_nm, temperature_c (optional),
               stator_slots (optional), rotor_poles (optional), air_gap_mm (optional)

        Example: {"rpm": 3000, "load_nm": 50, "temperature_c": 60}

        Returns: Predicted NVH dB level, severity, assessment, and BPF frequency.
        """
        try:
            params = json.loads(input_json)
            result = predict_nvh_level(**params)
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e), "hint": "Check input JSON format"})

except ImportError:
    # langchain not installed — tool decorator not available
    nvh_surrogate_tool = None


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN + TEST  (runs when you execute this file directly)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Training NVH Surrogate Model")
    print("=" * 60)

    print("\n[1/3] Loading training data...")
    X, y = load_training_data()

    print("\n[2/3] Training models...")
    result = train_surrogate(X, y)

    print("\n[3/3] Saving champion model...")
    save_model(result)

    print("\n" + "=" * 60)
    print("SURROGATE MODEL TESTS")
    print("=" * 60)

    test_cases = [
        {"rpm": 3000, "load_nm": 80, "temperature_c": 60,
         "description": "High load, elevated temp — expect high NVH"},
        {"rpm": 1000, "load_nm": 10, "temperature_c": 25,
         "description": "Low speed, light load — expect low NVH"},
        {"rpm": 6000, "load_nm": 120, "temperature_c": 100,
         "description": "Maximum speed + load + temp — expect critical NVH"},
        {"rpm": 3000, "load_nm": 50, "air_gap_mm": 1.5,
         "description": "Large air gap — EM forces reduced, lower NVH"},
        {"rpm": 3000, "load_nm": 50, "air_gap_mm": 0.3,
         "description": "Small air gap — high EM forces, higher NVH"},
    ]

    for tc in test_cases:
        desc = tc.pop("description")
        result_pred = predict_nvh_level(**tc)
        print(f"\n  {desc}")
        print(f"  Input: {tc}")
        print(f"  → {result_pred['predicted_nvh_db']} dB | Severity {result_pred['severity']}")
        print(f"  → {result_pred['assessment']}")
        print(f"  → BPF: {result_pred['bpf_hz']} Hz")

    print("\n" + "=" * 60)
    print("✅ Surrogate model ready!")
    print("   Next: Build the LangGraph StateGraph (Step 3)")
    print("=" * 60)
