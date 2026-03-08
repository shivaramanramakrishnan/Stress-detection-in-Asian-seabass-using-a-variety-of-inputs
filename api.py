from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import torch
from train_lstm import StressLSTM

from synthetic_data import AquacultureSyntheticGenerator, SyntheticConfig

# Paths to saved models (same as in train_models.py)
REG_MODEL_PATH = Path("stress_regressor.joblib")
CLF_MODEL_PATH = Path("stress_classifier.joblib")
LSTM_MODEL_PATH = Path("stress_lstm.pt")
LSTM_META_PATH  = Path("stress_lstm_meta.joblib")
lstm_model = None
lstm_meta  = None

app = FastAPI(title="Aquaculture Stress Early-Warning API")

# Allow frontend on localhost:3000 to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

reg_model = None
clf_model = None
NUM_FEATURES: List[str] = []
CAT_FEATURES: List[str] = []


def _extract_feature_lists_from_model(model) -> None:
    """
    Read numeric and categorical feature names from the ColumnTransformer
    inside the scikit-learn Pipeline.
    """
    global NUM_FEATURES, CAT_FEATURES
    ct = model.named_steps["preprocess"]
    num_cols = []
    cat_cols = []
    for name, transformer, cols in ct.transformers_:
        if name == "num":
            num_cols = list(cols)
        elif name == "cat":
            cat_cols = list(cols)
    NUM_FEATURES = num_cols
    CAT_FEATURES = cat_cols


def _ensure_models_trained_and_loaded() -> None:
    """
    If saved models do not exist, call the training script functions.
    Then load both models into memory and cache feature lists.
    """
    global reg_model, clf_model

    # Train if needed
    if not (REG_MODEL_PATH.exists() and CLF_MODEL_PATH.exists()):
        from train_models import (
            generate_if_needed,
            load_data,
            build_preprocessor,
            train_regressor,
            train_classifier,
        )

        print("No saved models found. Training models...")
        generate_if_needed()
        df = load_data()
        preprocessor, num_feats, cat_feats = build_preprocessor(df)
        train_regressor(df, preprocessor, num_feats, cat_feats)
        train_classifier(df, preprocessor, num_feats, cat_feats)
        print("Training complete.")

    # Load models
    print("Loading models from disk...")
    reg_model = joblib.load(REG_MODEL_PATH)
    clf_model = joblib.load(CLF_MODEL_PATH)

    # Load LSTM if available
    global lstm_model, lstm_meta
    if LSTM_MODEL_PATH.exists() and LSTM_META_PATH.exists():
        lstm_meta = joblib.load(LSTM_META_PATH)
        input_size = lstm_meta["mean"].shape[-1]
        lstm_model = StressLSTM(input_size=input_size)
        lstm_model.load_state_dict(
            torch.load(LSTM_MODEL_PATH, map_location="cpu", weights_only=True)
        )
        lstm_model.eval()
        print("LSTM model loaded.")
    else:
        print("No LSTM model found. Run train_lstm.py to enable /lstm-forecast.")

    # Extract feature lists
    _extract_feature_lists_from_model(reg_model)
    print("Models loaded. Ready to serve requests.")


@app.on_event("startup")
def startup_event():
    _ensure_models_trained_and_loaded()


@app.get("/health")
def health_check() -> Dict[str, Any]:
    return {"status": "ok"}


def _interpret_risk(stress_score: float) -> Dict[str, Any]:
    if stress_score < 0.33:
        level = "low"
        msg = "Fish are likely in a low-stress state. Conditions look broadly acceptable."
    elif stress_score < 0.66:
        level = "moderate"
        msg = (
            "Early stress signals present. Consider checking water quality and feeding behaviour more closely."
        )
    else:
        level = "high"
        msg = (
            "High stress risk detected. Early warning for potential health issues or performance losses."
        )
    return {"risk_level": level, "message": msg}


@app.get("/stress-ontology")
def stress_ontology() -> Dict[str, Any]:
    """
    Very simple ontology-like structure to show categories and observables.
    """
    return {
        "categories": {
            "environmental_abiotic": [
                "do_mg_per_l",
                "ph",
                "nh3_mg_per_l",
                "no2_mg_per_l",
                "co2_mg_per_l",
                "orp_mv",
                "temp_c",
                "salinity_psu",
            ],
            "behavioural": [
                "swimming_distance_m",
                "tailbeat_hz",
                "crowding_index",
                "biomass_kg",
                "fish_count",
            ],
            "biological": [
                "cortisol_ng_ml",
                "microbiome_dysbiosis",
                "feeding_grams",
                "feeding_events",
            ],
            "environmental_context": [
                "ambient_noise_db",
                "vibration_level",
                "weather",
                "system_type",
                "human_presence_count",
                "human_dwell_minutes",
            ],
            "growth_lifecycle": [
                "mortality_events",
                "grading_events",
                "awg_grams",
                "stocking_density",
            ],
        }
    }

@app.get("/lstm-forecast")
def lstm_forecast(steps: int = Query(24, ge=6, le=48)) -> Dict[str, Any]:
    """
    Generate a starting synthetic state, run it through the LSTM,
    and return a stress trajectory over `steps` timesteps.
    """
    global lstm_model, lstm_meta

    if lstm_model is None:
        return {"error": "LSTM model not loaded. Run train_lstm.py first."}

    # Generate one base sequence
    cfg = SyntheticConfig(n_samples=1, random_state=None)
    gen = AquacultureSyntheticGenerator(cfg)
    X, _, _ = gen.generate_sequences(n_sequences=1, timesteps=steps)

    # Normalize
    X_norm = (X - lstm_meta["mean"]) / (lstm_meta["std"] + 1e-8)
    x_tensor = torch.tensor(X_norm, dtype=torch.float32)

    with torch.no_grad():
        preds = lstm_model(x_tensor)[0].numpy()  # shape: (steps,)

    trajectory = []
    for i, score in enumerate(preds):
        score = float(np.clip(score, 0.0, 1.0))
        interp = _interpret_risk(score)
        trajectory.append(
            {
                "step": i + 1,
                "stress_score": score,
                "risk_level": interp["risk_level"],
            }
        )

    return {"forecast_steps": steps, "trajectory": trajectory}

@app.get("/synthetic-sample")
def synthetic_sample() -> Dict[str, Any]:
    """
    Generate a single synthetic scenario, run both models, and
    return features + predictions + simple interpretation.
    """
    global reg_model, clf_model, NUM_FEATURES, CAT_FEATURES

    # Generate 1-row synthetic dataset
    cfg = SyntheticConfig(n_samples=1, random_state=None)
    gen = AquacultureSyntheticGenerator(cfg)
    df = gen.generate(1)

    # Prepare input for models
    X = df[NUM_FEATURES + CAT_FEATURES]

    stress_score = float(reg_model.predict(X)[0])
    high_stress = int(clf_model.predict(X)[0])

    interpretation = _interpret_risk(stress_score)

    key_features = {
        "do_mg_per_l": float(df["do_mg_per_l"].iloc[0]),
        "ph": float(df["ph"].iloc[0]),
        "temp_c": float(df["temp_c"].iloc[0]),
        "nh3_mg_per_l": float(df["nh3_mg_per_l"].iloc[0]),
        "cortisol_ng_ml": float(df["cortisol_ng_ml"].iloc[0]),
        "microbiome_dysbiosis": float(df["microbiome_dysbiosis"].iloc[0]),
        "fish_count": float(df["fish_count"].iloc[0]),
        "crowding_index": float(df["crowding_index"].iloc[0]),
        "mortality_events": float(df["mortality_events"].iloc[0]),
        "weather": df["weather"].iloc[0],
        "system_type": df["system_type"].iloc[0],
    }

    return {
        "key_features": key_features,
        "all_features": df.to_dict(orient="records")[0],
        "predictions": {
            "stress_score": stress_score,
            "high_stress": bool(high_stress),
            **interpretation,
        },
    }


def _apply_simple_worsening_scenario(row: pd.Series, step_index: int) -> pd.Series:
    """
    For forecast: given a base feature row and a time step (1..T),
    apply a simple 'worsening water quality & density' scenario.
    This keeps things intentionally simple for the mini-project.
    """
    r = row.copy()
    t = float(step_index)

    # Gradually reduce DO, raise temp and ammonia
    r["do_mg_per_l"] = max(1.0, r["do_mg_per_l"] - 0.2 * t)
    r["temp_c"] = r["temp_c"] + 0.2 * t
    r["nh3_mg_per_l"] = r["nh3_mg_per_l"] + 0.03 * t

    # Slightly increase crowding index to simulate stress from density / behaviour
    if "crowding_index" in r:
        r["crowding_index"] = r["crowding_index"] + 0.02 * t

    # Keep other fields as-is for simplicity
    return r


@app.get("/scenario-forecast")
def scenario_forecast(
    steps: int = Query(12, ge=3, le=48),
) -> Dict[str, Any]:
    """
    Starting from one synthetic state, simulate a simple 'worsening conditions'
    scenario over the next `steps` time steps and run the existing models
    at each step to obtain a small stress risk trajectory.
    """
    global reg_model, clf_model, NUM_FEATURES, CAT_FEATURES

    # Base synthetic state
    cfg = SyntheticConfig(n_samples=1, random_state=None)
    gen = AquacultureSyntheticGenerator(cfg)
    df0 = gen.generate(1)
    base_row = df0.iloc[0]

    trajectory = []

    for i in range(1, steps + 1):
        scenario_row = _apply_simple_worsening_scenario(base_row, i)
        df_step = pd.DataFrame([scenario_row])
        X_step = df_step[NUM_FEATURES + CAT_FEATURES]

        stress_score = float(reg_model.predict(X_step)[0])
        high_stress = int(clf_model.predict(X_step)[0])
        interp = _interpret_risk(stress_score)

        trajectory.append(
            {
                "step": i,
                "stress_score": stress_score,
                "high_stress": bool(high_stress),
                "risk_level": interp["risk_level"],
                "message": interp["message"],
                "do_mg_per_l": float(df_step["do_mg_per_l"].iloc[0]),
                "temp_c": float(df_step["temp_c"].iloc[0]),
                "nh3_mg_per_l": float(df_step["nh3_mg_per_l"].iloc[0]),
            }
        )

    return {
        "base_state": df0.to_dict(orient="records")[0],
        "forecast_steps": steps,
        "trajectory": trajectory,
    }
