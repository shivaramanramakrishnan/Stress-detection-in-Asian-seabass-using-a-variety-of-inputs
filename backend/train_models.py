# train_models.py

import pandas as pd
from pathlib import Path

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

from synthetic_data import AquacultureSyntheticGenerator, SyntheticConfig


DATA_PATH      = Path("aquaculture_synthetic.csv")
REG_MODEL_PATH = Path("stress_regressor.joblib")
CLF_MODEL_PATH = Path("stress_classifier.joblib")


def generate_if_needed():
    from synthetic_data import NUMERIC_COLS

    if DATA_PATH.exists():
        existing_cols = set(pd.read_csv(DATA_PATH, nrows=0).columns)
        expected_cols = set(NUMERIC_COLS + ["system_type", "weather", "stress_score", "high_stress"])
        if expected_cols.issubset(existing_cols):
            print(f"{DATA_PATH} already exists and schema matches, skipping generation.")
            return
        else:
            print("Schema mismatch detected — regenerating data.")

    cfg = SyntheticConfig(n_samples=10_000, random_state=42)
    gen = AquacultureSyntheticGenerator(cfg)
    df  = gen.generate()
    df.to_csv(DATA_PATH, index=False)
    print(f"Generated and saved synthetic data to {DATA_PATH}")


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def build_preprocessor(df: pd.DataFrame):
    categorical_features = ["system_type", "weather"]

    # mortality_events excluded — it is an outcome of stress, not a cause.
    # Using it as a predictor creates circular reasoning.
    OUTCOME_FEATURES = ["mortality_events"]

    numeric_features = [
        c for c in df.columns
        if c not in categorical_features + ["stress_score", "high_stress"] + OUTCOME_FEATURES
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("scaler", StandardScaler())]),          numeric_features),
            ("cat", Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
        ]
    )

    return preprocessor, numeric_features, categorical_features


def train_regressor(df: pd.DataFrame, preprocessor, numeric_features, categorical_features):
    X = df[numeric_features + categorical_features]
    y = df["stress_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # clone() gives this pipeline its own independent preprocessor copy —
    # prevents the classifier's fit() from overwriting this scaler's parameters
    model = Pipeline(steps=[
        ("preprocess", clone(preprocessor)),
        ("regressor",  RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2   = r2_score(y_test, y_pred)

    print("=== Regression (stress_score) ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

    # Feature importance
    feature_names = (
        numeric_features +
        model.named_steps["preprocess"]
        .named_transformers_["cat"]
        .get_feature_names_out(categorical_features).tolist()
    )
    importances = model.named_steps["regressor"].feature_importances_
    top_10 = sorted(zip(feature_names, importances), key=lambda x: -x[1])[:10]
    print("\nTop 10 features by importance (regressor):")
    for name, imp in top_10:
        print(f"  {name:<35} {imp:.4f}")

    joblib.dump(model, REG_MODEL_PATH)
    print(f"\nSaved regression model to {REG_MODEL_PATH}")


def train_classifier(df: pd.DataFrame, preprocessor, numeric_features, categorical_features):
    X = df[numeric_features + categorical_features]
    y = df["high_stress"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123, stratify=y
    )

    # clone() gives this pipeline its own independent preprocessor copy
    model = Pipeline(steps=[
        ("preprocess", clone(preprocessor)),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight="balanced",
            random_state=123,
            n_jobs=-1,
        )),
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)

    print("=== Classification (high_stress) ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 score: {f1:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=3))

    joblib.dump(model, CLF_MODEL_PATH)
    print(f"Saved classifier model to {CLF_MODEL_PATH}")


if __name__ == "__main__":
    generate_if_needed()

    df = load_data()

    preprocessor, num_feats, cat_feats = build_preprocessor(df)

    train_regressor(df, preprocessor, num_feats, cat_feats)
    train_classifier(df, preprocessor, num_feats, cat_feats)