# train_models.py

import pandas as pd
from pathlib import Path

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


DATA_PATH = Path("aquaculture_synthetic.csv")
REG_MODEL_PATH = Path("stress_regressor.joblib")
CLF_MODEL_PATH = Path("stress_classifier.joblib")


def generate_if_needed():
    if DATA_PATH.exists():
        print(f"{DATA_PATH} already exists, skipping generation.")
        return

    cfg = SyntheticConfig(n_samples=10_000, random_state=42)
    gen = AquacultureSyntheticGenerator(cfg)
    df = gen.generate()
    df.to_csv(DATA_PATH, index=False)
    print(f"Generated and saved synthetic data to {DATA_PATH}")


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def build_preprocessor(df: pd.DataFrame):
    # Separate feature types
    categorical_features = ["system_type", "weather"]
    numeric_features = [
        c
        for c in df.columns
        if c not in categorical_features + ["stress_score", "high_stress"]
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor, numeric_features, categorical_features


def train_regressor(df: pd.DataFrame, preprocessor, numeric_features, categorical_features):
    X = df[numeric_features + categorical_features]
    y = df["stress_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    regressor = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regressor", regressor),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)


    print("=== Regression (stress_score) ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2:  {r2:.4f}")

    joblib.dump(model, REG_MODEL_PATH)
    print(f"Saved regression model to {REG_MODEL_PATH}")


def train_classifier(df: pd.DataFrame, preprocessor, numeric_features, categorical_features):
    X = df[numeric_features + categorical_features]
    y = df["high_stress"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123, stratify=y
    )

    classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=123,
        n_jobs=-1,
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", classifier),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("=== Classification (high_stress) ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 score: {f1:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=3))

    joblib.dump(model, CLF_MODEL_PATH)
    print(f"Saved classifier model to {CLF_MODEL_PATH}")


if __name__ == "__main__":
    # 1. Generate synthetic data if it does not exist
    generate_if_needed()

    # 2. Load data
    df = load_data()

    # 3. Build shared preprocessor
    preprocessor, num_feats, cat_feats = build_preprocessor(df)

    # 4. Train models
    train_regressor(df, preprocessor, num_feats, cat_feats)
    train_classifier(df, preprocessor, num_feats, cat_feats)
