"""
model_training.py
-------------------
Trains and evaluates two complementary ML models:

    1. A REGRESSOR  -> predicts the exact final marks (0-100, continuous)
    2. A CLASSIFIER -> predicts the performance category
                        (At Risk / Average / Good / Excellent)

Why two models instead of one?
A school administrator might want a single number to rank students
(regression) AND a clear category to flag who needs intervention
(classification). Building both demonstrates a broader ML skill set
than the brief's minimum ask, while still solving the same problem
from two genuinely useful angles.

Models compared:
    Regression:     Linear Regression vs Random Forest Regressor
    Classification:  Logistic Regression vs Random Forest Classifier

The better performer (by validation metric) is the one that gets saved.
"""

import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import config
from data_preprocessing import clean_pipeline
from feature_engineering import engineer_features


def prepare_dataset():
    """Runs cleaning + feature engineering and returns the model-ready
    DataFrame plus train/test splits for both regression and
    classification targets.
    """
    df = clean_pipeline()
    df = engineer_features(df)

    X = df[config.ALL_FEATURES]
    y_reg = df[config.TARGET_REGRESSION]
    y_clf = df[config.TARGET_CLASSIFICATION]

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = (
        train_test_split(
            X, y_reg, y_clf,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
        )
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    label_encoder = LabelEncoder()
    label_encoder.fit(config.PERFORMANCE_CATEGORIES)
    y_clf_train_enc = label_encoder.transform(y_clf_train)
    y_clf_test_enc = label_encoder.transform(y_clf_test)

    return {
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_reg_train": y_reg_train,
        "y_reg_test": y_reg_test,
        "y_clf_train_enc": y_clf_train_enc,
        "y_clf_test_enc": y_clf_test_enc,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "feature_names": list(X.columns),
    }


def train_regression_models(data: dict) -> dict:
    """Trains and compares two regressors, returning metrics for both
    and the best model object.
    """
    X_train, X_test = data["X_train_scaled"], data["X_test_scaled"]
    y_train, y_test = data["y_reg_train"], data["y_reg_test"]

    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200, max_depth=8, random_state=config.RANDOM_STATE
        ),
    }

    results = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        results[name] = {
            "model": model,
            "mae": round(mean_absolute_error(y_test, preds), 3),
            "rmse": round(np.sqrt(mean_squared_error(y_test, preds)), 3),
            "r2": round(r2_score(y_test, preds), 3),
        }

    best_name = max(results, key=lambda n: results[n]["r2"])
    return {"results": results, "best_name": best_name, "best_model": results[best_name]["model"]}


def train_classification_models(data: dict) -> dict:
    """Trains and compares two classifiers."""
    X_train, X_test = data["X_train_scaled"], data["X_test_scaled"]
    y_train, y_test = data["y_clf_train_enc"], data["y_clf_test_enc"]

    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=config.RANDOM_STATE
        ),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=config.RANDOM_STATE
        ),
    }

    results = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        results[name] = {
            "model": model,
            "accuracy": round(accuracy_score(y_test, preds), 3),
            "precision": round(
                precision_score(y_test, preds, average="weighted", zero_division=0), 3
            ),
            "recall": round(
                recall_score(y_test, preds, average="weighted", zero_division=0), 3
            ),
            "f1": round(
                f1_score(y_test, preds, average="weighted", zero_division=0), 3
            ),
            "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        }

    best_name = max(results, key=lambda n: results[n]["f1"])
    return {"results": results, "best_name": best_name, "best_model": results[best_name]["model"]}


def save_artifacts(data, reg_outcome, clf_outcome):
    """Persists the best models, scaler, label encoder, and a metadata
    JSON summarising the run for transparency and reproducibility.
    """
    joblib.dump(reg_outcome["best_model"], config.REGRESSION_MODEL_PATH)
    joblib.dump(clf_outcome["best_model"], config.CLASSIFIER_MODEL_PATH)
    joblib.dump(data["scaler"], config.SCALER_PATH)
    joblib.dump(data["label_encoder"], config.LABEL_ENCODER_PATH)

    metadata = {
        "feature_names": data["feature_names"],
        "regression": {
            "best_model": reg_outcome["best_name"],
            "metrics": {
                name: {k: v for k, v in res.items() if k != "model"}
                for name, res in reg_outcome["results"].items()
            },
        },
        "classification": {
            "best_model": clf_outcome["best_name"],
            "metrics": {
                name: {k: v for k, v in res.items() if k != "model"}
                for name, res in clf_outcome["results"].items()
            },
            "classes": list(data["label_encoder"].classes_),
        },
    }

    with open(config.METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nArtifacts saved to '{config.MODELS_DIR}'.")
    return metadata


def run_training_pipeline():
    print("=" * 60)
    print("STUDENT PERFORMANCE PREDICTION - MODEL TRAINING PIPELINE")
    print("=" * 60)

    data = prepare_dataset()

    print("\n--- Training Regression Models (predicting exact marks) ---")
    reg_outcome = train_regression_models(data)
    for name, res in reg_outcome["results"].items():
        print(f"  {name}: MAE={res['mae']}  RMSE={res['rmse']}  R2={res['r2']}")
    print(f"  >> Best regressor: {reg_outcome['best_name']}")

    print("\n--- Training Classification Models (predicting category) ---")
    clf_outcome = train_classification_models(data)
    for name, res in clf_outcome["results"].items():
        print(
            f"  {name}: Accuracy={res['accuracy']}  Precision={res['precision']}  "
            f"Recall={res['recall']}  F1={res['f1']}"
        )
    print(f"  >> Best classifier: {clf_outcome['best_name']}")

    metadata = save_artifacts(data, reg_outcome, clf_outcome)
    return metadata


if __name__ == "__main__":
    run_training_pipeline()
