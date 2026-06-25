"""
ml_pipeline.py
---------------
Core machine learning logic for the Utility Usage Prediction Tool.

Pipeline steps:
1. Load data (via data_handler)
2. Preprocess  -> handle types, encode categorical columns
3. Feature engineering -> derive a couple of extra useful features
4. Train/test split
5. Train a regression model (RandomForestRegressor)
6. Evaluate (MAE, RMSE, R2)
7. Save model + encoders + metrics to disk for later reuse
"""

import json
import pickle
import os

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

from config import (
    MODEL_DIR, MODEL_PATH, ENCODER_PATH, METRICS_PATH,
    CATEGORICAL_COLUMNS, TARGET_COLUMN,
)
from data_handler import load_dataset
from exceptions import DatasetError, ModelNotTrainedError


NUMERIC_COLUMNS = [
    "household_size", "home_area_sqft", "avg_temp_c",
    "ac_usage_hours", "num_appliances", "prev_month_units",
]


def _rows_to_matrix(rows, encoders=None, fit_encoders=False):
    """Convert list-of-dict CSV rows into a numeric feature matrix X
    and target vector y (y is None if TARGET_COLUMN absent, e.g. for
    a single prediction request).

    Also performs basic feature engineering:
    - area_per_person: home_area_sqft / household_size
    - appliance_density: num_appliances / household_size
    These tend to carry more predictive signal than the raw columns alone.
    """
    if encoders is None:
        encoders = {}

    X_rows = []
    y_values = []

    for row in rows:
        try:
            household_size = float(row["household_size"])
            home_area_sqft = float(row["home_area_sqft"])
            avg_temp_c = float(row["avg_temp_c"])
            ac_usage_hours = float(row["ac_usage_hours"])
            num_appliances = float(row["num_appliances"])
            prev_month_units = float(row["prev_month_units"])
        except (KeyError, ValueError) as e:
            raise DatasetError(f"Invalid numeric value in row: {e}")

        # --- feature engineering ---
        area_per_person = home_area_sqft / household_size if household_size else 0
        appliance_density = num_appliances / household_size if household_size else 0

        # --- encode categorical columns ---
        encoded_cats = []
        for col in CATEGORICAL_COLUMNS:
            raw_val = row.get(col, "Unknown")
            if fit_encoders:
                if col not in encoders:
                    encoders[col] = LabelEncoder()
                    encoders[col].fit_classes = set()
                # collect classes first pass; actual fit happens after loop
                encoders[col].fit_classes.add(raw_val)
                encoded_cats.append(raw_val)  # placeholder, fixed below
            else:
                le = encoders.get(col)
                if le is None:
                    raise ModelNotTrainedError(
                        "Encoders not available. Train the model first."
                    )
                if raw_val not in le.classes_:
                    # unseen category at prediction time -> fall back to
                    # the first known class rather than crashing
                    raw_val = le.classes_[0]
                encoded_cats.append(le.transform([raw_val])[0])

        feature_row = [
            household_size, home_area_sqft, avg_temp_c,
            ac_usage_hours, num_appliances, prev_month_units,
            area_per_person, appliance_density,
        ] + (encoded_cats if not fit_encoders else [])

        X_rows.append((feature_row, encoded_cats if fit_encoders else None, row))

        if TARGET_COLUMN in row and row[TARGET_COLUMN] not in (None, ""):
            y_values.append(float(row[TARGET_COLUMN]))

    if fit_encoders:
        # Now properly fit LabelEncoders on the collected classes
        for col in CATEGORICAL_COLUMNS:
            encoders[col] = LabelEncoder()
            classes = sorted({row.get(col, "Unknown") for row in rows})
            encoders[col].fit(classes)

        # Re-encode with fitted encoders
        final_X = []
        for feature_row, _, row in X_rows:
            encoded_cats = [
                encoders[col].transform([row.get(col, "Unknown")])[0]
                for col in CATEGORICAL_COLUMNS
            ]
            final_X.append(feature_row + encoded_cats)
        X = np.array(final_X, dtype=float)
    else:
        X = np.array([fr for fr, _, _ in X_rows], dtype=float)

    y = np.array(y_values, dtype=float) if y_values else None
    return X, y, encoders


def preprocess_and_engineer(rows, encoders=None, fit_encoders=False):
    """Public wrapper around _rows_to_matrix for clarity in calling code."""
    return _rows_to_matrix(rows, encoders=encoders, fit_encoders=fit_encoders)


def train_model(model_type="random_forest", test_size=0.2, random_state=42):
    """Train a regression model on the dataset and persist it to disk.

    Returns a dict of evaluation metrics.
    """
    rows = load_dataset()

    # Filter out rows with missing target (shouldn't normally happen)
    rows = [r for r in rows if r.get(TARGET_COLUMN) not in (None, "")]
    if len(rows) < 10:
        raise DatasetError("Not enough valid rows to train a model (need >= 10).")

    X, y, encoders = preprocess_and_engineer(rows, fit_encoders=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if model_type == "linear_regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(
            n_estimators=150, max_depth=12, random_state=random_state
        )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "model_type": model_type,
        "mae": round(float(mae), 3),
        "rmse": round(rmse, 3),
        "r2_score": round(float(r2), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(encoders, f)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def load_model():
    """Load the trained model and encoders from disk.
    Raises ModelNotTrainedError if not found."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        raise ModelNotTrainedError(
            "No trained model found. Please train the model first (Menu option)."
        )

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(ENCODER_PATH, "rb") as f:
        encoders = pickle.load(f)

    return model, encoders


def load_metrics():
    """Return the metrics dict saved after the last training run, or None."""
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


def predict_single(input_dict, model=None, encoders=None):
    """Predict units_consumed for a single household input dict.
    Loads model/encoders from disk if not provided."""
    if model is None or encoders is None:
        model, encoders = load_model()

    X, _, _ = preprocess_and_engineer([input_dict], encoders=encoders, fit_encoders=False)
    prediction = model.predict(X)[0]
    return round(float(prediction), 1)
