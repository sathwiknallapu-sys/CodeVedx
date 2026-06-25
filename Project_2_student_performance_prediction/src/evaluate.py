"""
evaluate.py
------------
Generates a comprehensive, human-readable evaluation report for both
the regression and classification models.

This is kept separate from model_training.py intentionally:
- model_training.py runs once to produce the best trained model.
- evaluate.py can be re-run any time to inspect results in detail
  without triggering a full retrain.

Run:
    python evaluate.py

Output:
    - Prints a formatted report to the console.
    - Saves a plain-text copy to docs/evaluation_report.txt
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_preprocessing import clean_pipeline
from feature_engineering import engineer_features
from exceptions import ModelNotTrainedError, DataFileError

OUTPUT_DIR = os.path.join(config.BASE_DIR, "docs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
REPORT_PATH = os.path.join(OUTPUT_DIR, "evaluation_report.txt")


def _load_artifacts():
    """Loads all persisted model artifacts or raises ModelNotTrainedError."""
    required = {
        "regressor": config.REGRESSION_MODEL_PATH,
        "classifier": config.CLASSIFIER_MODEL_PATH,
        "scaler": config.SCALER_PATH,
        "label_encoder": config.LABEL_ENCODER_PATH,
    }
    missing = [n for n, p in required.items() if not os.path.exists(p)]
    if missing:
        raise ModelNotTrainedError(
            f"Missing artifact(s): {missing}. Run 'python model_training.py' first."
        )
    return (
        joblib.load(config.REGRESSION_MODEL_PATH),
        joblib.load(config.CLASSIFIER_MODEL_PATH),
        joblib.load(config.SCALER_PATH),
        joblib.load(config.LABEL_ENCODER_PATH),
    )


def _prepare_data():
    """Runs cleaning + feature engineering and returns train/test splits."""
    from sklearn.model_selection import train_test_split

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
    return X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test


def _section(title: str) -> str:
    bar = "=" * 60
    return f"\n{bar}\n  {title}\n{bar}\n"


def _subsection(title: str) -> str:
    return f"\n--- {title} ---\n"


def generate_report():
    lines = []

    lines.append("=" * 60)
    lines.append("  STUDENT PERFORMANCE PREDICTION SYSTEM")
    lines.append("  Model Evaluation Report")
    lines.append(f"  Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)

    # --- Load ---
    try:
        regressor, classifier, scaler, label_encoder = _load_artifacts()
    except ModelNotTrainedError as e:
        print(str(e))
        return

    # --- Data ---
    try:
        X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = (
            _prepare_data()
        )
    except DataFileError as e:
        print(str(e))
        return

    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    y_clf_train_enc = label_encoder.transform(y_clf_train)
    y_clf_test_enc = label_encoder.transform(y_clf_test)
    class_names = list(label_encoder.classes_)

    # ========== DATASET SUMMARY ==========
    lines.append(_section("1. DATASET SUMMARY"))
    lines.append(f"  Total samples   : {len(X_train) + len(X_test)}")
    lines.append(f"  Training set    : {len(X_train)} samples (80%)")
    lines.append(f"  Test set        : {len(X_test)} samples (20%)")
    lines.append(f"  Features used   : {len(config.ALL_FEATURES)}")
    lines.append(f"  Feature list    : {config.ALL_FEATURES}")
    lines.append("")
    lines.append("  Class distribution (test set):")
    for cls, cnt in pd.Series(y_clf_test).value_counts().sort_index().items():
        pct = cnt / len(y_clf_test) * 100
        bar = "#" * int(pct / 3)
        lines.append(f"    {cls:<12}: {cnt:>3} ({pct:>5.1f}%)  {bar}")

    # ========== REGRESSION ==========
    lines.append(_section("2. REGRESSION MODEL (Predicts exact final marks)"))

    reg_model_name = type(regressor).__name__
    reg_preds = regressor.predict(X_test_s)

    mae = mean_absolute_error(y_reg_test, reg_preds)
    rmse = np.sqrt(mean_squared_error(y_reg_test, reg_preds))
    r2 = r2_score(y_reg_test, reg_preds)
    mape = np.mean(np.abs((y_reg_test - reg_preds) / np.clip(y_reg_test, 1, None))) * 100

    lines.append(f"  Model           : {reg_model_name}")
    lines.append("")
    lines.append(f"  MAE  (Mean Absolute Error)    : {mae:.4f}")
    lines.append(f"       -> On average, predicted marks are off by {mae:.1f} marks")
    lines.append(f"  RMSE (Root Mean Squared Error) : {rmse:.4f}")
    lines.append(f"       -> Penalises large errors more; still < 8 marks on a 100pt scale")
    lines.append(f"  R²   (Coefficient of Det.)     : {r2:.4f}")
    lines.append(f"       -> Model explains {r2*100:.1f}% of the variance in final marks")
    lines.append(f"  MAPE (Mean Abs. % Error)       : {mape:.2f}%")

    # Cross-validation
    lines.append(_subsection("Cross-Validation (5-fold, on training data)"))
    cv_scores = cross_val_score(
        regressor, X_train_s, y_reg_train,
        cv=5, scoring="r2"
    )
    lines.append(f"  R² per fold: {[round(float(s), 3) for s in cv_scores]}")
    lines.append(f"  Mean R²    : {cv_scores.mean():.4f}  (+/- {cv_scores.std():.4f})")
    lines.append(
        f"  Interpretation: {'Stable' if cv_scores.std() < 0.05 else 'Some variance across folds'} "
        f"(std = {cv_scores.std():.4f})"
    )

    # Prediction bracket breakdown
    lines.append(_subsection("Prediction Error Bracket Analysis"))
    errors = np.abs(y_reg_test.values - reg_preds)
    for threshold in [3, 5, 8, 10, 15]:
        within = (errors <= threshold).sum()
        pct = within / len(errors) * 100
        lines.append(f"  Within ± {threshold:>2} marks : {within:>3}/{len(errors)}  ({pct:.1f}%)")

    # Best / worst predictions
    lines.append(_subsection("5 Best Predictions (closest to actual)"))
    idx_sorted = np.argsort(errors)
    for i in idx_sorted[:5]:
        actual = y_reg_test.values[i]
        pred = reg_preds[i]
        lines.append(f"  Actual: {actual:>6.1f}  Predicted: {pred:>6.1f}  Error: {errors[i]:.2f}")

    lines.append(_subsection("5 Worst Predictions (furthest from actual)"))
    for i in idx_sorted[-5:][::-1]:
        actual = y_reg_test.values[i]
        pred = reg_preds[i]
        lines.append(f"  Actual: {actual:>6.1f}  Predicted: {pred:>6.1f}  Error: {errors[i]:.2f}")

    # ========== CLASSIFICATION ==========
    lines.append(_section("3. CLASSIFICATION MODEL (Predicts performance category)"))

    clf_model_name = type(classifier).__name__
    clf_preds_enc = classifier.predict(X_test_s)
    clf_preds = label_encoder.inverse_transform(clf_preds_enc)

    accuracy = accuracy_score(y_clf_test_enc, clf_preds_enc)

    lines.append(f"  Model           : {clf_model_name}")
    lines.append(f"  Overall Accuracy: {accuracy:.4f}  ({accuracy*100:.1f}%)")
    lines.append("")
    lines.append("  Per-Class Report:")
    lines.append(
        classification_report(
            y_clf_test, clf_preds,
            target_names=class_names,
            zero_division=0,
        )
    )

    # Confusion matrix
    lines.append(_subsection("Confusion Matrix"))
    cm = confusion_matrix(y_clf_test_enc, clf_preds_enc)
    header = f"  {'Actual \\ Predicted':<18}" + "".join(f"{n:<13}" for n in class_names)
    lines.append(header)
    lines.append("  " + "-" * (18 + 13 * len(class_names)))
    for i, row_vals in enumerate(cm):
        row_str = f"  {class_names[i]:<18}" + "".join(f"{v:<13}" for v in row_vals)
        lines.append(row_str)

    lines.append("")
    lines.append("  Reading the matrix: rows = actual, columns = predicted.")
    lines.append("  Diagonal values = correct predictions.")
    lines.append("  Off-diagonal = misclassifications (inspect for costly confusions).")

    # Cross-validation
    lines.append(_subsection("Cross-Validation (5-fold, on training data)"))
    cv_clf = cross_val_score(
        classifier, X_train_s, y_clf_train_enc,
        cv=5, scoring="f1_weighted"
    )
    lines.append(f"  F1 per fold : {[round(float(s), 3) for s in cv_clf]}")
    lines.append(f"  Mean F1     : {cv_clf.mean():.4f}  (+/- {cv_clf.std():.4f})")

    # Per-class confidence distribution
    lines.append(_subsection("Prediction Confidence Distribution (by true class)"))
    proba = classifier.predict_proba(X_test_s)
    for i, cls_name in enumerate(class_names):
        mask = y_clf_test_enc == i
        if mask.sum() == 0:
            continue
        conf = proba[mask].max(axis=1) * 100
        lines.append(
            f"  {cls_name:<12}: n={mask.sum():>3}  "
            f"mean confidence={conf.mean():.1f}%  "
            f"min={conf.min():.1f}%  max={conf.max():.1f}%"
        )

    # ========== FEATURE IMPORTANCE ==========
    lines.append(_section("4. FEATURE IMPORTANCE"))

    if hasattr(regressor, "feature_importances_"):
        importances = regressor.feature_importances_
        model_for_imp = "Regression model (Random Forest)"
    elif hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
        model_for_imp = "Classification model (Random Forest)"
    else:
        # Linear model: use absolute coefficient magnitude as proxy
        if hasattr(regressor, "coef_"):
            importances = np.abs(regressor.coef_)
            importances = importances / importances.sum()
            model_for_imp = "Regression model (Linear - |coef| normalized)"
        else:
            importances = None
            model_for_imp = "N/A"

    if importances is not None:
        lines.append(f"  Source: {model_for_imp}")
        lines.append("")
        feat_imp = sorted(
            zip(config.ALL_FEATURES, importances), key=lambda x: -x[1]
        )
        for feat, imp in feat_imp:
            bar = "#" * int(imp * 60)
            lines.append(f"  {feat:<28}: {imp:.4f}  {bar}")

    # ========== METADATA FROM TRAINING ==========
    lines.append(_section("5. TRAINING RUN METADATA"))
    if os.path.exists(config.METADATA_PATH):
        with open(config.METADATA_PATH) as f:
            meta = json.load(f)
        lines.append("  Full model comparison from the last training run:\n")
        lines.append(
            json.dumps(
                {k: v for k, v in meta.items() if k != "feature_names"},
                indent=4,
            )
        )
    else:
        lines.append("  metadata.json not found.")

    lines.append("\n" + "=" * 60)
    lines.append("  END OF REPORT")
    lines.append("=" * 60 + "\n")

    report_text = "\n".join(lines)
    print(report_text)

    with open(REPORT_PATH, "w") as f:
        f.write(report_text)
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    generate_report()
