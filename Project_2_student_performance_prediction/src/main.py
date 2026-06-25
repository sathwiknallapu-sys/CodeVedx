"""
main.py
--------
Console-based user interface for the Student Performance Prediction
System. This is the entry point a user/evaluator actually runs.

Run:
    python main.py
"""

import os
import sys
import csv
from datetime import datetime

import config
from exceptions import (
    StudentPerformanceError,
    ModelNotTrainedError,
    DataValidationError,
)
from predictor import StudentPerformancePredictor

HISTORY_PATH = os.path.join(config.DATA_DIR, "prediction_history.csv")

FIELD_PROMPTS = [
    ("attendance_percentage", "Attendance percentage (0-100)", float, 0, 100),
    ("study_hours_per_day", "Study hours per day (0-24)", float, 0, 24),
    ("previous_sem_marks", "Previous semester marks (0-100)", float, 0, 100),
    ("assignments_completed", "Assignments completed out of 10 (0-10)", int, 0, 10),
    ("extracurricular_hours", "Extracurricular hours per week (0-40)", float, 0, 40),
    ("sleep_hours", "Average sleep hours per night (0-24)", float, 0, 24),
    ("parental_support", "Parental support level, 1=low to 5=high (1-5)", int, 1, 5),
    ("internet_access", "Internet access at home? 1=Yes, 0=No", int, 0, 1),
]


def print_banner():
    print("=" * 62)
    print("   STUDENT PERFORMANCE PREDICTION SYSTEM".center(62))
    print("   CodeVedX AI/ML Internship - Project 2".center(62))
    print("=" * 62)


def print_menu():
    print("\nMAIN MENU")
    print("  1. Predict performance for a new student")
    print("  2. Batch predict from a CSV file")
    print("  3. View prediction history")
    print("  4. View model performance summary")
    print("  5. Exit")


def get_validated_input(prompt_text, cast_type, low, high):
    """Repeatedly prompts the user until a valid value is entered.
    Handles non-numeric input and out-of-range values gracefully
    instead of crashing.
    """
    while True:
        raw = input(f"  {prompt_text}: ").strip()
        try:
            value = cast_type(raw)
        except ValueError:
            print(f"  Invalid input. Please enter a valid {cast_type.__name__}.")
            continue

        if not (low <= value <= high):
            print(f"  Value must be between {low} and {high}. Try again.")
            continue

        return value


def collect_student_input() -> dict:
    print("\nEnter student details below:")
    student_data = {}
    for field, prompt_text, cast_type, low, high in FIELD_PROMPTS:
        student_data[field] = get_validated_input(prompt_text, cast_type, low, high)
    return student_data


def log_prediction(student_input: dict, result: dict):
    """Appends each prediction to a CSV history file, creating the
    file with a header on first use.
    """
    file_exists = os.path.exists(HISTORY_PATH)
    with open(HISTORY_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ["timestamp"] + list(student_input.keys()) + [
                "predicted_marks", "predicted_category", "confidence_percent"
            ]
            writer.writerow(header)
        row = (
            [datetime.now().isoformat(timespec="seconds")]
            + list(student_input.values())
            + [result["predicted_marks"], result["predicted_category"], result["confidence_percent"]]
        )
        writer.writerow(row)


def display_result(result: dict):
    print("\n" + "-" * 45)
    print("  PREDICTION RESULT")
    print("-" * 45)
    print(f"  Predicted Final Marks   : {result['predicted_marks']} / 100")
    print(f"  Performance Category    : {result['predicted_category']}")
    print(f"  Confidence              : {result['confidence_percent']}%")
    print("\n  Category Probability Breakdown:")
    for category, prob in sorted(
        result["category_probabilities"].items(), key=lambda x: -x[1]
    ):
        bar = "#" * int(prob / 5)
        print(f"    {category:<10}: {prob:>5.1f}%  {bar}")
    print("-" * 45)

    if result["predicted_category"] == "At Risk":
        print("  NOTE: This student may benefit from academic support.")


def handle_single_prediction(predictor: StudentPerformancePredictor):
    try:
        student_input = collect_student_input()
        result = predictor.predict(student_input)
        display_result(result)
        log_prediction(student_input, result)
        print("  (Saved to prediction history.)")
    except DataValidationError as e:
        print(f"\n  Input error: {e}")
    except StudentPerformanceError as e:
        print(f"\n  Unexpected error: {e}")


def handle_batch_prediction(predictor: StudentPerformancePredictor):
    path = input("\n  Enter path to CSV file: ").strip()
    if not os.path.exists(path):
        print(f"  File not found: {path}")
        return

    import pandas as pd
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"  Could not read CSV: {e}")
        return

    required_cols = [f[0] for f in FIELD_PROMPTS]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"  CSV is missing required column(s): {missing}")
        return

    print(f"\n  Processing {len(df)} student record(s)...\n")
    success_count, error_count = 0, 0

    for idx, row in df.iterrows():
        student_input = {col: row[col] for col in required_cols}
        try:
            result = predictor.predict(student_input)
            log_prediction(student_input, result)
            print(
                f"  Row {idx + 1}: {result['predicted_marks']}/100 "
                f"-> {result['predicted_category']} ({result['confidence_percent']}%)"
            )
            success_count += 1
        except DataValidationError as e:
            print(f"  Row {idx + 1}: SKIPPED - {e}")
            error_count += 1

    print(f"\n  Batch complete. Success: {success_count}, Skipped: {error_count}")


def view_history():
    if not os.path.exists(HISTORY_PATH):
        print("\n  No prediction history yet.")
        return

    import pandas as pd
    df = pd.read_csv(HISTORY_PATH)
    print(f"\n  Total predictions made: {len(df)}")
    print(df.tail(10).to_string(index=False))


def view_model_summary():
    import json
    if not os.path.exists(config.METADATA_PATH):
        print("\n  No trained model metadata found. Run model_training.py first.")
        return

    with open(config.METADATA_PATH) as f:
        meta = json.load(f)

    print("\n  REGRESSION MODEL (predicts exact marks)")
    print(f"    Best model: {meta['regression']['best_model']}")
    for name, m in meta["regression"]["metrics"].items():
        print(f"    {name}: MAE={m['mae']}  RMSE={m['rmse']}  R2={m['r2']}")

    print("\n  CLASSIFICATION MODEL (predicts category)")
    print(f"    Best model: {meta['classification']['best_model']}")
    for name, m in meta["classification"]["metrics"].items():
        print(
            f"    {name}: Accuracy={m['accuracy']}  Precision={m['precision']}  "
            f"Recall={m['recall']}  F1={m['f1']}"
        )


def main():
    print_banner()

    try:
        predictor = StudentPerformancePredictor()
    except ModelNotTrainedError as e:
        print(f"\n  ERROR: {e}")
        sys.exit(1)

    while True:
        print_menu()
        choice = input("\nEnter your choice (1-5): ").strip()

        if choice == "1":
            handle_single_prediction(predictor)
        elif choice == "2":
            handle_batch_prediction(predictor)
        elif choice == "3":
            view_history()
        elif choice == "4":
            view_model_summary()
        elif choice == "5":
            print("\nThank you for using the Student Performance Prediction System!")
            break
        else:
            print("\n  Invalid choice. Please select 1-5.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSession interrupted by user. Goodbye!")
        sys.exit(0)
