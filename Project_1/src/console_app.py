"""
console_app.py
----------------
Entry point for the Utility Usage Prediction Tool.

A menu-driven console application that lets a user:
1. View existing usage records
2. Add a new usage record
3. Update an existing record
4. Train / retrain the ML model
5. Predict utility usage for a new household
6. View last model performance metrics
0. Exit

Run with:  python src/console_app.py
"""

import sys

from config import VALID_HOME_TYPES, VALID_SEASONS, TARGET_COLUMN
from exceptions import (
    UtilityToolError, InvalidInputError, DatasetError, ModelNotTrainedError
)
from data_handler import (
    load_dataset, append_record, get_next_record_id,
    update_record, dataset_row_count,
)
from validators import (
    validate_int_range, validate_float_range,
    validate_home_type, validate_season,
)
from ml_pipeline import train_model, load_metrics, predict_single


MENU_TEXT = """
==================================================
   UTILITY USAGE PREDICTION TOOL  (CodeVedX AI/ML)
==================================================
  1. View usage records
  2. Add a new usage record
  3. Update an existing record
  4. Train / retrain prediction model
  5. Predict usage for a household
  6. View last model performance
  0. Exit
==================================================
"""


def print_header():
    print(MENU_TEXT)
    try:
        count = dataset_row_count()
        print(f"  Dataset currently has {count} records.\n")
    except Exception:
        pass


def prompt(field_label):
    """Small wrapper around input() so every prompt looks consistent."""
    return input(f"  > {field_label}: ").strip()


# ---------------------------------------------------------------------------
# Menu option handlers
# ---------------------------------------------------------------------------

def handle_view_records():
    try:
        rows = load_dataset()
    except DatasetError as e:
        print(f"\n[ERROR] {e}\n")
        return

    print(f"\nShowing last 10 of {len(rows)} records:\n")
    header = f"{'ID':<5}{'Size':<6}{'Area':<8}{'Type':<18}{'Season':<10}{'AC hrs':<8}{'Units':<8}{'Bill':<10}"
    print(header)
    print("-" * len(header))

    for row in rows[-10:]:
        print(
            f"{row.get('record_id',''):<5}"
            f"{row.get('household_size',''):<6}"
            f"{row.get('home_area_sqft',''):<8}"
            f"{row.get('home_type',''):<18}"
            f"{row.get('season',''):<10}"
            f"{row.get('ac_usage_hours',''):<8}"
            f"{row.get('units_consumed',''):<8}"
            f"{row.get('bill_amount',''):<10}"
        )
    print()


def _collect_household_input():
    """Shared input-collection logic used by both Add and Predict flows.
    Returns a dict of validated values. Raises InvalidInputError on bad input."""
    data = {}
    data["household_size"] = validate_int_range(
        prompt("Household size (1-10)"), "Household size", 1, 10
    )
    data["home_area_sqft"] = validate_int_range(
        prompt("Home area in sqft (200-6000)"), "Home area", 200, 6000
    )
    data["home_type"] = validate_home_type(
        prompt(f"Home type {VALID_HOME_TYPES}")
    )
    data["season"] = validate_season(
        prompt(f"Season {VALID_SEASONS}")
    )
    data["avg_temp_c"] = validate_float_range(
        prompt("Average temperature in Celsius (-5 to 50)"), "Average temperature", -5, 50
    )
    data["ac_usage_hours"] = validate_float_range(
        prompt("Daily AC usage hours (0-24)"), "AC usage hours", 0, 24
    )
    data["num_appliances"] = validate_int_range(
        prompt("Number of major appliances (1-30)"), "Number of appliances", 1, 30
    )
    data["prev_month_units"] = validate_float_range(
        prompt("Previous month's units consumed (0-2000)"), "Previous month units", 0, 2000
    )
    return data


def handle_add_record():
    print("\n-- Add New Usage Record --")
    try:
        data = _collect_household_input()
        actual_units = validate_float_range(
            prompt("Actual units consumed this month (0-2000)"), "Units consumed", 0, 2000
        )
        actual_bill = validate_float_range(
            prompt("Actual bill amount (0-50000)"), "Bill amount", 0, 50000
        )
    except InvalidInputError as e:
        print(f"\n[INVALID INPUT] {e}\n")
        return

    record = data.copy()
    record["units_consumed"] = actual_units
    record["bill_amount"] = actual_bill
    record["record_id"] = get_next_record_id()

    try:
        append_record(record)
    except DatasetError as e:
        print(f"\n[ERROR] {e}\n")
        return

    print(f"\nRecord added successfully with ID {record['record_id']}.\n")


def handle_update_record():
    print("\n-- Update Existing Record --")
    record_id = prompt("Enter record ID to update")

    try:
        field = prompt("Field to update (e.g. ac_usage_hours, num_appliances, units_consumed)")
        new_value = prompt("New value")
        updated = update_record(record_id, {field: new_value})
    except DatasetError as e:
        print(f"\n[ERROR] {e}\n")
        return

    if updated:
        print(f"\nRecord {record_id} updated successfully.\n")
    else:
        print(f"\nNo record found with ID {record_id}.\n")


def handle_train_model():
    print("\n-- Train / Retrain Model --")
    choice = prompt("Model type: [1] Random Forest  [2] Linear Regression (default 1)")
    model_type = "linear_regression" if choice.strip() == "2" else "random_forest"

    try:
        print("\nTraining in progress...")
        metrics = train_model(model_type=model_type)
    except DatasetError as e:
        print(f"\n[ERROR] {e}\n")
        return

    print("\nTraining complete. Performance on held-out test data:")
    print(f"  Model type : {metrics['model_type']}")
    print(f"  MAE        : {metrics['mae']} units")
    print(f"  RMSE       : {metrics['rmse']} units")
    print(f"  R^2 Score  : {metrics['r2_score']}")
    print(f"  Train/Test : {metrics['n_train']} / {metrics['n_test']} rows\n")


def handle_predict():
    print("\n-- Predict Utility Usage --")
    try:
        data = _collect_household_input()
    except InvalidInputError as e:
        print(f"\n[INVALID INPUT] {e}\n")
        return

    try:
        prediction = predict_single(data)
    except ModelNotTrainedError as e:
        print(f"\n[ERROR] {e}\n")
        return

    print(f"\nPredicted electricity usage: ~{prediction} units this month.")

    # rough cost estimate using same slab logic as dataset generator,
    # just so the user gets a sense of expected bill too
    def estimate_bill(u):
        if u <= 100:
            return u * 4.0
        elif u <= 300:
            return 100 * 4.0 + (u - 100) * 6.5
        else:
            return 100 * 4.0 + 200 * 6.5 + (u - 300) * 8.2

    est_bill = round(estimate_bill(prediction), 2)
    print(f"Estimated bill amount: ~Rs. {est_bill}\n")


def handle_view_metrics():
    print("\n-- Last Model Performance --")
    metrics = load_metrics()
    if metrics is None:
        print("No model has been trained yet. Use option 4 first.\n")
        return

    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    while True:
        print_header()
        choice = prompt("Select an option")

        try:
            if choice == "1":
                handle_view_records()
            elif choice == "2":
                handle_add_record()
            elif choice == "3":
                handle_update_record()
            elif choice == "4":
                handle_train_model()
            elif choice == "5":
                handle_predict()
            elif choice == "6":
                handle_view_metrics()
            elif choice == "0":
                print("\nExiting Utility Usage Prediction Tool. Goodbye!\n")
                sys.exit(0)
            else:
                print("\nInvalid menu option. Please choose a number from the menu.\n")
        except UtilityToolError as e:
            # Catch-all safety net for any custom exception not handled
            # inside a specific handler above.
            print(f"\n[ERROR] {e}\n")
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting.\n")
            sys.exit(0)
        except Exception as e:
            # Last-resort handler so the whole console app never crashes
            # with a raw traceback in front of the user.
            print(f"\n[UNEXPECTED ERROR] {type(e).__name__}: {e}\n")


if __name__ == "__main__":
    main()
