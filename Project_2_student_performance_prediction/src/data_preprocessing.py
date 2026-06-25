"""
data_preprocessing.py
----------------------
Handles loading, cleaning, and exploratory analysis of the raw student
dataset. This module is intentionally separated from feature
engineering and model training so each stage of the ML pipeline can be
tested, modified, or re-run independently (modular programming).

Responsibilities:
    1. Load raw CSV safely (with error handling)
    2. Handle missing values
    3. Validate value ranges
    4. Provide basic EDA summaries
    5. Save the cleaned dataset
"""

import os
import pandas as pd
import numpy as np

import config
from exceptions import DataFileError, DataValidationError


def load_raw_data(path: str = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Loads the raw CSV file with proper error handling.

    Raises:
        DataFileError: if the file does not exist or cannot be parsed.
    """
    if not os.path.exists(path):
        raise DataFileError(
            f"Dataset not found at '{path}'. "
            f"Run 'python generate_dataset.py' first to create it."
        )
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise DataFileError(f"Dataset file at '{path}' is empty.") from exc
    except pd.errors.ParserError as exc:
        raise DataFileError(f"Dataset file at '{path}' is malformed.") from exc

    if df.empty:
        raise DataFileError("Loaded dataset contains zero rows.")

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fills missing numeric values using median imputation.

    Median is chosen over mean because it is more robust to outliers,
    which is appropriate for attendance/study-hour data that can have
    skewed distributions.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    return df


def validate_data(df: pd.DataFrame) -> None:
    """Sanity-checks value ranges and raises DataValidationError on
    anything physically impossible (e.g. negative attendance).
    """
    checks = {
        "attendance_percentage": (0, 100),
        "study_hours_per_day": (0, 24),
        "previous_sem_marks": (0, 100),
        "sleep_hours": (0, 24),
        "final_marks": (0, 100),
    }

    for col, (low, high) in checks.items():
        if col not in df.columns:
            continue
        out_of_range = df[(df[col] < low) | (df[col] > high)]
        if not out_of_range.empty:
            raise DataValidationError(
                f"Column '{col}' has {len(out_of_range)} value(s) "
                f"outside the valid range [{low}, {high}]."
            )

    required_cols = config.NUMERIC_FEATURES + [
        config.TARGET_REGRESSION,
        config.TARGET_CLASSIFICATION,
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Missing required columns: {missing_cols}")


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Removes duplicate student records based on student_id."""
    before = len(df)
    df = df.drop_duplicates(subset="student_id", keep="first")
    removed = before - len(df)
    if removed > 0:
        print(f"  Removed {removed} duplicate record(s).")
    return df


def clean_pipeline(
    raw_path: str = config.RAW_DATA_PATH,
    save_path: str = config.CLEAN_DATA_PATH,
) -> pd.DataFrame:
    """Runs the full cleaning pipeline end-to-end and saves the result."""
    print("Step 1: Loading raw data...")
    df = load_raw_data(raw_path)
    print(f"  Loaded {df.shape[0]} rows, {df.shape[1]} columns.")

    print("Step 2: Removing duplicates...")
    df = remove_duplicates(df)

    print("Step 3: Handling missing values...")
    missing_before = df.isnull().sum().sum()
    df = handle_missing_values(df)
    missing_after = df.isnull().sum().sum()
    print(f"  Missing values: {missing_before} -> {missing_after}")

    print("Step 4: Validating data ranges...")
    validate_data(df)
    print("  All values within expected ranges.")

    df.to_csv(save_path, index=False)
    print(f"Step 5: Clean data saved to '{save_path}'.")

    return df


def get_eda_summary(df: pd.DataFrame) -> dict:
    """Returns a dictionary of basic exploratory statistics, useful for
    the README/report and for sanity-checking the dataset.
    """
    summary = {
        "n_records": len(df),
        "n_features": df.shape[1],
        "class_distribution": df[config.TARGET_CLASSIFICATION]
        .value_counts()
        .to_dict(),
        "mean_final_marks": round(df[config.TARGET_REGRESSION].mean(), 2),
        "correlation_with_target": df.select_dtypes(include=[np.number])
        .corr()[config.TARGET_REGRESSION]
        .sort_values(ascending=False)
        .round(3)
        .to_dict(),
    }
    return summary


if __name__ == "__main__":
    cleaned_df = clean_pipeline()
    print("\n--- EDA Summary ---")
    for key, value in get_eda_summary(cleaned_df).items():
        print(f"{key}: {value}")
