"""
feature_engineering.py
------------------------
Creates derived features that help the ML models capture relationships
that raw columns alone don't express well. This is what separates a
"medium-level, thoughtfully built" project from a bare-minimum one
that just feeds raw columns into a model.

Engineered features:
    1. study_attendance_ratio - combines study effort with classroom
       presence into a single "engagement" signal.
    2. effort_index - a weighted composite of study hours, assignment
       completion and attendance, capturing overall student effort
       in one number.
"""

import pandas as pd
import numpy as np

import config


def add_study_attendance_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """A student who studies a lot AND attends class regularly should
    score higher on this ratio than one who only does one of the two.
    Adding 1 to attendance avoids division-by-zero issues.
    """
    df = df.copy()
    df["study_attendance_ratio"] = (
        df["study_hours_per_day"] * df["attendance_percentage"] / 100
    )
    return df


def add_effort_index(df: pd.DataFrame) -> pd.DataFrame:
    """A composite 0-10 scale combining three effort-related signals,
    each normalized to comparable ranges before combining.
    """
    df = df.copy()

    norm_study = (df["study_hours_per_day"] / 9) * 10          # max ~9 hrs
    norm_attendance = (df["attendance_percentage"] / 100) * 10
    norm_assignments = (df["assignments_completed"] / 10) * 10

    df["effort_index"] = (
        0.45 * norm_study + 0.30 * norm_attendance + 0.25 * norm_assignments
    ).round(2)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Applies all feature engineering steps in sequence."""
    df = add_study_attendance_ratio(df)
    df = add_effort_index(df)
    return df


if __name__ == "__main__":
    raw_df = pd.read_csv(config.CLEAN_DATA_PATH)
    engineered_df = engineer_features(raw_df)

    print("New columns added:", config.ENGINEERED_FEATURES)
    print(engineered_df[config.ENGINEERED_FEATURES].describe())

    corr = engineered_df.select_dtypes(include=[np.number]).corr()[
        config.TARGET_REGRESSION
    ]
    print("\nEngineered feature correlation with final_marks:")
    print(corr[config.ENGINEERED_FEATURES])
