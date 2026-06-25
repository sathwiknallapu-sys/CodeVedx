"""
config.py
---------
Central configuration for the Student Performance Prediction System.
Keeping paths and constants in one place avoids "magic strings"
scattered across the codebase and makes the project easy to reconfigure.
"""

import os

# --- Path configuration ---------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

RAW_DATA_PATH = os.path.join(DATA_DIR, "student_performance.csv")
CLEAN_DATA_PATH = os.path.join(DATA_DIR, "student_performance_clean.csv")

REGRESSION_MODEL_PATH = os.path.join(MODELS_DIR, "marks_regressor.pkl")
CLASSIFIER_MODEL_PATH = os.path.join(MODELS_DIR, "performance_classifier.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "feature_scaler.pkl")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "metadata.json")

# --- Feature configuration -------------------------------------------
NUMERIC_FEATURES = [
    "attendance_percentage",
    "study_hours_per_day",
    "previous_sem_marks",
    "assignments_completed",
    "extracurricular_hours",
    "sleep_hours",
    "parental_support",
    "internet_access",
]

ENGINEERED_FEATURES = [
    "study_attendance_ratio",
    "effort_index",
]

ALL_FEATURES = NUMERIC_FEATURES + ENGINEERED_FEATURES

TARGET_REGRESSION = "final_marks"
TARGET_CLASSIFICATION = "performance_category"

PERFORMANCE_CATEGORIES = ["At Risk", "Average", "Good", "Excellent"]

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Ensure directories exist when this module is imported
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
