"""
config.py
---------
Central place for file paths and constant values used across the app.
Keeping these in one module avoids hard-coded strings scattered across
the codebase, and makes it easy to change paths later.
"""

import os

# Base directory = the project root (one level up from src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

DATASET_PATH = os.path.join(DATA_DIR, "utility_usage_data.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "utility_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoders.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

# Columns expected in the dataset. Used for validation when reading CSVs.
REQUIRED_COLUMNS = [
    "record_id",
    "household_size",
    "home_area_sqft",
    "home_type",
    "season",
    "avg_temp_c",
    "ac_usage_hours",
    "num_appliances",
    "prev_month_units",
    "units_consumed",
    "bill_amount",
]

CATEGORICAL_COLUMNS = ["home_type", "season"]
TARGET_COLUMN = "units_consumed"

VALID_HOME_TYPES = ["Apartment", "Independent House", "Villa"]
VALID_SEASONS = ["Summer", "Winter", "Monsoon", "Spring"]
