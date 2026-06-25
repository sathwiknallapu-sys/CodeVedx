"""
---------------------------------------------------------
AI Based Fake News Detection Tool
Helper Module
---------------------------------------------------------

Author : Your Name

Description:
This module contains reusable utility functions used
throughout the project.

Functions:
1. Read CSV dataset
2. Save ML model
3. Load ML model
4. Create folders
5. Format confidence score
6. Print model metrics
---------------------------------------------------------
"""

import os
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from utils.logger import (
    log_info,
    log_error
)


# ---------------------------------------------------------
# Create Folder
# ---------------------------------------------------------

def create_directory(folder_path: str):
    """
    Creates a directory if it doesn't already exist.
    """

    try:
        os.makedirs(folder_path, exist_ok=True)
        log_info(f"Directory created: {folder_path}")

    except Exception as e:
        log_error(str(e))
        raise


# ---------------------------------------------------------
# Load CSV Dataset
# ---------------------------------------------------------

def load_dataset(csv_path: str):
    """
    Loads dataset from CSV file.
    """

    try:

        df = pd.read_csv(csv_path)

        log_info("Dataset loaded successfully.")

        return df

    except FileNotFoundError:

        log_error("Dataset file not found.")

        raise

    except Exception as e:

        log_error(str(e))

        raise


# ---------------------------------------------------------
# Save Model
# ---------------------------------------------------------

def save_model(model, filepath: str):
    """
    Save trained ML model.
    """

    try:

        joblib.dump(model, filepath)

        log_info(f"Model saved at {filepath}")

    except Exception as e:

        log_error(str(e))

        raise


# ---------------------------------------------------------
# Load Model
# ---------------------------------------------------------

def load_model(filepath: str):
    """
    Load trained ML model.
    """

    try:

        model = joblib.load(filepath)

        log_info("Model loaded successfully.")

        return model

    except Exception as e:

        log_error(str(e))

        raise


# ---------------------------------------------------------
# Confidence Score
# ---------------------------------------------------------

def confidence_percentage(score: float):
    """
    Convert probability into percentage.

    Example:
        0.93 -> 93.00%
    """

    return f"{score * 100:.2f}%"


# ---------------------------------------------------------
# Print Evaluation Metrics
# ---------------------------------------------------------

def evaluate_predictions(y_true, y_pred):
    """
    Prints classification metrics.
    """

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred
    )

    recall = recall_score(
        y_true,
        y_pred
    )

    f1 = f1_score(
        y_true,
        y_pred
    )

    print("\n========== MODEL PERFORMANCE ==========\n")

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nClassification Report\n")

    print(classification_report(
        y_true,
        y_pred
    ))

    print("\nConfusion Matrix\n")

    print(confusion_matrix(
        y_true,
        y_pred
    ))


# ---------------------------------------------------------
# Validate User Input
# ---------------------------------------------------------

def validate_news(news: str):
    """
    Validates user input.
    """

    if not isinstance(news, str):
        return False

    if len(news.strip()) < 20:
        return False

    return True


# ---------------------------------------------------------
# Manual Testing
# ---------------------------------------------------------

if __name__ == "__main__":

    print(validate_news("India won the World Cup."))

    print(confidence_percentage(0.9234))