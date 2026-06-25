"""
predictor.py
-------------
Loads the trained models/scaler/encoder from disk and exposes a clean
`predict_student_performance()` function that the console UI (and
potentially a future Flask API) can call.

This module deliberately knows nothing about "where input comes from"
(console, web form, CSV row) - it only validates and predicts. This
separation of concerns is what makes the project genuinely modular
rather than one giant script.
"""

import os
import joblib
import numpy as np
import pandas as pd

import config
from exceptions import ModelNotTrainedError, DataValidationError
from feature_engineering import engineer_features


class StudentPerformancePredictor:
    """Wraps the trained regression + classification models behind a
    single, simple prediction interface.
    """

    def __init__(self):
        self.regressor = None
        self.classifier = None
        self.scaler = None
        self.label_encoder = None
        self._load_artifacts()

    def _load_artifacts(self):
        required_files = {
            "regressor": config.REGRESSION_MODEL_PATH,
            "classifier": config.CLASSIFIER_MODEL_PATH,
            "scaler": config.SCALER_PATH,
            "label_encoder": config.LABEL_ENCODER_PATH,
        }
        missing = [name for name, path in required_files.items() if not os.path.exists(path)]
        if missing:
            raise ModelNotTrainedError(
                f"Missing trained artifact(s): {missing}. "
                f"Run 'python model_training.py' first."
            )

        self.regressor = joblib.load(config.REGRESSION_MODEL_PATH)
        self.classifier = joblib.load(config.CLASSIFIER_MODEL_PATH)
        self.scaler = joblib.load(config.SCALER_PATH)
        self.label_encoder = joblib.load(config.LABEL_ENCODER_PATH)

    @staticmethod
    def validate_input(student_input: dict) -> None:
        """Validates a single student's raw input dictionary before
        prediction. Raises DataValidationError with a clear message
        on the first problem found.
        """
        bounds = {
            "attendance_percentage": (0, 100),
            "study_hours_per_day": (0, 24),
            "previous_sem_marks": (0, 100),
            "assignments_completed": (0, 10),
            "extracurricular_hours": (0, 40),
            "sleep_hours": (0, 24),
            "parental_support": (1, 5),
            "internet_access": (0, 1),
        }

        for field, (low, high) in bounds.items():
            if field not in student_input:
                raise DataValidationError(f"Missing required field: '{field}'")
            value = student_input[field]
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise DataValidationError(
                    f"Field '{field}' must be numeric, got {type(value).__name__}."
                )
            if not (low <= value <= high):
                raise DataValidationError(
                    f"Field '{field}' = {value} is outside valid range [{low}, {high}]."
                )

    def predict(self, student_input: dict) -> dict:
        """Takes a dict of raw student feature values and returns a
        dict with both the predicted exact marks and category, plus
        a basic confidence score for the category prediction.
        """
        self.validate_input(student_input)

        row = pd.DataFrame([student_input])
        row = engineer_features(row)
        row = row[config.ALL_FEATURES]

        scaled = self.scaler.transform(row)

        predicted_marks = float(np.clip(self.regressor.predict(scaled)[0], 0, 100))

        class_probs = self.classifier.predict_proba(scaled)[0]
        class_idx = int(np.argmax(class_probs))
        predicted_category = str(self.label_encoder.inverse_transform([class_idx])[0])
        confidence = float(round(class_probs[class_idx] * 100, 1))

        return {
            "predicted_marks": round(predicted_marks, 1),
            "predicted_category": predicted_category,
            "confidence_percent": confidence,
            "category_probabilities": {
                str(label): round(float(p) * 100, 1)
                for label, p in zip(self.label_encoder.classes_, class_probs)
            },
        }


if __name__ == "__main__":
    # Quick smoke test with a sample "good" student
    predictor = StudentPerformancePredictor()

    sample_student = {
        "attendance_percentage": 88,
        "study_hours_per_day": 4.5,
        "previous_sem_marks": 78,
        "assignments_completed": 9,
        "extracurricular_hours": 2,
        "sleep_hours": 7,
        "parental_support": 4,
        "internet_access": 1,
    }

    result = predictor.predict(sample_student)
    print("Sample prediction:")
    for k, v in result.items():
        print(f"  {k}: {v}")
