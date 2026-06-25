"""
test_project.py
-----------------
Unit tests for the Student Performance Prediction System.

Run:
    python -m pytest test_project.py -v
or simply:
    python test_project.py
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import config
from data_preprocessing import (
    handle_missing_values,
    validate_data,
    remove_duplicates,
    load_raw_data,
)
from feature_engineering import (
    add_study_attendance_ratio,
    add_effort_index,
    engineer_features,
)
from exceptions import DataValidationError, DataFileError, ModelNotTrainedError
from predictor import StudentPerformancePredictor


class TestDataPreprocessing(unittest.TestCase):

    def setUp(self):
        self.sample_df = pd.DataFrame({
            "student_id": ["S1", "S2", "S3"],
            "attendance_percentage": [80.0, np.nan, 95.0],
            "study_hours_per_day": [3.0, 4.0, np.nan],
            "previous_sem_marks": [70.0, 60.0, 90.0],
            "assignments_completed": [8, 6, 9],
            "extracurricular_hours": [2.0, 1.0, 3.0],
            "sleep_hours": [7.0, 6.5, 8.0],
            "parental_support": [3, 4, 5],
            "internet_access": [1, 1, 0],
            "final_marks": [72.0, 58.0, 88.0],
            "performance_category": ["Good", "Average", "Excellent"],
        })

    def test_missing_value_imputation(self):
        result = handle_missing_values(self.sample_df)
        self.assertEqual(result.isnull().sum().sum(), 0)

    def test_missing_value_uses_median(self):
        result = handle_missing_values(self.sample_df)
        expected_median = self.sample_df["attendance_percentage"].median()
        self.assertAlmostEqual(result.loc[1, "attendance_percentage"], expected_median)

    def test_validate_data_passes_on_clean_data(self):
        clean = handle_missing_values(self.sample_df)
        try:
            validate_data(clean)
        except DataValidationError:
            self.fail("validate_data() raised unexpectedly on valid data.")

    def test_validate_data_rejects_out_of_range(self):
        bad_df = self.sample_df.copy()
        bad_df.loc[0, "attendance_percentage"] = 150.0
        bad_df = handle_missing_values(bad_df)
        with self.assertRaises(DataValidationError):
            validate_data(bad_df)

    def test_remove_duplicates(self):
        dup_df = pd.concat([self.sample_df, self.sample_df.iloc[[0]]], ignore_index=True)
        result = remove_duplicates(dup_df)
        self.assertEqual(len(result), 3)

    def test_load_raw_data_missing_file_raises(self):
        with self.assertRaises(DataFileError):
            load_raw_data("/tmp/this_file_does_not_exist_xyz.csv")


class TestFeatureEngineering(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            "attendance_percentage": [80.0, 100.0, 0.0],
            "study_hours_per_day": [4.0, 0.0, 9.0],
            "assignments_completed": [8, 0, 10],
        })

    def test_study_attendance_ratio_calculation(self):
        result = add_study_attendance_ratio(self.df)
        expected_first = 4.0 * 80.0 / 100
        self.assertAlmostEqual(result.loc[0, "study_attendance_ratio"], expected_first)

    def test_study_attendance_ratio_zero_attendance(self):
        result = add_study_attendance_ratio(self.df)
        self.assertEqual(result.loc[2, "study_attendance_ratio"], 0.0)

    def test_effort_index_within_expected_bounds(self):
        result = add_effort_index(self.df)
        self.assertTrue((result["effort_index"] >= 0).all())
        self.assertTrue((result["effort_index"] <= 10).all())

    def test_engineer_features_adds_both_columns(self):
        result = engineer_features(self.df)
        for col in config.ENGINEERED_FEATURES:
            self.assertIn(col, result.columns)


class TestPredictor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # These tests assume model_training.py has already been run.
        try:
            cls.predictor = StudentPerformancePredictor()
        except ModelNotTrainedError:
            cls.predictor = None

    def setUp(self):
        if self.predictor is None:
            self.skipTest("Trained models not found - run model_training.py first.")

    def test_valid_prediction_returns_expected_keys(self):
        sample = {
            "attendance_percentage": 85, "study_hours_per_day": 4,
            "previous_sem_marks": 75, "assignments_completed": 8,
            "extracurricular_hours": 2, "sleep_hours": 7,
            "parental_support": 4, "internet_access": 1,
        }
        result = self.predictor.predict(sample)
        for key in ["predicted_marks", "predicted_category", "confidence_percent", "category_probabilities"]:
            self.assertIn(key, result)

    def test_predicted_marks_within_valid_range(self):
        sample = {
            "attendance_percentage": 95, "study_hours_per_day": 6,
            "previous_sem_marks": 90, "assignments_completed": 10,
            "extracurricular_hours": 1, "sleep_hours": 8,
            "parental_support": 5, "internet_access": 1,
        }
        result = self.predictor.predict(sample)
        self.assertGreaterEqual(result["predicted_marks"], 0)
        self.assertLessEqual(result["predicted_marks"], 100)

    def test_predicted_category_is_valid_label(self):
        sample = {
            "attendance_percentage": 50, "study_hours_per_day": 1,
            "previous_sem_marks": 40, "assignments_completed": 3,
            "extracurricular_hours": 5, "sleep_hours": 5,
            "parental_support": 2, "internet_access": 0,
        }
        result = self.predictor.predict(sample)
        self.assertIn(result["predicted_category"], config.PERFORMANCE_CATEGORIES)

    def test_missing_field_raises_validation_error(self):
        incomplete = {"attendance_percentage": 80, "study_hours_per_day": 3}
        with self.assertRaises(DataValidationError):
            self.predictor.predict(incomplete)

    def test_out_of_range_value_raises_validation_error(self):
        bad_input = {
            "attendance_percentage": 999, "study_hours_per_day": 4,
            "previous_sem_marks": 75, "assignments_completed": 8,
            "extracurricular_hours": 2, "sleep_hours": 7,
            "parental_support": 4, "internet_access": 1,
        }
        with self.assertRaises(DataValidationError):
            self.predictor.predict(bad_input)

    def test_wrong_type_raises_validation_error(self):
        bad_input = {
            "attendance_percentage": "very high", "study_hours_per_day": 4,
            "previous_sem_marks": 75, "assignments_completed": 8,
            "extracurricular_hours": 2, "sleep_hours": 7,
            "parental_support": 4, "internet_access": 1,
        }
        with self.assertRaises(DataValidationError):
            self.predictor.predict(bad_input)

    def test_high_effort_student_predicts_better_than_low_effort(self):
        """Sanity/regression test: a high-effort student profile should
        always score higher than a low-effort one. This protects
        against silent model degradation in future retraining.
        """
        high_effort = {
            "attendance_percentage": 95, "study_hours_per_day": 6,
            "previous_sem_marks": 88, "assignments_completed": 10,
            "extracurricular_hours": 1, "sleep_hours": 7,
            "parental_support": 4, "internet_access": 1,
        }
        low_effort = {
            "attendance_percentage": 45, "study_hours_per_day": 0.5,
            "previous_sem_marks": 35, "assignments_completed": 2,
            "extracurricular_hours": 2, "sleep_hours": 6,
            "parental_support": 2, "internet_access": 0,
        }
        high_result = self.predictor.predict(high_effort)
        low_result = self.predictor.predict(low_effort)
        self.assertGreater(high_result["predicted_marks"], low_result["predicted_marks"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
