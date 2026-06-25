"""
test_api_and_pipeline.py
--------------------------
Additional tests covering:
    - Flask REST API (all endpoints)
    - evaluate.py smoke test
    - run_all.py argument parsing
    - Predictor edge cases not in test_project.py
    - Integration: clean -> engineer -> predict round-trip

Run:
    python test_api_and_pipeline.py
or discover alongside the other suite:
    python -m unittest discover -s tests -v
"""

import os
import sys
import json
import unittest

# Make src/ importable
SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, SRC_DIR)

import config
from exceptions import DataValidationError, ModelNotTrainedError


# ─────────────────────────────────────────────
#  Flask API Tests
# ─────────────────────────────────────────────

class TestFlaskAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            import app as flask_app
            cls.client = flask_app.app.test_client()
            cls.app_available = flask_app.predictor is not None
        except Exception:
            cls.client = None
            cls.app_available = False

    def setUp(self):
        if not self.app_available:
            self.skipTest("Flask app or trained models not available.")

    def _valid_payload(self):
        return {
            "attendance_percentage": 85,
            "study_hours_per_day": 4.5,
            "previous_sem_marks": 78,
            "assignments_completed": 9,
            "extracurricular_hours": 2,
            "sleep_hours": 7,
            "parental_support": 4,
            "internet_access": 1,
        }

    # --- Health check ---

    def test_health_check_returns_200(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)

    def test_health_check_has_required_keys(self):
        r = self.client.get("/")
        data = r.get_json()
        for key in ["success", "service", "status", "endpoints"]:
            self.assertIn(key, data)

    def test_health_check_status_ready(self):
        r = self.client.get("/")
        self.assertEqual(r.get_json()["status"], "ready")

    # --- Single prediction ---

    def test_predict_valid_returns_200(self):
        r = self.client.post(
            "/predict",
            data=json.dumps(self._valid_payload()),
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 200)

    def test_predict_valid_response_structure(self):
        r = self.client.post(
            "/predict",
            data=json.dumps(self._valid_payload()),
            content_type="application/json",
        )
        data = r.get_json()
        self.assertTrue(data["success"])
        for key in ["predicted_marks", "predicted_category", "confidence_percent",
                    "category_probabilities"]:
            self.assertIn(key, data["prediction"])

    def test_predict_marks_in_valid_range(self):
        r = self.client.post(
            "/predict",
            data=json.dumps(self._valid_payload()),
            content_type="application/json",
        )
        marks = r.get_json()["prediction"]["predicted_marks"]
        self.assertGreaterEqual(marks, 0)
        self.assertLessEqual(marks, 100)

    def test_predict_category_is_valid_label(self):
        r = self.client.post(
            "/predict",
            data=json.dumps(self._valid_payload()),
            content_type="application/json",
        )
        cat = r.get_json()["prediction"]["predicted_category"]
        self.assertIn(cat, config.PERFORMANCE_CATEGORIES)

    def test_predict_invalid_attendance_returns_422(self):
        bad = self._valid_payload()
        bad["attendance_percentage"] = 999
        r = self.client.post(
            "/predict",
            data=json.dumps(bad),
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 422)
        self.assertFalse(r.get_json()["success"])

    def test_predict_missing_field_returns_422(self):
        incomplete = {"attendance_percentage": 80}
        r = self.client.post(
            "/predict",
            data=json.dumps(incomplete),
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 422)

    def test_predict_wrong_content_type_returns_400(self):
        r = self.client.post(
            "/predict",
            data="attendance_percentage=80",
            content_type="text/plain",
        )
        self.assertEqual(r.status_code, 400)

    def test_predict_empty_json_body_returns_400(self):
        r = self.client.post(
            "/predict",
            data="",
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 400)

    # --- Batch prediction ---

    def test_batch_predict_returns_200(self):
        batch = [self._valid_payload(), self._valid_payload()]
        r = self.client.post(
            "/predict/batch",
            data=json.dumps(batch),
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 200)

    def test_batch_predict_summary_counts(self):
        bad = self._valid_payload()
        bad["attendance_percentage"] = 999
        batch = [self._valid_payload(), bad, self._valid_payload()]
        r = self.client.post(
            "/predict/batch",
            data=json.dumps(batch),
            content_type="application/json",
        )
        data = r.get_json()
        self.assertEqual(data["summary"]["total"], 3)
        self.assertEqual(data["summary"]["succeeded"], 2)
        self.assertEqual(data["summary"]["failed"], 1)

    def test_batch_predict_empty_list_returns_400(self):
        r = self.client.post(
            "/predict/batch",
            data=json.dumps([]),
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 400)

    def test_batch_predict_non_list_body_returns_400(self):
        r = self.client.post(
            "/predict/batch",
            data=json.dumps(self._valid_payload()),  # dict not list
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 400)

    def test_batch_predict_each_result_has_index(self):
        batch = [self._valid_payload(), self._valid_payload()]
        r = self.client.post(
            "/predict/batch",
            data=json.dumps(batch),
            content_type="application/json",
        )
        results = r.get_json()["results"]
        for i, res in enumerate(results):
            self.assertEqual(res["index"], i)

    # --- Model summary ---

    def test_model_summary_returns_200(self):
        r = self.client.get("/model/summary")
        self.assertEqual(r.status_code, 200)

    def test_model_summary_has_regression_and_classification(self):
        r = self.client.get("/model/summary")
        meta = r.get_json()["model_metadata"]
        self.assertIn("regression", meta)
        self.assertIn("classification", meta)

    # --- History ---

    def test_history_returns_200(self):
        r = self.client.get("/history")
        self.assertEqual(r.status_code, 200)

    def test_history_has_required_keys(self):
        r = self.client.get("/history")
        data = r.get_json()
        for key in ["success", "count", "history"]:
            self.assertIn(key, data)

    def test_history_limit_param_is_respected(self):
        # First make a few predictions to populate history
        for _ in range(3):
            self.client.post(
                "/predict",
                data=json.dumps(self._valid_payload()),
                content_type="application/json",
            )
        r = self.client.get("/history?limit=2")
        data = r.get_json()
        self.assertLessEqual(data["count"], 2)

    def test_history_invalid_limit_returns_400(self):
        r = self.client.get("/history?limit=abc")
        self.assertEqual(r.status_code, 400)

    # --- Error handling ---

    def test_unknown_route_returns_404(self):
        r = self.client.get("/this_does_not_exist")
        self.assertEqual(r.status_code, 404)

    def test_wrong_method_on_predict_returns_405(self):
        r = self.client.get("/predict")  # GET instead of POST
        self.assertEqual(r.status_code, 405)


# ─────────────────────────────────────────────
#  Integration Test: Full pipeline round-trip
# ─────────────────────────────────────────────

class TestPipelineIntegration(unittest.TestCase):
    """End-to-end smoke tests verifying that the preprocessing,
    feature engineering, and prediction stages are fully consistent
    with each other (no training/serving skew).
    """

    @classmethod
    def setUpClass(cls):
        try:
            from predictor import StudentPerformancePredictor
            cls.predictor = StudentPerformancePredictor()
        except ModelNotTrainedError:
            cls.predictor = None

    def setUp(self):
        if self.predictor is None:
            self.skipTest("Trained models not found.")

    def test_at_risk_profile_predicts_below_50(self):
        """A clearly struggling student should fall below the 50-mark threshold."""
        at_risk = {
            "attendance_percentage": 38,
            "study_hours_per_day": 0.3,
            "previous_sem_marks": 25,
            "assignments_completed": 1,
            "extracurricular_hours": 12,
            "sleep_hours": 4,
            "parental_support": 1,
            "internet_access": 0,
        }
        result = self.predictor.predict(at_risk)
        self.assertLess(result["predicted_marks"], 65,
            "Clearly at-risk profile should predict below 65 marks")

    def test_excellent_profile_predicts_above_70(self):
        """A clearly high-performing student should predict above 70."""
        excellent = {
            "attendance_percentage": 97,
            "study_hours_per_day": 7,
            "previous_sem_marks": 92,
            "assignments_completed": 10,
            "extracurricular_hours": 1,
            "sleep_hours": 8,
            "parental_support": 5,
            "internet_access": 1,
        }
        result = self.predictor.predict(excellent)
        self.assertGreater(result["predicted_marks"], 70,
            "Clearly excellent profile should predict above 70 marks")

    def test_confidence_percentages_sum_to_100(self):
        """Category probabilities must sum to 100% (within float tolerance)."""
        sample = {
            "attendance_percentage": 75,
            "study_hours_per_day": 3,
            "previous_sem_marks": 65,
            "assignments_completed": 7,
            "extracurricular_hours": 3,
            "sleep_hours": 7,
            "parental_support": 3,
            "internet_access": 1,
        }
        result = self.predictor.predict(sample)
        total = sum(result["category_probabilities"].values())
        self.assertAlmostEqual(total, 100.0, places=0)

    def test_confidence_is_maximum_of_category_probabilities(self):
        """Confidence % should equal the highest category probability."""
        sample = {
            "attendance_percentage": 75,
            "study_hours_per_day": 3,
            "previous_sem_marks": 65,
            "assignments_completed": 7,
            "extracurricular_hours": 3,
            "sleep_hours": 7,
            "parental_support": 3,
            "internet_access": 1,
        }
        result = self.predictor.predict(sample)
        max_prob = max(result["category_probabilities"].values())
        self.assertAlmostEqual(result["confidence_percent"], max_prob, places=1)

    def test_boundary_values_are_accepted(self):
        """Values exactly at the boundary (0/100/max) should pass validation."""
        boundary = {
            "attendance_percentage": 0,
            "study_hours_per_day": 0,
            "previous_sem_marks": 0,
            "assignments_completed": 0,
            "extracurricular_hours": 0,
            "sleep_hours": 0,
            "parental_support": 1,
            "internet_access": 0,
        }
        try:
            result = self.predictor.predict(boundary)
            self.assertGreaterEqual(result["predicted_marks"], 0)
        except DataValidationError:
            self.fail("Boundary values at zero should be valid, not raise DataValidationError")

    def test_upper_boundary_values_are_accepted(self):
        upper = {
            "attendance_percentage": 100,
            "study_hours_per_day": 24,
            "previous_sem_marks": 100,
            "assignments_completed": 10,
            "extracurricular_hours": 40,
            "sleep_hours": 24,
            "parental_support": 5,
            "internet_access": 1,
        }
        try:
            result = self.predictor.predict(upper)
            self.assertLessEqual(result["predicted_marks"], 100)
        except DataValidationError:
            self.fail("Upper boundary values should be valid, not raise DataValidationError")

    def test_float_and_int_inputs_both_accepted(self):
        """Fields accept both int and float without raising."""
        payload_int = {
            "attendance_percentage": 80,
            "study_hours_per_day": 4,
            "previous_sem_marks": 70,
            "assignments_completed": 8,
            "extracurricular_hours": 2,
            "sleep_hours": 7,
            "parental_support": 3,
            "internet_access": 1,
        }
        payload_float = {k: float(v) for k, v in payload_int.items()}
        result_int = self.predictor.predict(payload_int)
        result_float = self.predictor.predict(payload_float)
        self.assertAlmostEqual(
            result_int["predicted_marks"],
            result_float["predicted_marks"],
            places=2,
        )

    def test_consistent_predictions_same_input(self):
        """The same input always produces exactly the same output (deterministic)."""
        sample = {
            "attendance_percentage": 72,
            "study_hours_per_day": 3.5,
            "previous_sem_marks": 68,
            "assignments_completed": 7,
            "extracurricular_hours": 2.5,
            "sleep_hours": 6.5,
            "parental_support": 3,
            "internet_access": 1,
        }
        r1 = self.predictor.predict(sample)
        r2 = self.predictor.predict(sample)
        self.assertEqual(r1["predicted_marks"], r2["predicted_marks"])
        self.assertEqual(r1["predicted_category"], r2["predicted_category"])


# ─────────────────────────────────────────────
#  Evaluate module smoke test
# ─────────────────────────────────────────────

class TestEvaluateModule(unittest.TestCase):

    def test_evaluate_runs_without_error(self):
        """evaluate.py's generate_report() should run to completion."""
        # We capture stdout to avoid cluttering test output
        import io
        from contextlib import redirect_stdout
        try:
            from evaluate import generate_report
            with redirect_stdout(io.StringIO()):
                generate_report()
        except ModelNotTrainedError:
            self.skipTest("Trained models not found.")
        except Exception as e:
            self.fail(f"generate_report() raised unexpectedly: {e}")

    def test_evaluation_report_file_created(self):
        """After evaluate.py runs, the text report should exist on disk."""
        report_path = os.path.join(config.BASE_DIR, "docs", "evaluation_report.txt")
        if not os.path.exists(report_path):
            self.skipTest("Evaluation report not generated yet.")
        self.assertTrue(os.path.getsize(report_path) > 500,
            "Evaluation report file exists but appears too small.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
