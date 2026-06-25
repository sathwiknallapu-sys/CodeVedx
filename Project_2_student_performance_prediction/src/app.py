"""
app.py
-------
Flask REST API for the Student Performance Prediction System.

This gives the project a web interface alongside the console UI,
making it deployable as a real service rather than a local script.

Endpoints:
    GET  /                     - API info / health check
    POST /predict              - predict a single student
    POST /predict/batch        - predict multiple students from a JSON list
    GET  /model/summary        - view model performance metrics
    GET  /history              - view recent prediction history

Run:
    python app.py
    # or for production:
    # gunicorn -w 2 -b 0.0.0.0:5000 app:app

Then test with curl:
    curl http://localhost:5000/
    curl -X POST http://localhost:5000/predict \
         -H "Content-Type: application/json" \
         -d '{"attendance_percentage":85,"study_hours_per_day":4.5,
              "previous_sem_marks":78,"assignments_completed":9,
              "extracurricular_hours":2,"sleep_hours":7,
              "parental_support":4,"internet_access":1}'
"""

import os
import sys
import json
import csv
from datetime import datetime

from flask import Flask, request, jsonify

# Allow running from the src/ directory directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from predictor import StudentPerformancePredictor
from exceptions import (
    DataValidationError,
    ModelNotTrainedError,
    StudentPerformanceError,
)

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False  # keep response key order readable

# --- Load predictor once at startup (not per-request) ----------------
try:
    predictor = StudentPerformancePredictor()
except ModelNotTrainedError as e:
    print(f"[ERROR] {e}")
    print("Run 'python model_training.py' first, then restart the API.")
    predictor = None


# --- Helpers ---------------------------------------------------------

def _log_prediction(student_input: dict, result: dict, source: str = "api"):
    """Appends prediction to the shared CSV history log."""
    history_path = os.path.join(config.DATA_DIR, "prediction_history.csv")
    file_exists = os.path.exists(history_path)
    try:
        with open(history_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                header = (
                    ["timestamp", "source"]
                    + list(student_input.keys())
                    + ["predicted_marks", "predicted_category", "confidence_percent"]
                )
                writer.writerow(header)
            row = (
                [datetime.now().isoformat(timespec="seconds"), source]
                + list(student_input.values())
                + [
                    result["predicted_marks"],
                    result["predicted_category"],
                    result["confidence_percent"],
                ]
            )
            writer.writerow(row)
    except OSError:
        pass  # history logging is best-effort, never fail a request over it


def _error(message: str, status: int = 400):
    return jsonify({"success": False, "error": message}), status


def _check_ready():
    """Returns an error response if models aren't loaded, else None."""
    if predictor is None:
        return _error(
            "Models are not loaded. Run 'python model_training.py' first, "
            "then restart the API.",
            status=503,
        )
    return None


# --- Routes ----------------------------------------------------------

@app.route("/", methods=["GET"])
def health():
    """Health check / API info endpoint."""
    return jsonify({
        "success": True,
        "service": "Student Performance Prediction System",
        "version": "1.0.0",
        "status": "ready" if predictor else "models_not_loaded",
        "endpoints": {
            "POST /predict": "Predict a single student's performance",
            "POST /predict/batch": "Predict multiple students (JSON array)",
            "GET /model/summary": "View model performance metrics",
            "GET /history": "View recent prediction history (last 20)",
        },
    })


@app.route("/predict", methods=["POST"])
def predict_single():
    """Predicts performance for a single student.

    Request body (JSON):
    {
        "attendance_percentage": 85,
        "study_hours_per_day": 4.5,
        "previous_sem_marks": 78,
        "assignments_completed": 9,
        "extracurricular_hours": 2,
        "sleep_hours": 7,
        "parental_support": 4,
        "internet_access": 1
    }
    """
    not_ready = _check_ready()
    if not_ready:
        return not_ready

    data = request.get_json(silent=True)
    if not data:
        return _error("Request body must be valid JSON with Content-Type: application/json.")

    try:
        result = predictor.predict(data)
        _log_prediction(data, result, source="api-single")
        return jsonify({
            "success": True,
            "input": data,
            "prediction": result,
        })
    except DataValidationError as e:
        return _error(str(e), status=422)
    except StudentPerformanceError as e:
        return _error(str(e), status=500)


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """Predicts performance for multiple students.

    Request body (JSON array):
    [
        {"attendance_percentage": 85, "study_hours_per_day": 4.5, ...},
        {"attendance_percentage": 60, "study_hours_per_day": 1.5, ...}
    ]

    Returns each prediction alongside the input, with errors for
    invalid rows reported inline (they are skipped, not fatal).
    """
    not_ready = _check_ready()
    if not_ready:
        return not_ready

    data = request.get_json(silent=True)
    if not data or not isinstance(data, list):
        return _error("Request body must be a JSON array of student objects.")

    if len(data) == 0:
        return _error("Received an empty list. Provide at least one student.")

    if len(data) > 500:
        return _error("Batch size exceeds the limit of 500 students per request.")

    results = []
    success_count = 0
    error_count = 0

    for idx, student in enumerate(data):
        try:
            result = predictor.predict(student)
            _log_prediction(student, result, source="api-batch")
            results.append({
                "index": idx,
                "success": True,
                "input": student,
                "prediction": result,
            })
            success_count += 1
        except DataValidationError as e:
            results.append({
                "index": idx,
                "success": False,
                "error": str(e),
                "input": student,
            })
            error_count += 1

    return jsonify({
        "success": True,
        "summary": {
            "total": len(data),
            "succeeded": success_count,
            "failed": error_count,
        },
        "results": results,
    })


@app.route("/model/summary", methods=["GET"])
def model_summary():
    """Returns the stored model comparison metrics from the last training run."""
    not_ready = _check_ready()
    if not_ready:
        return not_ready

    if not os.path.exists(config.METADATA_PATH):
        return _error(
            "No model metadata found. Run 'python model_training.py' first.",
            status=404,
        )

    with open(config.METADATA_PATH) as f:
        metadata = json.load(f)

    return jsonify({"success": True, "model_metadata": metadata})


@app.route("/history", methods=["GET"])
def prediction_history():
    """Returns the most recent predictions logged by any interface
    (console UI or this API).  Supports ?limit=N (default 20, max 100).
    """
    history_path = os.path.join(config.DATA_DIR, "prediction_history.csv")

    try:
        limit = min(int(request.args.get("limit", 20)), 100)
    except ValueError:
        return _error("'limit' must be an integer.")

    if not os.path.exists(history_path):
        return jsonify({
            "success": True,
            "count": 0,
            "history": [],
            "message": "No predictions made yet.",
        })

    rows = []
    try:
        with open(history_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
    except OSError as e:
        return _error(f"Could not read history file: {e}", status=500)

    recent = rows[-limit:][::-1]  # most recent first

    return jsonify({
        "success": True,
        "total_in_log": len(rows),
        "count": len(recent),
        "history": recent,
    })


# --- Error handlers --------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    return _error(f"Endpoint not found. GET / for available routes.", status=404)


@app.errorhandler(405)
def method_not_allowed(e):
    return _error(f"Method not allowed on this endpoint.", status=405)


@app.errorhandler(500)
def internal_error(e):
    return _error("An unexpected internal error occurred.", status=500)


# --- Entry point -----------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    print(f"\nStudent Performance Prediction API")
    print(f"Running on http://localhost:{port}")
    print(f"Health check: GET http://localhost:{port}/")
    print(f"Press Ctrl+C to stop.\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
