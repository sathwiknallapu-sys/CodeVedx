"""
test_ml_pipeline.py
---------------------
Tests for the ML training, evaluation, and prediction logic.

These tests run against the real dataset bundled with the project
(data/utility_usage_data.csv), so they double as a basic integration
check that the whole pipeline still works end-to-end.

Run with:  python -m pytest tests/test_ml_pipeline.py -v
"""

import os
import pytest

from ml_pipeline import train_model, load_model, predict_single, load_metrics
from exceptions import ModelNotTrainedError


SAMPLE_INPUT = {
    "household_size": 4,
    "home_area_sqft": 1500,
    "home_type": "Apartment",
    "season": "Summer",
    "avg_temp_c": 36,
    "ac_usage_hours": 5,
    "num_appliances": 9,
    "prev_month_units": 280,
}


@pytest.fixture(scope="module", autouse=True)
def trained_model():
    """Train the model once for the whole test module."""
    metrics = train_model(model_type="random_forest")
    yield metrics


def test_train_model_returns_expected_keys(trained_model):
    expected_keys = {"model_type", "mae", "rmse", "r2_score", "n_train", "n_test"}
    assert expected_keys.issubset(trained_model.keys())


def test_train_model_reasonable_r2(trained_model):
    # We expect a moderately good fit on this synthetic dataset, not
    # a suspiciously perfect score (which would suggest a data leak).
    assert 0.5 < trained_model["r2_score"] < 0.99


def test_train_model_positive_errors(trained_model):
    assert trained_model["mae"] > 0
    assert trained_model["rmse"] > 0


def test_model_and_encoders_load_after_training():
    model, encoders = load_model()
    assert model is not None
    assert "home_type" in encoders
    assert "season" in encoders


def test_predict_single_returns_positive_float():
    prediction = predict_single(SAMPLE_INPUT)
    assert isinstance(prediction, float)
    assert prediction > 0


def test_predict_single_unseen_category_does_not_crash():
    """If a category not seen during training is passed in, the pipeline
    should fall back gracefully instead of raising a KeyError."""
    bad_input = SAMPLE_INPUT.copy()
    bad_input["home_type"] = "Treehouse"  # not in training data
    prediction = predict_single(bad_input)
    assert prediction > 0


def test_load_metrics_matches_last_training_run(trained_model):
    metrics = load_metrics()
    assert metrics["model_type"] == trained_model["model_type"]
    assert metrics["mae"] == trained_model["mae"]


def test_more_ac_hours_increases_prediction_generally():
    """Sanity/directional test: more AC usage, all else equal, should
    generally predict higher consumption. Not guaranteed for every single
    case with a tree-based model, but should hold for this clear contrast."""
    low_ac = SAMPLE_INPUT.copy()
    low_ac["ac_usage_hours"] = 0

    high_ac = SAMPLE_INPUT.copy()
    high_ac["ac_usage_hours"] = 10

    pred_low = predict_single(low_ac)
    pred_high = predict_single(high_ac)

    assert pred_high > pred_low
