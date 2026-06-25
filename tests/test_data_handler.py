"""
test_data_handler.py
-----------------------
Tests for CSV read/write/append/update logic in src/data_handler.py.

These tests use a temporary copy of the dataset so they never touch
or corrupt the real project data file.

Run with:  python -m pytest tests/test_data_handler.py -v
"""

import os
import shutil
import pytest

import config
import data_handler


@pytest.fixture
def temp_dataset(tmp_path, monkeypatch):
    """Copy the real dataset to a temp location and point config at it,
    so each test runs against an isolated, disposable CSV file."""
    original = config.DATASET_PATH
    temp_csv = tmp_path / "utility_usage_data.csv"
    shutil.copy(original, temp_csv)

    monkeypatch.setattr(config, "DATASET_PATH", str(temp_csv))
    monkeypatch.setattr(data_handler, "DATASET_PATH", str(temp_csv))

    yield str(temp_csv)


def test_load_dataset_returns_rows(temp_dataset):
    rows = data_handler.load_dataset()
    assert len(rows) > 0
    assert "household_size" in rows[0]


def test_load_dataset_missing_file_raises(tmp_path, monkeypatch):
    fake_path = str(tmp_path / "does_not_exist.csv")
    monkeypatch.setattr(data_handler, "DATASET_PATH", fake_path)

    with pytest.raises(data_handler.DatasetError):
        data_handler.load_dataset()


def test_append_record_increases_row_count(temp_dataset):
    before = len(data_handler.load_dataset())

    new_record = {
        "record_id": 9999,
        "household_size": 3,
        "home_area_sqft": 1200,
        "home_type": "Apartment",
        "season": "Summer",
        "avg_temp_c": 35.0,
        "ac_usage_hours": 4.0,
        "num_appliances": 8,
        "prev_month_units": 200.0,
        "units_consumed": 210.5,
        "bill_amount": 1100.25,
    }
    data_handler.append_record(new_record)

    after = data_handler.load_dataset()
    assert len(after) == before + 1
    assert after[-1]["record_id"] == "9999"


def test_get_next_record_id_increments(temp_dataset):
    rows = data_handler.load_dataset()
    max_id = max(int(r["record_id"]) for r in rows)

    next_id = data_handler.get_next_record_id()
    assert next_id == max_id + 1


def test_update_record_changes_value(temp_dataset):
    rows = data_handler.load_dataset()
    target_id = rows[0]["record_id"]

    updated = data_handler.update_record(target_id, {"ac_usage_hours": "7.5"})
    assert updated is True

    rows_after = data_handler.load_dataset()
    matching = [r for r in rows_after if r["record_id"] == target_id][0]
    assert matching["ac_usage_hours"] == "7.5"


def test_update_record_returns_false_for_unknown_id(temp_dataset):
    updated = data_handler.update_record("not_a_real_id", {"ac_usage_hours": "1.0"})
    assert updated is False


def test_dataset_row_count_matches_load(temp_dataset):
    count = data_handler.dataset_row_count()
    rows = data_handler.load_dataset()
    assert count == len(rows)
