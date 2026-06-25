"""
data_handler.py
----------------
Handles all reading/writing of the utility_usage_data.csv file.
Keeping CSV logic in one module means the rest of the app (console UI,
ML pipeline) never has to deal with file I/O directly.
"""

import csv
import os

from config import DATASET_PATH, REQUIRED_COLUMNS
from exceptions import DatasetError


def load_dataset():
    """Read the full dataset and return it as a list of dicts.
    Raises DatasetError if the file is missing, empty, or malformed."""

    if not os.path.exists(DATASET_PATH):
        raise DatasetError(
            f"Dataset not found at '{DATASET_PATH}'. "
            "Run data/generate_dataset.py first or place the CSV there."
        )

    rows = []
    try:
        with open(DATASET_PATH, "r", newline="") as f:
            reader = csv.DictReader(f)

            if reader.fieldnames is None:
                raise DatasetError("Dataset file is empty.")

            missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
            if missing:
                raise DatasetError(
                    f"Dataset is missing required columns: {', '.join(missing)}"
                )

            for row in reader:
                rows.append(row)

    except csv.Error as e:
        raise DatasetError(f"Could not parse dataset CSV: {e}")

    if len(rows) == 0:
        raise DatasetError("Dataset contains no data rows.")

    return rows


def append_record(record: dict):
    """Append a single new usage record to the CSV file.
    `record` keys must match REQUIRED_COLUMNS."""

    file_exists = os.path.exists(DATASET_PATH)

    try:
        with open(DATASET_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)
    except OSError as e:
        raise DatasetError(f"Could not write to dataset file: {e}")


def get_next_record_id():
    """Look at the existing dataset and return the next sequential record_id."""
    try:
        rows = load_dataset()
    except DatasetError:
        return 1

    existing_ids = []
    for row in rows:
        try:
            existing_ids.append(int(row["record_id"]))
        except (KeyError, ValueError):
            continue

    return max(existing_ids, default=0) + 1


def update_record(record_id, updated_fields: dict):
    """Update an existing record (matched by record_id) with new field values.
    Returns True if a matching record was found and updated, False otherwise."""

    rows = load_dataset()
    updated = False

    for row in rows:
        if str(row.get("record_id")) == str(record_id):
            row.update({k: str(v) for k, v in updated_fields.items()})
            updated = True
            break

    if not updated:
        return False

    try:
        with open(DATASET_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    except OSError as e:
        raise DatasetError(f"Could not write updated dataset: {e}")

    return True


def dataset_row_count():
    """Return the number of data rows currently in the dataset (for display)."""
    try:
        return len(load_dataset())
    except DatasetError:
        return 0
