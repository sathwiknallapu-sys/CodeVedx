"""
test_validators.py
--------------------
Unit tests for the input validation helpers in src/validators.py.

Run with:  python -m pytest tests/test_validators.py -v
(run from the project root, with src/ on the path — see conftest.py)
"""

import pytest
from validators import (
    validate_int_range, validate_float_range,
    validate_home_type, validate_season, validate_non_empty,
)
from exceptions import InvalidInputError


# ---- validate_int_range ----

def test_validate_int_range_valid():
    assert validate_int_range("4", "Household size", 1, 10) == 4

def test_validate_int_range_non_numeric_raises():
    with pytest.raises(InvalidInputError):
        validate_int_range("abc", "Household size", 1, 10)

def test_validate_int_range_below_min_raises():
    with pytest.raises(InvalidInputError):
        validate_int_range("0", "Household size", 1, 10)

def test_validate_int_range_above_max_raises():
    with pytest.raises(InvalidInputError):
        validate_int_range("99", "Household size", 1, 10)

def test_validate_int_range_boundary_values():
    assert validate_int_range("1", "x", 1, 10) == 1
    assert validate_int_range("10", "x", 1, 10) == 10


# ---- validate_float_range ----

def test_validate_float_range_valid():
    assert validate_float_range("6.5", "AC usage hours", 0, 24) == 6.5

def test_validate_float_range_non_numeric_raises():
    with pytest.raises(InvalidInputError):
        validate_float_range("xyz", "AC usage hours", 0, 24)

def test_validate_float_range_out_of_bounds_raises():
    with pytest.raises(InvalidInputError):
        validate_float_range("30", "AC usage hours", 0, 24)


# ---- validate_home_type / validate_season ----

def test_validate_home_type_valid():
    assert validate_home_type("apartment") == "Apartment"

def test_validate_home_type_invalid_raises():
    with pytest.raises(InvalidInputError):
        validate_home_type("Treehouse")

def test_validate_season_valid_case_insensitive():
    assert validate_season("WINTER") == "Winter"

def test_validate_season_invalid_raises():
    with pytest.raises(InvalidInputError):
        validate_season("Autumn-ish")


# ---- validate_non_empty ----

def test_validate_non_empty_valid():
    assert validate_non_empty("  hello  ", "Field") == "hello"

def test_validate_non_empty_blank_raises():
    with pytest.raises(InvalidInputError):
        validate_non_empty("   ", "Field")

def test_validate_non_empty_none_raises():
    with pytest.raises(InvalidInputError):
        validate_non_empty(None, "Field")
