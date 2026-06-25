"""
validators.py
--------------
Small reusable input-validation helpers for the console interface.
Each function raises InvalidInputError with a clear message on bad input,
so the calling code (console_app.py) can catch one exception type and
re-prompt the user instead of crashing.
"""

from exceptions import InvalidInputError
from config import VALID_HOME_TYPES, VALID_SEASONS


def validate_int_range(value_str, field_name, min_val, max_val):
    """Convert a string to int and check it falls inside [min_val, max_val]."""
    try:
        value = int(value_str)
    except ValueError:
        raise InvalidInputError(f"{field_name} must be a whole number.")

    if value < min_val or value > max_val:
        raise InvalidInputError(
            f"{field_name} must be between {min_val} and {max_val}."
        )
    return value


def validate_float_range(value_str, field_name, min_val, max_val):
    """Convert a string to float and check it falls inside [min_val, max_val]."""
    try:
        value = float(value_str)
    except ValueError:
        raise InvalidInputError(f"{field_name} must be a number.")

    if value < min_val or value > max_val:
        raise InvalidInputError(
            f"{field_name} must be between {min_val} and {max_val}."
        )
    return value


def validate_choice(value_str, field_name, valid_options):
    """Check that a string matches one of a fixed set of valid options
    (case-insensitive), and return the correctly-cased version."""
    cleaned = value_str.strip()
    for option in valid_options:
        if cleaned.lower() == option.lower():
            return option

    raise InvalidInputError(
        f"{field_name} must be one of: {', '.join(valid_options)}."
    )


def validate_home_type(value_str):
    return validate_choice(value_str, "Home type", VALID_HOME_TYPES)


def validate_season(value_str):
    return validate_choice(value_str, "Season", VALID_SEASONS)


def validate_non_empty(value_str, field_name):
    if value_str is None or value_str.strip() == "":
        raise InvalidInputError(f"{field_name} cannot be empty.")
    return value_str.strip()
