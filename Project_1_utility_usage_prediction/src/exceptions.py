"""
exceptions.py
-------------
Custom exception classes for the Utility Usage Prediction Tool.

Using specific exception types (instead of generic Exception/ValueError
everywhere) makes error handling clearer for whoever reads the code,
and lets the console UI show more helpful messages to the user.
"""


class UtilityToolError(Exception):
    """Base exception for all custom errors raised by this application."""
    pass


class InvalidInputError(UtilityToolError):
    """Raised when a user enters a value that fails validation
    (wrong type, out-of-range, blank, invalid choice, etc.)."""
    pass


class DatasetError(UtilityToolError):
    """Raised when the dataset CSV is missing, empty, malformed,
    or missing required columns."""
    pass


class ModelNotTrainedError(UtilityToolError):
    """Raised when a prediction is requested but no trained model
    file is found on disk."""
    pass
