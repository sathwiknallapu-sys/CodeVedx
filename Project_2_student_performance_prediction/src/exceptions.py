"""
exceptions.py
-------------
Custom exception classes for the Student Performance Prediction System.
Using specific exceptions (instead of generic Exception/ValueError
everywhere) makes error handling clearer and easier to debug, and is
considered good practice in production-grade Python projects.
"""


class StudentPerformanceError(Exception):
    """Base exception for all custom errors raised by this project."""
    pass


class DataValidationError(StudentPerformanceError):
    """Raised when input data fails validation checks
    (e.g. out-of-range values, wrong types, missing required fields)."""
    pass


class ModelNotTrainedError(StudentPerformanceError):
    """Raised when a prediction is attempted before a model
    has been trained or loaded from disk."""
    pass


class DataFileError(StudentPerformanceError):
    """Raised when a required data file is missing, empty, or corrupted."""
    pass
