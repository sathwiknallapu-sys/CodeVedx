"""
---------------------------------------------------------
AI Based Fake News Detection Tool
Logger Module
---------------------------------------------------------

Author : Your Name

Description:
This module creates and manages application logs.
It records:

• Application Start
• Model Training
• Predictions
• Errors
• Warnings
• Debug Information

---------------------------------------------------------
"""

import logging
import os
from datetime import datetime


# ---------------------------------------------------------
# Create logs directory
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOG_FOLDER = os.path.join(BASE_DIR, "logs")

os.makedirs(LOG_FOLDER, exist_ok=True)


# ---------------------------------------------------------
# Log File Name
# ---------------------------------------------------------

LOG_FILE = os.path.join(
    LOG_FOLDER,
    f"fake_news_{datetime.now().strftime('%Y%m%d')}.log"
)


# ---------------------------------------------------------
# Configure Logger
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)


logger = logging.getLogger("FakeNewsLogger")


# ---------------------------------------------------------
# Logging Functions
# ---------------------------------------------------------

def log_info(message: str):
    """
    Log information message.
    """
    logger.info(message)


def log_warning(message: str):
    """
    Log warning message.
    """
    logger.warning(message)


def log_error(message: str):
    """
    Log error message.
    """
    logger.error(message)


def log_debug(message: str):
    """
    Log debug message.
    """
    logger.debug(message)


# ---------------------------------------------------------
# Manual Testing
# ---------------------------------------------------------

if __name__ == "__main__":

    log_info("Application Started")

    log_info("Dataset Loaded Successfully")

    log_warning("Missing values detected.")

    log_error("Model file not found.")

    log_debug("Debug message.")

    print("Logger Working Successfully.")