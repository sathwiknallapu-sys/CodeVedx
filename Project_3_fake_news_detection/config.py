"""
Configuration file

Stores all project paths and constants.
"""

import os

# -------------------------
# Base Directory
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------
# Dataset
# -------------------------
DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "fake_news.csv"
)

# -------------------------
# Model Folder
# -------------------------
MODEL_FOLDER = os.path.join(BASE_DIR, "model")

MODEL_PATH = os.path.join(
    MODEL_FOLDER,
    "fake_news_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    MODEL_FOLDER,
    "tfidf_vectorizer.pkl"
)

# -------------------------
# Reports Folder
# -------------------------
REPORT_FOLDER = os.path.join(BASE_DIR, "reports")

# -------------------------
# Random State
# -------------------------
RANDOM_STATE = 42

# -------------------------
# Train Test Split
# -------------------------
TEST_SIZE = 0.20

# -------------------------
# TF-IDF Parameters
# -------------------------
MAX_FEATURES = 6000

NGRAM_RANGE = (1, 2)

MIN_DF = 2

MAX_DF = 0.90