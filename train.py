"""
---------------------------------------------------------
AI Based Fake News Detection Tool
Main Training Script
---------------------------------------------------------

Description:
This script loads the dataset, validates it,
trains the machine learning model,
and saves the trained model and TF-IDF vectorizer.

Run:
    python train.py
---------------------------------------------------------
"""

import os

from utils.helper import load_dataset
from utils.logger import log_info, log_error
from ml.trainer import ModelTrainer

from config import (
    DATASET_PATH,
    MODEL_FOLDER
)


def check_dataset(df):
    """
    Validate dataset structure.
    """

    required_columns = ["text", "label"]

    for column in required_columns:

        if column not in df.columns:
            raise ValueError(
                f"Missing required column: {column}"
            )

    if df.empty:
        raise ValueError("Dataset is empty.")


def main():

    print("=" * 60)
    print(" AI Based Fake News Detection Tool ")
    print("=" * 60)

    try:

        os.makedirs(MODEL_FOLDER, exist_ok=True)

        log_info("Loading dataset...")

        dataset = load_dataset(DATASET_PATH)

        check_dataset(dataset)

        print(f"\nDataset Loaded Successfully")
        print(f"Total Records : {len(dataset)}")

        trainer = ModelTrainer(dataset)

        trainer.train()

        print("\nTraining Completed Successfully.")

        print("\nModel Saved Successfully.")

        print("\nLocation:")
        print(MODEL_FOLDER)

        log_info("Training completed.")

    except Exception as error:

        log_error(str(error))

        print("\nTraining Failed")

        print(error)


if __name__ == "__main__":

    main()