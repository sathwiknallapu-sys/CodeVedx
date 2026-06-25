"""
---------------------------------------------------------
AI Based Fake News Detection Tool
Feature Engineering Module
---------------------------------------------------------

Author : Your Name

Description:
This module converts cleaned news text into numerical
features using TF-IDF Vectorization.

Functions:
1. Create TF-IDF Vectorizer
2. Fit and Transform Training Data
3. Transform New Data
4. Save Vectorizer
5. Load Vectorizer
---------------------------------------------------------
"""

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from config import (
    MAX_FEATURES,
    MIN_DF,
    MAX_DF,
    NGRAM_RANGE,
    VECTORIZER_PATH
)

from utils.logger import (
    log_info,
    log_error
)


class FeatureEngineering:
    """
    Handles TF-IDF Vectorization.
    """

    def __init__(self):

        self.vectorizer = TfidfVectorizer(
            max_features=MAX_FEATURES,
            stop_words="english",
            min_df=MIN_DF,
            max_df=MAX_DF,
            ngram_range=NGRAM_RANGE
        )

    # --------------------------------------------------

    def fit_transform(self, text_data):

        try:

            features = self.vectorizer.fit_transform(text_data)

            log_info("TF-IDF training completed.")

            return features

        except Exception as e:

            log_error(str(e))

            raise

    # --------------------------------------------------

    def transform(self, text_data):

        try:

            return self.vectorizer.transform(text_data)

        except Exception as e:

            log_error(str(e))

            raise

    # --------------------------------------------------

    def save_vectorizer(self):

        try:

            joblib.dump(
                self.vectorizer,
                VECTORIZER_PATH
            )

            log_info("Vectorizer saved successfully.")

        except Exception as e:

            log_error(str(e))

            raise

    # --------------------------------------------------

    @staticmethod
    def load_vectorizer():

        try:

            vectorizer = joblib.load(
                VECTORIZER_PATH
            )

            log_info("Vectorizer loaded.")

            return vectorizer

        except Exception as e:

            log_error(str(e))

            raise

    # --------------------------------------------------

    def get_feature_names(self):

        return self.vectorizer.get_feature_names_out()


# ------------------------------------------------------
# Manual Testing
# ------------------------------------------------------

if __name__ == "__main__":

    sample_news = pd.Series([

        "India wins cricket world cup after exciting match.",

        "Scientists discover new medicine for cancer treatment.",

        "Aliens landed in New York yesterday according to rumors."

    ])

    feature = FeatureEngineering()

    X = feature.fit_transform(sample_news)

    feature.save_vectorizer()

    print("\nShape of TF-IDF Matrix")

    print(X.shape)

    print("\nSample Features")

    print(feature.get_feature_names()[:20])