"""
---------------------------------------------------------
AI Based Fake News Detection Tool
Model Trainer
---------------------------------------------------------

Description:
Train multiple machine learning models, compare them,
evaluate performance, and save the best model.
---------------------------------------------------------
"""

import joblib

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from preprocessing.clean_text import clean_news_text
from ml.feature_engineering import FeatureEngineering

from config import (
    MODEL_PATH,
    TEST_SIZE,
    RANDOM_STATE
)

from utils.logger import log_info


class ModelTrainer:

    def __init__(self, dataframe):

        self.df = dataframe.copy()

        self.vectorizer = FeatureEngineering()

        self.best_model = None

    # -------------------------------------------------

    def preprocess(self):

        log_info("Cleaning dataset...")

        self.df["clean_text"] = self.df["text"].apply(
            clean_news_text
        )

    # -------------------------------------------------

    def prepare_data(self):

        X = self.vectorizer.fit_transform(
            self.df["clean_text"]
        )

        y = self.df["label"]

        return train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )

    # -------------------------------------------------

    def train(self):

        self.preprocess()

        X_train, X_test, y_train, y_test = self.prepare_data()

        models = {

            "Logistic Regression":
                LogisticRegression(max_iter=1000),

            "Naive Bayes":
                MultinomialNB(),

            "Linear SVM":
                LinearSVC()

        }

        best_accuracy = 0

        print("\n")
        print("=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)

        for name, model in models.items():

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            accuracy = accuracy_score(
                y_test,
                predictions
            )

            print(f"{name:<25} {accuracy:.4f}")

            if accuracy > best_accuracy:

                best_accuracy = accuracy

                self.best_model = model

                best_name = name

                best_predictions = predictions

        print("=" * 60)

        print(f"\nBest Model : {best_name}")

        print(f"Accuracy   : {best_accuracy:.4f}")

        print("\nClassification Report\n")

        print(
            classification_report(
                y_test,
                best_predictions
            )
        )

        print("\nConfusion Matrix\n")

        print(
            confusion_matrix(
                y_test,
                best_predictions
            )
        )

        joblib.dump(
            self.best_model,
            MODEL_PATH
        )

        self.vectorizer.save_vectorizer()

        log_info("Model saved successfully.")

        return self.best_model