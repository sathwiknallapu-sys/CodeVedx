"""
---------------------------------------------------------
Model Evaluation Module
---------------------------------------------------------
"""

import matplotlib.pyplot as plt

from sklearn.metrics import (

    accuracy_score,

    confusion_matrix,

    classification_report,

    ConfusionMatrixDisplay

)


class Evaluator:

    @staticmethod
    def evaluate(model, X_test, y_test):

        predictions = model.predict(X_test)

        accuracy = accuracy_score(

            y_test,

            predictions

        )

        print("\nAccuracy")

        print(accuracy)

        print("\nClassification Report\n")

        print(

            classification_report(

                y_test,

                predictions

            )

        )

        cm = confusion_matrix(

            y_test,

            predictions

        )

        display = ConfusionMatrixDisplay(

            confusion_matrix=cm,

            display_labels=["Real", "Fake"]

        )

        display.plot()

        plt.savefig("reports/confusion_matrix.png")

        plt.show()

        return accuracy