"""
---------------------------------------------------------
AI Based Fake News Detection Tool
Flask Application
---------------------------------------------------------
"""

from flask import Flask, render_template, request
from predict import FakeNewsPredictor
from utils.helper import validate_news
from utils.logger import log_info, log_error

app = Flask(__name__)

# Load the trained model once
try:
    predictor = FakeNewsPredictor()
except Exception as e:
    predictor = None
    print(f"Error loading model: {e}")


@app.route("/")
def home():
    """
    Home Page
    """
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_news():
    """
    Predict whether the news is Fake or Real.
    """

    if predictor is None:
        return render_template(
            "index.html",
            error="Model not loaded. Please train the model first."
        )

    try:

        news = request.form.get("news", "").strip()

        if not validate_news(news):
            return render_template(
                "index.html",
                error="Please enter at least 20 characters."
            )

        result = predictor.predict(news)

        log_info("Prediction completed successfully.")

        return render_template(
            "result.html",
            news=news,
            prediction=result["prediction"],
            confidence=result["confidence"]
        )

    except Exception as e:

        log_error(str(e))

        return render_template(
            "index.html",
            error=f"Error: {str(e)}"
        )


@app.errorhandler(404)
def page_not_found(error):
    return "<h2>404 - Page Not Found</h2>", 404


@app.errorhandler(500)
def internal_error(error):
    return "<h2>500 - Internal Server Error</h2>", 500


if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )