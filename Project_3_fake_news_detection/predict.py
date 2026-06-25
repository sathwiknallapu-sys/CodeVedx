import joblib

from preprocessing.clean_text import clean_news_text
from config import MODEL_PATH, VECTORIZER_PATH


class FakeNewsPredictor:

    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.vectorizer = joblib.load(VECTORIZER_PATH)

    def predict(self, news):

        cleaned_news = clean_news_text(news)

        vector = self.vectorizer.transform([cleaned_news])

        prediction = self.model.predict(vector)[0]

        if hasattr(self.model, "predict_proba"):
            confidence = max(self.model.predict_proba(vector)[0])
        else:
            confidence = 0.99

        if prediction == 1:
            label = "Fake News"
        else:
            label = "Real News"

        return {
            "prediction": label,
            "confidence": round(confidence * 100, 2)
        }