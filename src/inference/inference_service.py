import os
import joblib

from src.features.cleaner import BasicTextCleaner


class InferenceService:

    def __init__(self):

        model_path, vectorizer_path = self.load_latest_model()

        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

        self.cleaner = BasicTextCleaner()

    def load_latest_model(self):

        files = [
            f for f in os.listdir("models")
            if f.startswith("baseline_v") and f.endswith(".pkl") and "vectorizer" not in f
        ]

        latest = sorted(files)[-1]

        model_path = f"models/{latest}"
        vectorizer_path = f"models/{latest.replace('.pkl','')}_vectorizer.pkl"

        return model_path, vectorizer_path

    def preprocess(self, text):

        return self.cleaner.clean(text)

    def predict(self, text):

        cleaned = self.preprocess(text)

        vector = self.vectorizer.transform([cleaned])

        prediction = self.model.predict(vector)[0]

        probability = None

        if hasattr(self.model, "predict_proba"):
            probability = self.model.predict_proba(vector)[0].max()

        label = "FAKE" if prediction == 1 else "REAL"

        return label, probability