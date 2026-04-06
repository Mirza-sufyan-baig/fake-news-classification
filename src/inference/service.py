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
        model_dir = "models"
        
        # 1. Safety Check: Ensure the directory exists
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"The directory '{model_dir}' does not exist.")

        # 2. Filter files
        files = [
            f for f in os.listdir(model_dir)
            if f.startswith("baseline_v") and f.endswith(".pkl") and "vectorizer" not in f
        ]

        # 3. Safety Check: Ensure we actually found matching files
        if not files:
            raise FileNotFoundError(
                "No models found matching 'baseline_v*.pkl'. "
                "Did you run the training script successfully?"
            )

        # 4. Sort and pick the latest
        # Note: sorting strings like 'v10' vs 'v2' can be tricky. 
        # This basic sort works if your versions are padded (e.g., v01, v02)
        latest = sorted(files)[-1]

        model_path = os.path.join(model_dir, latest)
        
        # 5. Correct the vectorizer path naming logic
        # If the model is 'baseline_v1.pkl', the vectorizer is likely 'baseline_v1_vectorizer.pkl'
        version_name = latest.replace(".pkl", "")
        vectorizer_path = os.path.join(model_dir, f"{version_name}_vectorizer.pkl")

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