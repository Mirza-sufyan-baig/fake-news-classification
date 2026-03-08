import joblib

class FakeNewsPredictor:
    
    def __init__(self):
        
        self.model = joblib.load("models/baseline_model.pkl")
        self.vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
        
    def predict(self, text):
        
        vector = self.vectorizer.transform([text])
        
        prediction = self.model.predict(vector)[0]
        
        probability = None
        
        if hasattr(self.model, "predict_proba"):
            probability = self.model.predict_proba(vector)[0].max()
            
        return prediction, probability