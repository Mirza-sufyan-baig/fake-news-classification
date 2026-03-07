from fastapi import FastAPI
from pydantic import BaseModel
import joblib

#i am loading the model artifacts

model = joblib.load("models/baseline_model.pkl")

vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

app = FastAPI()

#schema request

class NewRequest(BaseModel):
    text : str
    
#health check
@app.get("/")
def home():
    return{"message": "Fake News detection API running"}

#prediction endpoint

@app.post("/predict")
def predict(request: NewRequest):
    text_vector = vectorizer.transform([request.text])
    
    prediction = model.predict(text_vector)[0]
    
    if prediction == 1:
        label = "Fake"
    else:
        label = "Real"
        
        return {
            "prediction" : label,
            "status" :"success"
        }
    
 