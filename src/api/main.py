from fastapi import FastAPI
from pydantic import BaseModel
from src.inference.predict import FakeNewsPredictor
import joblib

#i am loading the model artifacts



app = FastAPI()

predictor = FakeNewsPredictor()

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
    
    
    #prediction = model.predict(text_vector)[0]
    prediction, probability = predictor.predict(request.text)
    
    if prediction == 1:
        label = "Fake"
    else:
        label = "Real"
        
    return {
            "prediction" : "FAKE" if prediction == 1 else "REAL",
            "probability" : probability #if probability else None
        }
    
 