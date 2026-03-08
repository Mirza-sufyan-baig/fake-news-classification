import streamlit as st
from src.inference.predict import FakeNewsPredictor
import joblib

predictor = FakeNewsPredictor()

st.title("!Fake News Detection AI")

st.write("paste a news article below and the model will predict if it is: Fake or Real")

user_input = st.text_area("Enter news teext")

if st.button("predict"):
    
    if user_input.strip() == "":
        st.warning("please enter some text")
        
    else:
        prediction, probability = predictor.predict(user_input)
        
        if prediction == 1:
            st.error("The news is predicted as Fake❌")
            
        else:
            st.success("The news is predicted as real🚀")