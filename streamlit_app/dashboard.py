import streamlit as st
import joblib

model = joblib.load("models/baseline_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

st.title("!Fake News Detection AI")

st.write("paste a news article below and the model will predict if it is: Fake or Real")

user_input = st.text_area("Enter news teext")

if st.button("predict"):
    
    if user_input.strip() == "":
        st.warning("please enter some text")
        
    else:
        text_vector = vectorizer.transform([user_input])
        prediction = model.predict(text_vector)[0]
        
        if prediction == 1:
            st.error("The news is predicted as Fake❌")
            
        else:
            st.success("The news is predicted as real🚀")