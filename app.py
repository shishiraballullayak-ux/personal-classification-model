import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("personal_classification_model.pkl")

st.title("🎯 Personal Career Classification Model")

st.write("Rate your skills from 1 to 10.")

coding = st.slider("Coding", 1, 10, 5)
problem_solving = st.slider("Problem Solving", 1, 10, 5)
communication = st.slider("Communication", 1, 10, 5)
testing = st.slider("Testing", 1, 10, 5)
data_analysis = st.slider("Data Analysis", 1, 10, 5)

if st.button("Predict Career"):

    data = pd.DataFrame({
        "Coding": [coding],
        "Problem_Solving": [problem_solving],
        "Communication": [communication],
        "Testing": [testing],
        "Data_Analysis": [data_analysis]
    })

    prediction = model.predict(data)[0]

    st.success(f"Predicted Career: {prediction}")
