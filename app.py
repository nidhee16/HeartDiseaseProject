import streamlit as st
import numpy as np
import joblib

# Load trained model & scaler
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define input fields
st.title("Heart Disease Prediction App")
st.write("Enter the following health parameters to predict the risk of heart disease:")

# User inputs
age = st.number_input("Age", min_value=20, max_value=100, value=50)
sex = st.radio("Sex (1 = Male, 0 = Female)", [1, 0])
cp = st.slider("Chest Pain Type (0-3)", 0, 3, 1)
trtbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
chol = st.number_input("Cholesterol Level", min_value=100, max_value=500, value=200)
fbs = st.radio("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)", [1, 0])
restecg = st.slider("Resting ECG Results (0-2)", 0, 2, 1)
thalachh = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
exng = st.radio("Exercise Induced Angina (1 = Yes, 0 = No)", [1, 0])
oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slp = st.slider("Slope of ST Segment (0-2)", 0, 2, 1)
caa = st.slider("Number of Major Vessels (0-4)", 0, 4, 1)
thall = st.slider("Thalassemia (0-3)", 0, 3, 1)

# Prepare input for model
features = np.array([[age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall]])
scaled_features = scaler.transform(features)  # Apply MinMaxScaler

# Predict
if st.button("Predict"):
    prediction = model.predict(scaled_features)
    result = "Heart Disease Detected 😟" if prediction[0] == 1 else "No Heart Disease 😊"
    st.subheader(result)
