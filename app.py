import streamlit as st
import pickle
import pandas as pd

# Load model artifacts
with open("model.pkl", "rb") as f:
    artifacts = pickle.load(f)

model = artifacts["model"]
scaler = artifacts["scaler"]
features = artifacts["features"]

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("📊 Customer Churn Prediction")
st.write("Enter customer information to predict churn.")

# ---- INPUT FORM ----
inputs = {}

for feature in features:
    inputs[feature] = st.number_input(
        label=feature,
        value=0.0
    )

# Convert input to DataFrame
input_df = pd.DataFrame([inputs])

# ---- PREDICTION ----
if st.button("Predict Churn"):
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to churn (Probability: {probability:.2%})")
    else:
        st.success(f"✅ Customer is not likely to churn (Probability: {probability:.2%})")

