import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="Churn Prediction", layout="centered")

with open("model.pkl", "rb") as f:
    artifacts = pickle.load(f)

model = artifacts["model"]
features = artifacts["features"]

st.title("📊 Customer Churn Prediction")

inputs = {}
for col in features:
    inputs[col] = st.number_input(col, value=0.0)

input_df = pd.DataFrame([inputs])

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Likely to churn ({probability:.2%})")
    else:
        st.success(f"✅ Not likely to churn ({probability:.2%})")

