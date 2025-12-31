 import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

# Load model
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
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.error(f"⚠️ Likely to churn ({proba:.2%})")
    else:
        st.success(f"✅ Not likely to churn ({proba:.2%})")

