import gradio as gr
import pickle
import pandas as pd

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# 🔹 Update these feature names to MATCH your training data
FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "InternetService",
    "MonthlyCharges",
    "TotalCharges"
]

def predict_churn(*inputs):
    data = pd.DataFrame([inputs], columns=FEATURES)
    prediction = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]

    label = "Churn" if prediction == 1 else "No Churn"
    return f"{label} (Probability: {prob:.2f})"

inputs = [
    gr.Dropdown(["Male", "Female"], label="Gender"),
    gr.Dropdown([0, 1], label="Senior Citizen"),
    gr.Dropdown(["Yes", "No"], label="Partner"),
    gr.Dropdown(["Yes", "No"], label="Dependents"),
    gr.Number(label="Tenure (months)"),
    gr.Dropdown(["Yes", "No"], label="Phone Service"),
    gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service"),
    gr.Number(label="Monthly Charges"),
    gr.Number(label="Total Charges"),
]

app = gr.Interface(
    fn=predict_churn,
    inputs=inputs,
    outputs=gr.Textbox(label="Prediction"),
    title="Customer Churn Prediction",
    description="Predict whether a customer is likely to churn using a trained ML model."
)

app.launch()

