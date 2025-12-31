import streamlit as st
import pandas as pd
import joblib
from PIL import Image
from pathlib import Path

# ===========================
# Page Configuration
# ===========================
st.set_page_config(
    page_title="Dashen Bank | Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

# ===========================
# Constants
# ===========================
BRAND_GREEN = "#0B5E2E"
BRAND_LIGHT = "#1F8F4A"
HIGH_RISK = "#C62828"

MODEL_PATH = "final_churn_bundle.joblib"
LOGO_PATH = "assets/dashen_logo.png"

# ===========================
# Load Model Bundle (SAFE)
# ===========================
@st.cache_resource
def load_bundle():
    if not Path(MODEL_PATH).exists():
        st.error("❌ Model file not found. Please upload final_churn_bundle.joblib")
        st.stop()
    return joblib.load(MODEL_PATH)

bundle = load_bundle()
model = bundle["model"]          # sklearn Pipeline
threshold = float(bundle["threshold"])
features = bundle["features"]

# ===========================
# Branding & Styling
# ===========================
st.markdown(
    f"""
    <style>
        .title {{
            color: {BRAND_GREEN};
            font-size: 40px;
            font-weight: 700;
        }}
        .subtitle {{
            color: {BRAND_LIGHT};
            font-size: 18px;
        }}
        .risk-high {{
            color: {HIGH_RISK};
            font-size: 22px;
            font-weight: bold;
        }}
        .risk-low {{
            color: {BRAND_GREEN};
            font-size: 22px;
            font-weight: bold;
        }}
        .metric-box {{
            background-color: #F5F9F6;
            padding: 15px;
            border-radius: 10px;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# ===========================
# Header
# ===========================
col1, col2 = st.columns([1, 6])

with col1:
    if Path(LOGO_PATH).exists():
        st.image(Image.open(LOGO_PATH), width=120)

with col2:
    st.markdown('<div class="title">Dashen Bank</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Customer Churn Risk Assessment System</div>',
        unsafe_allow_html=True
    )

st.divider()

# ===========================
# Sidebar — Customer Inputs
# ===========================
st.sidebar.header("Customer Information")

input_data = {}

for feature in features:
    # Smart defaults
    if feature.lower().endswith(("age", "tenure", "balance", "salary")):
        input_data[feature] = st.sidebar.number_input(
            feature, value=0.0, step=1.0
        )
    else:
        input_data[feature] = st.sidebar.text_input(
            feature, ""
        )

input_df = pd.DataFrame([input_data])

# ===========================
# ETB Retention Cost Calculator
# ===========================
st.sidebar.divider()
st.sidebar.subheader("ETB Retention Cost Calculator")

cost_fp = st.sidebar.number_input(
    "Cost of Unnecessary Retention (False Positive) – ETB",
    value=150.0
)

cost_fn = st.sidebar.number_input(
    "Cost of Lost Customer (False Negative) – ETB",
    value=1000.0
)

# ===========================
# Prediction
# ===========================
if st.button("🔍 Predict Churn Risk", use_container_width=True):
    try:
        prob = model.predict_proba(input_df)[0, 1]
        churn = prob >= threshold

        st.subheader("Prediction Result")

        colA, colB = st.columns(2)

        with colA:
            st.metric(
                label="Churn Probability",
                value=f"{prob:.2%}"
            )

        with colB:
            expected_cost = (
                prob * cost_fn + (1 - prob) * cost_fp
            )
            st.metric(
                label="Expected Retention Cost (ETB)",
                value=f"{expected_cost:,.2f}"
            )

        st.markdown("<div class='metric-box'>", unsafe_allow_html=True)

        if churn:
            st.markdown(
                "<div class='risk-high'>⚠ HIGH RISK — Immediate Retention Action Required</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='risk-low'>✅ LOW RISK — Customer Likely to Stay</div>",
                unsafe_allow_html=True
            )

        st.caption(
            f"Decision threshold: {threshold:.2f} | "
            "Model optimized using ETB-based cost minimization"
        )

        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error("❌ Prediction failed")
        st.exception(e)

# ===========================
# Footer
# ===========================
st.divider()
st.caption(
    "© Dashen Bank | AI-powered churn prediction | Academic & Demonstration Use"
)

