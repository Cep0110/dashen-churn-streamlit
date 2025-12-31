 import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Dashen Bank – Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

# ---------------------------
# Branding
# ---------------------------
logo = Image.open("assets/dashen_logo.png")

st.markdown("""
<style>
.title { color: #0B5E2E; font-size: 40px; font-weight: 700; }
.subtitle { color: #1F8F4A; font-size: 18px; }
.risk-high { color: #C62828; font-weight: bold; font-size: 22px; }
.risk-low { color: #0B5E2E; font-weight: bold; font-size: 22px; }
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 5])
with col1:
    st.image(logo, width=120)
with col2:
    st.markdown('<div class="title">Dashen Bank</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Customer Churn Risk Assessment System</div>', unsafe_allow_html=True)

st.divider()

# ---------------------------
# Load model bundle (CACHED)
# ---------------------------
@st.cache_resource
def load_bundle():
    return joblib.load("final_churn_bundle.joblib")

bundle = load_bundle()

model = bundle["model"]
threshold = bundle["threshold"]
features = bundle["features"]

# ---------------------------
# Sidebar Inputs
# ---------------------------
st.sidebar.header("Customer Information")

input_data = {}
for f in features:
    input_data[f] = st.sidebar.text_input(f)

input_df = pd.DataFrame([input_data])

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict Churn Risk"):
    try:
        prob = model.predict_proba(input_df)[0, 1]

        st.metric("Churn Probability", f"{prob:.2%}")

        if prob >= threshold:
            st.markdown('<div class="risk-high">⚠ HIGH RISK — Retention Required</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="risk-low">✅ LOW RISK — Likely to Stay</div>', unsafe_allow_html=True)

        st.caption(f"Decision threshold: {threshold:.2f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ---------------------------
# Footer
# ---------------------------
st.divider()
st.caption("© Dashen Bank | Academic Demonstration System")

