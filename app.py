import streamlit as st
import joblib
import pandas as pd
import numpy as np
import warnings

# Suppress minor warnings for clean output
warnings.filterwarnings('ignore')

# Define the features the model expects (must match the training order!)
MODEL_FEATURES = [
    'Credit Utilization Ratio',
    'Payment History',
    'Number of Credit Accounts',
    'Loan Amount',
    'Interest Rate',
    'Loan Term'
]

# Load the saved model and scaler
try:
    KMEANS_MODEL = joblib.load('kmeans_credit_model.joblib')
    SCALER = joblib.load('scaler_credit_data.joblib')
    st.sidebar.success("Model and Scaler loaded successfully.")
except FileNotFoundError:
    st.error("Error: Deployment assets not found.")
    st.info("Please ensure 'kmeans_credit_model.joblib' and 'scaler_credit_data.joblib' are in the same folder.")
    st.stop()
except Exception as e:
    st.error(f"Error loading assets: {e}")
    st.stop()

# Define the segment interpretation for user-friendly display
SEGMENT_LABELS = {
    0: "Segment 0 (High Risk) 🔴",
    1: "Segment 1 (Medium Risk) 🟠",
    2: "Segment 2 (Low Risk) 🟡",
    3: "Segment 3 (Excellent) 🟢"
}

def get_credit_segment(input_data: pd.DataFrame) -> int:
    """
    Takes customer input data, scales it, and predicts the credit segment ID.
    """
    # 1. Apply the saved scaler to the new data
    scaled_input = SCALER.transform(input_data)

    # 2. Use the saved model to predict the segment
    segment_id = KMEANS_MODEL.predict(scaled_input)[0]

    return int(segment_id)

st.set_page_config(page_title="Credit Scoring Predictor", layout="centered")

st.markdown("""
    <style>
        .main-header {
            font-size: 32px;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            padding-bottom: 20px;
        }
        .subheader {
            font-size: 18px;
            text-align: center;
            color: #333;
        }
    </style>
    <div class="main-header">K-Means Credit Scoring Predictor</div>
    <div class="subheader">Determine a customer's credit score segment (High Risk to Excellent) based on key financial metrics.</div>
    <hr>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    credit_utilization_ratio = st.slider(
        "1. Credit Utilization Ratio (0.0 to 1.0)",
        min_value=0.0, max_value=1.0, value=0.5, step=0.01
    )
    number_of_credit_accounts = st.number_input(
        "3. Number of Credit Accounts",
        min_value=1, max_value=20, value=5, step=1
    )
    interest_rate = st.slider(
        "5. Interest Rate (%)",
        min_value=1.0, max_value=20.0, value=10.0, step=0.1
    )

with col2:
    payment_history = st.number_input(
        "2. Payment History (e.g., Total payments made)",
        min_value=100.0, max_value=3000.0, value=1500.0, step=10.0
    )
    loan_amount = st.number_input(
        "4. Loan Amount (in local currency)",
        min_value=10000.0, max_value=5000000.0, value=2500000.0, step=10000.0
    )
    loan_term = st.number_input(
        "6. Loan Term (in months)",
        min_value=12, max_value=60, value=36, step=12
    )

if st.button('Predict Credit Segment', use_container_width=True):
    # 1. Gather all inputs into a dictionary
    raw_data = {
        'Credit Utilization Ratio': credit_utilization_ratio,
        'Payment History': payment_history,
        'Number of Credit Accounts': number_of_credit_accounts,
        'Loan Amount': loan_amount,
        'Interest Rate': interest_rate,
        'Loan Term': loan_term
    }

    # 2. Convert dictionary to DataFrame (ensures correct feature order)
    input_df = pd.DataFrame([raw_data], columns=MODEL_FEATURES)

    # 3. Get prediction
    segment_id = get_credit_segment(input_df)
    segment_label = SEGMENT_LABELS.get(segment_id, f"Segment {segment_id}")

    # 4. Display Result
    st.markdown("### Prediction Result")

    if segment_id == 3:
        st.success(f"**Classification:** {segment_label}", icon="✅")
        st.balloons()
    elif segment_id == 0:
        st.error(f"**Classification:** {segment_label}", icon="❌")
    else:
        st.warning(f"**Classification:** {segment_label}", icon="⚠️")

    st.markdown(f"The customer is classified into **Segment {segment_id}**.")
    st.markdown("---")
    st.markdown(
        "The segments represent customer groups with similar credit behavior patterns based on K-Means clustering.")
