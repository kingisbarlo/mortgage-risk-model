import streamlit as st
import pandas as pd
import numpy as np
import cloudpickle

# Load the trained model pipeline (assumes it's saved as 'xgb_pipeline.pkl')
@st.cache_resource
def load_model():
    with open("xgb_pipeline.pkl", "rb") as f:
        return cloudpickle.load(f)

model = load_model()

st.title("Mortgage Risk Classifier")
st.write("Enter borrower and loan information to estimate risk level.")

# Sidebar inputs
st.sidebar.header("Applicant Info")

fico_pr = st.sidebar.slider("FICO Score (Primary)", 500, 850, 720)
dti = st.sidebar.slider("Debt-to-Income Ratio (%)", 0.0, 60.0, 35.0)
ltv = st.sidebar.slider("Loan-to-Value (%)", 0.0, 150.0, 85.0)
monthly_income = st.sidebar.number_input("Monthly Income", value=5000)
loan_amount = st.sidebar.number_input("Loan Amount", value=200000)
interest_rate = st.sidebar.slider("Interest Rate (%)", 2.0, 10.0, 4.5)
decision_fico = st.sidebar.slider("Decision FICO", 500, 850, 720)
fico_cb = st.sidebar.slider("FICO CB", 500, 850, 720)
experian_cb = st.sidebar.slider("Experian - CB", 500, 850, 720)
equifax_cb = st.sidebar.slider("Equifax - CB", 500, 850, 720)
tu_cb = st.sidebar.slider("TransUnion - CB", 500, 850, 720)
total_monthly_payment = st.sidebar.number_input("Total Monthly Payment", value=1800)
mi = st.sidebar.selectbox("Mortgage Insurance", ["yes", "no"])
lien = st.sidebar.selectbox("2nd Lien", ["yes", "no"])
loan_type = st.sidebar.selectbox("Loan Type", ["Conventional", "FHA", "VA"])
loan_purpose = st.sidebar.selectbox("Loan Purpose", ["Purchase", "Refinance"])


# Credit scores
experian = st.sidebar.slider("Experian Score", 500, 850, 720)
equifax = st.sidebar.slider("Equifax Score", 500, 850, 720)
tu = st.sidebar.slider("TransUnion Score", 500, 850, 720)

# Additional flags (if present in your training data)
status = st.sidebar.selectbox("Loan Status", ["Funded", "Cancelled"])


# Construct single-row dataframe for prediction
input_dict = {
    "fico pr": fico_pr,
    "debt-to-income": dti,
    "loan-to-value": ltv,
    "monthly income": monthly_income,
    "loan amount": loan_amount,
    "interest rate": interest_rate,
    "experian": experian,
    "equifax": equifax,
    "tu": tu,
    "status": status,
    "decision fico": decision_fico,
    "fico cb": fico_cb,
    "experian - cb": experian_cb,
    "equifax - cb": equifax_cb,
    "tu - cb": tu_cb,
    "total monthly payment": total_monthly_payment,
    "mi $": mi,
    "2nd lien": lien,
    "loan type": loan_type,
    "loan purpose": loan_purpose,
}

input_df = pd.DataFrame([input_dict])

# Predict
if st.button("Predict Risk"):
    proba = model.predict_proba(input_df)[0][1]  # Probability of high risk
    label = "HIGH RISK" if proba >= 0.5 else "LOW RISK"

    st.subheader("Prediction")
    st.metric("Risk Level", label, delta=f"{proba*100:.2f}% probability of high risk")

    st.caption("This prediction is based on a trained XGBoost model using real-world lending logic.")
