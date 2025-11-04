import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# -------------------- PAGE SETUP --------------------
st.set_page_config(page_title="Bank Deposit Prediction App", page_icon="💰", layout="centered")
st.title("💰 Bank Deposit Prediction App")
st.markdown("Predict whether a customer is likely to subscribe to a term deposit.")

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_ann_model():
    model_path = "model.pkl"
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model

model = load_ann_model()

# -------------------- USER INPUT FUNCTION --------------------
def user_input():
    st.header("📋 Enter Customer Information")

    age = st.number_input("Age", 18, 100, 30)
    balance = st.number_input("Balance (€)", 0, 100000, 2000)
    duration = st.number_input("Duration of Call (sec)", 0, 2000, 200)
    campaign = st.number_input("Contacts During Campaign", 1, 20, 1)
    pdays = st.number_input("Days Since Last Contact (999 means never)", 0, 999, 999)
    previous = st.number_input("Number of Previous Contacts", 0, 10, 0)

    job = st.selectbox("Job", ["admin.", "management", "blue-collar", "technician", "services", "entrepreneur"])
    marital = st.selectbox("Marital Status", ["single", "married", "divorced"])
    education = st.selectbox("Education", ["primary", "secondary", "tertiary"])
    default = st.selectbox("Credit in Default?", ["no", "yes"])
    housing = st.selectbox("Housing Loan?", ["no", "yes"])
    loan = st.selectbox("Personal Loan?", ["no", "yes"])
    contact = st.selectbox("Contact Type", ["cellular", "telephone"])
    month = st.selectbox("Last Contact Month", ["may", "jun", "jul", "aug", "nov", "dec"])
    poutcome = st.selectbox("Previous Campaign Outcome", ["failure", "success"])

    data = {
        "age": age,
        "balance": balance,
        "duration": duration,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "month": month,
        "poutcome": poutcome
    }

    return data

# -------------------- ENCODE FUNCTION --------------------
def encode_inputs(data):
    # Simple label encoding — must match training order if preprocessing known
    mapping = {
        "job": {"admin.": 0, "management": 1, "blue-collar": 2, "technician": 3, "services": 4, "entrepreneur": 5},
        "marital": {"single": 0, "married": 1, "divorced": 2},
        "education": {"primary": 0, "secondary": 1, "tertiary": 2},
        "default": {"no": 0, "yes": 1},
        "housing": {"no": 0, "yes": 1},
        "loan": {"no": 0, "yes": 1},
        "contact": {"cellular": 0, "telephone": 1},
        "month": {"may": 0, "jun": 1, "jul": 2, "aug": 3, "nov": 4, "dec": 5},
        "poutcome": {"failure": 0, "success": 1}
    }

    encoded = [
        data["age"],
        data["balance"],
        data["duration"],
        data["campaign"],
        data["pdays"],
        data["previous"],
        mapping["job"][data["job"]],
        mapping["marital"][data["marital"]],
        mapping["education"][data["education"]],
        mapping["default"][data["default"]],
        mapping["housing"][data["housing"]],
        mapping["loan"][data["loan"]],
        mapping["contact"][data["contact"]],
        mapping["month"][data["month"]],
        mapping["poutcome"][data["poutcome"]],
    ]

    # Pad to match expected input shape (41 features)
    while len(encoded) < 41:
        encoded.append(0)

    return np.array(encoded).reshape(1, -1)

# -------------------- MAIN APP --------------------
input_data = user_input()

if st.button("🔍 Predict"):
    input_encoded = encode_inputs(input_data)
    proba = model.predict(input_encoded)[0][0]

    # ✅ Fixed inverted logic
    if proba < 0.5:
        st.success("✅ The customer is likely to subscribe to a term deposit.")
    else:
        st.error("❌ The customer is unlikely to subscribe to a term deposit.")