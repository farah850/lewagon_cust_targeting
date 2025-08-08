import sys
import pathlib

# Adding the project root to sys.path for imports to work
project_root = pathlib.Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


import streamlit as st
import pickle
import pandas as pd

@st.cache_resource
def load_model():
    file_path = '../models/log_pipeline_20250807.pkl'
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model

def show_prediction():
    st.title("Customer Investment Prediction")

    st.write("### Enter customer information:")

    # Input fields based on model features
    age = st.number_input("Age", min_value=18, max_value=100, value=30)

    job = st.selectbox("Job", [
        "admin.", "blue-collar", "technician", "services", "management", "retired",
        "unemployed", "self-employed", "entrepreneur", "housemaid", "student"
    ])

    marital = st.selectbox("Marital Status", ["married", "single", "divorced"])

    education = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])

    default = st.selectbox("Has credit default?", ["yes", "no"])

    balance = st.number_input("Balance", value=0)

    housing = st.selectbox("Has housing loan?", ["yes", "no"])

    loan = st.selectbox("Has personal loan?", ["yes", "no"])

    contact = st.selectbox("Contact communication type", ["cellular", "telephone", "unknown"])

    month = st.selectbox("Last contact month", [
        "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"
    ])

    day = st.number_input("Last contact day of month", min_value=1, max_value=31, value=5)

    campaign = st.number_input("Number of contacts during campaign", min_value=1, max_value=50, value=1)

    pdays = st.number_input("Days passed since last contact (-1 means never)", value=-1)

    previous = st.number_input("Number of contacts before this campaign", min_value=0, max_value=50, value=0)

    poutcome = st.selectbox("Outcome of previous campaign", ["failure", "nonexistent", "success"])

    if st.button("Predict"):
        model = load_model()

        input_data = pd.DataFrame([{
            "age": age,
            "job": job,
            "marital": marital,
            "education": education,
            "default": default,
            "balance": balance,
            "housing": housing,
            "loan": loan,
            "contact": contact,
            "day": day,
            "month": month,
            "campaign": campaign,
            "pdays": pdays,
            "previous": previous,
            "poutcome": poutcome
        }])

        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]

        st.success(f"Prediction: {'Will Invest' if prediction == 1 else 'Will Not Invest'}")
        st.info(f"Probability of Investing: {proba:.2%}")
