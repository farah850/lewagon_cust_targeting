import sys
import pathlib
import streamlit as st
import pickle
import pandas as pd

# Adding the project root to sys.path for imports to work
project_root = pathlib.Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

@st.cache_resource
def load_model():
    file_path = '../models/log_pipeline_20250807.pkl'
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model

def show_prediction():
    st.title("Customer Investment Prediction")
    st.write("### Enter customer information:")

    # Row 1
    col1, col2 = st.columns(2)
    age = col1.number_input("Age", min_value=18, max_value=100, value=30)
    job = col2.selectbox("Job", [
        "admin.", "blue-collar", "technician", "services", "management", "retired",
        "unemployed", "self-employed", "entrepreneur", "housemaid", "student"
    ])

    # Row 2
    col1, col2 = st.columns(2)
    marital = col1.selectbox("Marital Status", ["married", "single", "divorced"])
    education = col2.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])

    # Row 3
    col1, col2, col3 = st.columns(3)
    default = col1.selectbox("Has credit default?", ["yes", "no"])
    balance = col2.number_input("Balance", value=0)
    housing = col3.selectbox("Has housing loan?", ["yes", "no"])

    # Row 4
    col1, col2, col3 = st.columns(3)
    loan = col1.selectbox("Has personal loan?", ["yes", "no"])
    contact = col2.selectbox("Contact communication type", ["cellular", "telephone", "unknown"])
    month = col3.selectbox("Last contact month", [
        "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"
    ])

    # Row 5
    col1, col2, col3 = st.columns(3)
    day = col1.number_input("Last contact day of month", min_value=1, max_value=31, value=5)
    campaign = col2.number_input("Number of contacts during campaign", min_value=1, max_value=50, value=1)
    pdays = col3.number_input("Days passed since last contact (-1 means never)", value=-1)

    # Row 6
    col1, col2 = st.columns(2)
    previous = col1.number_input("Number of contacts before this campaign", min_value=0, max_value=50, value=0)
    poutcome = col2.selectbox("Outcome of previous campaign", ["failure", "nonexistent", "success"])

    # Prediction button
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
