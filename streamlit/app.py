import streamlit as st
import pandas as pd
from streamliteda import show_eda
from pathlib import Path

@st.cache_data
def load_data():
    try:
        data_path = Path(__file__).resolve().parent.parent / "data" / "bank-full.csv"
        return pd.read_csv(data_path, sep=";")
    except FileNotFoundError:
        st.error(f"Data file not found at {data_path}")
        return pd.DataFrame()

def main():
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["EDA", "Prediction"])

    data = load_data()

    if page == "EDA":
        st.title("Exploratory Data Analysis")
        show_eda(data)

    # TODO: Add Prediction page functionality here

if __name__ == "__main__":
    main()
