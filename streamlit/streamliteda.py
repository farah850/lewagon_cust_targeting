
#Streamlit wrapper function for the EDA package
from matplotlib import pyplot as plt
import seaborn as sns
import streamlit as st
#from eda_package import clean_data, plot_distributions, plot_correlations # Package created by Son for the EDA tasks

def show_eda(data):
    st.write("## Exploratory Data Analysis")

    # 1. Show a preview of the data
    st.write("### Dataset Preview")
    st.dataframe(data.head())

    # 2. Plot distributions of numeric features
    st.write("### Distributions of Numerical Features")
    numeric_data = data.select_dtypes(include='number')
    for col in numeric_data.columns:
        st.write(f"#### Distribution of {col}")
        fig, ax = plt.subplots()
        sns.histplot(data[col], kde=True, ax=ax)
        st.pyplot(fig)

    # 3. Correlation heatmap
    st.write("### Correlation Heatmap")
    corr = numeric_data.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
