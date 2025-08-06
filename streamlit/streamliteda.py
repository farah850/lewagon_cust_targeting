
#Streamlit wrapper function for the EDA package
import streamlit as st
#from eda_package import clean_data, plot_distributions, plot_correlations # Package created by Son for the EDA tasks

def show_eda(data):


    # 1. Calling the clean data function
    st.write("### Cleaned Data Preview")
    st.dataframe(data.head())

    # 2. Plotting distributions
    # st.write("### Feature Distributions")
    # #dist_fig = plot_distributions(data)
    # st.pyplot(dist_fig)

    # # 3. Plotting correlations
    # st.write("### Correlation Matrix")
    # corr_fig = plot_correlations(data)
    # st.pyplot(corr_fig)
