#Necessary imports for EDA

from matplotlib import pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import pandas as pd

def show_eda(data):

    # Create descriptive labels for housing + loan combo
    housing_map = {'yes': 'Housing', 'no': 'No Housing'}
    loan_map = {'yes': 'Other Loan', 'no': 'No Other Loan'}
    data['housing_loan_combo'] = data['housing'].map(housing_map) + " + " + data['loan'].map(loan_map)

    # Pleasant pastel coloring instead of red and green
    pleasant_palette = px.colors.qualitative.Set2

    #descriptive text
    st.write("""
    ### Housing & Loan Status Analysis
    We explore how combinations of housing and loan statuses relate to customers' decisions to invest.
    The first chart shows the absolute number of customers by housing + loan combo, grouped by investment decision.
    The second chart presents the proportions of investment decisions within each housing + loan combo for better comparison.
    """)
    #adding in a main center title
    st.markdown(
        "<h3 style='text-align: center;'>Housing & Loan Investment Decisions: Counts and Proportions</h3>",
        unsafe_allow_html=True
    )
    #having the fist two charts side by side
    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.histogram(
            data,
            x='housing_loan_combo',
            color='y',
            barmode='group',
            color_discrete_sequence=pleasant_palette,
            labels={
                'housing_loan_combo': 'Housing + Loan Status',
                'count': 'Number of Customers',
                'y': 'Will Invest?'
            }
        )
        st.plotly_chart(fig1, use_container_width=True)

    # Calculate proportions for combo
    combo_counts = (
        data.groupby(['housing_loan_combo', 'y'])
        .size()
        .reset_index(name='count')
    )
    combo_counts['proportion'] = combo_counts.groupby('housing_loan_combo')['count'].transform(lambda x: x / x.sum())

    with col2:
        fig2 = px.bar(
            combo_counts,
            x='housing_loan_combo',
            y='proportion',
            color='y',
            barmode='stack',
            color_discrete_sequence=pleasant_palette,
            labels={
                'housing_loan_combo': 'Housing + Loan Status',
                'proportion': 'Proportion',
                'y': 'Will Invest?'
            }
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Marital status analysis with explanation
    st.write("### Proportion of Investment Decisions by Marital Status")

    st.write("""
    Let's explore how marital status relates to the customers' investment decisions.
    This chart shows the proportion of customers who decided to invest or not within each marital status group.
    """)

    marital_counts = (
        data.groupby(['marital', 'y'])
        .size()
        .reset_index(name='count')
    )
    marital_counts['proportion'] = marital_counts.groupby('marital')['count'].transform(lambda x: x / x.sum())

    fig3 = px.bar(
        marital_counts,
        x='marital',
        y='proportion',
        color='y',
        barmode='stack',
        color_discrete_sequence=pleasant_palette,
        title="Proportion of Investment Decisions by Marital Status",
        labels={
            'marital': 'Marital Status',
            'proportion': 'Proportion',
            'y': 'Will Invest?'
        }
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.write("### Interactive Variable Distribution")

    var_options = ["age", "job", "marital", "education"]
    selected_var = st.selectbox("Select variable to visualize:", var_options)

    if pd.api.types.is_numeric_dtype(data[selected_var]):
        fig = px.histogram(
            data,
            x=selected_var,
            nbins=30,
            title=f"{selected_var.capitalize()} — Distribution"

        )
    else:
        fig = px.histogram(
            data,
            x=selected_var,
            color="y",
            barmode="stack",
            color_discrete_sequence=pleasant_palette,
            title=f"{selected_var.capitalize()} by Subscription (y)"
        )
        fig.update_layout(xaxis_tickangle=-45)

    st.plotly_chart(fig, use_container_width=True)
