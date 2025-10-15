import streamlit as st
import pandas as pd


def show():
    """ML Insights page (placeholder, consistent UI)."""
    st.header("Machine Learning Insights")

    if 'data_loader' not in st.session_state or st.session_state.data_loader.data is None:
        st.warning("Please upload a data file first using the sidebar.")
        return

    df: pd.DataFrame = st.session_state.data_loader.data

    st.subheader("Setup")
    st.info("This page will host PCA, clustering, regression, and classification pipelines.")
    st.dataframe(df.head(10))

    st.markdown("<br>", unsafe_allow_html=True)

