import streamlit as st
import pandas as pd


def show():
    """Statistical Analysis page (placeholder, consistent UI)."""
    st.header("Statistical Analysis")

    if 'data_loader' not in st.session_state or st.session_state.data_loader.data is None:
        st.warning("Please upload a data file first using the sidebar.")
        return

    df: pd.DataFrame = st.session_state.data_loader.data

    st.subheader("Overview")
    st.dataframe(df.head(20))

    st.subheader("Descriptive Statistics")
    numeric = df.select_dtypes(include=['number'])
    if not numeric.empty:
        st.dataframe(numeric.describe().T)
    else:
        st.info("No numeric columns detected.")

    st.markdown("<br>", unsafe_allow_html=True)

