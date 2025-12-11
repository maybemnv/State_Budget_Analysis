import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Import modules
from src.data_loader import DataLoader
from src.ui.visualizer import Visualizer
from src.analyzers.statistical_analyzer import StatisticalAnalyzer
from src.analyzers.ml_analyzer import MLAnalyzer
from src.analyzers.gemini_analyzer import GeminiAnalyzer
from src.time_series.analyzer import TimeSeriesAnalyzer

# Import UI Views
from src.ui.views.overview import render_overview
from src.ui.views.visualizations import render_visualizations
from src.ui.views.statistics import render_statistics
from src.ui.views.ml_insights import render_ml_insights
from src.ui.views.time_series import render_time_series
from src.ui.views.gemini_ai import render_gemini_ai

# Set page configuration
st.set_page_config(
    page_title="Data Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2196F3;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-text {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize modules
data_loader = DataLoader()
visualizer = Visualizer(data_loader)
statistical_analyzer = StatisticalAnalyzer(data_loader)
ml_analyzer = MLAnalyzer(data_loader)
gemini_analyzer = GeminiAnalyzer(data_loader)
time_series_analyzer = TimeSeriesAnalyzer()

# Main header
st.markdown('<h1 class="main-header">CSV Data Analyzer</h1>', unsafe_allow_html=True)

# Sidebar for file upload and settings
with st.sidebar:
    st.header("Upload & Settings")

    # File upload
    uploaded_file = data_loader.create_upload_widget()

    if uploaded_file is not None:
        success = data_loader.load_data(uploaded_file)
        if success:
            st.success(f"Successfully loaded: {data_loader.get_filename()}")

    # Gemini API key input
    st.markdown("---")
    st.subheader("Gemini AI Integration")
    gemini_configured = gemini_analyzer.create_api_key_input()

# Main content area
if data_loader.get_data() is not None:
    # Display tabs for different analysis options
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "📊 Overview",
            "📈 Visualizations",
            "🧮 Statistical Analysis",
            "🤖 ML Insights",
            "📅 Time Series Analysis",
            "🧠 Gemini AI Analysis",
        ]
    )

    # Tab 1: Overview
    with tab1:
        render_overview(data_loader)

    # Tab 2: Visualizations
    with tab2:
        render_visualizations(data_loader, visualizer)

    # Tab 3: Statistical Analysis
    with tab3:
        render_statistics(data_loader, statistical_analyzer, visualizer)

    # Tab 4: ML Insights
    with tab4:
        render_ml_insights(data_loader, ml_analyzer)

    # Tab 5: Time Series Analysis
    with tab5:
        render_time_series(data_loader, time_series_analyzer)

    # Tab 6: Gemini AI Analysis
    with tab6:
        render_gemini_ai(gemini_analyzer)
    
else:
    # Display welcome message when no data is loaded
    st.markdown('<div class="info-text">', unsafe_allow_html=True)
    st.markdown("<h3>Welcome to the Data Analyzer!</h3>", unsafe_allow_html=True)
    st.write("Please upload a CSV or Excel file using the sidebar to begin analysis.")
    st.write("This tool allows you to:")
    st.markdown(
        """
    *   **Overview:** View basic information and preview your data.
    *   **Visualizations:** Explore data distributions and relationships with various plots.
    *   **Statistical Analysis:** perform descriptive statistics and group-by operations.
    *   **ML Insights:** Gain insights using clustering, regression and PCA.
    *   **Time Series:** Analyze, transform, and forecast time-dependent data.
    *   **Gemini AI:** Use Google's Gemini AI to ask natural language questions about your data.
    """
    )
    st.markdown("</div>", unsafe_allow_html=True)
