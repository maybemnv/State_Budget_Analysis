import streamlit as st
from pathlib import Path
import sys

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent))

# Import application modules
from src.config import APP_CONFIG, PLOT_CONFIG
from src.data_loader import DataLoader

# Set page configuration
st.set_page_config(
    page_title="State Budget Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        max-width: 1200px;
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
    }
    .stSelectbox, .stTextInput, .stNumberInput {
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding: 0 16px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loader' not in st.session_state:
    st.session_state.data_loader = DataLoader()

# Sidebar for file upload and navigation
with st.sidebar:
    st.title("📊 State Budget Analysis")
    st.markdown("---")
    
    # File uploader
    st.subheader("1. Upload Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your budget data file"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            data = st.session_state.data_loader.load_from_streamlit_uploader(uploaded_file)
            if data is not None:
                st.success(f"✅ Successfully loaded {uploaded_file.name}")
                
                # Display data summary
                st.subheader("Data Summary")
                st.write(f"- Rows: {data.shape[0]}")
                st.write(f"- Columns: {data.shape[1]}")
                
                # Display column information
                st.subheader("Columns")
                for col in data.columns:
                    col_type = "📅" if col in st.session_state.data_loader.get_date_columns() else "🔢"
                    st.write(f"{col_type} {col} ({data[col].dtype})")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    st.markdown("---")
    st.markdown("### Navigation")
    
    # Navigation buttons
    if st.button("🏠 Home"):
        st.session_state.current_page = "home"
    if st.button("📈 Time Series Analysis"):
        st.session_state.current_page = "time_series"
    if st.button("📊 Statistical Analysis"):
        st.session_state.current_page = "stats"
    if st.button("🤖 ML Insights"):
        st.session_state.current_page = "ml"
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("State Budget Analysis Tool")
    st.markdown("Version 1.0.0")

# Main content area
st.title("State Budget Analysis")

# Initialize current page if not set
if 'current_page' not in st.session_state:
    st.session_state.current_page = "home"

# Page routing
if st.session_state.current_page == "home":
    st.markdown("""
    ## Welcome to State Budget Analysis Tool
    
    This application helps you analyze and visualize state budget data. 
    Get started by uploading your data using the sidebar.
    
    ### Features:
    - 📊 Interactive data exploration
    - 📈 Time series analysis and forecasting
    - 📊 Statistical analysis
    - 🤖 Machine learning insights
    
    ### How to use:
    1. Upload your budget data file (CSV or Excel)
    2. Navigate to the desired analysis section
    3. Explore the insights and visualizations
    """)

elif st.session_state.current_page == "time_series":
    st.header("Time Series Analysis")
    
    if st.session_state.data_loader.data is None:
        st.warning("Please upload a data file first using the sidebar.")
    else:
        # Time series analysis page will be loaded from a separate module
        from src.pages import time_series_analysis
        time_series_analysis.show()

elif st.session_state.current_page == "stats":
    st.header("Statistical Analysis")
    
    if st.session_state.data_loader.data is None:
        st.warning("Please upload a data file first using the sidebar.")
    else:
        # Statistical analysis page will be loaded from a separate module
        from src.pages import statistical_analysis
        statistical_analysis.show()

elif st.session_state.current_page == "ml":
    st.header("Machine Learning Insights")
    
    if st.session_state.data_loader.data is None:
        st.warning("Please upload a data file first using the sidebar.")
    else:
        # ML insights page will be loaded from a separate module
        from src.pages import ml_insights
        ml_insights.show()

# Add some spacing at the bottom
st.markdown("<br><br>", unsafe_allow_html=True)
