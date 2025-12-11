import streamlit as st
import pandas as pd
from src.data_loader import DataLoader

def render_overview(data_loader: DataLoader):
    """
    Render the Data Overview tab.
    
    Args:
        data_loader: The data loader instance containing the data.
    """
    st.markdown('<h2 class="sub-header">Data Overview</h2>', unsafe_allow_html=True)

    # Get data info
    data_info = data_loader.get_data_info()

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Filename:** {data_info['filename']}")
        st.write(f"**Rows:** {data_info['rows']}")
        st.write(f"**Columns:** {data_info['columns']}")

    with col2:
        st.write("**Memory Usage:**")
        st.write(data_info["memory_usage"])

        # Missing values summary
        st.write(f"**Missing Values:** {data_info['missing_values']}")

    st.markdown('<h3 class="sub-header">Data Preview</h3>', unsafe_allow_html=True)
    st.dataframe(data_loader.get_data().head(10))

    st.markdown(
        '<h3 class="sub-header">Column Information</h3>', unsafe_allow_html=True
    )

    # Column information
    col_info = data_loader.get_column_info()
    st.dataframe(pd.DataFrame(col_info))

    # Download processed data
    st.markdown(
        '<h3 class="sub-header">Download Processed Data</h3>',
        unsafe_allow_html=True,
    )
    download_link = data_loader.create_download_link()
    if download_link:
        st.markdown(download_link, unsafe_allow_html=True)
