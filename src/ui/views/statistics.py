import streamlit as st
import pandas as pd
from src.data_loader import DataLoader
from src.ui.visualizer import Visualizer
from src.analyzers.statistical_analyzer import StatisticalAnalyzer

def render_statistics(data_loader: DataLoader, statistical_analyzer: StatisticalAnalyzer, visualizer: Visualizer):
    """
    Render the Statistical Analysis tab.
    
    Args:
        data_loader: The data loader instance.
        statistical_analyzer: The statistical analyzer instance.
        visualizer: The visualizer instance for plotting.
    """
    st.markdown(
        '<h2 class="sub-header">Statistical Analysis</h2>', unsafe_allow_html=True
    )

    numeric_cols = data_loader.get_numeric_columns()
    categorical_cols = data_loader.get_categorical_columns()

    if len(numeric_cols) > 0:
        # Descriptive statistics
        st.markdown(
            '<h3 class="sub-header">Descriptive Statistics</h3>',
            unsafe_allow_html=True,
        )
        desc_stats = statistical_analyzer.get_descriptive_statistics()
        if desc_stats is not None:
            st.dataframe(desc_stats)

        # Group by analysis
        if len(categorical_cols) > 0:
            st.markdown(
                '<h3 class="sub-header">Group By Analysis</h3>',
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns(2)

            with col1:
                group_col = st.selectbox(
                    "Select column to group by:", categorical_cols
                )

            with col2:
                agg_col = st.selectbox("Select column to aggregate:", numeric_cols)

            agg_func = st.selectbox(
                "Select aggregation function:",
                ["Mean", "Median", "Sum", "Min", "Max", "Count", "Std Dev"],
            )

            # Perform group by analysis
            grouped_data = statistical_analyzer.get_group_by_statistics(
                group_col, agg_col, agg_func
            )

            if grouped_data is not None:
                # Display results
                st.write(f"**{agg_func} of {agg_col} grouped by {group_col}:**")
                st.dataframe(grouped_data)

                # Create group by plot
                grouped_data, group_fig = visualizer.create_group_by_plot(
                    group_col, agg_col, agg_func
                )
                if group_fig:
                    st.pyplot(group_fig)

        # Missing values analysis
        st.markdown(
            '<h3 class="sub-header">Missing Values Analysis</h3>',
            unsafe_allow_html=True,
        )
        missing_summary = statistical_analyzer.get_missing_values_summary()
        if missing_summary is not None and len(missing_summary) > 0:
            st.dataframe(missing_summary)
        else:
            st.info("No missing values found in the dataset.")

        # Outlier analysis
        st.markdown(
            '<h3 class="sub-header">Outlier Analysis</h3>', unsafe_allow_html=True
        )

        outlier_method = st.radio(
            "Select outlier detection method:", ["IQR", "Z-Score"], horizontal=True
        )
        method = "iqr" if outlier_method == "IQR" else "zscore"

        outlier_summary = statistical_analyzer.get_outliers_summary(method=method)
        if outlier_summary is not None and len(outlier_summary) > 0:
            st.dataframe(outlier_summary)
        else:
            st.info("No outliers found using the selected method.")
