import streamlit as st
from src.data_loader import DataLoader
from src.ui.visualizer import Visualizer

def render_visualizations(data_loader: DataLoader, visualizer: Visualizer):
    """
    Render the Data Visualizations tab.
    
    Args:
        data_loader: The data loader instance.
        visualizer: The visualizer instance.
    """
    st.markdown(
        '<h2 class="sub-header">Data Visualizations</h2>', unsafe_allow_html=True
    )

    # Get column lists
    numeric_cols = data_loader.get_numeric_columns()
    categorical_cols = data_loader.get_categorical_columns()

    # Distribution plots for numeric columns
    if len(numeric_cols) > 0:
        st.markdown(
            '<h3 class="sub-header">Distribution Plots</h3>', unsafe_allow_html=True
        )

        selected_num_col = st.selectbox(
            "Select a numeric column for distribution analysis:", numeric_cols
        )

        col1, col2 = st.columns(2)

        # Create distribution plots
        hist_fig, box_fig = visualizer.create_distribution_plots(selected_num_col)

        with col1:
            if hist_fig:
                st.pyplot(hist_fig)

        with col2:
            if box_fig:
                st.pyplot(box_fig)

        # Correlation heatmap
        if len(numeric_cols) > 1:
            st.markdown(
                '<h3 class="sub-header">Correlation Heatmap</h3>',
                unsafe_allow_html=True,
            )

            corr_fig = visualizer.create_correlation_heatmap()
            if corr_fig:
                st.pyplot(corr_fig)

    # Categorical analysis
    if len(categorical_cols) > 0:
        st.markdown(
            '<h3 class="sub-header">Categorical Analysis</h3>',
            unsafe_allow_html=True,
        )

        selected_cat_col = st.selectbox(
            "Select a categorical column:", categorical_cols
        )

        # Create categorical plot
        cat_fig = visualizer.create_categorical_plot(selected_cat_col)
        if cat_fig:
            st.pyplot(cat_fig)

    # Scatter plot for numeric columns
    if len(numeric_cols) >= 2:
        st.markdown(
            '<h3 class="sub-header">Scatter Plot</h3>', unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            x_col = st.selectbox("Select X-axis:", numeric_cols, index=0)

        with col2:
            remaining_cols = [col for col in numeric_cols if col != x_col]
            y_col = st.selectbox(
                "Select Y-axis:",
                remaining_cols,
                index=0 if len(remaining_cols) > 0 else None,
            )

        if len(remaining_cols) > 0:
            scatter_fig = visualizer.create_scatter_plot(x_col, y_col)
            if scatter_fig:
                st.pyplot(scatter_fig)
