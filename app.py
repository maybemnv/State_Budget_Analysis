import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import base64

# Import modules
from data_loader import DataLoader
from visualizer import Visualizer
from statistical_analyzer import StatisticalAnalyzer
from ml_analyzer import MLAnalyzer
from gemini_analyzer import GeminiAnalyzer
from time_series_analyzer import TimeSeriesAnalyzer

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
            "🧠 Gemini AI Analysis",
            "📅 Time Series Analysis",
        ]
    )

    # Tab 1: Overview
    with tab1:
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

    # Tab 2: Visualizations
    with tab2:
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

    # Tab 3: Statistical Analysis
    with tab3:
        st.markdown(
            '<h2 class="sub-header">Statistical Analysis</h2>', unsafe_allow_html=True
        )

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

    # Tab 4: ML Insights
    with tab4:
        st.markdown(
            '<h2 class="sub-header">Machine Learning Insights</h2>',
            unsafe_allow_html=True,
        )

        if len(numeric_cols) >= 2:
            st.markdown(
                '<div class="info-text">This section provides machine learning insights from your data.</div>',
                unsafe_allow_html=True,
            )

            # Create tabs for different ML analyses
            ml_tab1, ml_tab2, ml_tab3, ml_tab4 = st.tabs(
                ["PCA Analysis", "Clustering", "Regression", "Classification"]
            )

            # PCA Analysis tab
            with ml_tab1:
                st.markdown(
                    '<h3 class="sub-header">Principal Component Analysis (PCA)</h3>',
                    unsafe_allow_html=True,
                )

                # Select columns for PCA
                pca_cols = st.multiselect(
                    "Select numeric columns for PCA:",
                    numeric_cols,
                    default=numeric_cols[: min(5, len(numeric_cols))],
                )

                if len(pca_cols) >= 2:
                    # Number of components
                    n_components = st.slider(
                        "Number of components:",
                        min_value=2,
                        max_value=min(len(pca_cols), 10),
                        value=min(3, len(pca_cols)),
                    )

                    if st.button("Run PCA Analysis"):
                        with st.spinner("Performing PCA analysis..."):
                            # Perform PCA
                            pca_result = ml_analyzer.perform_pca(pca_cols, n_components)

                            if pca_result is not None:
                                # Display explained variance
                                explained_variance = pca_result["explained_variance"]

                                st.write(
                                    "**Explained Variance by Principal Components:**"
                                )
                                for i, var in enumerate(explained_variance):
                                    st.write(f"PC{i+1}: {var:.2f}%")

                                # Create PCA plots
                                pca_plots = ml_analyzer.create_pca_plots(pca_result)

                                if pca_plots is not None:
                                    # Display plots
                                    if "explained_variance" in pca_plots:
                                        st.pyplot(pca_plots["explained_variance"])

                                    if "scatter" in pca_plots:
                                        st.pyplot(pca_plots["scatter"])

                                    # Feature importance
                                    st.markdown(
                                        "<h4>Feature Importance</h4>",
                                        unsafe_allow_html=True,
                                    )

                                    st.dataframe(pca_result["loadings"])

                                    if "importance" in pca_plots:
                                        st.pyplot(pca_plots["importance"])

            # Clustering tab
            with ml_tab2:
                st.markdown(
                    '<h3 class="sub-header">K-Means Clustering</h3>',
                    unsafe_allow_html=True,
                )

                # Select columns for clustering
                cluster_cols = st.multiselect(
                    "Select numeric columns for clustering:",
                    numeric_cols,
                    default=numeric_cols[: min(3, len(numeric_cols))],
                    key="cluster_cols",
                )

                if len(cluster_cols) >= 2:
                    # Number of clusters
                    k = st.slider(
                        "Number of clusters (k):", min_value=2, max_value=10, value=3
                    )

                    if st.button("Run Clustering Analysis"):
                        with st.spinner("Performing clustering analysis..."):
                            # Perform clustering
                            clustering_result = ml_analyzer.perform_clustering(
                                cluster_cols, k
                            )

                            if clustering_result is not None:
                                # Display cluster centers
                                centers = clustering_result["centers"]

                                st.write("**Cluster Centers:**")
                                st.dataframe(centers)

                                # Create clustering plots
                                clustering_plots = ml_analyzer.create_clustering_plots(
                                    clustering_result
                                )

                                if clustering_plots is not None:
                                    # Display plots
                                    if "scatter" in clustering_plots:
                                        st.pyplot(clustering_plots["scatter"])

                                    if "distribution" in clustering_plots:
                                        st.pyplot(clustering_plots["distribution"])

                                    if "parallel" in clustering_plots:
                                        st.pyplot(clustering_plots["parallel"])

                                # Download clustered data
                                st.markdown(
                                    "<h4>Download Clustered Data</h4>",
                                    unsafe_allow_html=True,
                                )

                                download_link = ml_analyzer.create_download_link(
                                    clustering_result["cluster_df"],
                                    "clustered_data.csv",
                                )

                                if download_link:
                                    st.markdown(download_link, unsafe_allow_html=True)

            # Regression tab
            with ml_tab3:
                st.markdown(
                    '<h3 class="sub-header">Regression Analysis</h3>',
                    unsafe_allow_html=True,
                )

                # Select target column
                target_col = st.selectbox(
                    "Select target column for regression:",
                    numeric_cols,
                    key="reg_target",
                )

                # Select feature columns
                feature_cols = st.multiselect(
                    "Select feature columns (leave empty to use all numeric columns except target):",
                    [col for col in numeric_cols if col != target_col],
                    default=[],
                    key="reg_features",
                )

                # Use all numeric columns except target if none selected
                if len(feature_cols) == 0:
                    feature_cols = None

                # Test size
                test_size = (
                    st.slider(
                        "Test size (%):", min_value=10, max_value=50, value=20, step=5
                    )
                    / 100
                )

                if st.button("Run Regression Analysis"):
                    with st.spinner("Training regression model..."):
                        # Train regression model
                        regression_result = ml_analyzer.train_regression_model(
                            target_col, feature_cols, test_size
                        )

                        if regression_result is not None:
                            # Display metrics
                            st.write("**Regression Metrics:**")
                            metrics_df = pd.DataFrame(
                                {
                                    "Metric": [
                                        "Mean Squared Error",
                                        "Root Mean Squared Error",
                                        "R² Score",
                                    ],
                                    "Value": [
                                        regression_result["mse"],
                                        regression_result["rmse"],
                                        regression_result["r2"],
                                    ],
                                }
                            )
                            st.dataframe(metrics_df)

                            # Create regression plots
                            regression_plots = ml_analyzer.create_regression_plots(
                                regression_result
                            )

                            if regression_plots is not None:
                                # Display plots
                                col1, col2 = st.columns(2)

                                with col1:
                                    if "actual_vs_predicted" in regression_plots:
                                        st.pyplot(
                                            regression_plots["actual_vs_predicted"]
                                        )

                                with col2:
                                    if "residuals" in regression_plots:
                                        st.pyplot(regression_plots["residuals"])

                                # Feature importance
                                st.markdown(
                                    "<h4>Feature Importance</h4>",
                                    unsafe_allow_html=True,
                                )

                                st.dataframe(regression_result["feature_importance"])

                                if "importance" in regression_plots:
                                    st.pyplot(regression_plots["importance"])

            # Classification tab
            with ml_tab4:
                st.markdown(
                    '<h3 class="sub-header">Classification Analysis</h3>',
                    unsafe_allow_html=True,
                )

                # Get categorical columns with limited unique values
                potential_targets = []
                for col in data_loader.get_data().columns:
                    if col in categorical_cols:
                        n_unique = data_loader.get_data()[col].nunique()
                        if 2 <= n_unique <= 10:  # Reasonable number of classes
                            potential_targets.append(col)
                    elif col in numeric_cols:
                        n_unique = data_loader.get_data()[col].nunique()
                        if 2 <= n_unique <= 10:  # Reasonable number of classes
                            potential_targets.append(col)

                if len(potential_targets) > 0:
                    # Select target column
                    target_col = st.selectbox(
                        "Select target column for classification:",
                        potential_targets,
                        key="class_target",
                    )

                    # Select feature columns
                    feature_cols = st.multiselect(
                        "Select feature columns (leave empty to use all numeric columns except target):",
                        [col for col in numeric_cols if col != target_col],
                        default=[],
                        key="class_features",
                    )

                    # Use all numeric columns except target if none selected
                    if len(feature_cols) == 0:
                        feature_cols = None

                    # Test size
                    test_size = (
                        st.slider(
                            "Test size (%):",
                            min_value=10,
                            max_value=50,
                            value=20,
                            step=5,
                            key="class_test_size",
                        )
                        / 100
                    )

                    if st.button("Run Classification Analysis"):
                        with st.spinner("Training classification model..."):
                            # Train classification model
                            classification_result = ml_analyzer.train_classification_model(
                                target_col, feature_cols, test_size
                            )

                            if classification_result is not None:
                                # Display metrics
                                st.write("**Classification Metrics:**")
                                st.write(
                                    f"Accuracy: {classification_result['accuracy']:.4f}"
                                )

                                # Create classification plots
                                classification_plots = ml_analyzer.create_classification_plots(
                                    classification_result
                                )

                                if classification_plots is not None:
                                    # Display plots
                                    col1, col2 = st.columns(2)

                                    with col1:
                                        if "confusion_matrix" in classification_plots:
                                            st.pyplot(
                                                classification_plots["confusion_matrix"]
                                            )

                                    with col2:
                                        if "class_report" in classification_plots:
                                            st.pyplot(
                                                classification_plots["class_report"]
                                            )

                                    # Feature importance
                                    st.markdown(
                                        "<h4>Feature Importance</h4>",
                                        unsafe_allow_html=True,
                                    )

                                    st.dataframe(
                                        classification_result["feature_importance"]
                                    )

                                    if "importance" in classification_plots:
                                        st.pyplot(classification_plots["importance"])
                else:
                    st.info(
                        "No suitable target columns found for classification. Target columns should have between 2 and 10 unique values."
                    )

    # Tab 5: Gemini AI Analysis
    with tab5:
        st.markdown(
            '<h2 class="sub-header">Gemini AI Analysis</h2>', unsafe_allow_html=True
        )
        
        if not gemini_analyzer.is_configured():
            st.warning(
                "Please configure your Gemini API key in the sidebar to use AI analysis."
            )
        else:
            st.markdown(
                '<div class="info-text">This section uses Google\'s Gemini AI to provide insights about your data.</div>',
                unsafe_allow_html=True,
            )

            # Analysis options
            analysis_type = st.selectbox(
                "Select analysis type:", gemini_analyzer.get_analysis_types()
            )

            # Custom question for custom analysis
            custom_question = None
            if analysis_type == "Custom Analysis":
                custom_question = st.text_area(
                    "Enter your specific question about the data:",
                    "What are the most interesting insights from this dataset and what actions would you recommend based on them?",
                )

            # Run analysis button
            if st.button("Run AI Analysis"):
                with st.spinner("Gemini AI is analyzing your data..."):
                    # Generate analysis
                    response = gemini_analyzer.analyze_data(
                        analysis_type, custom_question
                    )

                    if response:
                        # Display response
                        st.markdown(
                            '<h3 class="sub-header">AI Analysis Results</h3>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(response)

                        # Save analysis to file option
                        st.download_button(
                            label="Download Analysis",
                            data=response,
                            file_name="gemini_analysis.txt",
                            mime="text/plain",
                        )
        
    # Tab 6: Time Series Analysis
    with tab6:
        st.markdown('<h2 class="sub-header">Time Series Analysis</h2>', unsafe_allow_html=True)
        
        st.markdown(
            '<div class="info-text">Analyze time series data with automatic frequency detection, decomposition, stationarity testing, and forecasting. '
            'Works with date columns, year columns (2019, 2020...), or any sequential data!</div>',
            unsafe_allow_html=True,
        )
        
        # Detect what kind of time series data we have
        ts_capabilities = time_series_analyzer.detect_time_series_capable_data(data_loader.get_data())
        
        # Setup section
        st.markdown("### ⚙️ Setup Time Series Data")
        
        # Determine data mode
        data_modes = []
        if ts_capabilities['has_date_columns']:
            data_modes.append("📅 Use Date Column")
        if ts_capabilities['has_year_columns']:
            data_modes.append("📊 Use Year Columns (Financial Data)")
        if ts_capabilities['numeric_columns']:
            data_modes.append("🔢 Use Sequential Data (Create Synthetic Dates)")
        
        if not data_modes:
            st.error(
                "❌ No time series capable data detected. Your data needs either:\n\n"
                "1. A date/time column\n"
                "2. Year columns (2019, 2020, 2021, etc.)\n"
                "3. Numeric columns with sequential data"
            )
        else:
            # Select date column
            date_col = st.selectbox(
                "Select Date Column:",
                potential_date_cols,
                help="Choose the column containing dates/timestamps"
            )
            
            # Select value column
            value_col = st.selectbox(
                "Select Value Column:",
                numeric_cols,
                help="Choose the numeric column to analyze over time"
            )
            
            if st.button("🚀 Run Time Series Analysis", type="primary"):
                try:
                    with st.spinner("🔄 Preparing and cleaning time series data..."):
                        # Prepare data
                        series = time_series_analyzer.load_and_prepare_data(
                            data_loader.get_data(), date_col, value_col
                        )
                    
                    # Show data info
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Data Points", len(series))
                    with col2:
                        st.metric("Frequency", time_series_analyzer.frequency or "Unknown")
                    with col3:
                        date_range = f"{series.index.min().date()} to {series.index.max().date()}"
                        st.metric("Date Range", "")
                        st.caption(date_range)
                    
                    st.success("✅ Data prepared successfully!")
                    
                    # Create tabs for different analyses
                    ts_tab1, ts_tab2, ts_tab3 = st.tabs([
                        "📊 Decomposition",
                        "📈 Stationarity",
                        "🔮 Forecasting"
                    ])
                    
                    # Tab 1: Decomposition Analysis
                    with ts_tab1:
                        st.markdown("### Time Series Decomposition")
                        st.info(
                            "Decomposition breaks down the time series into trend, seasonal, and residual components."
                        )
                        
                        with st.spinner("Decomposing time series..."):
                            decomp_fig, error = time_series_analyzer.decompose_series(series)
                        
                        if error:
                            st.error(f"❌ {error}")
                            if "Insufficient data" in error:
                                st.info(
                                    "💡 **Tip:** Decomposition requires at least 2 full periods of data. "
                                    "For monthly data, you need at least 24 months."
                                )
                        else:
                            st.plotly_chart(decomp_fig, use_container_width=True)
                    
                    # Tab 2: Stationarity Test
                    with ts_tab2:
                        st.markdown("### Stationarity Analysis")
                        st.info(
                            "Stationarity test checks if the statistical properties of the series remain constant over time. "
                            "Stationary data is important for many forecasting models."
                        )
                        
                        with st.spinner("Running stationarity test..."):
                            stationarity_results, error = time_series_analyzer.check_stationarity(series)
                        
                        if error:
                            st.error(f"❌ {error}")
                        else:
                            st.write("**Augmented Dickey-Fuller Test Results:**")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Test Statistic", f"{stationarity_results['test_statistic']:.4f}")
                            with col2:
                                st.metric("p-value", f"{stationarity_results['p_value']:.4f}")
                            
                            st.write("\n**Critical Values:**")
                            crit_cols = st.columns(len(stationarity_results['critical_values']))
                            for idx, (key, value) in enumerate(stationarity_results['critical_values'].items()):
                                with crit_cols[idx]:
                                    st.metric(key, f"{value:.4f}")
                                
                            # Interpretation
                            st.markdown("---")
                            if stationarity_results['is_stationary']:
                                st.success(
                                    "✅ **The time series is STATIONARY** (p-value < 0.05)\n\n"
                                    "This means the series has constant mean and variance over time, "
                                    "making it suitable for most forecasting models."
                                )
                            else:
                                st.warning(
                                    "⚠️ **The time series is NON-STATIONARY** (p-value ≥ 0.05)\n\n"
                                    "This means the statistical properties change over time. "
                                    "Consider differencing or transformation before forecasting."
                                )
                    
                    # Tab 3: Forecasting
                    with ts_tab3:
                        st.markdown("### Time Series Forecasting")
                        st.info(
                            "Forecast future values based on historical patterns. "
                            "**ARIMA** is good for stationary data, while **Prophet** handles seasonality and trends automatically."
                        )
                        
                        # Select forecasting method
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            method = st.radio(
                                "Forecasting Method:",
                                ["Prophet", "ARIMA"],
                                help="Prophet is more robust and handles missing data better. ARIMA gives more control."
                            )
                        with col2:
                            # Number of periods to forecast
                            freq_label = "periods"
                            if time_series_analyzer.frequency:
                                freq_map = {'D': 'days', 'W': 'weeks', 'M': 'months', 'Q': 'quarters', 'Y': 'years'}
                                freq_label = freq_map.get(time_series_analyzer.frequency, 'periods')
                            
                            periods = st.slider(
                                f"Forecast {freq_label}:",
                                min_value=1,
                                max_value=min(48, len(series)),
                                value=min(12, len(series) // 4),
                                help=f"Number of {freq_label} to predict into the future"
                            )
                        
                        if method == "ARIMA":
                            # ARIMA parameters
                            st.markdown("**ARIMA Parameters:**")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                p = st.number_input(
                                    "AR(p)",
                                    min_value=0,
                                    max_value=5,
                                    value=1,
                                    help="Autoregressive order - number of lag observations"
                                )
                            with col2:
                                d = st.number_input(
                                    "I(d)",
                                    min_value=0,
                                    max_value=2,
                                    value=1,
                                    help="Degree of differencing - number of times to difference the series"
                                )
                            with col3:
                                q = st.number_input(
                                    "MA(q)",
                                    min_value=0,
                                    max_value=5,
                                    value=1,
                                    help="Moving average order - size of the moving average window"
                                )
                            
                            st.caption("💡 Common starting points: (1,1,1) for non-stationary data, (1,0,1) for stationary data")
                                
                            if st.button("🚀 Generate ARIMA Forecast", type="primary"):
                                with st.spinner(f"Training ARIMA({p},{d},{q}) model and generating forecast..."):
                                    # Fit ARIMA model
                                    model, error = time_series_analyzer.fit_arima(series, order=(p,d,q))
                                    
                                    if error:
                                        st.error(f"❌ {error}")
                                        st.info(
                                            "💡 **Tips:**\n"
                                            "- Try different parameter combinations\n"
                                            "- Ensure you have enough data points\n"
                                            "- Check if your data needs differencing (use I(d)=1 or 2)"
                                        )
                                    else:
                                        forecast, forecast_error = time_series_analyzer.forecast_arima(steps=periods)
                                        
                                        if forecast_error:
                                            st.error(f"❌ {forecast_error}")
                                        else:
                                            # Determine frequency
                                            freq = time_series_analyzer.frequency or 'M'
                                            
                                            # Create forecast dataframe
                                            forecast_index = pd.date_range(
                                                start=series.index[-1],
                                                periods=periods+1,
                                                freq=freq
                                            )[1:]
                                            
                                            # Calculate confidence intervals
                                            forecast_stderr = model.forecast(steps=periods, alpha=0.05)
                                            if hasattr(forecast_stderr, 'conf_int'):
                                                conf_int = forecast_stderr.conf_int()
                                                lower = conf_int[:, 0]
                                                upper = conf_int[:, 1]
                                            else:
                                                # Fallback to simple stderr estimation
                                                stderr = np.std(series) * 1.96
                                                lower = forecast - stderr
                                                upper = forecast + stderr
                                            
                                            forecast_df = pd.DataFrame({
                                                'ds': forecast_index,
                                                'yhat': forecast,
                                                'yhat_lower': lower,
                                                'yhat_upper': upper
                                            })
                                            
                                            # Plot forecast
                                            fig = time_series_analyzer.plot_forecast(
                                                series, 
                                                forecast_df,
                                                f"ARIMA({p},{d},{q}) Forecast"
                                            )
                                            st.plotly_chart(fig, use_container_width=True)
                                            
                                            # Show model summary
                                            with st.expander("📊 View Model Summary"):
                                                st.text(model.summary())
                                            
                                            # Download forecast
                                            csv = forecast_df.to_csv(index=False)
                                            st.download_button(
                                                label="💾 Download Forecast Data",
                                                data=csv,
                                                file_name="arima_forecast.csv",
                                                mime="text/csv"
                                            )
                                    
                        else:  # Prophet
                            st.markdown(
                                "**Prophet Forecast:** Prophet automatically handles seasonality, "
                                "holidays, and missing data. No parameter tuning required!"
                            )
                            
                            if st.button("🚀 Generate Prophet Forecast", type="primary"):
                                with st.spinner("Training Prophet model and generating forecast..."):
                                    # Fit Prophet model and generate forecast
                                    forecast_df, error = time_series_analyzer.fit_prophet(series, periods)
                                    
                                    if error:
                                        st.error(f"❌ {error}")
                                        st.info(
                                            "💡 **Tips:**\n"
                                            "- Ensure you have enough data points (at least 10)\n"
                                            "- Check that your date column is properly formatted\n"
                                            "- Prophet works best with daily or monthly data"
                                        )
                                    else:
                                        # Plot forecast
                                        fig = time_series_analyzer.plot_forecast(
                                            series, 
                                            forecast_df,
                                            "Prophet Forecast"
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Show forecast components
                                        st.markdown("### Forecast Components")
                                        st.caption("Prophet decomposes the forecast into trend and seasonal components")
                                        
                                        # Create components plot
                                        component_cols = ['trend']
                                        for col in ['yearly', 'weekly', 'daily']:
                                            if col in forecast_df.columns:
                                                component_cols.append(col)
                                        
                                        if len(component_cols) > 0:
                                            fig_components = make_subplots(
                                                rows=len(component_cols),
                                                cols=1,
                                                subplot_titles=[c.title() for c in component_cols]
                                            )
                                            
                                            for idx, col in enumerate(component_cols, 1):
                                                fig_components.add_trace(
                                                    go.Scatter(
                                                        x=forecast_df['ds'],
                                                        y=forecast_df[col],
                                                        name=col.title(),
                                                        line=dict(color='blue')
                                                    ),
                                                    row=idx,
                                                    col=1
                                                )
                                            
                                            fig_components.update_layout(
                                                height=300 * len(component_cols),
                                                showlegend=False
                                            )
                                            st.plotly_chart(fig_components, use_container_width=True)
                                        
                                        # Download forecast
                                        forecast_csv = forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False)
                                        st.download_button(
                                            label="💾 Download Forecast Data",
                                            data=forecast_csv,
                                            file_name="prophet_forecast.csv",
                                            mime="text/csv"
                                        )
                
                except Exception as e:
                    st.error(f"❌ Error in time series analysis: {str(e)}")
                    st.info(
                        "💡 **Common issues:**\n"
                        "- Date column format not recognized\n"
                        "- Insufficient data points\n"
                        "- Missing or invalid values in the series\n\n"
                        "Try selecting a different date or value column."
                    )

else:
    # Display welcome message when no data is loaded
    st.markdown('<div class="info-text">', unsafe_allow_html=True)
    st.markdown(
        """
    # Welcome to CSV Data Analyzer!
    
    This application helps you analyze CSV data files with powerful visualization and AI-powered insights.
    
    ## Features:
    
    - **Data Overview**: Get a quick summary of your data structure and content
    - **Visualizations**: Generate various charts and plots to understand patterns
    - **Statistical Analysis**: Perform detailed statistical analysis on your data
    - **Machine Learning Insights**: Discover clusters and patterns with ML algorithms
    - **Gemini AI Analysis**: Get AI-powered insights and recommendations
    
    ## Getting Started:
    
    1. Upload your CSV file using the sidebar on the left
    2. Optionally, add your Gemini API key for AI-powered analysis
    3. Explore the different tabs to analyze your data
    
    Upload a CSV file to begin!
    """
    )
    st.markdown("</div>", unsafe_allow_html=True)
