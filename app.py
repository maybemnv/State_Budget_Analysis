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
from src.time_series.analyzer import TimeSeriesAnalyzer
from src.time_series.stationarity import make_stationary

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
            '<div class="info-text">Analyze time series with a clean, unified flow: prepare your series, then Analyze, Forecast, or check Stationarity. Works with date columns, year columns, or sequential data.</div>',
            unsafe_allow_html=True,
        )

        analyzer = TimeSeriesAnalyzer()
        df = data_loader.get_data()

        # Setup: choose mode and prepare series
        ts_cap = analyzer.detect_time_series_capable_data(df)
        modes = []
        if ts_cap['has_date_columns']:
            modes.append("📅 Date Column")
        if ts_cap['has_year_columns']:
            modes.append("📊 Year Columns (Wide → Long)")
        if len(ts_cap['numeric_columns']) > 0:
            modes.append("🔢 Sequential (Synthetic Dates)")

        if not modes:
            st.error("No time-series capable data detected. Provide a date column, year columns, or a numeric column.")
        else:
            st.markdown('<h3 class="sub-header">⚙️ Setup Time Series Data</h3>', unsafe_allow_html=True)
            mode = st.radio("Data Mode", options=modes)

            prepared_series = None
            if mode == "📅 Date Column":
                date_col = st.selectbox("Date column", ts_cap['date_columns'])
                value_col = st.selectbox("Value column", ts_cap['numeric_columns'])
                if st.button("Prepare Series", type="primary"):
                    try:
                        prepared_series = analyzer.load_and_prepare(df, date_col, value_col)
                        st.success("Series prepared")
                    except Exception as e:
                        st.error(str(e))
            elif mode == "📊 Year Columns (Wide → Long)":
                year_like = [c for c in df.columns if str(c).strip().isdigit()]
                year_cols = st.multiselect("Year columns", year_like, default=year_like[:min(5, len(year_like))])
                selector = st.text_input("Row selector (optional)")
                if st.button("Prepare Series", type="primary"):
                    series, err = analyzer.create_time_series_from_year_columns(df, year_cols, selector or None)
                    if err:
                        st.error(err)
                    else:
                        prepared_series = series
                        st.success("Series prepared")
            else:  # Sequential
                value_col = st.selectbox("Value column", ts_cap['numeric_columns'])
                freq = st.selectbox("Frequency", ["D", "W", "M", "Q", "Y"], index=2)
                start_date = st.date_input("Start date", pd.to_datetime("2000-01-01")).strftime("%Y-%m-%d")
                if st.button("Prepare Series", type="primary"):
                    series, err = analyzer.create_time_series_from_sequential(df, value_col, freq=freq, start_date=start_date)
                    if err:
                        st.error(err)
                    else:
                        prepared_series = series
                        st.success("Series prepared")

            if prepared_series is not None:
                st.markdown("---")
                tab_analyze, tab_forecast, tab_stationarity = st.tabs(["📈 Analyze", "🔮 Forecast", "📊 Stationarity"]) 

                with tab_analyze:
                    st.markdown('<h3 class="sub-header">Decomposition</h3>', unsafe_allow_html=True)
                    model = st.selectbox("Model", ["additive", "multiplicative"], index=0)
                    if st.button("Run Decomposition"):
                        with st.spinner("Decomposing time series..."):
                            try:
                                result = analyzer.decompose_series(prepared_series, model=model)
                                st.plotly_chart(result['plot'], use_container_width=True)
                            except Exception as e:
                                st.error(str(e))

                with tab_stationarity:
                    st.markdown('<h3 class="sub-header">Stationarity Tests</h3>', unsafe_allow_html=True)
                    if st.button("Run Tests"):
                        with st.spinner("Running ADF/KPSS..."):
                            try:
                                res = analyzer.check_stationarity(prepared_series)
                                c1, c2 = st.columns(2)
                                with c1:
                                    st.metric("ADF p-value", f"{res['p_value']:.4f}")
                                with c2:
                                    st.write("Critical Values:")
                                    st.write({k: float(v) for k, v in res['critical_values'].items()})
                                st.success("Stationary" if res['is_stationary'] else "Non-stationary")
                            except Exception as e:
                                st.error(str(e))

                    st.markdown("#### Transform if needed")
                    method_label = st.selectbox("Transformation", ["Differencing", "Log Transform", "Log + Differencing", "Percent Change"], index=0)
                    method_map = {"Differencing": "diff", "Log Transform": "log", "Log + Differencing": "log_diff", "Percent Change": "pct_change"}
                    if st.button("Apply Transformation"):
                        try:
                            transformed, _ = make_stationary(prepared_series, method=method_map[method_label], periods=1)
                            st.line_chart(pd.DataFrame({"Original": prepared_series, "Transformed": transformed}))
                            st.session_state.transformed_series = transformed
                            st.session_state.is_transformed = True
                        except Exception as e:
                            st.error(str(e))

                with tab_forecast:
                    st.markdown('<h3 class="sub-header">Forecasting</h3>', unsafe_allow_html=True)
                    model_choice = st.radio("Model", ["ARIMA", "Prophet"], horizontal=True)
                    steps = st.number_input("Forecast periods", min_value=1, max_value=60, value=12)
                    if st.button("Generate Forecast", type="primary"):
                        try:
                            working = st.session_state.get('transformed_series', prepared_series)
                            if model_choice == "ARIMA":
                                fit = analyzer.fit_arima(working, order=(1, 1, 1))
                                if not fit['success']:
                                    st.error(fit['message'])
                                else:
                                    fc = analyzer.forecast_arima(fit['model'], steps=steps)
                                    if not fc['success']:
                                        st.error(fc['message'])
                                    else:
                                        last = working.index[-1]
                                        freq = pd.infer_freq(working.index) or 'M'
                                        idx = pd.date_range(start=last, periods=steps+1, freq=freq)[1:]
                                        yhat = pd.Series(fc['forecast']['forecast'].values, index=idx)
                                        fig = analyzer.plot_forecast(working, yhat, fc['forecast'][['lower', 'upper']])
                                        st.plotly_chart(fig, use_container_width=True)
                                        st.dataframe(fc['forecast'])
                            else:
                                fit = analyzer.fit_prophet(working)
                                if not fit['success']:
                                    st.error(fit['message'])
                                else:
                                    freq = pd.infer_freq(working.index) or 'M'
                                    future_idx = pd.date_range(start=working.index[-1], periods=steps+1, freq=freq)[1:]
                                    model = fit['model']
                                    future_df = pd.DataFrame({'ds': future_idx})
                                    forecast_df = model.predict(future_df)
                                    fig = analyzer.plot_forecast(working, forecast_df.set_index('ds')['yhat'], forecast_df.set_index('ds')[['yhat_lower', 'yhat_upper']].rename(columns={'yhat_lower': 'lower', 'yhat_upper': 'upper'}))
                                    st.plotly_chart(fig, use_container_width=True)
                                    st.dataframe(forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={'ds': 'Date'}))
                        except Exception as e:
                            st.error(str(e))

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
