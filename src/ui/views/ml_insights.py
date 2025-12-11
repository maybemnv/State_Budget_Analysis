import streamlit as st
import pandas as pd
from src.data_loader import DataLoader
from src.analyzers.ml_analyzer import MLAnalyzer

def render_ml_insights(data_loader: DataLoader, ml_analyzer: MLAnalyzer):
    """
    Render the Machine Learning Insights tab.
    
    Args:
        data_loader: The data loader instance.
        ml_analyzer: The ML analyzer instance.
    """
    st.markdown(
        '<h2 class="sub-header">Machine Learning Insights</h2>',
        unsafe_allow_html=True,
    )

    numeric_cols = data_loader.get_numeric_columns()
    categorical_cols = data_loader.get_categorical_columns()

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
    else:
        st.warning("Not enough numeric columns for Machine Learning analysis. Please ensure your dataset has at least 2 numeric columns.")
