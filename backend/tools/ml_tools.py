from typing import Optional
from langchain_core.tools import tool
from ..schemas import PCAInput, ClusteringInput, RegressionInput, ClassificationInput, AnomalyDetectionInput
from ..analyzers import ml
from .guards import require_df


@tool("run_pca", args_schema=PCAInput)
def run_pca(session_id: Optional[str] = None, columns: list[str] | None = None, n_components: int | None = None) -> dict:
    """Run PCA dimensionality reduction. Returns component loadings, explained variance, and 2D/3D coordinates."""
    df, err = require_df(session_id)
    if err:
        return err
    try:
        return ml.perform_pca(df, columns=columns, n_components=n_components)
    except ValueError as e:
        return {"error": str(e)}


@tool("run_kmeans", args_schema=ClusteringInput)
def run_kmeans(session_id: Optional[str] = None, columns: list[str] | None = None, n_clusters: int = 3) -> dict:
    """Cluster data into K groups using K-means. Returns labels, centroids, and silhouette score."""
    df, err = require_df(session_id)
    if err:
        return err
    try:
        return ml.perform_clustering(df, columns=columns, n_clusters=n_clusters)
    except ValueError as e:
        return {"error": str(e)}


@tool("detect_anomalies", args_schema=AnomalyDetectionInput)
def detect_anomalies(session_id: Optional[str] = None, columns: list[str] | None = None, contamination: float = 0.05) -> dict:
    """Detect anomalous rows using Isolation Forest. Returns anomaly indices and scores."""
    df, err = require_df(session_id)
    if err:
        return err
    try:
        return ml.detect_anomalies(df, columns=columns, contamination=contamination)
    except Exception as e:
        return {"error": str(e)}


@tool("run_regression", args_schema=RegressionInput)
def run_regression(
    session_id: Optional[str] = None,
    target_column: Optional[str] = None,
    feature_columns: list[str] | None = None,
    test_size: float = 0.2,
) -> dict:
    """Train a Random Forest regression model. Returns R², RMSE, and feature importance."""
    if not target_column:
        return {"error": "target_column is required"}

    df, err = require_df(session_id)
    if err:
        return err
    try:
        return ml.train_regression_model(df, target_column, feature_columns=feature_columns, test_size=test_size)
    except ValueError as e:
        return {"error": str(e)}


@tool("run_classification", args_schema=ClassificationInput)
def run_classification(
    session_id: Optional[str] = None,
    target_column: Optional[str] = None,
    feature_columns: list[str] | None = None,
    test_size: float = 0.2,
) -> dict:
    """Train a Random Forest classification model. Returns accuracy and per-class metrics."""
    if not target_column:
        return {"error": "target_column is required"}

    df, err = require_df(session_id)
    if err:
        return err
    try:
        return ml.train_classification_model(df, target_column, feature_columns=feature_columns, test_size=test_size)
    except ValueError as e:
        return {"error": str(e)}
