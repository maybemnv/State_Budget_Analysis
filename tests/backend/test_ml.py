import pandas as pd
import numpy as np
import pytest
from backend.analyzers import ml


@pytest.fixture
def df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "x1": rng.normal(0, 1, 80),
        "x2": rng.normal(5, 2, 80),
        "x3": rng.normal(-2, 0.5, 80),
        "target_reg": rng.normal(10, 3, 80),
        "target_cls": rng.choice([0, 1], 80),
    })


def test_pca_returns_expected_keys(df):
    result = ml.perform_pca(df, columns=["x1", "x2", "x3"])
    assert "explained_variance_pct" in result
    assert "loadings" in result
    assert "coordinates" in result
    assert result["n_components"] <= 3


def test_pca_explained_variance_sums_to_one(df):
    result = ml.perform_pca(df, columns=["x1", "x2", "x3"])
    total = sum(result["explained_variance_pct"])
    assert abs(total - 1.0) < 1e-6


def test_pca_too_few_columns(df):
    with pytest.raises(ValueError):
        ml.perform_pca(df, columns=["x1"])


def test_clustering_returns_labels_and_score(df):
    result = ml.perform_clustering(df, columns=["x1", "x2", "x3"], n_clusters=3)
    assert "labels" in result
    assert len(result["labels"]) == len(df)
    assert "silhouette_score" in result
    assert -1.0 <= result["silhouette_score"] <= 1.0


def test_detect_anomalies(df):
    result = ml.detect_anomalies(df, columns=["x1", "x2"], contamination=0.1)
    assert "anomaly_count" in result
    assert result["anomaly_count"] > 0
    assert 0 < result["anomaly_pct"] <= 100


def test_regression_r2_in_valid_range(df):
    result = ml.train_regression_model(df, "target_reg", feature_columns=["x1", "x2", "x3"])
    assert "r2" in result
    assert "rmse" in result
    assert "feature_importance" in result
    assert result["rmse"] >= 0


def test_regression_invalid_target(df):
    with pytest.raises(ValueError):
        ml.train_regression_model(df, "nonexistent")


def test_classification_accuracy_in_range(df):
    result = ml.train_classification_model(df, "target_cls", feature_columns=["x1", "x2", "x3"])
    assert "accuracy" in result
    assert 0.0 <= result["accuracy"] <= 1.0
    assert "feature_importance" in result
    assert "classification_report" in result
