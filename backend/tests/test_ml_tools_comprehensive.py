"""Comprehensive tests for ml_tools.py"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, AsyncMock

from backend.tools import ml_tools


@pytest.fixture
def sample_df():
    """Create a sample DataFrame suitable for ML tasks."""
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame({
        "feature1": rng.normal(0, 1, n),
        "feature2": rng.normal(5, 2, n),
        "feature3": rng.normal(-2, 0.5, n),
        "target_reg": rng.normal(10, 3, n),
        "target_cls": rng.choice([0, 1], n),
        "target_multi": rng.choice(["A", "B", "C"], n),
    })


@pytest.fixture
def small_df():
    """Create a small DataFrame for edge cases."""
    return pd.DataFrame({
        "x1": [1, 2, 3, 4, 5],
        "x2": [2, 4, 6, 8, 10],
        "y": [1, 2, 3, 4, 5],
    })


@pytest.fixture
def highly_correlated_df():
    """Create a DataFrame with highly correlated features."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)
    return pd.DataFrame({
        "x1": x,
        "x2": x * 0.95 + rng.normal(0, 0.1, 100),  # Highly correlated with x1
        "x3": x * 0.90 + rng.normal(0, 0.2, 100),  # Correlated with x1
        "target": rng.normal(0, 1, 100),
    })


@pytest.fixture
def df_with_outliers():
    """Create a DataFrame with obvious outliers for anomaly detection."""
    rng = np.random.default_rng(42)
    normal = rng.normal(0, 1, 95)
    outliers = rng.uniform(10, 20, 5)  # Clear outliers
    return pd.DataFrame({
        "a": np.concatenate([normal, outliers]),
        "b": rng.normal(0, 1, 100),
    })


class TestRunPCA:
    """Comprehensive tests for run_pca tool."""

    @pytest.mark.asyncio
    async def test_pca_success_default_components(self, sample_df):
        """Test PCA with default number of components."""
        with patch("backend.tools.ml_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await ml_tools.run_pca.ainvoke({"session_id": "test"})
            
            assert "explained_variance_pct" in result
            assert "loadings" in result
            assert "coordinates" in result
            assert "n_components" in result
            # Default is min(n_samples, n_features) - 1
            assert result["n_components"] > 0

    @pytest.mark.asyncio
    async def test_pca_success_specific_components(self, sample_df):
        """Test PCA with specific number of components."""
        with patch("backend.tools.ml_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await ml_tools.run_pca.ainvoke({
                "session_id": "test",
                "columns": ["feature1", "feature2", "feature3"],
                "n_components": 2,
            })
            
            assert result["n_components"] == 2
            assert len(result["explained_variance_pct"]) == 2

    @pytest.mark.asyncio
    async def test_pca_explained_variance_sums_to_one(self, sample_df):
        """Test that explained variance sums to 1 (or close)."""
        with patch("backend.tools.ml_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await ml_tools.run_pca.ainvoke({
                "session_id": "test",
                "columns": ["feature1", "feature2", "feature3"],
            })
            
            total = sum(result["explained_variance_pct"])
            assert abs(total - 1.0) < 1e-6 or total <= 1.0

    @pytest.mark.asyncio
    async def test_pca_coordinates_shape(self, sample_df):
        """Test that coordinates have correct shape."""
        with patch("backend.tools.ml_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await ml_tools.run_pca.ainvoke({
                "session_id": "test",
                "columns": ["feature1", "feature2"],
                "n_components": 2,
            })
            
            # Coordinates should have n_samples rows
            assert len(result["coordinates"]) == len(sample_df)

    @pytest.mark.asyncio
    async def test_pca_single_column(self, sample_df):
        """Test error with single column (need at least 2)."""
        with patch("backend.tools.ml_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await ml_tools.run_pca.ainvoke({
                "session_id": "test",
                "columns": ["feature1"],
            })
            
            assert "error" in result

    @pytest.mark.asyncio
    async def test_pca_nonexistent_columns(self, sample_df):
        """Test error with nonexistent columns."""
        with patch("backend.tools.ml_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await ml_tools.run_pca.ainvoke({
                "session_id": "test",
                "columns": ["nonexistent1", "nonexistent2"],
            })
            
            assert "error" in result

    @pytest.mark.asyncio
    async def test_pca_session_not_found(self):
        """Test error when session doesn't exist."""
        with patch("backend.tools.ml_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (None, {"error": "Session not found"})
            result = await ml_tools.run_pca.ainvoke({"session_id": "invalid"})
            
            assert "error" in result

    @pytest.mark.asyncio
    async def test_pca_too_many_components(self, sample_df):
        """Test PCA requesting more components than features."""
        with patch("backend.tools.ml_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await ml_tools.run_pca.ainvoke({
                "session_id": "test",
                "columns": ["feature1", "feature2"],
                "n_components": 10,  # More than available features
            })
            
            # Should either error or limit to available components
            assert "error" in result or result["n_components"] <= 2


class TestRunKMeans:
    """Comprehensive tests for run_kmeans tool."""

    @pytest.mark.asyncio
    async def test_kmeans_success(self, sample_df):
        """Test successful K-means clustering."""
        with patch("backend.tools.ml_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await ml_tools.run_kmeans.ainvoke({
                "session_id": "test",
                "columns": ["feature1", "feature2", "feature3"],
                "n_clusters": 3,
            })
            
            assert "labels" in result
            assert "cluster_centers" in result
            assert "silhouette_score" in result
            assert len(result["labels"]) == len(sample_df)
            assert -1.0 <= result["silhouette_score"] <= 1.0

    @pytest.mark.asyncio
    async def test_kmeans_different_cluster_counts(self, sample_df):
        """Test K-means with different numbers of clusters."""
        with patch("backend.tools.ml_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            
            for n_clusters in [2, 3, 5]:
                result = await ml_tools.run_kmeans.ainvoke({
                    "session_id": "test",
                    "columns": ["feature1", "feature2"],
                    "n_clusters": n_clusters,
                })
                
                assert "labels" in result
                assert len(set(result["labels"])) <= n_clusters

    @pytest.mark.asyncio
    async def test_kmeans_default_clusters(self, sample_df):
        """Test K-means with default cluster count (3)."""
        with patch("backend.tools.ml_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await ml_tools.run_kmeans.ainvoke({
                "session_id": "test",
                "columns": ["feature1", "feature2"],
            })
            
            assert "labels" in result

    @pytest.mark.asyncio
    async def test_kmeans_single_column(self, sample_df):
        """Test K-means with single column - requires at least 2 columns."""
        with patch("backend.tools.ml_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await ml_tools.run_kmeans.ainvoke({
                "session_id": "test",
                "columns": ["feature1"],
                "n_clusters": 2,
            })
            
            # Should error with single column (needs at least 2)
            assert "error" in result

    @pytest.mark.asyncio
    async def test_kmeans_nonexistent_columns(self, sample_df):
        """Test error with nonexistent columns."""
        with patch("backend.tools.ml_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await ml_tools.run_kmeans.ainvoke({
                "session_id": "test",
                "columns": ["nonexistent"],
            })
            
            assert "error" in result

    @pytest.mark.asyncio
    async def test_kmeans_session_not_found(self):
        """Test error when session doesn't exist."""
        with patch("backend.tools.ml_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (None, {"error": "Session not found"})
            result = await ml_tools.run_kmeans.ainvoke({"session_id": "invalid"})
            
            assert "error" in result

    @pytest.mark.asyncio
    async def test_kmeans_small_dataset(self, small_df):
        """Test K-means with very small dataset."""
        with patch("backend.tools.ml_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (small_df, None)
            result = await ml_tools.run_kmeans.ainvoke({
                "session_id": "test",
                "columns": ["x1", "x2"],
                "n_clusters": 2,
            })
            
            # Should work even with small dataset
            assert "labels" in result


class TestDetectAnomalies:
    """Comprehensive tests for detect_anomalies tool."""

    @pytest.mark.asyncio
    async def test_detect_anomalies_success(self, df_with_outliers):
        """Test successful anomaly detection."""
        with patch("backend.tools.ml_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (df_with_outliers, None)
            result = await ml_tools.detect_anomalies.ainvoke({
                "session_id": "test",
                "columns": ["a", "b"],
                "contamination": 0.05,
            })
            
            assert "anomaly_count" in result
            assert "anomaly_pct" in result
            assert "anomaly_indices" in result
            assert "scores" in result
            assert result["anomaly_count"] > 0
            assert 0 < result["anomaly_pct"] <= 100

    @pytest.mark.asyncio
    async def test_detect_anomalies_different_contamination(self, df_with_outliers):
        """Test anomaly detection with different contamination levels."""
        with patch("backend.tools.ml_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (df_with_outliers, None)
            
            result_05 = await ml_tools.detect_anomalies.ainvoke({
                "session_id": "test",
                "columns": ["a", "b"],
                "contamination": 0.05,
            })
            result_10 = await ml_tools.detect_anomalies.ainvoke({
                "session_id": "test",
                "columns": ["a", "b"],
                "contamination": 0.10,
            })
            
            # Higher contamination should detect more anomalies
            assert result_10["anomaly_count"] >= result_05["anomaly_count"]

    @pytest.mark.asyncio
    async def test_detect_anomalies_all_columns(self, sample_df):
        """Test anomaly detection on all columns."""
        with patch("backend.tools.ml_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await ml_tools.detect_anomalies.ainvoke({
                "session_id": "test",
                "contamination": 0.1,
            })
            
            assert "anomaly_count" in result

    @pytest.mark.asyncio
    async def test_detect_anomalies_single_column(self, df_with_outliers):
        """Test anomaly detection on single column."""
        with patch("backend.tools.ml_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (df_with_outliers, None)
            result = await ml_tools.detect_anomalies.ainvoke({
                "session_id": "test",
                "columns": ["a"],
                "contamination": 0.05,
            })
            
            assert "anomaly_count" in result

    @pytest.mark.asyncio
    async def test_detect_anomalies_no_outliers(self, sample_df):
        """Test with data that has no clear outliers."""
        with patch("backend.tools.ml_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await ml_tools.detect_anomalies.ainvoke({
                "session_id": "test",
                "columns": ["feature1"],
                "contamination": 0.01,
            })
            
            assert "anomaly_count" in result
            # Might detect 0 or very few

    @pytest.mark.asyncio
    async def test_detect_anomalies_session_not_found(self):
        """Test error when session doesn't exist."""
        with patch("backend.tools.ml_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (None, {"error": "Session not found"})
            result = await ml_tools.detect_anomalies.ainvoke({"session_id": "invalid"})
            
            assert "error" in result

    @pytest.mark.asyncio
    async def test_detect_anomalies_high_contamination(self, sample_df):
        """Test with high contamination value."""
        with patch("backend.tools.ml_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await ml_tools.detect_anomalies.ainvoke({
                "session_id": "test",
                "columns": ["feature1"],
                "contamination": 0.5,
            })
            
            # Should still work with high contamination
            assert "anomaly_count" in result


class TestRunRegression:
    """Comprehensive tests for run_regression tool."""

    @pytest.mark.asyncio
    async def test_regression_success(self, sample_df):
        """Test successful regression training."""
        with patch("backend.tools.ml_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await ml_tools.run_regression.ainvoke({
                "session_id": "test",
                "target_column": "target_reg",
                "feature_columns": ["feature1", "feature2", "feature3"],
                "test_size": 0.2,
            })
            
            assert "r2" in result
            assert "rmse" in result
            assert "feature_importance" in result
            assert "predictions_vs_actual" in result
            # R² can be negative for poor models, just check it's a number
            assert isinstance(result["r2"], (int, float))
            assert result["rmse"] >= 0

    @pytest.mark.asyncio
    async def test_regression_default_features(self, sample_df):
        """Test regression with default features (all other columns)."""
        with patch("backend.tools.ml_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await ml_tools.run_regression.ainvoke({
                "session_id": "test",
                "target_column": "target_reg",
            })
            
            assert "r2" in result
            assert "feature_importance" in result

    @pytest.mark.asyncio
    async def test_regression_different_test_sizes(self, sample_df):
        """Test regression with different test sizes."""
        with patch("backend.tools.ml_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            
            for test_size in [0.1, 0.2, 0.3]:
                result = await ml_tools.run_regression.ainvoke({
                    "session_id": "test",
                    "target_column": "target_reg",
                    "feature_columns": ["feature1", "feature2"],
                    "test_size": test_size,
                })
                
                assert "r2" in result

    @pytest.mark.asyncio
    async def test_regression_missing_target_column(self, sample_df):
        """Test error when target_column is missing."""
        with patch("backend.tools.ml_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await ml_tools.run_regression.ainvoke({
                "session_id": "test",
                "target_column": None,
            })
            
            assert "error" in result
            assert "target_column is required" in result["error"]

    @pytest.mark.asyncio
    async def test_regression_nonexistent_target(self, sample_df):
        """Test error when target column doesn't exist."""
        with patch("backend.tools.ml_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await ml_tools.run_regression.ainvoke({
                "session_id": "test",
                "target_column": "nonexistent",
            })
            
            assert "error" in result

    @pytest.mark.asyncio
    async def test_regression_nonexistent_features(self, sample_df):
        """Test error when feature columns don't exist."""
        with patch("backend.tools.ml_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await ml_tools.run_regression.ainvoke({
                "session_id": "test",
                "target_column": "target_reg",
                "feature_columns": ["nonexistent1", "nonexistent2"],
            })
            
            assert "error" in result

    @pytest.mark.asyncio
    async def test_regression_session_not_found(self):
        """Test error when session doesn't exist."""
        with patch("backend.tools.ml_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (None, {"error": "Session not found"})
            result = await ml_tools.run_regression.ainvoke({
                "session_id": "invalid",
                "target_column": "y",
            })
            
            assert "error" in result

    @pytest.mark.asyncio
    async def test_regression_small_dataset(self, small_df):
        """Test regression with very small dataset."""
        with patch("backend.tools.ml_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (small_df, None)
            result = await ml_tools.run_regression.ainvoke({
                "session_id": "test",
                "target_column": "y",
                "feature_columns": ["x1", "x2"],
                "test_size": 0.2,
            })
            
            # Should still work even with small dataset
            assert "r2" in result


class TestRunClassification:
    """Comprehensive tests for run_classification tool."""

    @pytest.mark.asyncio
    async def test_classification_binary_success(self, sample_df):
        """Test successful binary classification."""
        with patch("backend.tools.ml_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await ml_tools.run_classification.ainvoke({
                "session_id": "test",
                "target_column": "target_cls",
                "feature_columns": ["feature1", "feature2", "feature3"],
                "test_size": 0.2,
            })
            
            assert "accuracy" in result
            assert "feature_importance" in result
            assert "classification_report" in result
            assert 0.0 <= result["accuracy"] <= 1.0

    @pytest.mark.asyncio
    async def test_classification_multiclass(self, sample_df):
        """Test multiclass classification."""
        with patch("backend.tools.ml_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await ml_tools.run_classification.ainvoke({
                "session_id": "test",
                "target_column": "target_multi",
                "feature_columns": ["feature1", "feature2"],
            })
            
            assert "accuracy" in result
            assert "classification_report" in result

    @pytest.mark.asyncio
    async def test_classification_default_features(self, sample_df):
        """Test classification with default features."""
        with patch("backend.tools.ml_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await ml_tools.run_classification.ainvoke({
                "session_id": "test",
                "target_column": "target_cls",
            })
            
            assert "accuracy" in result
            assert "feature_importance" in result

    @pytest.mark.asyncio
    async def test_classification_different_test_sizes(self, sample_df):
        """Test classification with different test sizes."""
        with patch("backend.tools.ml_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            
            for test_size in [0.1, 0.2, 0.3]:
                result = await ml_tools.run_classification.ainvoke({
                    "session_id": "test",
                    "target_column": "target_cls",
                    "feature_columns": ["feature1", "feature2"],
                    "test_size": test_size,
                })
                
                assert "accuracy" in result

    @pytest.mark.asyncio
    async def test_classification_missing_target_column(self, sample_df):
        """Test error when target_column is missing."""
        with patch("backend.tools.ml_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await ml_tools.run_classification.ainvoke({
                "session_id": "test",
                "target_column": None,
            })
            
            assert "error" in result
            assert "target_column is required" in result["error"]

    @pytest.mark.asyncio
    async def test_classification_nonexistent_target(self, sample_df):
        """Test error when target column doesn't exist."""
        with patch("backend.tools.ml_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await ml_tools.run_classification.ainvoke({
                "session_id": "test",
                "target_column": "nonexistent",
            })
            
            assert "error" in result

    @pytest.mark.asyncio
    async def test_classification_session_not_found(self):
        """Test error when session doesn't exist."""
        with patch("backend.tools.ml_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (None, {"error": "Session not found"})
            result = await ml_tools.run_classification.ainvoke({
                "session_id": "invalid",
                "target_column": "y",
            })
            
            assert "error" in result

    @pytest.mark.asyncio
    async def test_classification_small_dataset(self, small_df):
        """Test classification with very small dataset."""
        # Add a classification target
        small_df["target"] = [0, 1, 0, 1, 0]
        
        with patch("backend.tools.ml_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (small_df, None)
            result = await ml_tools.run_classification.ainvoke({
                "session_id": "test",
                "target_column": "target",
                "feature_columns": ["x1", "x2"],
                "test_size": 0.2,
            })
            
            # Small dataset may error due to stratify issues or return results
            assert "accuracy" in result or "error" in result


class TestMLToolsEdgeCases:
    """Edge case tests for ML tools."""

    @pytest.mark.asyncio
    async def test_pca_highly_correlated_data(self, highly_correlated_df):
        """Test PCA with highly correlated features."""
        with patch("backend.tools.ml_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (highly_correlated_df, None)
            result = await ml_tools.run_pca.ainvoke({
                "session_id": "test",
                "columns": ["x1", "x2", "x3"],
                "n_components": 2,
            })
            
            # First component should explain most variance
            assert result["explained_variance_pct"][0] > result["explained_variance_pct"][1]

    @pytest.mark.asyncio
    async def test_kmeans_perfect_clusters(self):
        """Test K-means with data that has clear clusters."""
        # Create data with 3 clear clusters
        df = pd.DataFrame({
            "x": [1, 1.1, 0.9, 5, 5.1, 4.9, 10, 10.1, 9.9],
            "y": [1, 1.2, 0.8, 5, 4.8, 5.2, 10, 9.8, 10.2],
        })
        
        with patch("backend.tools.ml_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (df, None)
            result = await ml_tools.run_kmeans.ainvoke({
                "session_id": "test",
                "columns": ["x", "y"],
                "n_clusters": 3,
            })
            
            # Should have good silhouette score for clear clusters
            assert result["silhouette_score"] > 0.5

    @pytest.mark.asyncio
    async def test_regression_constant_target(self, sample_df):
        """Test regression when target is constant."""
        df = sample_df.copy()
        df["constant_target"] = 5.0  # All same value
        
        with patch("backend.tools.ml_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (df, None)
            result = await ml_tools.run_regression.ainvoke({
                "session_id": "test",
                "target_column": "constant_target",
                "feature_columns": ["feature1", "feature2"],
            })
            
            # R² might be 0 or NaN for constant target
            assert "r2" in result

    @pytest.mark.asyncio
    async def test_classification_single_class(self, sample_df):
        """Test classification with only one class."""
        df = sample_df.copy()
        df["single_class"] = "A"  # All same class
        
        with patch("backend.tools.ml_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (df, None)
            result = await ml_tools.run_classification.ainvoke({
                "session_id": "test",
                "target_column": "single_class",
                "feature_columns": ["feature1", "feature2"],
            })
            
            # Should error or handle gracefully
            assert "error" in result or "accuracy" in result
