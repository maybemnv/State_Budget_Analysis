"""Comprehensive tests for statistical_tools.py"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from backend.tools import statistical_tools


@pytest.fixture
def sample_df():
    """Create a sample DataFrame with various data types."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "numeric_a": rng.normal(100, 15, 100),
        "numeric_b": rng.normal(50, 10, 100),
        "numeric_c": rng.uniform(0, 1000, 100),
        "category": rng.choice(["X", "Y", "Z"], 100),
        "small_ints": rng.integers(1, 10, 100),
    })


@pytest.fixture
def df_with_missing():
    """Create a DataFrame with missing values."""
    return pd.DataFrame({
        "a": [1.0, 2.0, None, 4.0, 5.0],
        "b": [None, 2.0, 3.0, None, 5.0],
        "c": ["x", None, "z", "w", None],
        "cat": ["A", "A", "B", "B", "A"],
    })


@pytest.fixture
def empty_df():
    """Create an empty DataFrame."""
    return pd.DataFrame()


@pytest.fixture
def single_row_df():
    """Create a DataFrame with only one row."""
    return pd.DataFrame({
        "a": [1.0],
        "b": [2.0],
        "cat": ["A"],
    })


@pytest.fixture
def constant_df():
    """Create a DataFrame with constant values (std=0)."""
    return pd.DataFrame({
        "constant": [5.0] * 100,
        "normal": np.random.default_rng(42).normal(0, 1, 100),
    })


class TestDescriptiveStats:
    """Comprehensive tests for descriptive_stats tool."""

    @pytest.mark.asyncio
    async def test_descriptive_stats_success_all_columns(self, sample_df):
        """Test getting stats for all numeric columns."""
        with patch("backend.tools.statistical_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await statistical_tools.descriptive_stats.ainvoke({"session_id": "test"})
            
            assert "mean" in result
            assert "std" in result
            assert "min" in result
            assert "max" in result
            assert "skew" in result
            assert "kurtosis" in result
            assert "count" in result
            # Check numeric columns are present
            assert "numeric_a" in result["mean"]
            assert "numeric_b" in result["mean"]

    @pytest.mark.asyncio
    async def test_descriptive_stats_specific_columns(self, sample_df):
        """Test getting stats for specific columns only."""
        with patch("backend.tools.statistical_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await statistical_tools.descriptive_stats.ainvoke({
                "session_id": "test",
                "columns": ["numeric_a", "numeric_b"],
            })
            
            assert "numeric_a" in result["mean"]
            assert "numeric_b" in result["mean"]
            assert "numeric_c" not in result["mean"]

    @pytest.mark.asyncio
    async def test_descriptive_stats_empty_columns_list(self, sample_df):
        """Test with empty columns list - should return all numeric columns."""
        with patch("backend.tools.statistical_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await statistical_tools.descriptive_stats.ainvoke({
                "session_id": "test",
                "columns": [],
            })
            
            # Should return empty result since no columns specified
            assert result == {}

    @pytest.mark.asyncio
    async def test_descriptive_stats_with_missing_values(self, df_with_missing):
        """Test with DataFrame containing missing values."""
        with patch("backend.tools.statistical_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (df_with_missing, None)
            result = await statistical_tools.descriptive_stats.ainvoke({"session_id": "test"})
            
            # Should handle missing values gracefully
            assert "mean" in result
            assert "a" in result["mean"]

    @pytest.mark.asyncio
    async def test_descriptive_stats_constant_column(self, constant_df):
        """Test with constant column (std=0)."""
        with patch("backend.tools.statistical_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (constant_df, None)
            result = await statistical_tools.descriptive_stats.ainvoke({"session_id": "test"})
            
            assert result["std"]["constant"] == 0.0
            assert result["mean"]["constant"] == 5.0

    @pytest.mark.asyncio
    async def test_descriptive_stats_session_not_found(self):
        """Test error when session doesn't exist."""
        with patch("backend.tools.statistical_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (None, {"error": "Session not found"})
            result = await statistical_tools.descriptive_stats.ainvoke({"session_id": "invalid"})
            
            assert "error" in result

    @pytest.mark.asyncio
    async def test_descriptive_stats_nonexistent_column(self, sample_df):
        """Test with nonexistent column."""
        with patch("backend.tools.statistical_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await statistical_tools.descriptive_stats.ainvoke({
                "session_id": "test",
                "columns": ["nonexistent"],
            })
            
            # Should return error since column doesn't exist
            assert "error" in result


class TestGroupByStats:
    """Comprehensive tests for group_by_stats tool."""

    @pytest.mark.asyncio
    async def test_group_by_stats_mean_success(self, sample_df):
        """Test successful group by with mean aggregation."""
        with patch("backend.tools.statistical_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await statistical_tools.group_by_stats.ainvoke({
                "session_id": "test",
                "group_column": "category",
                "agg_column": "numeric_a",
                "agg_func": "mean",
            })
            
            assert "result" in result
            assert isinstance(result["result"], list)
            assert len(result["result"]) == 3  # X, Y, Z

    @pytest.mark.asyncio
    async def test_group_by_stats_sum(self, sample_df):
        """Test group by with sum aggregation."""
        with patch("backend.tools.statistical_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await statistical_tools.group_by_stats.ainvoke({
                "session_id": "test",
                "group_column": "category",
                "agg_column": "numeric_a",
                "agg_func": "sum",
            })
            
            assert "result" in result

    @pytest.mark.asyncio
    async def test_group_by_stats_count(self, sample_df):
        """Test group by with count aggregation."""
        with patch("backend.tools.statistical_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await statistical_tools.group_by_stats.ainvoke({
                "session_id": "test",
                "group_column": "category",
                "agg_column": "numeric_a",
                "agg_func": "count",
            })
            
            assert "result" in result

    @pytest.mark.asyncio
    async def test_group_by_stats_median(self, sample_df):
        """Test group by with median aggregation."""
        with patch("backend.tools.statistical_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await statistical_tools.group_by_stats.ainvoke({
                "session_id": "test",
                "group_column": "category",
                "agg_column": "numeric_a",
                "agg_func": "median",
            })
            
            assert "result" in result

    @pytest.mark.asyncio
    async def test_group_by_stats_min_max(self, sample_df):
        """Test group by with min and max aggregation."""
        with patch("backend.tools.statistical_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            
            result_min = await statistical_tools.group_by_stats.ainvoke({
                "session_id": "test",
                "group_column": "category",
                "agg_column": "numeric_a",
                "agg_func": "min",
            })
            result_max = await statistical_tools.group_by_stats.ainvoke({
                "session_id": "test",
                "group_column": "category",
                "agg_column": "numeric_a",
                "agg_func": "max",
            })
            
            assert "result" in result_min
            assert "result" in result_max

    @pytest.mark.asyncio
    async def test_group_by_stats_std(self, sample_df):
        """Test group by with std aggregation."""
        with patch("backend.tools.statistical_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await statistical_tools.group_by_stats.ainvoke({
                "session_id": "test",
                "group_column": "category",
                "agg_column": "numeric_a",
                "agg_func": "std",
            })
            
            assert "result" in result

    @pytest.mark.asyncio
    async def test_group_by_stats_missing_group_column(self, sample_df):
        """Test error when group_column is missing."""
        with patch("backend.tools.statistical_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await statistical_tools.group_by_stats.ainvoke({
                "session_id": "test",
                "group_column": None,
                "agg_column": "numeric_a",
            })
            
            assert "error" in result
            assert "group_column and agg_column are both required" in result["error"]

    @pytest.mark.asyncio
    async def test_group_by_stats_missing_agg_column(self, sample_df):
        """Test error when agg_column is missing."""
        with patch("backend.tools.statistical_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await statistical_tools.group_by_stats.ainvoke({
                "session_id": "test",
                "group_column": "category",
                "agg_column": None,
            })
            
            assert "error" in result

    @pytest.mark.asyncio
    async def test_group_by_stats_invalid_group_column(self, sample_df):
        """Test error when group_column doesn't exist."""
        with patch("backend.tools.statistical_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await statistical_tools.group_by_stats.ainvoke({
                "session_id": "test",
                "group_column": "nonexistent",
                "agg_column": "numeric_a",
            })
            
            assert "error" in result

    @pytest.mark.asyncio
    async def test_group_by_stats_invalid_agg_column(self, sample_df):
        """Test error when agg_column doesn't exist."""
        with patch("backend.tools.statistical_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await statistical_tools.group_by_stats.ainvoke({
                "session_id": "test",
                "group_column": "category",
                "agg_column": "nonexistent",
            })
            
            assert "error" in result

    @pytest.mark.asyncio
    async def test_group_by_stats_session_not_found(self):
        """Test error when session doesn't exist."""
        with patch("backend.tools.statistical_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (None, {"error": "Session not found"})
            result = await statistical_tools.group_by_stats.ainvoke({
                "session_id": "invalid",
                "group_column": "a",
                "agg_column": "b",
            })
            
            assert "error" in result


class TestCorrelationMatrix:
    """Comprehensive tests for correlation_matrix tool."""

    @pytest.mark.asyncio
    async def test_correlation_matrix_all_numeric(self, sample_df):
        """Test correlation matrix for all numeric columns."""
        with patch("backend.tools.statistical_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await statistical_tools.correlation_matrix.ainvoke({"session_id": "test"})
            
            # Should have all numeric columns
            assert "numeric_a" in result
            assert "numeric_b" in result
            assert "numeric_c" in result
            # Diagonal should be 1.0
            assert abs(result["numeric_a"]["numeric_a"] - 1.0) < 1e-9

    @pytest.mark.asyncio
    async def test_correlation_matrix_specific_columns(self, sample_df):
        """Test correlation matrix for specific columns."""
        with patch("backend.tools.statistical_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await statistical_tools.correlation_matrix.ainvoke({
                "session_id": "test",
                "columns": ["numeric_a", "numeric_b"],
            })
            
            assert "numeric_a" in result
            assert "numeric_b" in result
            assert "numeric_c" not in result

    @pytest.mark.asyncio
    async def test_correlation_matrix_symmetric(self, sample_df):
        """Test that correlation matrix is symmetric."""
        with patch("backend.tools.statistical_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await statistical_tools.correlation_matrix.ainvoke({"session_id": "test"})
            
            # Correlation should be symmetric
            assert abs(result["numeric_a"]["numeric_b"] - result["numeric_b"]["numeric_a"]) < 1e-9

    @pytest.mark.asyncio
    async def test_correlation_matrix_single_column(self, sample_df):
        """Test error with single column (need at least 2)."""
        with patch("backend.tools.statistical_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await statistical_tools.correlation_matrix.ainvoke({
                "session_id": "test",
                "columns": ["numeric_a"],
            })
            
            assert "error" in result

    @pytest.mark.asyncio
    async def test_correlation_matrix_nonexistent_column(self, sample_df):
        """Test error with nonexistent column."""
        with patch("backend.tools.statistical_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await statistical_tools.correlation_matrix.ainvoke({
                "session_id": "test",
                "columns": ["nonexistent"],
            })
            
            assert "error" in result

    @pytest.mark.asyncio
    async def test_correlation_matrix_no_numeric_columns(self):
        """Test with DataFrame having no numeric columns."""
        df = pd.DataFrame({
            "a": ["x", "y", "z"],
            "b": ["a", "b", "c"],
        })
        
        with patch("backend.tools.statistical_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (df, None)
            result = await statistical_tools.correlation_matrix.ainvoke({"session_id": "test"})
            
            assert "error" in result

    @pytest.mark.asyncio
    async def test_correlation_matrix_session_not_found(self):
        """Test error when session doesn't exist."""
        with patch("backend.tools.statistical_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (None, {"error": "Session not found"})
            result = await statistical_tools.correlation_matrix.ainvoke({"session_id": "invalid"})
            
            assert "error" in result


class TestValueCounts:
    """Comprehensive tests for value_counts tool."""

    @pytest.mark.asyncio
    async def test_value_counts_success(self, sample_df):
        """Test successful value counts."""
        with patch("backend.tools.statistical_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await statistical_tools.value_counts.ainvoke({
                "session_id": "test",
                "column": "category",
            })
            
            assert isinstance(result, dict)
            assert sum(result.values()) == 100
            assert "X" in result
            assert "Y" in result
            assert "Z" in result

    @pytest.mark.asyncio
    async def test_value_counts_normalize(self, sample_df):
        """Test normalized value counts (percentages)."""
        with patch("backend.tools.statistical_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await statistical_tools.value_counts.ainvoke({
                "session_id": "test",
                "column": "category",
                "normalize": True,
            })
            
            # Values should sum to approximately 1.0
            assert abs(sum(result.values()) - 1.0) < 0.01

    @pytest.mark.asyncio
    async def test_value_counts_with_limit(self, sample_df):
        """Test value counts with limit."""
        # Create DataFrame with many unique values
        df = pd.DataFrame({
            "many_vals": [f"val_{i % 50}" for i in range(1000)],
        })
        
        with patch("backend.tools.statistical_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (df, None)
            result = await statistical_tools.value_counts.ainvoke({
                "session_id": "test",
                "column": "many_vals",
                "limit": 10,
            })
            
            assert len(result) <= 10

    @pytest.mark.asyncio
    async def test_value_counts_missing_column_param(self, sample_df):
        """Test error when column parameter is missing."""
        with patch("backend.tools.statistical_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await statistical_tools.value_counts.ainvoke({
                "session_id": "test",
                "column": None,
            })
            
            assert "error" in result
            assert "column is required" in result["error"]

    @pytest.mark.asyncio
    async def test_value_counts_invalid_column(self, sample_df):
        """Test error when column doesn't exist."""
        with patch("backend.tools.statistical_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await statistical_tools.value_counts.ainvoke({
                "session_id": "test",
                "column": "nonexistent",
            })
            
            assert "error" in result

    @pytest.mark.asyncio
    async def test_value_counts_session_not_found(self):
        """Test error when session doesn't exist."""
        with patch("backend.tools.statistical_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (None, {"error": "Session not found"})
            result = await statistical_tools.value_counts.ainvoke({
                "session_id": "invalid",
                "column": "a",
            })
            
            assert "error" in result


class TestOutliersSummary:
    """Comprehensive tests for outliers_summary tool."""

    @pytest.mark.asyncio
    async def test_outliers_summary_iqr_success(self, sample_df):
        """Test IQR outlier detection."""
        with patch("backend.tools.statistical_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await statistical_tools.outliers_summary.ainvoke({
                "session_id": "test",
                "columns": ["numeric_a", "numeric_b"],
                "method": "iqr",
                "threshold": 1.5,
            })
            
            assert "outliers" in result
            assert isinstance(result["outliers"], list)

    @pytest.mark.asyncio
    async def test_outliers_summary_zscore(self, sample_df):
        """Test Z-score outlier detection."""
        with patch("backend.tools.statistical_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await statistical_tools.outliers_summary.ainvoke({
                "session_id": "test",
                "columns": ["numeric_a"],
                "method": "zscore",
                "threshold": 3.0,
            })
            
            assert "outliers" in result

    @pytest.mark.asyncio
    async def test_outliers_summary_all_columns(self, sample_df):
        """Test outlier detection on all columns (no columns specified)."""
        with patch("backend.tools.statistical_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await statistical_tools.outliers_summary.ainvoke({
                "session_id": "test",
            })
            
            assert "outliers" in result

    @pytest.mark.asyncio
    async def test_outliers_summary_with_obvious_outliers(self):
        """Test with obvious outliers injected."""
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5, 1000],  # 1000 is clear outlier
            "b": [10, 11, 12, 13, 14, 15],
        })
        
        with patch("backend.tools.statistical_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (df, None)
            result = await statistical_tools.outliers_summary.ainvoke({
                "session_id": "test",
                "columns": ["a"],
                "method": "iqr",
            })
            
            assert "outliers" in result
            # Should find at least one outlier
            assert any(r.get("column") == "a" for r in result["outliers"])

    @pytest.mark.asyncio
    async def test_outliers_summary_session_not_found(self):
        """Test error when session doesn't exist."""
        with patch("backend.tools.statistical_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (None, {"error": "Session not found"})
            result = await statistical_tools.outliers_summary.ainvoke({"session_id": "invalid"})
            
            assert "error" in result


class TestStatisticalToolsEdgeCases:
    """Edge case tests for statistical tools."""

    @pytest.mark.asyncio
    async def test_group_by_stats_non_numeric_agg_column(self, sample_df):
        """Test error when trying to aggregate non-numeric column."""
        with patch("backend.tools.statistical_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await statistical_tools.group_by_stats.ainvoke({
                "session_id": "test",
                "group_column": "category",
                "agg_column": "category",  # Can't sum/mean a categorical column
                "agg_func": "mean",
            })
            
            assert "error" in result

    @pytest.mark.asyncio
    async def test_value_counts_numeric_column(self, sample_df):
        """Test value counts on numeric column."""
        with patch("backend.tools.statistical_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await statistical_tools.value_counts.ainvoke({
                "session_id": "test",
                "column": "small_ints",
            })
            
            # Should work on numeric column too
            assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_correlation_matrix_with_nan(self, df_with_missing):
        """Test correlation matrix with missing values."""
        with patch("backend.tools.statistical_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (df_with_missing, None)
            result = await statistical_tools.correlation_matrix.ainvoke({
                "session_id": "test",
                "columns": ["a", "b"],
            })
            
            # Should handle NaN values
            assert "error" in result or ("a" in result and "b" in result)
