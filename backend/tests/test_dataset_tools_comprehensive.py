"""Comprehensive tests for dataset_tools.py"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from backend.tools import dataset_tools


@pytest.fixture
def sample_df():
    """Create a sample DataFrame with various data types."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=50, freq="ME").astype(str),
        "revenue": rng.normal(1000, 100, 50).round(2),
        "category": rng.choice(["A", "B", "C"], 50),
        "amount": rng.uniform(10, 1000, 50),
        "is_active": rng.choice([True, False], 50),
    })


@pytest.fixture
def empty_df():
    """Create an empty DataFrame."""
    return pd.DataFrame()


@pytest.fixture
def df_with_missing():
    """Create a DataFrame with missing values."""
    return pd.DataFrame({
        "a": [1, None, 3, 4, None],
        "b": [None, 2, 3, None, 5],
        "c": ["x", "y", None, "z", "w"],
    })


@pytest.fixture
def large_df():
    """Create a large DataFrame to test MAX_CHART_DATA_ROWS limit."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "x": rng.normal(0, 1, 3000),
        "y": rng.normal(0, 1, 3000),
        "category": rng.choice(["A", "B"], 3000),
    })


@pytest.fixture
def single_row_df():
    """Create a DataFrame with only one row."""
    return pd.DataFrame({
        "a": [1],
        "b": ["test"],
        "c": [3.14],
    })


class TestDescribeDataset:
    """Comprehensive tests for describe_dataset tool."""

    @pytest.mark.asyncio
    async def test_describe_dataset_success(self, sample_df):
        """Test successful dataset description."""
        session_id = "test-session-456"
        
        with patch("backend.tools.dataset_tools.get_session") as mock_get_session:
            mock_get_session.return_value = {
                "df": sample_df,
                "filename": "test.csv",
                "schema": {
                    "filename": "test.csv",
                    "shape": sample_df.shape,
                    "columns": list(sample_df.columns),
                    "numeric_columns": ["revenue", "amount"],
                    "categorical_columns": ["category", "date"],
                    "dtypes": sample_df.dtypes.astype(str).to_dict(),
                    "missing_values": 0,
                },
            }
            
            with patch("backend.tools.dataset_tools.require_df") as mock_require_df:
                mock_require_df.return_value = (sample_df, None)
                result = await dataset_tools.describe_dataset.ainvoke({"session_id": session_id})
                
                assert "filename" in result
                assert "shape" in result
                assert "columns" in result
                assert "numeric_columns" in result
                assert "categorical_columns" in result
                assert "dtypes" in result
                assert "sample" in result
                assert "numeric_summary" in result
                assert result["shape"] == (50, 5)
                assert len(result["sample"]) <= 3  # MAX_SAMPLE_ROWS

    @pytest.mark.asyncio
    async def test_describe_dataset_session_not_found(self):
        """Test when session doesn't exist."""
        with patch("backend.tools.dataset_tools.get_session") as mock_get_session:
            mock_get_session.return_value = None
            result = await dataset_tools.describe_dataset.ainvoke({"session_id": "invalid-session"})
            assert "error" in result
            assert "not found" in result["error"]

    @pytest.mark.skip(reason="Requires database infrastructure")
    @pytest.mark.asyncio
    async def test_describe_dataset_none_session_id(self):
        """Test with None session_id."""
        with patch("backend.session.get_session", new_callable=MagicMock) as mock_session:
            mock_session.return_value = None
            result = await dataset_tools.describe_dataset.ainvoke({"session_id": None})
            assert "error" in result

    @pytest.mark.asyncio
    async def test_describe_dataset_empty_dataframe(self, empty_df):
        """Test with empty DataFrame."""
        session_id = "test-empty"
        
        with patch("backend.tools.dataset_tools.get_session") as mock_get_session:
            mock_get_session.return_value = {
                "df": empty_df,
                "filename": "empty.csv",
                "schema": {
                    "filename": "empty.csv",
                    "shape": (0, 0),
                    "columns": [],
                    "numeric_columns": [],
                    "categorical_columns": [],
                    "dtypes": {},
                    "missing_values": 0,
                },
            }
            
            with patch("backend.tools.dataset_tools.require_df") as mock_require_df:
                mock_require_df.return_value = (empty_df, None)
                result = await dataset_tools.describe_dataset.ainvoke({"session_id": session_id})
                
                assert result["shape"] == (0, 0)
                assert result["sample"] == []
                assert result["numeric_summary"] == {}

    @pytest.mark.asyncio
    async def test_describe_dataset_with_missing_values(self, df_with_missing):
        """Test with DataFrame containing missing values."""
        session_id = "test-missing"
        
        with patch("backend.tools.dataset_tools.get_session") as mock_get_session:
            mock_get_session.return_value = {
                "df": df_with_missing,
                "filename": "missing.csv",
                "schema": {
                    "filename": "missing.csv",
                    "shape": df_with_missing.shape,
                    "columns": list(df_with_missing.columns),
                    "numeric_columns": ["a", "b"],
                    "categorical_columns": ["c"],
                    "dtypes": df_with_missing.dtypes.astype(str).to_dict(),
                    "missing_values": 3,
                },
            }
            
            with patch("backend.tools.dataset_tools.require_df") as mock_require_df:
                mock_require_df.return_value = (df_with_missing, None)
                result = await dataset_tools.describe_dataset.ainvoke({"session_id": session_id})
                
                assert "missing_by_column" in result
                assert result["missing_values_total"] == 3

    @pytest.mark.asyncio
    async def test_describe_dataset_single_row(self, single_row_df):
        """Test with single row DataFrame."""
        session_id = "test-single"
        
        with patch("backend.tools.dataset_tools.get_session") as mock_get_session:
            mock_get_session.return_value = {
                "df": single_row_df,
                "filename": "single.csv",
                "schema": {
                    "filename": "single.csv",
                    "shape": (1, 3),
                    "columns": list(single_row_df.columns),
                    "numeric_columns": ["a", "c"],
                    "categorical_columns": ["b"],
                    "dtypes": single_row_df.dtypes.astype(str).to_dict(),
                    "missing_values": 0,
                },
            }
            
            with patch("backend.tools.dataset_tools.require_df") as mock_require_df:
                mock_require_df.return_value = (single_row_df, None)
                result = await dataset_tools.describe_dataset.ainvoke({"session_id": session_id})
                
                assert result["shape"] == (1, 3)
                assert len(result["sample"]) == 1


class TestGenerateChartSpec:
    """Comprehensive tests for generate_chart_spec tool."""

    @pytest.mark.asyncio
    async def test_generate_chart_spec_bar_success(self, sample_df):
        """Test successful bar chart generation."""
        with patch("backend.tools.dataset_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await dataset_tools.generate_chart_spec.ainvoke({
                "session_id": "test",
                "chart_type": "bar",
                "x_column": "category",
                "y_column": "revenue",
            })
            
            assert "chart_spec" in result
            assert result["chart_type"] == "bar"
            assert "mark" in result["chart_spec"]
            assert "encoding" in result["chart_spec"]
            assert result["chart_spec"]["mark"] == "bar"
            assert "x" in result["chart_spec"]["encoding"]
            assert "y" in result["chart_spec"]["encoding"]

    @pytest.mark.asyncio
    async def test_generate_chart_spec_scatter_success(self, sample_df):
        """Test successful scatter plot generation."""
        with patch("backend.tools.dataset_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await dataset_tools.generate_chart_spec.ainvoke({
                "session_id": "test",
                "chart_type": "scatter",
                "x_column": "revenue",
                "y_column": "amount",
            })
            
            assert result["chart_spec"]["mark"] == "point"
            assert result["chart_spec"]["encoding"]["x"]["type"] == "quantitative"
            assert result["chart_spec"]["encoding"]["y"]["type"] == "quantitative"

    @pytest.mark.asyncio
    async def test_generate_chart_spec_line_with_temporal(self, sample_df):
        """Test line chart with date column (temporal type)."""
        with patch("backend.tools.dataset_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await dataset_tools.generate_chart_spec.ainvoke({
                "session_id": "test",
                "chart_type": "line",
                "x_column": "date",
                "y_column": "revenue",
            })
            
            assert result["chart_spec"]["mark"] == "line"
            assert result["chart_spec"]["encoding"]["x"]["type"] == "temporal"

    @pytest.mark.asyncio
    async def test_generate_chart_spec_histogram(self, sample_df):
        """Test histogram chart generation."""
        with patch("backend.tools.dataset_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await dataset_tools.generate_chart_spec.ainvoke({
                "session_id": "test",
                "chart_type": "histogram",
                "x_column": "revenue",
            })
            
            assert result["chart_spec"]["mark"] == "bar"
            assert result["chart_spec"]["encoding"]["y"]["aggregate"] == "count"

    @pytest.mark.asyncio
    async def test_generate_chart_spec_boxplot(self, sample_df):
        """Test box plot generation."""
        with patch("backend.tools.dataset_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await dataset_tools.generate_chart_spec.ainvoke({
                "session_id": "test",
                "chart_type": "box",
                "x_column": "category",
                "y_column": "revenue",
            })
            
            assert result["chart_spec"]["mark"] == "boxplot"

    @pytest.mark.asyncio
    async def test_generate_chart_spec_heatmap(self, sample_df):
        """Test heatmap generation."""
        with patch("backend.tools.dataset_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await dataset_tools.generate_chart_spec.ainvoke({
                "session_id": "test",
                "chart_type": "heatmap",
                "x_column": "category",
                "y_column": "is_active",
            })
            
            assert result["chart_spec"]["mark"] == "rect"
            assert result["chart_spec"]["encoding"]["color"]["aggregate"] == "count"

    @pytest.mark.asyncio
    async def test_generate_chart_spec_with_color_column(self, sample_df):
        """Test chart with color encoding."""
        with patch("backend.tools.dataset_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await dataset_tools.generate_chart_spec.ainvoke({
                "session_id": "test",
                "chart_type": "scatter",
                "x_column": "revenue",
                "y_column": "amount",
                "color_column": "category",
            })
            
            assert "color" in result["chart_spec"]["encoding"]
            assert result["chart_spec"]["encoding"]["color"]["field"] == "category"

    @pytest.mark.asyncio
    async def test_generate_chart_spec_with_title(self, sample_df):
        """Test chart with custom title."""
        with patch("backend.tools.dataset_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await dataset_tools.generate_chart_spec.ainvoke({
                "session_id": "test",
                "chart_type": "bar",
                "x_column": "category",
                "y_column": "revenue",
                "title": "My Custom Chart",
            })
            
            assert result["chart_spec"]["title"] == "My Custom Chart"

    @pytest.mark.asyncio
    async def test_generate_chart_spec_missing_chart_type(self, sample_df):
        """Test error when chart_type is missing."""
        with patch("backend.tools.dataset_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await dataset_tools.generate_chart_spec.ainvoke({
                "session_id": "test",
                "chart_type": None,
            })
            
            assert "error" in result
            assert "chart_type is required" in result["error"]

    @pytest.mark.asyncio
    async def test_generate_chart_spec_invalid_x_column(self, sample_df):
        """Test error when x_column doesn't exist."""
        with patch("backend.tools.dataset_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await dataset_tools.generate_chart_spec.ainvoke({
                "session_id": "test",
                "chart_type": "bar",
                "x_column": "nonexistent",
                "y_column": "revenue",
            })
            
            assert "error" in result
            assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_generate_chart_spec_invalid_y_column(self, sample_df):
        """Test error when y_column doesn't exist."""
        with patch("backend.tools.dataset_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await dataset_tools.generate_chart_spec.ainvoke({
                "session_id": "test",
                "chart_type": "bar",
                "x_column": "category",
                "y_column": "nonexistent",
            })
            
            assert "error" in result
            assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_generate_chart_spec_invalid_color_column(self, sample_df):
        """Test error when color_column doesn't exist."""
        with patch("backend.tools.dataset_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await dataset_tools.generate_chart_spec.ainvoke({
                "session_id": "test",
                "chart_type": "scatter",
                "x_column": "revenue",
                "y_column": "amount",
                "color_column": "nonexistent",
            })
            
            assert "error" in result
            assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_generate_chart_spec_session_not_found(self):
        """Test error when session doesn't exist."""
        with patch("backend.tools.dataset_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (None, {"error": "Session not found"})
            result = await dataset_tools.generate_chart_spec.ainvoke({
                "session_id": "invalid",
                "chart_type": "bar",
                "x_column": "a",
            })
            
            assert "error" in result

    @pytest.mark.asyncio
    async def test_generate_chart_spec_data_limit(self, large_df):
        """Test that data is limited to MAX_CHART_DATA_ROWS."""
        with patch("backend.tools.dataset_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (large_df, None)
            result = await dataset_tools.generate_chart_spec.ainvoke({
                "session_id": "test",
                "chart_type": "scatter",
                "x_column": "x",
                "y_column": "y",
            })
            
            assert len(result["chart_spec"]["data"]["values"]) <= 2000

    @pytest.mark.asyncio
    async def test_generate_chart_spec_unknown_chart_type(self, sample_df):
        """Test with unknown chart type defaults to point."""
        with patch("backend.tools.dataset_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await dataset_tools.generate_chart_spec.ainvoke({
                "session_id": "test",
                "chart_type": "unknown_type",
                "x_column": "category",
                "y_column": "revenue",
            })
            
            # Should default to "point" for unknown types
            assert result["chart_spec"]["mark"] == "point"

    @pytest.mark.asyncio
    async def test_generate_chart_spec_no_columns(self, sample_df):
        """Test chart generation with minimal parameters."""
        with patch("backend.tools.dataset_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = await dataset_tools.generate_chart_spec.ainvoke({
                "session_id": "test",
                "chart_type": "bar",
            })
            
            # Should still work with minimal params
            assert "chart_spec" in result


class TestDatasetToolsEdgeCases:
    """Edge case tests for dataset tools."""

    @pytest.mark.asyncio
    async def test_describe_dataset_numeric_column_not_in_df(self, sample_df):
        """Test when numeric column in schema is not in DataFrame."""
        session_id = "test"
        
        # Create a modified sample_df without the 'amount' column that schema claims exists
        df_missing_col = sample_df.drop(columns=["amount"])
        
        with patch("backend.tools.dataset_tools.get_session") as mock_get_session:
            mock_get_session.return_value = {
                "df": df_missing_col,
                "filename": "test.csv",
                "schema": {
                    "filename": "test.csv",
                    "shape": df_missing_col.shape,
                    "columns": list(df_missing_col.columns),
                    "numeric_columns": ["revenue", "amount"],  # amount not in df
                    "categorical_columns": ["category", "date"],
                    "dtypes": df_missing_col.dtypes.astype(str).to_dict(),
                    "missing_values": 0,
                },
            }
            
            with patch("backend.tools.dataset_tools.require_df") as mock_require_df:
                mock_require_df.return_value = (df_missing_col, None)
                result = await dataset_tools.describe_dataset.ainvoke({"session_id": session_id})
                
                # Should not crash, just skip the missing column
                assert "amount" not in result["numeric_summary"]
                assert "revenue" in result["numeric_summary"]

    @pytest.mark.asyncio
    async def test_describe_dataset_all_missing_column(self):
        """Test when all values in a column are missing."""
        df = pd.DataFrame({
            "a": [None, None, None],
            "b": [1, 2, 3],
        })
        
        with patch("backend.tools.dataset_tools.get_session") as mock_get_session:
            mock_get_session.return_value = {
                "df": df,
                "filename": "test.csv",
                "schema": {
                    "filename": "test.csv",
                    "shape": df.shape,
                    "columns": list(df.columns),
                    "numeric_columns": ["a", "b"],
                    "categorical_columns": [],
                    "dtypes": df.dtypes.astype(str).to_dict(),
                    "missing_values": 3,
                },
            }
            
            with patch("backend.tools.dataset_tools.require_df") as mock_require_df:
                mock_require_df.return_value = (df, None)
                result = await dataset_tools.describe_dataset.ainvoke({"session_id": "test"})
                
                # Column with all NaN might be skipped or have NaN in summary
                assert "numeric_summary" in result
