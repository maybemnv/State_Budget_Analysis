import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from backend.tools import guards
from backend.tools import dataset_tools
from backend.session import get_df, get_session, create_session


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=50, freq="ME").astype(str),
            "revenue": rng.normal(1000, 100, 50).round(2),
            "category": rng.choice(["A", "B", "C"], 50),
            "amount": rng.uniform(10, 1000, 50),
        }
    )


@pytest.fixture
def mock_session(sample_df):
    session_id = "test-session-123"
    with patch("backend.tools.guards.get_df") as mock_get_df:
        mock_get_df.return_value = sample_df
        yield session_id, sample_df


class TestGuards:
    """Test suite for guard functions in tools/guards.py"""

    def test_require_df_success(self, mock_session):
        session_id, df = mock_session
        result_df, error = guards.require_df(session_id)
        assert error is None
        assert result_df is not None
        assert isinstance(result_df, pd.DataFrame)

    def test_require_df_no_session_id(self):
        result_df, error = guards.require_df(None)
        assert error is not None
        assert "session_id is required" in error["error"]
        assert result_df is None

    def test_require_df_session_not_found(self):
        with patch("backend.tools.guards.get_df") as mock_get_df:
            mock_get_df.return_value = None
            result_df, error = guards.require_df("nonexistent-session")
            assert error is not None
            assert "not found" in error["error"]
            assert result_df is None


class TestDatasetTools:
    """Test suite for dataset tools in tools/dataset_tools.py"""

    def test_describe_dataset_success(self, sample_df):
        session_id = "test-session-456"
        with patch("backend.tools.dataset_tools.get_session") as mock_get_session:
            mock_get_session.return_value = {
                "df": sample_df,
                "metadata": {
                    "filename": "test.csv",
                    "shape": sample_df.shape,
                    "columns": list(sample_df.columns),
                    "numeric_columns": ["revenue", "amount"],
                    "categorical_columns": ["category", "date"],
                    "dtypes": sample_df.dtypes.astype(str).to_dict(),
                    "missing_values": 0,
                },
            }
            result = dataset_tools.describe_dataset.invoke({"session_id": session_id})
            assert "filename" in result
            assert "shape" in result
            assert "columns" in result
            assert result["shape"] == (50, 4)

    def test_describe_dataset_session_not_found(self):
        with patch("backend.tools.dataset_tools.get_session") as mock_get_session:
            mock_get_session.return_value = None
            result = dataset_tools.describe_dataset.invoke({"session_id": "invalid"})
            assert "error" in result
            assert "not found" in result["error"]

    def test_generate_chart_spec_success(self, sample_df):
        session_id = "test-session-789"
        with patch("backend.tools.dataset_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = dataset_tools.generate_chart_spec.invoke(
                {
                    "session_id": session_id,
                    "chart_type": "bar",
                    "x_column": "category",
                    "y_column": "revenue",
                }
            )
            assert "chart_spec" in result
            assert result["chart_type"] == "bar"

    def test_generate_chart_spec_missing_chart_type(self, sample_df):
        with patch("backend.tools.dataset_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = dataset_tools.generate_chart_spec.invoke(
                {
                    "session_id": "test",
                    "chart_type": None,
                }
            )
            assert "error" in result

    def test_generate_chart_spec_invalid_column(self, sample_df):
        with patch("backend.tools.dataset_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (sample_df, None)
            result = dataset_tools.generate_chart_spec.invoke(
                {
                    "session_id": "test",
                    "chart_type": "bar",
                    "x_column": "nonexistent_column",
                }
            )
            assert "error" in result
            assert "not found" in result["error"]
