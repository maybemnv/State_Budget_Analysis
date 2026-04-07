"""Comprehensive tests for time_series_tools.py"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, AsyncMock

from backend.tools import time_series_tools


@pytest.fixture
def time_series_df():
    """Create a DataFrame with time series data."""
    dates = pd.date_range(start="2020-01-01", periods=48, freq="ME")
    values = np.sin(np.linspace(0, 6 * np.pi, 48)) * 10 + np.linspace(0, 20, 48)
    return pd.DataFrame({
        "date": dates,
        "value": values,
        "other_col": range(48),
    })


@pytest.fixture
def daily_ts_df():
    """Create a daily frequency time series."""
    dates = pd.date_range(start="2023-01-01", periods=365, freq="D")
    values = np.sin(np.linspace(0, 4 * np.pi, 365)) * 10 + 50
    return pd.DataFrame({
        "timestamp": dates,
        "sales": values + np.random.default_rng(42).normal(0, 2, 365),
    })


@pytest.fixture
def short_ts_df():
    """Create a very short time series for edge cases."""
    dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
    return pd.DataFrame({
        "date": dates,
        "value": [1, 2, 3, 4, 5],
    })


@pytest.fixture
def stationary_df():
    """Create a stationary time series (white noise)."""
    rng = np.random.default_rng(42)
    dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
    return pd.DataFrame({
        "date": dates,
        "value": rng.normal(0, 1, 100),
    })


@pytest.fixture
def trend_df():
    """Create a time series with strong trend."""
    dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
    return pd.DataFrame({
        "date": dates,
        "value": np.linspace(0, 100, 100) + np.random.default_rng(42).normal(0, 5, 100),
    })


@pytest.fixture
def seasonal_df():
    """Create a strongly seasonal time series."""
    dates = pd.date_range(start="2020-01-01", periods=24, freq="ME")
    values = np.sin(np.linspace(0, 4 * np.pi, 24)) * 20 + 50
    return pd.DataFrame({
        "date": dates,
        "value": values,
    })


class TestCheckStationarity:
    """Comprehensive tests for check_stationarity tool."""

    @pytest.mark.asyncio
    async def test_stationarity_success(self, time_series_df):
        """Test successful stationarity check."""
        with patch("backend.tools.time_series_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (time_series_df, None)
            result = await time_series_tools.check_stationarity.ainvoke({
                "session_id": "test",
                "date_column": "date",
                "value_column": "value",
            })
            
            assert "is_stationary" in result
            assert "adf" in result
            assert "kpss" in result
            assert "p_value" in result["adf"]
            assert "test_statistic" in result["adf"]
            assert "critical_values" in result["adf"]

    @pytest.mark.asyncio
    async def test_stationarity_stationary_data(self, stationary_df):
        """Test with stationary data (should indicate stationary)."""
        with patch("backend.tools.time_series_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (stationary_df, None)
            result = await time_series_tools.check_stationarity.ainvoke({
                "session_id": "test",
                "date_column": "date",
                "value_column": "value",
            })
            
            # Stationary data should have is_stationary = True
            assert result["is_stationary"] is True

    @pytest.mark.asyncio
    async def test_stationarity_trend_data(self, trend_df):
        """Test with trending data (should indicate non-stationary)."""
        with patch("backend.tools.time_series_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (trend_df, None)
            result = await time_series_tools.check_stationarity.ainvoke({
                "session_id": "test",
                "date_column": "date",
                "value_column": "value",
            })
            
            # Trending data should be non-stationary
            assert result["is_stationary"] is False

    @pytest.mark.asyncio
    async def test_stationarity_missing_date_column(self, time_series_df):
        """Test error when date_column is missing."""
        with patch("backend.tools.time_series_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (time_series_df, None)
            result = await time_series_tools.check_stationarity.ainvoke({
                "session_id": "test",
                "date_column": None,
                "value_column": "value",
            })
            
            assert "error" in result
            assert "date_column and value_column are both required" in result["error"]

    @pytest.mark.asyncio
    async def test_stationarity_missing_value_column(self, time_series_df):
        """Test error when value_column is missing."""
        with patch("backend.tools.time_series_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (time_series_df, None)
            result = await time_series_tools.check_stationarity.ainvoke({
                "session_id": "test",
                "date_column": "date",
                "value_column": None,
            })
            
            assert "error" in result

    @pytest.mark.asyncio
    async def test_stationarity_invalid_date_column(self, time_series_df):
        """Test error when date_column doesn't exist."""
        with patch("backend.tools.time_series_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (time_series_df, None)
            result = await time_series_tools.check_stationarity.ainvoke({
                "session_id": "test",
                "date_column": "nonexistent",
                "value_column": "value",
            })
            
            assert "error" in result

    @pytest.mark.asyncio
    async def test_stationarity_invalid_value_column(self, time_series_df):
        """Test error when value_column doesn't exist."""
        with patch("backend.tools.time_series_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (time_series_df, None)
            result = await time_series_tools.check_stationarity.ainvoke({
                "session_id": "test",
                "date_column": "date",
                "value_column": "nonexistent",
            })
            
            assert "error" in result

    @pytest.mark.asyncio
    async def test_stationarity_session_not_found(self):
        """Test error when session doesn't exist."""
        with patch("backend.tools.time_series_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (None, {"error": "Session not found"})
            result = await time_series_tools.check_stationarity.ainvoke({
                "session_id": "invalid",
                "date_column": "date",
                "value_column": "value",
            })
            
            assert "error" in result

    @pytest.mark.asyncio
    async def test_stationarity_too_short_series(self, short_ts_df):
        """Test error with very short time series."""
        with patch("backend.tools.time_series_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (short_ts_df, None)
            result = await time_series_tools.check_stationarity.ainvoke({
                "session_id": "test",
                "date_column": "date",
                "value_column": "value",
            })
            
            # Very short series should error
            assert "error" in result

    @pytest.mark.asyncio
    async def test_stationarity_with_missing_values(self, time_series_df):
        """Test with time series containing missing values."""
        df = time_series_df.copy()
        df.loc[5:10, "value"] = np.nan
        
        with patch("backend.tools.time_series_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (df, None)
            result = await time_series_tools.check_stationarity.ainvoke({
                "session_id": "test",
                "date_column": "date",
                "value_column": "value",
            })
            
            # Should handle missing values
            assert "is_stationary" in result or "error" in result


class TestRunForecast:
    """Comprehensive tests for run_forecast tool."""

    @pytest.mark.asyncio
    async def test_forecast_arima_success(self, time_series_df):
        """Test successful ARIMA forecast."""
        with patch("backend.tools.time_series_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (time_series_df, None)
            result = await time_series_tools.run_forecast.ainvoke({
                "session_id": "test",
                "date_column": "date",
                "value_column": "value",
                "steps": 12,
                "model": "arima",
            })
            
            assert "forecast" in result
            assert "lower" in result
            assert "upper" in result
            assert "model_type" in result
            assert result["model_type"] == "ARIMA"
            assert len(result["forecast"]) == 12
            assert len(result["lower"]) == 12
            assert len(result["upper"]) == 12

    @pytest.mark.asyncio
    async def test_forecast_prophet_success(self, daily_ts_df):
        """Test successful Prophet forecast."""
        with patch("backend.tools.time_series_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (daily_ts_df, None)
            result = await time_series_tools.run_forecast.ainvoke({
                "session_id": "test",
                "date_column": "timestamp",
                "value_column": "sales",
                "steps": 30,
                "model": "prophet",
            })
            
            assert "forecast" in result
            assert "lower" in result
            assert "upper" in result
            assert "model_type" in result
            assert result["model_type"] == "Prophet"
            assert len(result["forecast"]) == 30

    @pytest.mark.asyncio
    async def test_forecast_different_steps(self, time_series_df):
        """Test forecast with different step counts."""
        with patch("backend.tools.time_series_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (time_series_df, None)
            
            for steps in [6, 12, 24]:
                result = await time_series_tools.run_forecast.ainvoke({
                    "session_id": "test",
                    "date_column": "date",
                    "value_column": "value",
                    "steps": steps,
                    "model": "arima",
                })
                
                assert len(result["forecast"]) == steps

    @pytest.mark.asyncio
    async def test_forecast_default_model(self, time_series_df):
        """Test forecast with default model (arima)."""
        with patch("backend.tools.time_series_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (time_series_df, None)
            result = await time_series_tools.run_forecast.ainvoke({
                "session_id": "test",
                "date_column": "date",
                "value_column": "value",
                "steps": 12,
            })
            
            assert result["model_type"] == "ARIMA"

    @pytest.mark.asyncio
    async def test_forecast_missing_date_column(self, time_series_df):
        """Test error when date_column is missing."""
        with patch("backend.tools.time_series_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (time_series_df, None)
            result = await time_series_tools.run_forecast.ainvoke({
                "session_id": "test",
                "date_column": None,
                "value_column": "value",
                "steps": 12,
            })
            
            assert "error" in result
            assert "date_column and value_column are both required" in result["error"]

    @pytest.mark.asyncio
    async def test_forecast_missing_value_column(self, time_series_df):
        """Test error when value_column is missing."""
        with patch("backend.tools.time_series_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (time_series_df, None)
            result = await time_series_tools.run_forecast.ainvoke({
                "session_id": "test",
                "date_column": "date",
                "value_column": None,
                "steps": 12,
            })
            
            assert "error" in result

    @pytest.mark.asyncio
    async def test_forecast_invalid_column(self, time_series_df):
        """Test error when columns don't exist."""
        with patch("backend.tools.time_series_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (time_series_df, None)
            result = await time_series_tools.run_forecast.ainvoke({
                "session_id": "test",
                "date_column": "nonexistent",
                "value_column": "value",
                "steps": 12,
            })
            
            assert "error" in result

    @pytest.mark.asyncio
    async def test_forecast_session_not_found(self):
        """Test error when session doesn't exist."""
        with patch("backend.tools.time_series_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (None, {"error": "Session not found"})
            result = await time_series_tools.run_forecast.ainvoke({
                "session_id": "invalid",
                "date_column": "date",
                "value_column": "value",
            })
            
            assert "error" in result

    @pytest.mark.asyncio
    async def test_forecast_short_series(self, short_ts_df):
        """Test forecast with very short series."""
        with patch("backend.tools.time_series_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (short_ts_df, None)
            result = await time_series_tools.run_forecast.ainvoke({
                "session_id": "test",
                "date_column": "date",
                "value_column": "value",
                "steps": 3,
            })
            
            # Short series might error or return results
            assert "forecast" in result or "error" in result

    @pytest.mark.asyncio
    async def test_forecast_single_step(self, time_series_df):
        """Test forecast with single step."""
        with patch("backend.tools.time_series_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (time_series_df, None)
            result = await time_series_tools.run_forecast.ainvoke({
                "session_id": "test",
                "date_column": "date",
                "value_column": "value",
                "steps": 1,
            })
            
            assert "forecast" in result
            assert len(result["forecast"]) == 1


class TestDecomposeTimeSeries:
    """Comprehensive tests for decompose_time_series tool."""

    @pytest.mark.asyncio
    async def test_decompose_success(self, seasonal_df):
        """Test successful time series decomposition."""
        with patch("backend.tools.time_series_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (seasonal_df, None)
            result = await time_series_tools.decompose_time_series.ainvoke({
                "session_id": "test",
                "date_column": "date",
                "value_column": "value",
            })
            
            assert "trend" in result
            assert "seasonal" in result
            assert "resid" in result
            assert "observed" in result
            assert "period" in result

    @pytest.mark.asyncio
    async def test_decompose_trend_data(self, trend_df):
        """Test decomposition of trending data."""
        with patch("backend.tools.time_series_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (trend_df, None)
            result = await time_series_tools.decompose_time_series.ainvoke({
                "session_id": "test",
                "date_column": "date",
                "value_column": "value",
            })
            
            assert "trend" in result
            # Trend should capture the increasing pattern

    @pytest.mark.asyncio
    async def test_decompose_missing_date_column(self, time_series_df):
        """Test error when date_column is missing."""
        with patch("backend.tools.time_series_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (time_series_df, None)
            result = await time_series_tools.decompose_time_series.ainvoke({
                "session_id": "test",
                "date_column": None,
                "value_column": "value",
            })
            
            assert "error" in result
            assert "date_column and value_column are both required" in result["error"]

    @pytest.mark.asyncio
    async def test_decompose_missing_value_column(self, time_series_df):
        """Test error when value_column is missing."""
        with patch("backend.tools.time_series_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (time_series_df, None)
            result = await time_series_tools.decompose_time_series.ainvoke({
                "session_id": "test",
                "date_column": "date",
                "value_column": None,
            })
            
            assert "error" in result

    @pytest.mark.asyncio
    async def test_decompose_invalid_column(self, time_series_df):
        """Test error when columns don't exist."""
        with patch("backend.tools.time_series_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (time_series_df, None)
            result = await time_series_tools.decompose_time_series.ainvoke({
                "session_id": "test",
                "date_column": "nonexistent",
                "value_column": "value",
            })
            
            assert "error" in result

    @pytest.mark.asyncio
    async def test_decompose_session_not_found(self):
        """Test error when session doesn't exist."""
        with patch("backend.tools.time_series_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (None, {"error": "Session not found"})
            result = await time_series_tools.decompose_time_series.ainvoke({
                "session_id": "invalid",
                "date_column": "date",
                "value_column": "value",
            })
            
            assert "error" in result

    @pytest.mark.asyncio
    async def test_decompose_too_short_series(self, short_ts_df):
        """Test error with very short series."""
        with patch("backend.tools.time_series_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (short_ts_df, None)
            result = await time_series_tools.decompose_time_series.ainvoke({
                "session_id": "test",
                "date_column": "date",
                "value_column": "value",
            })
            
            assert "error" in result

    @pytest.mark.asyncio
    async def test_decompose_with_missing_values(self, time_series_df):
        """Test decomposition with missing values."""
        df = time_series_df.copy()
        df.loc[10:15, "value"] = np.nan
        
        with patch("backend.tools.time_series_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (df, None)
            result = await time_series_tools.decompose_time_series.ainvoke({
                "session_id": "test",
                "date_column": "date",
                "value_column": "value",
            })
            
            # Should handle missing values or return error
            assert "trend" in result or "error" in result


class TestTimeSeriesToolsEdgeCases:
    """Edge case tests for time series tools."""

    @pytest.mark.asyncio
    async def test_forecast_negative_steps(self, time_series_df):
        """Test forecast with negative steps."""
        with patch("backend.tools.time_series_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (time_series_df, None)
            result = await time_series_tools.run_forecast.ainvoke({
                "session_id": "test",
                "date_column": "date",
                "value_column": "value",
                "steps": -5,
            })
            
            # Should error on negative steps
            assert "error" in result

    @pytest.mark.asyncio
    async def test_forecast_zero_steps(self, time_series_df):
        """Test forecast with zero steps."""
        with patch("backend.tools.time_series_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (time_series_df, None)
            result = await time_series_tools.run_forecast.ainvoke({
                "session_id": "test",
                "date_column": "date",
                "value_column": "value",
                "steps": 0,
            })
            
            # Should error or return empty
            assert "error" in result or len(result.get("forecast", [])) == 0

    @pytest.mark.asyncio
    async def test_stationarity_constant_series(self):
        """Test stationarity check with constant series."""
        dates = pd.date_range(start="2020-01-01", periods=50, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "value": [5.0] * 50,
        })
        
        with patch("backend.tools.time_series_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (df, None)
            result = await time_series_tools.check_stationarity.ainvoke({
                "session_id": "test",
                "date_column": "date",
                "value_column": "value",
            })
            
            # Constant series should be stationary
            assert "is_stationary" in result

    @pytest.mark.asyncio
    async def test_decompose_constant_series(self):
        """Test decomposition with constant series."""
        dates = pd.date_range(start="2020-01-01", periods=50, freq="ME")
        df = pd.DataFrame({
            "date": dates,
            "value": [5.0] * 50,
        })
        
        with patch("backend.tools.time_series_tools.require_df") as mock_require_df:
            mock_require_df.return_value = (df, None)
            result = await time_series_tools.decompose_time_series.ainvoke({
                "session_id": "test",
                "date_column": "date",
                "value_column": "value",
            })
            
            # Should handle constant series
            assert "trend" in result or "error" in result
