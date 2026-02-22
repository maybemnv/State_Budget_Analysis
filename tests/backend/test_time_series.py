import pandas as pd
import numpy as np
import pytest
from backend.analyzers.time_series import (
    stationarity as stat_mod,
    forecasting as fc_mod,
    decomposition as dec_mod,
)
from backend.analyzers.time_series.analyzer import TimeSeriesAnalyzer


def make_series(n: int = 48, start: str = "2020-01-01") -> pd.Series:
    idx = pd.date_range(start=start, periods=n, freq="ME")
    vals = np.sin(np.linspace(0, 6 * np.pi, n)) * 10 + np.linspace(0, 5, n)
    return pd.Series(vals, index=idx, name="y")


def test_check_stationarity_returns_keys():
    s = make_series()
    result = stat_mod.check_stationarity(s)
    assert "is_stationary" in result
    assert "adf" in result
    assert "kpss" in result
    assert "p_value" in result["adf"]


def test_check_stationarity_too_short():
    with pytest.raises(ValueError):
        stat_mod.check_stationarity(pd.Series([1, 2, 3]))


def test_fit_arima_success():
    s = make_series()
    result = fc_mod.fit_arima(s, order=(1, 1, 1))
    assert "aic" in result
    assert "_fit" in result


def test_forecast_arima_length():
    s = make_series()
    fit = fc_mod.fit_arima(s, order=(1, 1, 1))
    fc = fc_mod.forecast_arima(fit, steps=6)
    assert len(fc["forecast"]) == 6
    assert len(fc["lower"]) == 6
    assert len(fc["upper"]) == 6


def test_decompose_series():
    s = make_series(n=36)
    result = dec_mod.decompose_series(s, period=12)
    assert "trend" in result
    assert "seasonal" in result
    assert "resid" in result
    assert result["period"] == 12


def test_decompose_series_too_short():
    s = make_series(n=5)
    with pytest.raises(ValueError):
        dec_mod.decompose_series(s, period=12)


def test_analyzer_load_and_prepare():
    s = make_series(n=36)
    df = pd.DataFrame({"date": s.index, "value": s.values})
    analyzer = TimeSeriesAnalyzer()
    series = analyzer.load_and_prepare(df, "date", "value")
    assert isinstance(series, pd.Series)
    assert isinstance(series.index, pd.DatetimeIndex)


def test_analyzer_detect_year_columns():
    df = pd.DataFrame({"Category": ["Revenue"], "2020": [100], "2021": [110], "2022": [120]})
    analyzer = TimeSeriesAnalyzer()
    caps = analyzer.detect_time_series_capable_data(df)
    assert caps["has_year_columns"] is True
    assert "2020" in caps["year_columns"]


@pytest.mark.parametrize("n", [12, 24])
def test_prophet_fit(n):
    s = make_series(n=n)
    result = fc_mod.fit_prophet(s)
    assert result["model_type"] == "Prophet"
    assert len(result["fitted_values"]) == n
