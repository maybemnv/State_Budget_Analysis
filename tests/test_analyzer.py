import pandas as pd
import numpy as np
import pytest

from src.time_series.analyzer import TimeSeriesAnalyzer


def make_monthly_series(n=36, start="2020-01-01"):
    idx = pd.date_range(start=start, periods=n, freq="M")
    values = np.sin(np.linspace(0, 6 * np.pi, n)) * 10 + np.linspace(0, 5, n)
    return pd.Series(values, index=idx, name="y")


def test_detect_time_series_capable_data_with_dates():
    df = pd.DataFrame({
        "date": pd.date_range("2021-01-01", periods=10, freq="D"),
        "value": np.arange(10)
    })
    analyzer = TimeSeriesAnalyzer()
    caps = analyzer.detect_time_series_capable_data(df)
    assert caps["has_date_columns"] is True
    assert "date" in caps["date_columns"]
    assert "value" in caps["numeric_columns"]


def test_create_time_series_from_year_columns():
    df = pd.DataFrame({
        "measure": ["Revenue"],
        "2019": [100],
        "2020": [120],
        "2021": [140],
    }).set_index("measure")
    analyzer = TimeSeriesAnalyzer()
    series, err = analyzer.create_time_series_from_year_columns(df, ["2019", "2020", "2021"], "Revenue")
    assert err is None
    assert isinstance(series, pd.Series)
    assert len(series) == 3
    assert series.index.is_monotonic_increasing


def test_create_time_series_from_sequential():
    df = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
    analyzer = TimeSeriesAnalyzer()
    series, err = analyzer.create_time_series_from_sequential(df, "value", freq="M", start_date="2020-01-01")
    assert err is None
    assert isinstance(series, pd.Series)
    assert len(series) == 5
    assert isinstance(series.index, pd.DatetimeIndex)


def test_load_and_prepare_and_decompose():
    s = make_monthly_series(36)
    df = pd.DataFrame({"date": s.index, "value": s.values})
    analyzer = TimeSeriesAnalyzer()
    series = analyzer.load_and_prepare(df, "date", "value")
    assert isinstance(series, pd.Series)
    result = analyzer.decompose_series(series)
    # Expect plot and keys present
    assert "plot" in result
    assert result["period"] >= 2


def test_stationarity_and_arima_fit():
    s = make_monthly_series(36)
    df = pd.DataFrame({"date": s.index, "value": s.values})
    analyzer = TimeSeriesAnalyzer()
    series = analyzer.load_and_prepare(df, "date", "value")
    stat, err = analyzer.check_stationarity(series)
    assert err is None
    assert "p_value" in stat
    fit = analyzer.fit_arima(series, order=(1, 1, 1))
    assert fit["success"] is True
    fc = analyzer.forecast_arima(fit["model"], steps=6)
    assert fc["success"] is True
    assert len(fc["forecast"]) == 6


@pytest.mark.parametrize("n", [12, 24])
def test_prophet_fit_and_forecast(n):
    # Prophet needs enough data; 12 monthly points should suffice
    s = make_monthly_series(n)
    df = pd.DataFrame({"date": s.index, "value": s.values})
    analyzer = TimeSeriesAnalyzer()
    series = analyzer.load_and_prepare(df, "date", "value")
    fit = analyzer.fit_prophet(series)
    assert fit["success"] is True

