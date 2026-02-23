from typing import Optional
from langchain_core.tools import tool
from ..schemas import TimeSeriesInput, ForecastInput, StationarityInput
from ..analyzers.time_series.analyzer import TimeSeriesAnalyzer
from ..analyzers.time_series import stationarity as stat_module
from ..analyzers.time_series import forecasting as fc_module
from .guards import require_df


@tool("check_stationarity", args_schema=StationarityInput)
def check_stationarity(
    session_id: Optional[str] = None,
    date_column: Optional[str] = None,
    value_column: Optional[str] = None,
) -> dict:
    """Run ADF + KPSS stationarity tests on a time series column."""
    if not date_column or not value_column:
        return {"error": "date_column and value_column are both required"}

    df, err = require_df(session_id)
    if err:
        return err
    try:
        series = TimeSeriesAnalyzer().load_and_prepare(df, date_column, value_column)
        return stat_module.check_stationarity(series)
    except Exception as e:
        return {"error": str(e)}


@tool("run_forecast", args_schema=ForecastInput)
def run_forecast(
    session_id: Optional[str] = None,
    date_column: Optional[str] = None,
    value_column: Optional[str] = None,
    steps: int = 12,
    model: str = "arima",
) -> dict:
    """Forecast a time series using ARIMA or Prophet. Returns point forecasts and confidence intervals."""
    if not date_column or not value_column:
        return {"error": "date_column and value_column are both required"}

    df, err = require_df(session_id)
    if err:
        return err
    try:
        series = TimeSeriesAnalyzer().load_and_prepare(df, date_column, value_column)
        if model == "prophet":
            fit = fc_module.fit_prophet(series)
            return fc_module.forecast_prophet(fit, steps=steps)
        fit = fc_module.fit_arima(series)
        return fc_module.forecast_arima(fit, steps=steps)
    except Exception as e:
        return {"error": str(e)}


@tool("decompose_time_series", args_schema=TimeSeriesInput)
def decompose_time_series(
    session_id: Optional[str] = None,
    date_column: Optional[str] = None,
    value_column: Optional[str] = None,
) -> dict:
    """Decompose a time series into trend, seasonal, and residual components."""
    if not date_column or not value_column:
        return {"error": "date_column and value_column are both required"}

    df, err = require_df(session_id)
    if err:
        return err
    try:
        analyzer = TimeSeriesAnalyzer()
        series = analyzer.load_and_prepare(df, date_column, value_column)
        return analyzer.decompose_series(series)
    except Exception as e:
        return {"error": str(e)}
