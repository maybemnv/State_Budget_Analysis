from langchain_core.tools import tool
from ..schemas import TimeSeriesInput, ForecastInput, StationarityInput
from ..session import get_df
from ..analyzers.time_series.analyzer import TimeSeriesAnalyzer
from ..analyzers.time_series import stationarity as stat_module
from ..analyzers.time_series import forecasting as fc_module


@tool("check_stationarity", args_schema=StationarityInput)
def check_stationarity(session_id: str, date_column: str, value_column: str) -> dict:
    """Run ADF + KPSS stationarity tests on a time series column."""
    df = get_df(session_id)
    if df is None:
        return {"error": f"Session {session_id!r} not found"}
    try:
        analyzer = TimeSeriesAnalyzer()
        series = analyzer.load_and_prepare(df, date_column, value_column)
        return stat_module.check_stationarity(series)
    except Exception as e:
        return {"error": str(e)}


@tool("run_forecast", args_schema=ForecastInput)
def run_forecast(
    session_id: str,
    date_column: str,
    value_column: str,
    steps: int = 12,
    model: str = "arima",
) -> dict:
    """Forecast a time series using ARIMA or Prophet. Returns point forecasts and confidence intervals."""
    df = get_df(session_id)
    if df is None:
        return {"error": f"Session {session_id!r} not found"}
    try:
        analyzer = TimeSeriesAnalyzer()
        series = analyzer.load_and_prepare(df, date_column, value_column)
        if model == "prophet":
            fit = fc_module.fit_prophet(series)
            return fc_module.forecast_prophet(fit, steps=steps)
        else:
            fit = fc_module.fit_arima(series)
            return fc_module.forecast_arima(fit, steps=steps)
    except Exception as e:
        return {"error": str(e)}


@tool("decompose_time_series", args_schema=TimeSeriesInput)
def decompose_time_series(session_id: str, date_column: str, value_column: str) -> dict:
    """Decompose a time series into trend, seasonal, and residual components."""
    df = get_df(session_id)
    if df is None:
        return {"error": f"Session {session_id!r} not found"}
    try:
        analyzer = TimeSeriesAnalyzer()
        series = analyzer.load_and_prepare(df, date_column, value_column)
        return analyzer.decompose_series(series)
    except Exception as e:
        return {"error": str(e)}
