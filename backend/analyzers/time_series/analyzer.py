from __future__ import annotations
from typing import Optional, Tuple
import re
import pandas as pd

from .preprocessing import clean_and_interpolate, infer_frequency
from .decomposition import decompose_series
from .stationarity import check_stationarity
from .forecasting import fit_arima, forecast_arima, fit_prophet, forecast_prophet


class TimeSeriesAnalyzer:
    """High-level orchestrator for time series workflows."""

    def __init__(self) -> None:
        self.frequency: Optional[str] = None

    def detect_time_series_capable_data(self, df: pd.DataFrame) -> dict:
        result: dict = {
            "has_date_columns": False,
            "has_year_columns": False,
            "date_columns": [],
            "year_columns": [],
            "numeric_columns": df.select_dtypes(include="number").columns.tolist(),
        }

        # Explicit datetime columns
        dt_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
        if dt_cols:
            result["has_date_columns"] = True
            result["date_columns"].extend(dt_cols)

        # String columns parseable as datetime
        for col in df.select_dtypes(include="object").columns:
            sample = df[col].dropna().astype(str).head(25)
            if sample.empty:
                continue
            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().mean() >= 0.8 and col not in result["date_columns"]:
                result["has_date_columns"] = True
                result["date_columns"].append(col)

            year_like = (
                sample.str.fullmatch(r"\d{4}").any()
                or sample.str.contains(r"FY\d{2}|FY\d{4}|\d{4}[-/]\d{2}", regex=True).any()
            )
            if year_like and col not in result["year_columns"]:
                result["has_year_columns"] = True
                result["year_columns"].append(col)

        # Wide-format year column headers
        for col in df.columns:
            col_str = str(col).strip().upper()
            if re.fullmatch(r"\d{4}", col_str) and col not in result["year_columns"]:
                result["has_year_columns"] = True
                result["year_columns"].append(col)
            elif re.search(r"FY|FISCAL|FINANCIAL", col_str) and col not in result["year_columns"]:
                result["has_year_columns"] = True
                result["year_columns"].append(col)

        return result

    def create_time_series_from_year_columns(
        self, df: pd.DataFrame, year_columns: list[str], row_selector: Optional[str] = None
    ) -> Tuple[Optional[pd.Series], Optional[str]]:
        try:
            work = df.copy()
            if row_selector is not None and row_selector in work.index:
                values = work.loc[row_selector, year_columns]
            else:
                values = work[year_columns].iloc[0]
            values = pd.to_numeric(values, errors="coerce")
            dates = pd.to_datetime([f"{c}-01-01" for c in year_columns])
            return pd.Series(values.values, index=dates, name=row_selector or "Value").sort_index(), None
        except Exception as e:
            return None, str(e)

    def create_time_series_from_sequential(
        self, df: pd.DataFrame, value_column: str, freq: str = "ME", start_date: str = "2000-01-01"
    ) -> Tuple[Optional[pd.Series], Optional[str]]:
        try:
            values = pd.to_numeric(df[value_column], errors="coerce")
            dates = pd.date_range(start=start_date, periods=len(values), freq=freq)
            return pd.Series(values.values, index=dates, name=value_column), None
        except Exception as e:
            return None, str(e)

    def load_and_prepare(
        self, df: pd.DataFrame, date_column: str, value_column: str, interpolate_method: str = "time"
    ) -> pd.Series:
        work = df.copy()
        work[date_column] = pd.to_datetime(work[date_column], errors="coerce")
        work = work.dropna(subset=[date_column]).set_index(date_column).sort_index()
        series = pd.to_numeric(work[value_column], errors="coerce")
        series = clean_and_interpolate(series, method=interpolate_method)
        self.frequency = infer_frequency(series)
        return series

    def decompose_series(self, series: pd.Series, model: str = "additive", period: Optional[int] = None) -> dict:
        return decompose_series(series, model=model, period=period)

    def check_stationarity(self, series: pd.Series) -> dict:
        return check_stationarity(series)

    def fit_arima(self, series: pd.Series, order=(1, 1, 1), seasonal_order=None) -> dict:
        return fit_arima(series, order=order, seasonal_order=seasonal_order)

    def forecast_arima(self, fit_result: dict, steps: int, alpha: float = 0.05) -> dict:
        return forecast_arima(fit_result, steps=steps, alpha=alpha)

    def fit_prophet(self, series: pd.Series) -> dict:
        return fit_prophet(series)

    def forecast_prophet(self, fit_result: dict, steps: int) -> dict:
        return forecast_prophet(fit_result, steps=steps)
