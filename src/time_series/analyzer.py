"""
Unified Time Series Analyzer

High-level wrapper that orchestrates preprocessing, decomposition, stationarity
testing, and forecasting across multiple data shapes (date column, year
columns, or sequential numeric data). This class composes functions from the
modular submodules in this package to provide a single, coherent API.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import re
import pandas as pd

from .preprocessing import clean_and_interpolate, infer_frequency
from .decomposition import decompose_series as _decompose_series
from .stationarity import check_stationarity as _check_stationarity
from .forecasting import fit_arima as _fit_arima, forecast_arima as _forecast_arima, fit_prophet as _fit_prophet
from .visualization import plot_forecast as _plot_forecast


class TimeSeriesAnalyzer:
    def __init__(self) -> None:
        self.frequency: Optional[str] = None
        self.model = None
        self.forecast_df: Optional[pd.DataFrame] = None

    # ---------- Data detection and shaping ----------
    def detect_time_series_capable_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect whether a dataframe can be used for time series analysis.
        Supports:
          - Explicit datetime columns
          - Year-like columns (2019, 2020, FY2020-21)
          - Sequential numeric columns for synthetic dating
        """
        suggestions: Dict[str, Any] = {
            'has_date_columns': False,
            'has_year_columns': False,
            'has_sequential_index': False,
            'date_columns': [],
            'year_columns': [],
            'numeric_columns': [],
        }

        # Datetime columns
        datetime_cols = df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns.tolist()
        if datetime_cols:
            suggestions['has_date_columns'] = True
            suggestions['date_columns'] = datetime_cols

        # String/object columns that parse as datetime in majority of sample
        for col in df.select_dtypes(include=['object']).columns:
            sample = df[col].dropna().astype(str).head(25)
            if sample.empty:
                continue
            parsed = pd.to_datetime(sample, errors='coerce', infer_datetime_format=True)
            if parsed.notna().mean() >= 0.8:  # 80% parsable
                suggestions['has_date_columns'] = True
                suggestions['date_columns'].append(col)

            # Year-like labels in values
            year_like = sample.str.fullmatch(r"\d{4}").any() or sample.str.contains(r"FY\d{2}|FY\d{4}|\d{4}[-/]\d{2}", regex=True).any()
            if year_like and col not in suggestions['year_columns']:
                suggestions['has_year_columns'] = True
                suggestions['year_columns'].append(col)

        # Year-like columns by header name (wide format budget tables)
        for col in df.columns:
            col_str = str(col).strip().upper()
            if re.fullmatch(r"\d{4}", col_str):
                suggestions['has_year_columns'] = True
                suggestions['year_columns'].append(col)
            if re.search(r"FY|FISCAL|FINANCIAL", col_str):
                if col not in suggestions['year_columns']:
                    suggestions['has_year_columns'] = True
                    suggestions['year_columns'].append(col)

        # Sequential index
        idx = df.index
        suggestions['has_sequential_index'] = getattr(idx, 'is_monotonic_increasing', False)

        # Numeric columns
        suggestions['numeric_columns'] = df.select_dtypes(include=['number']).columns.tolist()

        return suggestions

    def create_time_series_from_year_columns(self, df: pd.DataFrame, year_columns: List[str], row_selector: Optional[str] = None) -> Tuple[Optional[pd.Series], Optional[str]]:
        """
        Build a Series from wide year columns (2019, 2020, ...). If row_selector
        is provided and matches an index value, that row is used; otherwise the
        first row is used.
        """
        try:
            work = df.copy()
            if row_selector is not None:
                if row_selector in work.index:
                    values = work.loc[row_selector, year_columns]
                else:
                    # try matching a column label instead
                    if row_selector in work.columns:
                        values = work[row_selector]
                        values = values.reindex(year_columns)
                    else:
                        values = work[year_columns].iloc[0]
            else:
                values = work[year_columns].iloc[0]

            values = pd.to_numeric(values, errors='coerce')
            dates = pd.to_datetime([f"{c}-01-01" for c in year_columns])
            series = pd.Series(values.values, index=dates, name=row_selector or 'Value')
            series = series.sort_index()
            return series, None
        except Exception as e:
            return None, f"Error creating time series from year columns: {str(e)}"

    def create_time_series_from_sequential(self, df: pd.DataFrame, value_column: str, freq: str = 'M', start_date: str = '2000-01-01') -> Tuple[Optional[pd.Series], Optional[str]]:
        try:
            values = pd.to_numeric(df[value_column], errors='coerce')
            dates = pd.date_range(start=start_date, periods=len(values), freq=freq)
            series = pd.Series(values.values, index=dates, name=value_column)
            return series, None
        except Exception as e:
            return None, f"Error creating sequential time series: {str(e)}"

    def load_and_prepare(self, df: pd.DataFrame, date_column: str, value_column: str, interpolate_method: str = 'time') -> pd.Series:
        work = df.copy()
        work[date_column] = pd.to_datetime(work[date_column], errors='coerce')
        work = work.dropna(subset=[date_column])
        work = work.set_index(date_column).sort_index()
        series = pd.to_numeric(work[value_column], errors='coerce')
        series = clean_and_interpolate(series, method=interpolate_method)
        self.frequency = infer_frequency(series)
        return series

    # ---------- Analysis primitives ----------
    def decompose_series(self, series: pd.Series, model: str = 'additive', period: Optional[int] = None):
        return _decompose_series(series, model=model, period=period)

    def check_stationarity(self, series: pd.Series):
        return _check_stationarity(series)

    # ---------- Forecasting ----------
    def fit_arima(self, series: pd.Series, order: Tuple[int, int, int] = (1, 1, 1), seasonal_order: Optional[Tuple[int, int, int, int]] = None, **kwargs):
        return _fit_arima(series, order=order, seasonal_order=seasonal_order, **kwargs)

    def forecast_arima(self, model_result: Any, steps: int, alpha: float = 0.05):
        return _forecast_arima(model_result, steps=steps, alpha=alpha)

    def fit_prophet(self, series: pd.Series, **kwargs):
        return _fit_prophet(series, **kwargs)

    # ---------- Visualization ----------
    def plot_forecast(self, actual: pd.Series, forecast: Optional[pd.Series] = None, conf_int: Optional[pd.DataFrame] = None, title: str = 'Forecast'):
        return _plot_forecast(actual, forecast, conf_int, title=title)


