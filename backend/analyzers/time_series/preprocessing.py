from typing import Optional
import pandas as pd
import numpy as np
from .date_utils import parse_financial_year, extract_year_columns


def clean_and_interpolate(series: pd.Series, method: str = "linear", limit: int = 5) -> pd.Series:
    cleaned = series.copy()
    if not pd.api.types.is_numeric_dtype(cleaned):
        cleaned = pd.to_numeric(cleaned, errors="coerce")
    if len(cleaned) > 10:
        q1, q3 = cleaned.quantile(0.25), cleaned.quantile(0.75)
        iqr = q3 - q1
        cleaned = cleaned.clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    if method == "time":
        cleaned = cleaned.interpolate(method="time", limit=limit, limit_direction="both")
    else:
        cleaned = cleaned.interpolate(method=method, limit=limit, limit_direction="both")
    return cleaned.ffill().bfill()


def infer_frequency(series: pd.Series) -> str:
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Series must have a DatetimeIndex")
    diffs = series.index.to_series().diff().dropna()
    if diffs.empty:
        return "D"
    mode_diff = diffs.mode()
    if mode_diff.empty:
        return "D"
    days = mode_diff.iloc[0].days
    if days == 1:
        return "D"
    if 28 <= days <= 31:
        return "ME"
    if 365 <= days <= 366:
        return "YE"
    return f"{days}D"
