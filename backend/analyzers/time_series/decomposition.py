from typing import Optional
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose


def decompose_series(series: pd.Series, model: str = "additive", period: Optional[int] = None) -> dict:
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Series must have a DatetimeIndex")

    series = series.asfreq(series.index.inferred_freq or "D")

    if period is None:
        freq = series.index.inferred_freq or ""
        freq_up = freq.upper()
        if freq_up.startswith("D"):
            period = 7
        elif freq_up.startswith("W"):
            period = 52
        elif freq_up.startswith("M") or freq_up == "ME":
            period = 12
        elif freq_up.startswith("Q"):
            period = 4
        else:
            period = 12 if len(series) > 365 else 7

    if len(series) < 2 * period:
        raise ValueError(f"Need at least {2 * period} observations for period={period}")

    result = seasonal_decompose(series.dropna(), model=model, period=period, extrapolate_trend=True)

    return {
        "observed": result.observed.dropna().tolist(),
        "trend": result.trend.dropna().tolist(),
        "seasonal": result.seasonal.dropna().tolist(),
        "resid": result.resid.dropna().tolist(),
        "index": [str(d) for d in result.observed.index],
        "period": period,
        "model": model,
    }
