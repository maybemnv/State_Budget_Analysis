import warnings
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")


def check_stationarity(series: pd.Series, alpha: float = 0.05) -> dict:
    series = series.dropna()
    if len(series) < 10:
        raise ValueError("Need at least 10 data points for stationarity tests")

    adf_stat, adf_p, _, _, adf_cv, _ = adfuller(series, autolag="AIC")

    try:
        kpss_stat, kpss_p, _, kpss_cv = kpss(series, regression="c", nlags="auto")
    except Exception:
        kpss_stat, kpss_p, kpss_cv = float("nan"), float("nan"), {}

    adf_stationary = bool(adf_p <= alpha)
    kpss_stationary = bool(not np.isnan(kpss_p) and kpss_p > alpha)
    is_stationary = adf_stationary and kpss_stationary

    return {
        "is_stationary": is_stationary,
        "conclusion": "Stationary" if is_stationary else "Non-stationary",
        "adf": {
            "statistic": round(float(adf_stat), 4),
            "p_value": round(float(adf_p), 4),
            "critical_values": {k: round(v, 4) for k, v in adf_cv.items()},
            "stationary": adf_stationary,
        },
        "kpss": {
            "statistic": round(float(kpss_stat), 4) if not np.isnan(kpss_stat) else None,
            "p_value": round(float(kpss_p), 4) if not np.isnan(kpss_p) else None,
            "critical_values": {k: round(v, 4) for k, v in kpss_cv.items()},
            "stationary": kpss_stationary,
        },
    }
