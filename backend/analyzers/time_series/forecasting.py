import warnings
from typing import Optional, Tuple
import pandas as pd

warnings.filterwarnings("ignore")

# ─── Prophet: optional dependency with graceful fallback ────────────
try:
    from prophet import Prophet
    _HAS_PROPHET = True
except ImportError:
    Prophet = None
    _HAS_PROPHET = False


def fit_arima(
    series: pd.Series,
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Optional[Tuple[int, int, int, int]] = None,
) -> dict:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.arima.model import ARIMA

    series = series.asfreq(series.index.inferred_freq or "D").dropna()
    if len(series) < 10:
        raise ValueError("Need at least 10 observations for ARIMA")

    if seasonal_order:
        model = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
    else:
        model = ARIMA(series, order=order)

    fit = model.fit()
    return {
        "model_type": "SARIMAX" if seasonal_order else "ARIMA",
        "order": order,
        "seasonal_order": seasonal_order,
        "aic": round(float(fit.aic), 2),
        "bic": round(float(fit.bic), 2),
        "nobs": int(fit.nobs),
        "_fit": fit,
    }


def forecast_arima(fit_result: dict, steps: int, alpha: float = 0.05) -> dict:
    fit = fit_result["_fit"]
    fc = fit.get_forecast(steps=steps)
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=alpha)
    return {
        "forecast": mean.tolist(),
        "lower": ci.iloc[:, 0].tolist(),
        "upper": ci.iloc[:, 1].tolist(),
        "index": [str(d) for d in mean.index],
        "steps": steps,
    }


def fit_prophet(series: pd.Series) -> dict:
    """Fit Prophet model. Falls back to ARIMA if Prophet is not installed."""
    if not _HAS_PROPHET:
        # Graceful fallback: fit ARIMA and return compatible structure
        arima_result = fit_arima(series)
        fc = forecast_arima(arima_result, steps=0)
        return {
            "model_type": "ARIMA (Prophet unavailable)",
            "fitted_values": fc.get("forecast", []),
            "index": [str(d) for d in series.index],
            "_fit": arima_result["_fit"],
            "_series_freq": series.index.freqstr if hasattr(series.index, "freqstr") else None,
        }

    df = pd.DataFrame({"ds": series.index, "y": series.values})
    model = Prophet(daily_seasonality=False)
    model.fit(df)

    future = model.make_future_dataframe(periods=0)
    forecast = model.predict(future)

    return {
        "model_type": "Prophet",
        "fitted_values": forecast["yhat"].round(4).tolist(),
        "index": [str(d) for d in series.index],
        "_model": model,
        "_series_freq": series.index.freqstr if hasattr(series.index, "freqstr") else None,
    }


def forecast_prophet(fit_result: dict, steps: int) -> dict:
    """Forecast using Prophet or fallback ARIMA model."""
    # If it was an ARIMA fallback
    if "_fit" in fit_result and fit_result.get("model_type", "").startswith("ARIMA"):
        return forecast_arima(fit_result, steps=steps)

    model = fit_result["_model"]
    freq = fit_result.get("_series_freq") or "ME"
    future = model.make_future_dataframe(periods=steps, freq=freq)
    forecast = model.predict(future)
    fc = forecast.tail(steps)
    return {
        "forecast": fc["yhat"].round(4).tolist(),
        "lower": fc["yhat_lower"].round(4).tolist(),
        "upper": fc["yhat_upper"].round(4).tolist(),
        "index": [str(d) for d in fc["ds"]],
        "steps": steps,
    }
