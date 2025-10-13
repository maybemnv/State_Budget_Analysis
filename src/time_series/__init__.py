"""
Time Series Analysis Module

This module provides functionality for time series analysis, including
data preprocessing, decomposition, stationarity testing, and forecasting.
"""

from .analyzer import TimeSeriesAnalyzer
from .preprocessing import clean_and_interpolate, infer_frequency
from .decomposition import decompose_series
from .stationarity import check_stationarity
from .forecasting import fit_arima, forecast_arima, fit_prophet
from .visualization import plot_forecast

__all__ = [
    'TimeSeriesAnalyzer',
    'clean_and_interpolate',
    'infer_frequency',
    'decompose_series',
    'check_stationarity',
    'fit_arima',
    'forecast_arima',
    'fit_prophet',
    'plot_forecast'
]
