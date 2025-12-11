"""
Time Series Forecasting Module

This module provides functions for time series forecasting using various models
like ARIMA, SARIMA, and Prophet.
"""
from typing import Dict, Any, Tuple, Optional, Union
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima.model import ARIMA
try:
    from prophet import Prophet
except ImportError:
    Prophet = None
import warnings
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=Warning)

def fit_arima(series: pd.Series, order: Tuple[int, int, int], 
             seasonal_order: Optional[Tuple[int, int, int, int]] = None, 
             **kwargs) -> Dict[str, Any]:
    """
    Fit an ARIMA or SARIMA model to the time series.
    
    Args:
        series: Time series data
        order: (p,d,q) order of the ARIMA model
        seasonal_order: (P,D,Q,s) seasonal order of the SARIMA model
        **kwargs: Additional arguments to pass to the ARIMA/SARIMAX model
        
    Returns:
        Dictionary containing the fitted model and model information
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Series must have a DatetimeIndex")
    
    # Make sure the series is regularly spaced
    series = series.asfreq(series.index.inferred_freq or 'D')
    
    # Remove any remaining NaNs
    series = series.dropna()
    
    if len(series) < 10:  # Minimum number of observations
        raise ValueError("Insufficient data points for ARIMA modeling")
    
    model_info = {
        'model_type': 'SARIMAX' if seasonal_order else 'ARIMA',
        'order': order,
        'seasonal_order': seasonal_order,
        'nobs': len(series),
        'start_date': series.index[0],
        'end_date': series.index[-1],
        'freq': series.index.freqstr if hasattr(series.index, 'freqstr') else None
    }
    
    try:
        if seasonal_order:
            # Use SARIMAX for seasonal models
            model = SARIMAX(
                series,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
                **kwargs
            )
        else:
            # Use ARIMA for non-seasonal models
            model = ARIMA(
                series,
                order=order,
                **kwargs
            )
        
        # Fit the model
        model_fit = model.fit(disp=0)
        
        # Get model summary
        model_info['aic'] = model_fit.aic
        model_info['bic'] = model_fit.bic
        model_info['hqic'] = model_fit.hqic
        model_info['residuals'] = model_fit.resid
        model_info['fitted_values'] = model_fit.fittedvalues
        
        return {
            'model': model_fit,
            'model_info': model_info,
            'success': True,
            'message': 'Model fitted successfully'
        }
        
    except Exception as e:
        return {
            'model': None,
            'model_info': model_info,
            'success': False,
            'message': f'Error fitting model: {str(e)}'
        }

def forecast_arima(model_result, steps: int, alpha: float = 0.05) -> Dict[str, Any]:
    """
    Generate forecasts using a fitted ARIMA/SARIMA model.
    
    Args:
        model_result: Fitted ARIMA/SARIMA model
        steps: Number of steps ahead to forecast
        alpha: Significance level for confidence intervals
        
    Returns:
        Dictionary containing forecast results
    """
    if model_result is None or not hasattr(model_result, 'get_forecast'):
        raise ValueError("Invalid model result provided")
    
    try:
        # Get forecast
        forecast_result = model_result.get_forecast(steps=steps)
        forecast = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int(alpha=alpha)
        
        # Create a DataFrame with the forecast and confidence intervals
        forecast_df = pd.DataFrame({
            'forecast': forecast,
            'lower': conf_int.iloc[:, 0],
            'upper': conf_int.iloc[:, 1]
        })
        
        return {
            'forecast': forecast_df,
            'success': True,
            'message': 'Forecast generated successfully'
        }
        
    except Exception as e:
        return {
            'forecast': None,
            'success': False,
            'message': f'Error generating forecast: {str(e)}'
        }

def fit_prophet(series: pd.Series, **kwargs) -> Dict[str, Any]:
    """
    Fit a Prophet model to the time series.
    
    Args:
        series: Time series data
        **kwargs: Additional arguments to pass to the Prophet model
        
    Returns:
        Dictionary containing the fitted model and model information
    """
    if Prophet is None:
        return {
            'model': None,
            'model_info': None,
            'success': False,
            'message': 'Prophet library is not installed. Please install it using "pip install prophet" to use this feature.'
        }

    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Series must have a DatetimeIndex")
    
    # Prepare data for Prophet
    df = pd.DataFrame({
        'ds': series.index,
        'y': series.values
    })
    
    model_info = {
        'model_type': 'Prophet',
        'nobs': len(series),
        'start_date': series.index[0],
        'end_date': series.index[-1],
        'freq': series.index.freqstr if hasattr(series.index, 'freqstr') else None
    }
    
    try:
        # Initialize and fit model
        model = Prophet(**kwargs)
        model.fit(df)
        
        # Get fitted values
        forecast = model.predict(df)
        
        model_info['fitted_values'] = forecast['yhat'].values
        model_info['residuals'] = df['y'].values - forecast['yhat'].values
        
        return {
            'model': model,
            'model_info': model_info,
            'success': True,
            'message': 'Prophet model fitted successfully'
        }
        
    except Exception as e:
        return {
            'model': None,
            'model_info': model_info,
            'success': False,
            'message': f'Error fitting Prophet model: {str(e)}'
        }
