"""
Time Series Preprocessing Module

This module provides functions for cleaning, transforming, and preparing
time series data for analysis.
"""
from typing import Optional, Tuple, Dict, Any, Union
import pandas as pd
import numpy as np
from ..utils.date_utils import parse_financial_year, detect_date_format, extract_year_columns

def clean_and_interpolate(series: pd.Series, freq: Optional[str] = None, 
                         method: str = 'linear', limit: int = 5) -> pd.Series:
    """
    Clean and interpolate missing values in a time series.
    
    Args:
        series: Input time series with datetime index
        freq: Frequency string (e.g., 'D' for daily, 'M' for monthly)
        method: Interpolation method ('linear', 'time', 'spline', etc.)
        limit: Maximum number of consecutive NaNs to fill
        
    Returns:
        Cleaned and interpolated time series
    """
    if not isinstance(series, pd.Series):
        raise ValueError("Input must be a pandas Series")
    
    # Make a copy to avoid modifying the original
    cleaned = series.copy()
    
    # Convert to numeric, coercing errors to NaN
    if not pd.api.types.is_numeric_dtype(cleaned):
        cleaned = pd.to_numeric(cleaned, errors='coerce')
    
    # Handle outliers using IQR method
    if len(cleaned) > 10:  # Only if we have enough data points
        q1 = cleaned.quantile(0.25)
        q3 = cleaned.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Cap outliers instead of removing them
        cleaned = cleaned.clip(lower_bound, upper_bound)
    
    # Interpolate missing values
    if method == 'time':
        cleaned = cleaned.interpolate(method='time', limit=limit, limit_direction='both')
    else:
        cleaned = cleaned.interpolate(method=method, limit=limit, limit_direction='both')
    
    # Forward/backward fill any remaining NaNs at the ends
    cleaned = cleaned.ffill().bfill()
    
    return cleaned

def infer_frequency(series: pd.Series) -> str:
    """
    Infer the frequency of a time series.
    
    Args:
        series: Time series with datetime index
        
    Returns:
        Inferred frequency string (e.g., 'D', 'M', 'Y')
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Series must have a DatetimeIndex")
    
    # Calculate time differences between consecutive dates
    diffs = series.index.to_series().diff().dropna()
    
    if len(diffs) == 0:
        return 'D'  # Default to daily if we can't determine
    
    # Get the most common time difference
    mode_diff = diffs.mode()
    if not mode_diff.empty:
        mode_diff = mode_diff[0]
    else:
        return 'D'  # Default to daily if no mode
    
    # Convert to frequency string
    days = mode_diff.days
    seconds = mode_diff.seconds
    
    if days > 0:
        if days == 1:
            return 'D'  # Daily
        elif 28 <= days <= 31:
            return 'M'  # Monthly
        elif 365 <= days <= 366:
            return 'Y'  # Yearly
        else:
            return f'{days}D'  # Custom day frequency
    elif seconds > 0:
        if seconds % 86400 == 0:
            return f'{seconds//86400}D'  # Multiple days
        elif seconds % 3600 == 0:
            return f'{seconds//3600}H'  # Hourly
        elif seconds % 60 == 0:
            return f'{seconds//60}T'  # Minutely
        else:
            return f'{seconds}S'  # Secondly
    
    return 'D'  # Default to daily
