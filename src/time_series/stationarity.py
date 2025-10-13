"""
Stationarity Testing Module

This module provides functions for testing the stationarity of time series data
using statistical tests like ADF and KPSS.
"""
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss

def check_stationarity(series: pd.Series, alpha: float = 0.05) -> Dict[str, Any]:
    """
    Check stationarity of a time series using ADF and KPSS tests.
    
    Args:
        series: Time series to test
        alpha: Significance level for the tests
        
    Returns:
        Dictionary containing test results and stationarity conclusion
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    
    # Remove any remaining NaNs
    series = series.dropna()
    
    if len(series) < 10:  # Minimum number of observations for meaningful tests
        raise ValueError("Insufficient data points for stationarity testing")
    
    # ADF Test
    adf_result = adfuller(series, autolag='AIC')
    adf_stat, adf_pvalue = adf_result[0], adf_result[1]
    adf_critical_values = adf_result[4]
    
    # KPSS Test
    try:
        kpss_result = kpss(series, regression='c', nlags='auto')
        kpss_stat, kpss_pvalue = kpss_result[0], kpss_result[1]
        kpss_critical_values = kpss_result[3]
    except:
        # KPSS might fail for some series, especially if there are NaNs or Infs
        kpss_stat, kpss_pvalue = np.nan, np.nan
        kpss_critical_values = {}
    
    # Determine stationarity based on both tests
    is_stationary = False
    reason = []
    
    # ADF: Null hypothesis is that the series has a unit root (non-stationary)
    if adf_pvalue <= alpha:
        reason.append("ADF test indicates stationarity (p-value: {:.4f})".format(adf_pvalue))
    else:
        reason.append("ADF test indicates non-stationarity (p-value: {:.4f})".format(adf_pvalue))
    
    # KPSS: Null hypothesis is that the series is stationary
    if not np.isnan(kpss_pvalue):
        if kpss_pvalue > alpha:
            reason.append("KPSS test indicates stationarity (p-value: {:.4f})".format(kpss_pvalue))
            # If both ADF and KPSS suggest stationarity, we're more confident
            if adf_pvalue <= alpha:
                is_stationary = True
        else:
            reason.append("KPSS test indicates non-stationarity (p-value: {:.4f})".format(kpss_pvalue))
    else:
        reason.append("KPSS test could not be performed")
    
    # If tests disagree, we'll be conservative and say it's not stationary
    if not is_stationary and adf_pvalue <= alpha and (np.isnan(kpss_pvalue) or kpss_pvalue > alpha):
        reason.append("Tests disagree - assuming non-stationarity")
    
    return {
        'is_stationary': is_stationary,
        'reason': ' '.join(reason),
        'adf': {
            'statistic': adf_stat,
            'pvalue': adf_pvalue,
            'critical_values': adf_critical_values,
            'stationary': adf_pvalue <= alpha
        },
        'kpss': {
            'statistic': kpss_stat,
            'pvalue': kpss_pvalue,
            'critical_values': kpss_critical_values,
            'stationary': not np.isnan(kpss_pvalue) and kpss_pvalue > alpha
        },
        'conclusion': 'Stationary' if is_stationary else 'Non-stationary'
    }

def make_stationary(series: pd.Series, method: str = 'diff', **kwargs) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Apply transformations to make a time series stationary.
    
    Args:
        series: Input time series
        method: Transformation method ('diff' for differencing, 'log' for log transform, 'log_diff' for log then diff)
        **kwargs: Additional arguments for the transformation
            - log_shift: Small value to add before log transform (default: 1e-6)
            - periods: Number of periods to difference (default: 1)
    
    Returns:
        Tuple of (transformed_series, transformation_info)
    """
    transformation_info = {
        'method': method,
        'parameters': kwargs,
        'stationarity_before': check_stationarity(series)
    }
    
    transformed = series.copy()
    
    # Apply transformation
    if method == 'diff':
        periods = kwargs.get('periods', 1)
        transformed = transformed.diff(periods=periods).dropna()
        transformation_info['description'] = f"Differenced with period {periods}"
    
    elif method == 'log':
        shift = kwargs.get('log_shift', 1e-6)
        transformed = np.log(transformed + shift)
        transformation_info['description'] = f"Log transform with shift={shift}"
    
    elif method == 'log_diff':
        shift = kwargs.get('log_shift', 1e-6)
        periods = kwargs.get('periods', 1)
        transformed = np.log(transformed + shift).diff(periods=periods).dropna()
        transformation_info['description'] = f"Log transform (shift={shift}) then differenced with period {periods}"
    
    elif method == 'pct_change':
        periods = kwargs.get('periods', 1)
        transformed = transformed.pct_change(periods=periods).dropna()
        transformation_info['description'] = f"Percentage change with period {periods}"
    
    else:
        raise ValueError(f"Unknown transformation method: {method}")
    
    # Check stationarity after transformation
    transformation_info['stationarity_after'] = check_stationarity(transformed)
    
    return transformed, transformation_info
