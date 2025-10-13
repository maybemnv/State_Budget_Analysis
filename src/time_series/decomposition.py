"""
Time Series Decomposition Module

This module provides functions for decomposing time series data into trend,
seasonal, and residual components.
"""
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def decompose_series(series: pd.Series, model: str = 'additive', 
                   period: Optional[int] = None) -> Dict[str, Any]:
    """
    Decompose a time series into trend, seasonal, and residual components.
    
    Args:
        series: Input time series with datetime index
        model: Decomposition model ('additive' or 'multiplicative')
        period: Seasonal period. If None, will try to infer from the data
        
    Returns:
        Dictionary containing:
        - observed: Original series
        - trend: Trend component
        - seasonal: Seasonal component
        - resid: Residual component
        - period: Detected/used seasonal period
        - plot: Plotly figure object
    """
    if not isinstance(series, pd.Series):
        raise ValueError("Input must be a pandas Series")
    
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Series must have a DatetimeIndex")
    
    # Make sure the series is regularly spaced
    series = series.asfreq(series.index.inferred_freq or 'D')
    
    # Infer period if not provided
    if period is None:
        # Try to infer seasonality from the data
        # Common periods for different frequencies
        freq = series.index.inferred_freq
        if freq == 'D':  # Daily data
            period = 7  # Weekly seasonality
        elif freq in ['W', 'W-SUN', 'W-MON']:  # Weekly data
            period = 52  # Yearly seasonality
        elif freq == 'M':  # Monthly data
            period = 12  # Yearly seasonality
        elif freq == 'Q':  # Quarterly data
            period = 4  # Yearly seasonality
        else:
            # Default to 12 for monthly/quarterly data, 7 for daily/weekly
            period = 12 if len(series) > 365 else 7
    
    # Make sure we have enough data points for decomposition
    min_length = 2 * period
    if len(series) < min_length:
        raise ValueError(f"Need at least {min_length} observations for decomposition with period {period}")
    
    try:
        # Perform decomposition
        decomposition = seasonal_decompose(
            series.dropna(),  # Remove any remaining NaNs
            model=model,
            period=period,
            extrapolate_trend=True
        )
        
        # Create a plot
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=('Observed', 'Trend', 'Seasonal', 'Residuals'),
            vertical_spacing=0.1
        )
        
        # Add traces
        fig.add_trace(
            go.Scatter(x=series.index, y=decomposition.observed, name='Observed'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=series.index, y=decomposition.trend, name='Trend'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=series.index, y=decomposition.seasonal, name='Seasonal'),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=series.index, y=decomposition.resid, name='Residual'),
            row=4, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text=f"Time Series Decomposition ({model.capitalize()} Model)",
            showlegend=False
        )
        
        # Update y-axis titles
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Trend", row=2, col=1)
        fig.update_yaxes(title_text="Seasonal", row=3, col=1)
        fig.update_yaxes(title_text="Residual", row=4, col=1)
        
        # Return results
        return {
            'observed': decomposition.observed,
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'resid': decomposition.resid,
            'period': period,
            'model': model,
            'plot': fig
        }
        
    except Exception as e:
        raise ValueError(f"Error in decomposition: {str(e)}")
