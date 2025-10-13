"""
Time Series Visualization Module

This module provides functions for visualizing time series data,
including actuals, forecasts, and model diagnostics.
"""
from typing import Dict, Any, Optional, Tuple, List, Union
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def plot_forecast(actual: pd.Series, 
                 forecast: Optional[pd.Series] = None, 
                 conf_int: Optional[pd.DataFrame] = None,
                 title: str = 'Time Series Forecast',
                 xlabel: str = 'Date',
                 ylabel: str = 'Value') -> go.Figure:
    """
    Create an interactive plot of actual values and forecast.
    
    Args:
        actual: Series of actual values with datetime index
        forecast: Series of forecasted values
        conf_int: DataFrame with 'lower' and 'upper' columns for confidence intervals
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        
    Returns:
        Plotly Figure object
    """
    if not isinstance(actual, pd.Series):
        actual = pd.Series(actual)
    
    fig = go.Figure()
    
    # Plot actual values
    fig.add_trace(go.Scatter(
        x=actual.index,
        y=actual,
        mode='lines+markers',
        name='Actual',
        line=dict(color='#1f77b4')
    ))
    
    # Plot forecast if provided
    if forecast is not None:
        if not isinstance(forecast, pd.Series):
            forecast = pd.Series(forecast, index=actual.index[-len(forecast):])
        
        # Add forecast line
        fig.add_trace(go.Scatter(
            x=forecast.index,
            y=forecast,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#ff7f0e', dash='dash')
        ))
        
        # Add confidence interval if provided
        if conf_int is not None:
            if not isinstance(conf_int, pd.DataFrame):
                conf_int = pd.DataFrame(conf_int, index=forecast.index)
            
            # Ensure we have both lower and upper bounds
            if 'lower' in conf_int.columns and 'upper' in conf_int.columns:
                fig.add_trace(go.Scatter(
                    x=conf_int.index.tolist() + conf_int.index[::-1].tolist(),
                    y=conf_int['upper'].tolist() + conf_int['lower'][::-1].tolist(),
                    fill='toself',
                    fillcolor='rgba(255, 127, 14, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=False,
                    name='Confidence Interval'
                ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        template='plotly_white',
        showlegend=True,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Add range slider
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    
    return fig

def plot_decomposition(observed: pd.Series, 
                     trend: pd.Series, 
                     seasonal: pd.Series, 
                     resid: pd.Series,
                     title: str = 'Time Series Decomposition') -> go.Figure:
    """
    Create a decomposition plot showing observed, trend, seasonal, and residual components.
    
    Args:
        observed: Observed time series
        trend: Trend component
        seasonal: Seasonal component
        resid: Residual component
        title: Plot title
        
    Returns:
        Plotly Figure object with subplots
    """
    fig = make_subplots(
        rows=4, 
        cols=1, 
        subplot_titles=('Observed', 'Trend', 'Seasonal', 'Residuals'),
        vertical_spacing=0.08
    )
    
    # Observed
    fig.add_trace(
        go.Scatter(x=observed.index, y=observed, name='Observed'),
        row=1, col=1
    )
    
    # Trend
    fig.add_trace(
        go.Scatter(x=trend.index, y=trend, name='Trend'),
        row=2, col=1
    )
    
    # Seasonal
    fig.add_trace(
        go.Scatter(x=seasonal.index, y=seasonal, name='Seasonal'),
        row=3, col=1
    )
    
    # Residuals
    fig.add_trace(
        go.Scatter(x=resid.index, y=resid, name='Residuals'),
        row=4, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text=title,
        showlegend=False,
        template='plotly_white',
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Update y-axis titles
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Trend", row=2, col=1)
    fig.update_yaxes(title_text="Seasonal", row=3, col=1)
    fig.update_yaxes(title_text="Residual", row=4, col=1)
    
    return fig

def plot_acf_pacf(series: pd.Series, 
                 lags: int = 30, 
                 title: str = 'ACF and PACF Plots') -> go.Figure:
    """
    Create ACF and PACF plots for time series analysis.
    
    Args:
        series: Time series data
        lags: Number of lags to include in the plots
        title: Plot title
        
    Returns:
        Plotly Figure object with ACF and PACF subplots
    """
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    import matplotlib.pyplot as plt
    from plotly.tools import mpl_to_plotly
    
    # Create matplotlib figure with ACF and PACF
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot ACF
    plot_acf(series, lags=lags, ax=ax1, title='')
    ax1.set_title('Autocorrelation (ACF)')
    
    # Plot PACF
    plot_pacf(series, lags=lags, ax=ax2, title='', method='ywm')
    ax2.set_title('Partial Autocorrelation (PACF)')
    
    plt.tight_layout()
    
    # Convert to plotly
    plotly_fig = mpl_to_plotly(fig)
    plotly_fig.update_layout(title=title)
    
    return plotly_fig
