import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class TimeSeriesAnalyzer:
    def __init__(self):
        self.model = None
        self.forecast = None
        
    def load_and_prepare_data(self, df, date_column, value_column):
        """
        Prepare data for time series analysis
        """
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.set_index(date_column)
        df = df.sort_index()
        return df[value_column]
        
    def decompose_series(self, series):
        """
        Decompose time series into trend, seasonal, and residual components
        """
        decomposition = seasonal_decompose(series, period=12, model='additive')
        
        fig = make_subplots(rows=4, cols=1, subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'))
        
        # Original
        fig.add_trace(go.Scatter(x=series.index, y=series.values, name='Original'), row=1, col=1)
        # Trend
        fig.add_trace(go.Scatter(x=series.index, y=decomposition.trend, name='Trend'), row=2, col=1)
        # Seasonal
        fig.add_trace(go.Scatter(x=series.index, y=decomposition.seasonal, name='Seasonal'), row=3, col=1)
        # Residual
        fig.add_trace(go.Scatter(x=series.index, y=decomposition.resid, name='Residual'), row=4, col=1)
        
        fig.update_layout(height=900, title_text="Time Series Decomposition")
        return fig
        
    def check_stationarity(self, series):
        """
        Perform Augmented Dickey-Fuller test for stationarity
        """
        result = adfuller(series.dropna())
        return {
            'test_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4]
        }
        
    def fit_arima(self, series, order=(1,1,1)):
        """
        Fit ARIMA model and generate forecast
        """
        model = ARIMA(series, order=order)
        self.model = model.fit()
        return self.model
        
    def forecast_arima(self, steps=12):
        """
        Generate forecast using fitted ARIMA model
        """
        if self.model is None:
            raise ValueError("Model must be fitted first")
            
        forecast = self.model.forecast(steps=steps)
        return forecast
        
    def fit_prophet(self, series, future_periods=12):
        """
        Fit Prophet model and generate forecast
        """
        # Prepare data for Prophet
        df = pd.DataFrame({'ds': series.index, 'y': series.values})
        
        # Initialize and fit Prophet model
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model.fit(df)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=future_periods, freq='M')
        
        # Generate forecast
        self.forecast = model.predict(future)
        
        return self.forecast
        
    def plot_forecast(self, original_series, forecast_df, title="Forecast"):
        """
        Plot the original data and forecast
        """
        fig = go.Figure()
        
        # Plot original data
        fig.add_trace(go.Scatter(
            x=original_series.index,
            y=original_series.values,
            name='Historical Data',
            mode='lines'
        ))
        
        # Plot forecast
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['yhat'],
            name='Forecast',
            mode='lines',
            line=dict(dash='dash')
        ))
        
        # Add confidence intervals
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['yhat_upper'],
            fill=None,
            mode='lines',
            line_color='rgba(0,100,80,0.2)',
            name='Upper Bound'
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['yhat_lower'],
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,100,80,0.2)',
            name='Lower Bound'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Value",
            hovermode="x unified"
        )
        
        return fig