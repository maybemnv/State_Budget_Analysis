import re
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, Union, List
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ValueWarning, ConvergenceWarning
from prophet import Prophet
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

# Configure warnings
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')
warnings.filterwarnings('ignore', category=ValueWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

class TimeSeriesAnalysisError(Exception):
    """Custom exception for time series analysis errors."""
    pass

class TimeSeriesAnalyzer:
    def __init__(self):
        self.model = None
        self.forecast = None
        self.frequency = None
        self.original_data = None
        self.current_analysis = {}
        self.analysis_history = []
        
        # Common date formats to try when parsing dates
        self.date_formats = [
            '%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', '%Y/%m/%d',
            '%Y%m%d', '%d%m%Y', '%m%d%Y',
            '%Y-%m', '%Y',
            '%b %Y', '%B %Y',
            '%d-%b-%Y', '%d-%B-%Y',
            '%b-%y', '%B-%y',
            '%Y-Q%q', '%Y-Q%q',  # Quarterly formats
            'FY%y', 'FY%Y',      # Fiscal year formats
        ]
        
        # Financial year patterns (e.g., 2021-22, 2021/22, FY2021-22, etc.)
        self.financial_year_patterns = [
            (r'(\d{4})[-/](\d{2})$', lambda m: f"20{m.group(1)}-07-01"),  # 2021-22
            (r'FY(\d{2})[-/](\d{2})', lambda m: f"20{m.group(1)}-04-01"),  # FY21-22
            (r'(\d{4})-\d{2}$', lambda m: f"{m.group(1)}-04-01"),  # 2021-22 (full year)
        ]
        
    def _parse_financial_year(self, year_str: str) -> Optional[datetime]:
        """Parse financial year strings into datetime objects."""
        year_str = str(year_str).strip().upper()
        
        # Try standard date formats first
        for fmt in self.date_formats:
            try:
                return datetime.strptime(year_str, fmt)
            except ValueError:
                continue
        
        # Try financial year patterns
        for pattern, date_func in self.financial_year_patterns:
            match = re.match(pattern, year_str, re.IGNORECASE)
            if match:
                try:
                    return pd.to_datetime(date_func(match))
                except:
                    continue
        
        return None
    
    def detect_time_series_capable_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Enhanced detection of time series capable data with support for various date formats.
        
        Args:
            df: Input DataFrame to analyze
            
        Returns:
            Dict containing detection results with the following keys:
            - has_date_columns: bool
            - has_year_columns: bool
            - has_sequential_index: bool
            - date_columns: List of potential date columns
            - year_columns: List of columns that appear to be years
            - numeric_columns: List of numeric columns
            - date_column_candidates: List of potential date columns with confidence scores
        """
        suggestions = {
            'has_date_columns': False,
            'has_year_columns': False,
            'has_sequential_index': False,
            'date_columns': [],
            'year_columns': [],
            'numeric_columns': [],
            'date_column_candidates': [],
            'financial_year_columns': []
        }
        
        # 1. Check datetime columns
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        if datetime_cols:
            suggestions['has_date_columns'] = True
            suggestions['date_columns'] = datetime_cols
            suggestions['date_column_candidates'].extend([(col, 1.0) for col in datetime_cols])
        
        # 2. Check string columns that might be dates
        string_cols = df.select_dtypes(include=['object']).columns.tolist()
        date_confidence = []
        
        for col in string_cols:
            try:
                # Skip columns with very long strings
                if df[col].str.len().max() > 50:
                    continue
                    
                sample = df[col].dropna().head(20)  # Check a larger sample
                if len(sample) == 0:
                    continue
                
                # Try to parse as datetime
                parsed = pd.to_datetime(sample, errors='coerce', infer_datetime_format=True)
                valid_ratio = parsed.notna().sum() / len(sample)
                
                if valid_ratio > 0.8:  # At least 80% of values parse as dates
                    confidence = min(0.9, valid_ratio)  # Cap confidence at 0.9 for auto-detected
                    date_confidence.append((col, confidence))
                    suggestions['has_date_columns'] = True
                    suggestions['date_columns'].append(col)
                
                # Check for financial year patterns
                financial_year_count = 0
                for val in sample:
                    if self._parse_financial_year(val) is not None:
                        financial_year_count += 1
                
                if financial_year_count / len(sample) > 0.8:  # 80% match financial year pattern
                    suggestions['financial_year_columns'].append(col)
                    
            except Exception as e:
                continue
        
        # Sort date columns by confidence
        suggestions['date_column_candidates'].extend(date_confidence)
        suggestions['date_column_candidates'].sort(key=lambda x: x[1], reverse=True)
        
        # 3. Check for year-like columns (e.g., 2019, 2020, etc.)
        year_like_cols = []
        for col in df.columns:
            col_str = str(col).strip()
            
            # Check if column name is a year
            if col_str.isdigit() and 1900 <= int(col_str) <= 2100:
                suggestions['has_year_columns'] = True
                suggestions['year_columns'].append(col)
                year_like_cols.append(col)
            
            # Check for financial year in column name
            elif any(fy in col_str.upper() for fy in ['FY', 'FINANCIAL_YEAR', 'FISCAL_YEAR']):
                suggestions['financial_year_columns'].append(col)
        
        # 4. Check if index looks sequential (could be time periods)
        if (isinstance(df.index, pd.RangeIndex) or 
            (hasattr(df.index, 'is_monotonic_increasing') and df.index.is_monotonic_increasing) or
            (hasattr(df.index, 'is_monotonic') and df.index.is_monotonic)):
            suggestions['has_sequential_index'] = True
        
        # 5. Get numeric columns for analysis
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        suggestions['numeric_columns'] = [col for col in numeric_cols if col not in year_like_cols]
        
        return suggestions
    
    def create_time_series_from_columns(self, df, year_columns, value_column=None):
        """
        Create a time series from columns that represent time periods (like years).
        This is for financial data where columns are 2019, 2020, 2021, etc.
        """
        try:
            if value_column:
                # Extract the row for the specified value
                if value_column in df.index:
                    series_data = df.loc[value_column, year_columns]
                elif value_column in df.columns:
                    # Transpose if needed
                    series_data = df[year_columns].loc[value_column] if value_column in df.index else df[year_columns].iloc[0]
                else:
                    series_data = df[year_columns].iloc[0]
            else:
                # Use the first row
                series_data = df[year_columns].iloc[0]
            
            # Convert to numeric
            series_data = pd.to_numeric(series_data, errors='coerce')
            
            # Create datetime index from year columns
            dates = pd.to_datetime([f"{col}-01-01" for col in year_columns])
            
            # Create series
            series = pd.Series(series_data.values, index=dates)
            series.name = value_column if value_column else 'Value'
            
            return series, None
        except Exception as e:
            return None, f"Error creating time series: {str(e)}"
    
    def create_time_series_from_sequential_data(self, df, value_column, freq='M', start_date=None):
        """
        Create a time series from sequential data without explicit dates.
        Useful for data that has row index as time periods.
        """
        try:
            # Extract values
            if value_column not in df.columns:
                return None, f"Column '{value_column}' not found"
            
            values = df[value_column].copy()
            
            # Create a synthetic date range
            if start_date is None:
                start_date = '2000-01-01'
            
            dates = pd.date_range(start=start_date, periods=len(values), freq=freq)
            
            # Create series
            series = pd.Series(values.values, index=dates)
            series.name = value_column
            
            return series, None
        except Exception as e:
            return None, f"Error creating time series: {str(e)}"
    
    def infer_frequency(self, series):
        """
        Infer the frequency of a time series.
        Returns the inferred frequency or None.
        """
        try:
            # Try to infer frequency
            freq = pd.infer_freq(series.index)
            if freq:
                return freq
            
            # Calculate median time difference
            time_diffs = series.index.to_series().diff().dropna()
            if len(time_diffs) == 0:
                return None
                
            median_diff = time_diffs.median()
            
            # Map to common frequencies
            if median_diff <= pd.Timedelta(days=1):
                return 'D'  # Daily
            elif median_diff <= pd.Timedelta(days=7):
                return 'W'  # Weekly
            elif median_diff <= pd.Timedelta(days=31):
                return 'M'  # Monthly
            elif median_diff <= pd.Timedelta(days=92):
                return 'Q'  # Quarterly
            else:
                return 'Y'  # Yearly
        except:
            return None
    
    def auto_detect_period(self, series):
        """
        Automatically detect the period for seasonal decomposition.
        """
        n_points = len(series)
        
        if self.frequency is None:
            self.frequency = self.infer_frequency(series)
        
        # Map frequency to typical seasonal periods
        freq_period_map = {
            'D': 7,      # Daily data -> weekly seasonality
            'W': 52,     # Weekly data -> yearly seasonality
            'M': 12,     # Monthly data -> yearly seasonality
            'Q': 4,      # Quarterly data -> yearly seasonality
            'Y': 1,      # Yearly data -> no seasonality
        }
        
        period = freq_period_map.get(self.frequency, 12)
        
        # Ensure period is not larger than half the data length
        if period > n_points // 2:
            period = max(2, n_points // 4)
        
        return period
    
    def clean_and_interpolate(self, series):
        """
        Clean the time series by handling missing values and outliers.
        """
        series = series.copy()
        
        # Handle missing values
        if series.isna().sum() > 0:
            # Use interpolation for missing values
            series = series.interpolate(method='time', limit_direction='both')
            
            # If still have NaNs at edges, forward/backward fill
            series = series.fillna(method='ffill').fillna(method='bfill')
        
        # Handle outliers (optional - use IQR method)
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        # Don't remove outliers, just note them
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        outliers = (series < lower_bound) | (series > upper_bound)
        n_outliers = outliers.sum()
        
        return series, n_outliers
    
    def load_and_prepare_data(self, df, date_column, value_column):
        """
        Prepare data for time series analysis with automatic cleaning and preprocessing.
        """
        df = df.copy()
        
        # Parse date column
        try:
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        except Exception as e:
            raise ValueError(f"Failed to parse date column '{date_column}': {str(e)}")
        
        # Remove rows with invalid dates
        df = df.dropna(subset=[date_column])
        
        if len(df) == 0:
            raise ValueError(f"No valid dates found in column '{date_column}'")
        
        # Set date as index
        df = df.set_index(date_column)
        df = df.sort_index()
        
        # Extract the value column
        series = df[value_column].copy()
        
        # Convert to numeric if needed
        if not pd.api.types.is_numeric_dtype(series):
            series = pd.to_numeric(series, errors='coerce')
        
        # Store original data
        self.original_data = series.copy()
        
        # Clean and interpolate
        series, n_outliers = self.clean_and_interpolate(series)
        
        # Infer frequency
        self.frequency = self.infer_frequency(series)
        
        return series
        
    def decompose_series(self, series):
        """
        Decompose time series into trend, seasonal, and residual components.
        Automatically detects period and handles edge cases.
        """
        try:
            # Auto-detect period
            period = self.auto_detect_period(series)
            
            # Check if we have enough data points
            if len(series) < 2 * period:
                raise ValueError(f"Insufficient data for decomposition. Need at least {2 * period} points, but have {len(series)}")
            
            # Perform decomposition
            decomposition = seasonal_decompose(series, period=period, model='additive', extrapolate_trend='freq')
            
            fig = make_subplots(rows=4, cols=1, subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'))
            
            # Original
            fig.add_trace(go.Scatter(x=series.index, y=series.values, name='Original', line=dict(color='blue')), row=1, col=1)
            # Trend
            fig.add_trace(go.Scatter(x=series.index, y=decomposition.trend, name='Trend', line=dict(color='red')), row=2, col=1)
            # Seasonal
            fig.add_trace(go.Scatter(x=series.index, y=decomposition.seasonal, name='Seasonal', line=dict(color='green')), row=3, col=1)
            # Residual
            fig.add_trace(go.Scatter(x=series.index, y=decomposition.resid, name='Residual', line=dict(color='orange')), row=4, col=1)
            
            fig.update_layout(
                height=900, 
                title_text=f"Time Series Decomposition (Period: {period}, Frequency: {self.frequency or 'Unknown'})",
                showlegend=False
            )
            return fig, None
        except Exception as e:
            return None, str(e)
        
    def check_stationarity(self, series):
        """
        Perform Augmented Dickey-Fuller test for stationarity.
        """
        try:
            series_clean = series.dropna()
            if len(series_clean) < 10:
                return None, "Insufficient data for stationarity test (need at least 10 points)"
            
            result = adfuller(series_clean)
            return {
                'test_statistic': result[0],
                'p_value': result[1],
                'critical_values': result[4],
                'is_stationary': result[1] < 0.05
            }, None
        except Exception as e:
            return None, f"Error in stationarity test: {str(e)}"
        
    def fit_arima(self, series, order=(1,1,1)):
        """
        Fit ARIMA model with error handling.
        """
        try:
            series_clean = series.dropna()
            if len(series_clean) < 10:
                raise ValueError("Insufficient data for ARIMA model (need at least 10 points)")
            
            model = ARIMA(series_clean, order=order)
            self.model = model.fit()
            return self.model, None
        except Exception as e:
            return None, f"Error fitting ARIMA model: {str(e)}"
        
    def forecast_arima(self, steps=12):
        """
        Generate forecast using fitted ARIMA model.
        """
        try:
            if self.model is None:
                raise ValueError("Model must be fitted first")
                
            forecast = self.model.forecast(steps=steps)
            return forecast, None
        except Exception as e:
            return None, f"Error generating forecast: {str(e)}"
        
    def fit_prophet(self, series, future_periods=12):
        """
        Fit Prophet model and generate forecast with automatic frequency detection.
        """
        try:
            series_clean = series.dropna()
            if len(series_clean) < 10:
                raise ValueError("Insufficient data for Prophet model (need at least 10 points)")
            
            # Prepare data for Prophet
            df = pd.DataFrame({'ds': series_clean.index, 'y': series_clean.values})
            
            # Determine seasonality based on frequency
            yearly_seasonality = True
            weekly_seasonality = False
            daily_seasonality = False
            
            if self.frequency == 'D':
                daily_seasonality = True
                weekly_seasonality = True
            elif self.frequency == 'W':
                weekly_seasonality = False
                yearly_seasonality = True
            
            # Initialize and fit Prophet model
            model = Prophet(
                yearly_seasonality=yearly_seasonality,
                weekly_seasonality=weekly_seasonality,
                daily_seasonality=daily_seasonality,
                changepoint_prior_scale=0.05
            )
            model.fit(df)
            
            # Determine frequency for future dates
            freq = self.frequency or 'M'
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=future_periods, freq=freq)
            
            # Generate forecast
            self.forecast = model.predict(future)
            
            return self.forecast, None
        except Exception as e:
            return None, f"Error with Prophet model: {str(e)}"
        
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