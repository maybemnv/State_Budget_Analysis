import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import our time series modules
from src.time_series import TimeSeriesAnalyzer
from src.time_series.visualization import plot_forecast, plot_decomposition, plot_acf_pacf
from src.time_series.stationarity import check_stationarity, make_stationary
from src.time_series.forecasting import fit_arima, forecast_arima, fit_prophet

def show():
    """Display the Time Series Analysis page."""
    st.header("Time Series Analysis")
    
    # Get data from session state
    if 'data_loader' not in st.session_state or st.session_state.data_loader.data is None:
        st.warning("Please upload a data file first using the sidebar.")
        return
    
    data = st.session_state.data_loader.data
    date_columns = st.session_state.data_loader.get_date_columns()
    numeric_columns = st.session_state.data_loader.get_numeric_columns()
    
    # If no date columns found, try to infer them
    if not date_columns:
        st.warning("No date columns detected. Please select a column that contains dates.")
        
        # Let user select potential date columns
        potential_date_cols = st.multiselect(
            "Select potential date columns:",
            data.columns,
            help="Select columns that might contain date information"
        )
        
        if potential_date_cols:
            # Try to convert selected columns to datetime
            for col in potential_date_cols:
                try:
                    data[col] = pd.to_datetime(data[col], errors='coerce')
                    if data[col].notna().any():
                        date_columns.append(col)
                except:
                    pass
    
    # If still no date columns, show error
    if not date_columns:
        st.error("No valid date columns found. Time series analysis requires a datetime column.")
        return
    
    # Select date column
    date_col = st.selectbox(
        "Select Date Column:",
        date_columns,
        index=0,
        help="Select the column that contains the date/time information"
    )
    
    # Select value column
    value_col = st.selectbox(
        "Select Value Column:",
        numeric_columns,
        index=0 if numeric_columns else None,
        help="Select the column that contains the values to analyze"
    )
    
    if not value_col:
        st.error("No numeric columns found for analysis.")
        return
    
    # Create time series analyzer
    analyzer = TimeSeriesAnalyzer()
    
    # Set up tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["📈 Decomposition", "📊 Stationarity", "🔮 Forecasting"])
    
    with tab1:
        st.subheader("Time Series Decomposition")
        st.info(
            "Decompose the time series into trend, seasonal, and residual components. "
            "This helps identify patterns and anomalies in the data."
        )
        
        if st.button("🔍 Decompose Time Series"):
            with st.spinner("Analyzing time series components..."):
                try:
                    # Prepare the time series
                    series = data.set_index(date_col)[value_col].sort_index()
                    
                    # Decompose the series
                    decomposition = analyzer.decompose_series(
                        series,
                        model='additive',  # or 'multiplicative'
                        period=None  # Auto-detect period
                    )
                    
                    # Display decomposition plot
                    st.plotly_chart(
                        plot_decomposition(
                            observed=decomposition['observed'],
                            trend=decomposition['trend'],
                            seasonal=decomposition['seasonal'],
                            resid=decomposition['resid']
                        ),
                        use_container_width=True
                    )
                    
                except Exception as e:
                    st.error(f"Error during decomposition: {str(e)}")
    
    with tab2:
        st.subheader("Stationarity Analysis")
        st.info(
            "Stationarity tests check if the statistical properties of the series remain constant over time. "
            "Stationary data is important for many forecasting models."
        )
        
        if st.button("📊 Run Stationarity Tests"):
            with st.spinner("Checking stationarity..."):
                try:
                    # Prepare the time series
                    series = data.set_index(date_col)[value_col].sort_index()
                    
                    # Check stationarity
                    result = check_stationarity(series)
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("ADF Test p-value", f"{result['adf']['pvalue']:.4f}", 
                                "✅ Stationary" if result['adf']['pvalue'] <= 0.05 else "❌ Non-stationary")
                    
                    with col2:
                        st.metric("KPSS Test p-value", 
                                f"{result['kpss']['pvalue']:.4f}" if 'pvalue' in result['kpss'] else "N/A",
                                "✅ Stationary" if 'pvalue' in result['kpss'] and result['kpss']['pvalue'] > 0.05 else "❌ Non-stationary")
                    
                    # Display test details
                    with st.expander("View test details"):
                        st.json(result)
                    
                    # If not stationary, suggest transformations
                    if not result['is_stationary']:
                        st.warning("The time series appears to be non-stationary. Consider applying transformations.")
                        
                        transform_method = st.selectbox(
                            "Select transformation method:",
                            ["Differencing", "Log Transform", "Log + Differencing", "Percent Change"],
                            index=0
                        )
                        
                        if st.button("Apply Transformation"):
                            with st.spinner("Applying transformation..."):
                                try:
                                    method_map = {
                                        "Differencing": "diff",
                                        "Log Transform": "log",
                                        "Log + Differencing": "log_diff",
                                        "Percent Change": "pct_change"
                                    }
                                    
                                    transformed_series, transform_info = make_stationary(
                                        series,
                                        method=method_map[transform_method],
                                        periods=1
                                    )
                                    
                                    # Display transformed series
                                    st.plotly_chart(
                                        plot_forecast(
                                            series,
                                            transformed_series,
                                            title=f"Original vs. {transform_method} Transformed Series"
                                        ),
                                        use_container_width=True
                                    )
                                    
                                    # Update the series for further analysis
                                    st.session_state.transformed_series = transformed_series
                                    st.session_state.is_transformed = True
                                    
                                except Exception as e:
                                    st.error(f"Error during transformation: {str(e)}")
                    
                except Exception as e:
                    st.error(f"Error during stationarity analysis: {str(e)}")
    
    with tab3:
        st.subheader("Time Series Forecasting")
        st.info(
            "Generate forecasts using different models. "
            "ARIMA is good for stationary data, while Prophet handles seasonality and trends automatically."
        )
        
        # Model selection
        model_type = st.radio(
            "Select Forecasting Model:",
            ["ARIMA", "Prophet"],
            horizontal=True
        )
        
        # Forecast horizon
        forecast_steps = st.number_input(
            "Forecast Periods:",
            min_value=1,
            max_value=365,
            value=12,
            help="Number of future periods to forecast"
        )
        
        if st.button("🚀 Generate Forecast"):
            with st.spinner(f"Training {model_type} model and generating forecast..."):
                try:
                    # Prepare the time series
                    series = data.set_index(date_col)[value_col].sort_index()
                    
                    # Use transformed series if available
                    if hasattr(st.session_state, 'is_transformed') and st.session_state.is_transformed:
                        series = st.session_state.transformed_series
                    
                    if model_type == "ARIMA":
                        # Simple auto-ARIMA for demonstration
                        # In a real app, you'd want to implement proper parameter selection
                        model_result = fit_arima(
                            series,
                            order=(1, 1, 1),  # Simple ARIMA(1,1,1) for demo
                            seasonal_order=(0, 1, 1, 12)  # Simple seasonal component
                        )
                        
                        if model_result['success']:
                            # Generate forecast
                            forecast = forecast_arima(
                                model_result['model'],
                                steps=forecast_steps,
                                alpha=0.05  # 95% confidence interval
                            )
                            
                            if forecast['success']:
                                # Create index for forecast period
                                last_date = series.index[-1]
                                if isinstance(last_date, (int, float)):
                                    # If index is numeric (e.g., year)
                                    freq = 1  # Default frequency
                                    if len(series) > 1:
                                        freq = series.index[1] - series.index[0]
                                    forecast_index = pd.RangeIndex(
                                        start=last_date + freq,
                                        stop=last_date + (forecast_steps + 1) * freq,
                                        step=freq
                                    )
                                else:
                                    # If index is datetime
                                    freq = pd.infer_freq(series.index) or 'D'  # Default to daily
                                    forecast_index = pd.date_range(
                                        start=last_date + pd.Timedelta(days=1),
                                        periods=forecast_steps,
                                        freq=freq
                                    )
                                
                                # Create forecast series
                                forecast_series = pd.Series(
                                    forecast['forecast']['forecast'].values,
                                    index=forecast_index,
                                    name='Forecast'
                                )
                                
                                # Plot forecast
                                fig = plot_forecast(
                                    actual=series,
                                    forecast=forecast_series,
                                    conf_int=forecast['forecast'][['lower', 'upper']],
                                    title=f"{model_type} Forecast - {value_col}"
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Display forecast metrics
                                st.subheader("Model Performance")
                                st.metric("AIC", f"{model_result['model'].aic:.2f}")
                                st.metric("BIC", f"{model_result['model'].bic:.2f}")
                                
                                # Show forecast values
                                st.subheader("Forecast Values")
                                st.dataframe(forecast['forecast'].style.background_gradient(cmap='YlOrRd'))
                                
                            else:
                                st.error(f"Forecast error: {forecast['message']}")
                        else:
                            st.error(f"Model fitting error: {model_result['message']}")
                    
                    elif model_type == "Prophet":
                        # Prepare data for Prophet
                        df_prophet = pd.DataFrame({
                            'ds': series.index,
                            'y': series.values
                        })
                        
                        # Fit Prophet model
                        model = Prophet(
                            yearly_seasonality=True,
                            weekly_seasonality=True,
                            daily_seasonality=False
                        )
                        
                        try:
                            model.fit(df_prophet)
                            
                            # Create future dataframe
                            future = model.make_future_dataframe(
                                periods=forecast_steps,
                                freq=pd.infer_freq(series.index) or 'D'
                            )
                            
                            # Generate forecast
                            forecast_df = model.predict(future)
                            
                            # Plot forecast
                            st.subheader("Prophet Forecast")
                            
                            # Plot components
                            st.plotly_chart(
                                model.plot_components(forecast_df),
                                use_container_width=True
                            )
                            
                            # Plot forecast with actuals
                            fig = model.plot(forecast_df)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display forecast values
                            st.subheader("Forecast Values")
                            forecast_cols = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
                            st.dataframe(
                                forecast_df[forecast_cols].tail(forecast_steps)
                                .rename(columns={
                                    'ds': 'Date',
                                    'yhat': 'Forecast',
                                    'yhat_lower': 'Lower Bound',
                                    'yhat_upper': 'Upper Bound'
                                })
                                .style.background_gradient(cmap='YlOrRd')
                            )
                            
                        except Exception as e:
                            st.error(f"Error in Prophet forecasting: {str(e)}")
                    
                except Exception as e:
                    st.error(f"Error during forecasting: {str(e)}")
                    st.exception(e)  # Show full traceback for debugging
    
    # Add some spacing at the bottom
    st.markdown("<br><br>", unsafe_allow_html=True)
