import streamlit as st
import pandas as pd
from src.data_loader import DataLoader
from src.time_series.analyzer import TimeSeriesAnalyzer
from src.time_series.stationarity import make_stationary

def render_time_series(data_loader: DataLoader, analyzer: TimeSeriesAnalyzer):
    """
    Render the Time Series Analysis tab.
    
    Args:
        data_loader: The data loader instance.
        analyzer: The Time Series Analyzer instance.
    """
    st.markdown('<h2 class="sub-header">Time Series Analysis</h2>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-text">Analyze time series using best-fit detection. Choose from Date columns, Year columns, or Sequential patterns.</div>',
        unsafe_allow_html=True,
    )

    df = data_loader.get_data()

    # Setup: choose mode and prepare series
    ts_cap = analyzer.detect_time_series_capable_data(df)
    modes = []
    
    # Suggest specific modes based on detection
    if ts_cap['has_date_columns']:
        modes.append("📅 Date Column (Detected)")
    if ts_cap['has_year_columns']:
        modes.append("📊 Year Columns (Wide → Long)")
    if len(ts_cap['numeric_columns']) > 0:
        modes.append("🔢 Sequential (Synthetic Dates)")

    if not modes:
        st.warning("No standard time-series data detected. You can try using sequential mode with any numeric column.")
        modes.append("🔢 Sequential (Synthetic Dates)")
        
    st.markdown('<h3 class="sub-header">⚙️ Setup Time Series Data</h3>', unsafe_allow_html=True)
    
    # Auto-select the best mode based on what was detected first (priority: Date -> Year -> Sequential)
    initial_index = 0
    mode = st.radio("Data Mode", options=modes, index=initial_index)

    prepared_series = None
    
    if "📅 Date Column" in mode:
        date_col = st.selectbox("Date column", ts_cap['date_columns'])
        value_col = st.selectbox("Value column", ts_cap['numeric_columns'])
        if st.button("Prepare Series", type="primary"):
            try:
                prepared_series = analyzer.load_and_prepare(df, date_col, value_col)
                st.success(f"Series prepared from {date_col} and {value_col}")
            except Exception as e:
                st.error(str(e))
                
    elif "📊 Year Columns" in mode:
        year_like = [c for c in df.columns if str(c).strip().isdigit() or 'FY' in str(c).upper()]
        # Fallback to detected ones if list comprehension misses
        if not year_like:
             year_like = ts_cap['year_columns']
             
        if not year_like:
            st.warning("No explicit year columns found via direct check. Please select manually.")
            year_like = df.columns.tolist()
            
        year_cols = st.multiselect("Year columns", year_like, default=year_like[:min(5, len(year_like))])
        selector = st.text_input("Row selector (optional, e.g., row index or generic label)")
        if st.button("Prepare Series", type="primary"):
            series, err = analyzer.create_time_series_from_year_columns(df, year_cols, selector or None)
            if err:
                st.error(err)
            else:
                prepared_series = series
                st.success("Series prepared from year columns")
                
    else:  # Sequential
        value_col = st.selectbox("Value column", ts_cap['numeric_columns'])
        freq = st.selectbox("Frequency", ["D", "W", "M", "Q", "Y"], index=2)
        start_date = st.date_input("Start date", pd.to_datetime("2000-01-01")).strftime("%Y-%m-%d")
        if st.button("Prepare Series", type="primary"):
            series, err = analyzer.create_time_series_from_sequential(df, value_col, freq=freq, start_date=start_date)
            if err:
                st.error(err)
            else:
                prepared_series = series
                st.success("Series prepared sequentially")

    if prepared_series is not None:
        st.markdown("---")
        tab_analyze, tab_forecast, tab_stationarity = st.tabs(["📈 Analyze", "🔮 Forecast", "📊 Stationarity"]) 

        with tab_analyze:
            st.markdown('<h3 class="sub-header">Decomposition</h3>', unsafe_allow_html=True)
            model = st.selectbox("Model", ["additive", "multiplicative"], index=0)
            if st.button("Run Decomposition"):
                with st.spinner("Decomposing time series..."):
                    try:
                        result = analyzer.decompose_series(prepared_series, model=model)
                        st.plotly_chart(result['plot'], use_container_width=True)
                    except Exception as e:
                        st.error(str(e))

        with tab_stationarity:
            st.markdown('<h3 class="sub-header">Stationarity Tests</h3>', unsafe_allow_html=True)
            if st.button("Run Tests"):
                with st.spinner("Running ADF/KPSS..."):
                    try:
                        res = analyzer.check_stationarity(prepared_series)
                        c1, c2 = st.columns(2)
                        with c1:
                            st.metric("ADF p-value", f"{res['p_value']:.4f}")
                        with c2:
                            st.write("Critical Values:")
                            st.write({k: float(v) for k, v in res['critical_values'].items()})
                        st.success("Stationary" if res['is_stationary'] else "Non-stationary")
                    except Exception as e:
                        st.error(str(e))

            st.markdown("#### Transform if needed")
            method_label = st.selectbox("Transformation", ["Differencing", "Log Transform", "Log + Differencing", "Percent Change"], index=0)
            method_map = {"Differencing": "diff", "Log Transform": "log", "Log + Differencing": "log_diff", "Percent Change": "pct_change"}
            if st.button("Apply Transformation"):
                try:
                    transformed, _ = make_stationary(prepared_series, method=method_map[method_label], periods=1)
                    st.line_chart(pd.DataFrame({"Original": prepared_series, "Transformed": transformed}))
                    st.session_state.transformed_series = transformed
                    st.session_state.is_transformed = True
                except Exception as e:
                    st.error(str(e))

        with tab_forecast:
            st.markdown('<h3 class="sub-header">Forecasting</h3>', unsafe_allow_html=True)
            model_choice = st.radio("Model", ["ARIMA", "Prophet"], horizontal=True)
            steps = st.number_input("Forecast periods", min_value=1, max_value=60, value=12)
            if st.button("Generate Forecast", type="primary"):
                try:
                    working = st.session_state.get('transformed_series', prepared_series)
                    if model_choice == "ARIMA":
                         # Using smart default for order if possible, currently fixed to 1,1,1 for simplicity
                         # In future could auto-arima if desired
                         model_res = analyzer.fit_arima(working, order=(1,1,1))
                         if model_res['success']:
                             fcast = analyzer.forecast_arima(model_res['model'], steps=steps)
                             st.line_chart(fcast['forecast']['forecast'])
                         else:
                             st.error(model_res['message'])
                    else:
                         model_res = analyzer.fit_prophet(working)
                         if model_res['success']:
                             # Prophet is tricky with future df, need helper in analyzer but trying direct fit here
                             future = model_res['model'].make_future_dataframe(periods=steps)
                             forecast = model_res['model'].predict(future)
                             st.line_chart(forecast.set_index('ds')['yhat'])
                         else:
                             st.error(model_res['message'])

                except Exception as e:
                    st.error(f"Forecast error: {str(e)}")
