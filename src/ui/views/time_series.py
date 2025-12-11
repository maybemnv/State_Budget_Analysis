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
    # Header Section
    col_header, col_mode = st.columns([2, 1])
    with col_header:
        st.markdown('<h2 class="sub-header">Time Series Analysis</h2>', unsafe_allow_html=True)
        st.markdown(
            '<div class="info-text">Analyze time-dependent data. Supports Date columns, Year-like columns, or Sequential numeric data.</div>',
            unsafe_allow_html=True,
        )

    df = data_loader.get_data()
    
    # Initialize session state for persistence
    if 'time_series_data' not in st.session_state:
        st.session_state.time_series_data = None
    
    # -------------------------------------------------------------------------
    # 1. SETUP SECTION
    # -------------------------------------------------------------------------
    st.markdown("### 1. Data Setup")
    
    with st.expander("🛠️ Configuration & Detection", expanded=st.session_state.time_series_data is None):
        # Auto-detect capabilities
        ts_cap = analyzer.detect_time_series_capable_data(df)
        
        # Build modes map for cleaner UI
        modes_map = {}
        if ts_cap['has_date_columns']:
            modes_map["📅 Date Column"] = "date"
        if ts_cap['has_year_columns']:
            modes_map["📊 Year Columns (Wide)"] = "year"
        
        # Always allow sequential as fallback
        modes_map["🔢 Sequential (Index/Numeric)"] = "seq"
        
        mode_labels = list(modes_map.keys())
        
        # Default index logic
        default_ix = 0
        if not ts_cap['has_date_columns'] and not ts_cap['has_year_columns']:
            # If nothing specific detected, default to sequential (last item) and warn
            default_ix = len(mode_labels) - 1
            st.info("💡 No standard time/date columns detected. Defaulting to **Sequential** mode.")

        selected_label = st.radio("Select Data Source", mode_labels, index=default_ix, horizontal=True)
        mode_key = modes_map[selected_label]
        
        if mode_key == "date":
            c1, c2 = st.columns(2)
            with c1:
                date_col = st.selectbox("Select Date Column", ts_cap['date_columns'])
            with c2:
                value_col = st.selectbox("Select Value Column", ts_cap['numeric_columns'])
            
            if st.button("Load Series", type="primary", use_container_width=True):
                try:
                    series = analyzer.load_and_prepare(df, date_col, value_col)
                    st.session_state.time_series_data = series
                    st.session_state.transformed_series = None # Reset transformation on new load
                    st.success(f"Loaded series from **{date_col}**")
                    st.rerun() # Force rerun to collapse expander and show tabs
                except Exception as e:
                    st.error(f"Failed to load series: {e}")

        elif mode_key == "year":
            st.caption("Convert wide-format data (years as columns) into a time series.")
            year_like = [c for c in df.columns if str(c).strip().isdigit() or 'FY' in str(c).upper()]
            if not year_like: year_like = ts_cap['year_columns']
            if not year_like: year_like = df.columns.tolist()

            year_cols = st.multiselect("Select Year Columns", year_like, default=year_like[:min(10, len(year_like))])
            selector = st.text_input("Row Selector (Optional: Index value or unique identifier)")
            
            if st.button("Load Series", type="primary", use_container_width=True):
                series, err = analyzer.create_time_series_from_year_columns(df, year_cols, selector or None)
                if err:
                    st.error(err)
                else:
                    st.session_state.time_series_data = series
                    st.session_state.transformed_series = None
                    st.success("Loaded series from selected years")
                    st.rerun()

        elif mode_key == "seq":
            st.caption("Create a synthetic time index for sequential data.")
            c1, c2, c3 = st.columns(3)
            with c1:
                value_col = st.selectbox("Value Column", ts_cap['numeric_columns'])
            with c2:
                freq = st.selectbox("Frequency", ["D (Daily)", "W (Weekly)", "M (Monthly)", "Q (Quarterly)", "Y (Yearly)"], index=2)
            with c3:
                start_date = st.date_input("Start Date", pd.to_datetime("2000-01-01"))
            
            freq_code = freq.split()[0] # Extract 'D' from 'D (Daily)'
            
            if st.button("Load Series", type="primary", use_container_width=True):
                start_str = start_date.strftime("%Y-%m-%d")
                series, err = analyzer.create_time_series_from_sequential(df, value_col, freq=freq_code, start_date=start_str)
                if err:
                    st.error(err)
                else:
                    st.session_state.time_series_data = series
                    st.session_state.transformed_series = None
                    st.success("Generated sequential series")
                    st.rerun()

    # -------------------------------------------------------------------------
    # 2. ANALYSIS TABS (Only show if series is ready)
    # -------------------------------------------------------------------------
    prepared_series = st.session_state.time_series_data

    if prepared_series is not None:
        
        st.markdown("### 2. Analysis Playground")
        
        # Summary Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Start Date", prepared_series.index.min().strftime('%Y-%m-%d'))
        m2.metric("End Date", prepared_series.index.max().strftime('%Y-%m-%d'))
        m3.metric("Data Points", len(prepared_series))
        m4.metric("Frequency", getattr(prepared_series.index, 'freqstr', 'Unknown'))
        
        st.line_chart(prepared_series, use_container_width=True)
        
        tab_analyze, tab_stat, tab_forecast = st.tabs(["📉 Decomposition", "📊 Stationarity", "🔮 Forecasting"])
        
        # --- DECOMPOSITION ---
        with tab_analyze:
            c_ctrl, c_plot = st.columns([1, 3])
            with c_ctrl:
                st.markdown("#### Settings")
                model = st.selectbox("Model Type", ["additive", "multiplicative"], help="Additive for constant amplitude, Multiplicative for increasing amplitude.")
                run_decomp = st.button("Run Decomposition", use_container_width=True)
            
            with c_plot:
                if run_decomp:
                    with st.spinner("Analyzing..."):
                        try:
                            result = analyzer.decompose_series(prepared_series, model=model)
                            st.plotly_chart(result['plot'], use_container_width=True)
                        except Exception as e:
                            st.warning(f"Decomposition failed: {e}. Try a different model or ensure data has no zeros/negatives for multiplicative.")

        # --- STATIONARITY ---
        with tab_stat:
            st.markdown("#### Test Stationarity")
            if st.button("Check Stationarity (ADF Test)"):
                res = analyzer.check_stationarity(prepared_series)
                if res:
                    pval = res['adf']['pvalue']
                    st.info(f"**Stationary?** {'✅ Yes' if res['is_stationary'] else '❌ No'} (ADF p-value: {pval:.4f})")
                else:
                    st.error("Could not run test.")
            
            st.divider()
            st.markdown("#### Apply Transformations")
            
            c_trans, c_view = st.columns([1, 3])
            with c_trans:
                method_label = st.selectbox("Transform Method", 
                                            ["Differencing", "Log Transform", "Log + Diff", "Percent Change"])
                method_map = {"Differencing": "diff", "Log Transform": "log", "Log + Diff": "log_diff", "Percent Change": "pct_change"}
                
                apply_trans = st.button("Apply Transform", use_container_width=True)
                
            with c_view:
                if apply_trans:
                    try:
                        transformed, _ = make_stationary(prepared_series, method=method_map[method_label], periods=1)
                        st.session_state.transformed_series = transformed # persist for forecast
                        
                        chart_data = pd.DataFrame({
                            "Original": prepared_series,
                            "Transformed": transformed
                        })
                        st.line_chart(chart_data, use_container_width=True)
                        st.success("Transformation applied! This series will be used for forecasting below.")
                    except Exception as e:
                        st.error(f"Transformation failed: {e}")

        # --- FORECAST ---
        with tab_forecast:
            st.markdown("#### Forecast Future Values")
            
            # Check if using transformed or original
            use_transformed = "transformed_series" in st.session_state and st.session_state.transformed_series is not None
            working_series = st.session_state.transformed_series if use_transformed else prepared_series
            
            if use_transformed:
                st.info("⚠️ Using **Transformed** data for forecasting.")
            else:
                st.info("Using **Original** data for forecasting.")

            c_mod, c_res = st.columns([1, 3])
            
            with c_mod:
                model_choice = st.radio("Select Model", ["ARIMA", "Prophet"])
                steps = st.number_input("Forecast Steps", min_value=1, max_value=365, value=12)
                run_forecast = st.button("Generate Forecast", type="primary", use_container_width=True)
            
            with c_res:
                if run_forecast:
                    with st.spinner(f"Running {model_choice}..."):
                        try:
                            if model_choice == "ARIMA":
                                fit = analyzer.fit_arima(working_series, order=(1,1,1))
                                if fit['success']:
                                    fc = analyzer.forecast_arima(fit['model'], steps=steps)
                                    if fc['success']:
                                        # Plotting - handled by basic line chart for speed or custom plot
                                        # Combining history + forecast for context
                                        # Just showing forecast plot from analyzer wrapper
                                        # For custom chart we need valid dates
                                        
                                        # Create simple plot
                                        hist_df = pd.DataFrame({"Actual": working_series})
                                        # Forecast index needs to be inferred or generated
                                        # analyzer.forecast_arima returns a dataframe with 'forecast', 'lower', 'upper'
                                        
                                        forecast_vals = fc['forecast']['forecast']
                                        confidence = fc['forecast'][['lower', 'upper']]
                                        
                                        # Generate future index
                                        last_date = working_series.index[-1]
                                        freq = getattr(working_series.index, 'freqstr', 'D') # default daily
                                        if freq is None: freq = 'D'
                                        
                                        future_idx = pd.date_range(start=last_date, periods=steps+1, freq=freq)[1:]
                                        forecast_series = pd.Series(forecast_vals.values, index=future_idx)
                                        
                                        # Use analyzer plot helper
                                        fig = analyzer.plot_forecast(working_series, forecast_series, confidence)
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.error(fc['message'])
                                else:
                                    st.error(fit['message'])
                            
                            else: # Prophet
                                fit = analyzer.fit_prophet(working_series)
                                if fit['success']:
                                    # Need a dedicated helper or direct interaction
                                    model = fit['model']
                                    future = model.make_future_dataframe(periods=steps)
                                    forecast = model.predict(future)
                                    
                                    # Plot
                                    fig = analyzer.plot_forecast(
                                        working_series, 
                                        forecast.set_index('ds')['yhat'].tail(steps), 
                                        forecast.set_index('ds')[['yhat_lower', 'yhat_upper']].tail(steps).rename(columns={'yhat_lower':'lower', 'yhat_upper':'upper'})
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.error(fit['message'])
                        except Exception as e:
                            st.error(f"Forecast Error: {e}")
