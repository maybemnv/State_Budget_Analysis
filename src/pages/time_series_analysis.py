import streamlit as st
import pandas as pd

from src.time_series import TimeSeriesAnalyzer
from src.time_series.stationarity import make_stationary


def _build_series_ui(analyzer: TimeSeriesAnalyzer, data: pd.DataFrame):
    detection = analyzer.detect_time_series_capable_data(data)

    st.markdown("### ⚙️ Setup Time Series Data")

    mode_options = []
    if detection['has_date_columns']:
        mode_options.append("📅 Date Column")
    if detection['has_year_columns']:
        mode_options.append("📊 Year Columns (Wide → Long)")
    if len(detection['numeric_columns']) > 0:
        mode_options.append("🔢 Sequential (Synthetic Dates)")

    if not mode_options:
        st.error(
            "No time-series capable data detected. Provide a date column, year columns, or a numeric column for sequential analysis."
        )
        return None

    mode = st.radio("Data Mode", options=mode_options, horizontal=False)

    if mode == "📅 Date Column":
        date_col = st.selectbox("Date column", detection['date_columns'])
        value_col = st.selectbox("Value column", detection['numeric_columns'])
        if st.button("Prepare Series", type="primary"):
            try:
                series = analyzer.load_and_prepare(data, date_col, value_col)
                st.success("Series prepared")
                return series
            except Exception as e:
                st.error(str(e))
                return None

    if mode == "📊 Year Columns (Wide → Long)":
        # Pick columns that look like years (by header)
        year_like = [c for c in data.columns if str(c).strip().isdigit()]
        year_cols = st.multiselect("Year columns", year_like, default=year_like[:min(5, len(year_like))])
        row_selector = st.text_input("Row selector (optional)", help="Row label to extract if your table uses rows as measures")
        if st.button("Prepare Series", type="primary"):
            series, err = analyzer.create_time_series_from_year_columns(data, year_cols, row_selector or None)
            if err:
                st.error(err)
                return None
            st.success("Series prepared")
            return series

    if mode == "🔢 Sequential (Synthetic Dates)":
        value_col = st.selectbox("Value column", detection['numeric_columns'])
        freq = st.selectbox("Frequency", ["D", "W", "M", "Q", "Y"], index=2)
        start_date = st.date_input("Start date", pd.to_datetime("2000-01-01")).strftime("%Y-%m-%d")
        if st.button("Prepare Series", type="primary"):
            series, err = analyzer.create_time_series_from_sequential(data, value_col, freq=freq, start_date=start_date)
            if err:
                st.error(err)
                return None
            st.success("Series prepared")
            return series

    return None


def show():
    """Display the Time Series Analysis page (Analyze | Forecast | Stationarity)."""
    st.header("Time Series Analysis")

    if 'data_loader' not in st.session_state or st.session_state.data_loader.data is None:
        st.warning("Please upload a data file first using the sidebar.")
        return

    data = st.session_state.data_loader.data
    analyzer = TimeSeriesAnalyzer()

    # Data setup
    series = _build_series_ui(analyzer, data)
    if series is None:
        return

    st.markdown("---")
    tab_analyze, tab_forecast, tab_stationarity = st.tabs(["📈 Analyze", "🔮 Forecast", "📊 Stationarity"]) 

    with tab_analyze:
        st.subheader("Decomposition")
        model = st.selectbox("Model", ["additive", "multiplicative"], index=0)
        if st.button("Run Decomposition"):
            with st.spinner("Decomposing time series..."):
                try:
                    result = analyzer.decompose_series(series, model=model)
                    st.plotly_chart(result['plot'], use_container_width=True)
                except Exception as e:
                    st.error(str(e))

    with tab_stationarity:
        st.subheader("Stationarity Tests")
        if st.button("Run Tests"):
            with st.spinner("Running ADF/KPSS..."):
                try:
                    res = analyzer.check_stationarity(series)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("ADF p-value", f"{res['p_value']:.4f}")
                    with c2:
                        crit = res['critical_values']
                        st.write("Critical Values:")
                        st.write({k: float(v) for k, v in crit.items()})
                    st.success("Stationary" if res['is_stationary'] else "Non-stationary")
                except Exception as e:
                    st.error(str(e))

        st.markdown("#### Transform if needed")
        method_label = st.selectbox("Transformation", ["Differencing", "Log Transform", "Log + Differencing", "Percent Change"], index=0)
        method_map = {"Differencing": "diff", "Log Transform": "log", "Log + Differencing": "log_diff", "Percent Change": "pct_change"}
        if st.button("Apply Transformation"):
            try:
                transformed, info = make_stationary(series, method=method_map[method_label], periods=1)
                st.line_chart(pd.DataFrame({"Original": series, "Transformed": transformed}))
                st.session_state.transformed_series = transformed
                st.session_state.is_transformed = True
            except Exception as e:
                st.error(str(e))

    with tab_forecast:
        st.subheader("Forecasting")
        model_choice = st.radio("Model", ["ARIMA", "Prophet"], horizontal=True)
        steps = st.number_input("Forecast periods", min_value=1, max_value=60, value=12)

        run = st.button("Generate Forecast", type="primary")
        if run:
            try:
                working_series = st.session_state.get('transformed_series', series)
                if model_choice == "ARIMA":
                    fit = analyzer.fit_arima(working_series, order=(1, 1, 1))
                    if not fit['success']:
                        st.error(fit['message'])
                    else:
                        fc = analyzer.forecast_arima(fit['model'], steps=steps)
                        if not fc['success']:
                            st.error(fc['message'])
                        else:
                            # Build forecast series aligned to end
                            last = working_series.index[-1]
                            freq = pd.infer_freq(working_series.index) or 'M'
                            idx = pd.date_range(start=last, periods=steps+1, freq=freq)[1:]
                            yhat = pd.Series(fc['forecast']['forecast'].values, index=idx)
                            fig = analyzer.plot_forecast(working_series, yhat, fc['forecast'][['lower', 'upper']])
                            st.plotly_chart(fig, use_container_width=True)
                            st.dataframe(fc['forecast'])
                else:
                    fit = analyzer.fit_prophet(working_series)
                    if not fit['success']:
                        st.error(fit['message'])
                    else:
                        # Create future and predict using the fitted model
                        freq = pd.infer_freq(working_series.index) or 'M'
                        future_index = pd.date_range(start=working_series.index[-1], periods=steps+1, freq=freq)[1:]
                        # Prophet expects internal generation; reuse model.predict on future dataframe
                        model = fit['model']
                        future_df = pd.DataFrame({'ds': future_index})
                        forecast_df = model.predict(future_df)
                        display_df = forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={'ds': 'Date'})
                        fig = analyzer.plot_forecast(working_series, forecast_df.set_index('ds')['yhat'], forecast_df.set_index('ds')[['yhat_lower', 'yhat_upper']].rename(columns={'yhat_lower': 'lower', 'yhat_upper': 'upper'}))
                        st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(display_df)
            except Exception as e:
                st.error(str(e))

    st.markdown("<br>", unsafe_allow_html=True)
