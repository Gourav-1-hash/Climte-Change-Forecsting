"""Main Streamlit app for end-to-end climate change forecasting project."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.config import Config
from app.models.predictor import (
    forecast_time_series,
    get_feature_importance,
    get_model_metrics,
    predict_climate_scenario,
    retrain_models,
)
from app.services.weather_service import (
    get_city_climate_bundle,
    get_extreme_event_summary,
    get_temperature_trend_df,
)
from app.ui.charts import (
    build_aqi_gauge,
    build_aqi_trend,
    build_correlation_heatmap,
    build_extreme_events_chart,
    build_feature_importance,
    build_forecast_plot,
    build_long_term_trend,
    build_model_metrics_chart,
    build_pollutant_radar,
    build_scenario_result_chart,
    build_seasonality_heatmap,
    build_temperature_trend,
)
from app.ui.components import (
    aqi_badge,
    inject_css,
    metric_card,
    prediction_result_card,
    section_header,
)
from app.ui.sidebar import render_sidebar
from app.utils.helpers import (
    celsius_to_fahrenheit,
    current_timestamp,
    get_weather_description,
    get_weather_icon,
    risk_label,
    summarize_climate_frame,
)
from app.utils.report_generator import generate_climate_report


st.set_page_config(
    page_title=Config.APP_TITLE,
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()


@st.cache_data(ttl=600, show_spinner=False)
def _load_bundle(city: str, history_years: int) -> dict:
    return get_city_climate_bundle(city=city, history_years=history_years)


def _fmt_temp(value_c: float, unit: str) -> str:
    if value_c is None or pd.isna(value_c):
        return "N/A"
    return f"{celsius_to_fahrenheit(float(value_c)):.1f} F" if unit == "Fahrenheit" else f"{float(value_c):.1f} C"


def _render_metric_grid(cards: list[tuple[str, str, str, str]], columns_per_row: int) -> None:
    """Render metric cards in rows with a variable column count."""
    columns_per_row = max(1, int(columns_per_row))
    for start in range(0, len(cards), columns_per_row):
        row_cards = cards[start : start + columns_per_row]
        row_cols = st.columns(len(row_cards))
        for col, (icon, label, value, sub) in zip(row_cols, row_cards):
            with col:
                metric_card(icon, label, value, sub)


def _plot_chart(fig, mobile_layout: bool) -> None:
    """Render Plotly chart with compact height on mobile layouts."""
    if mobile_layout:
        current_height = int(getattr(fig.layout, "height", 320) or 320)
        fig.update_layout(height=max(220, int(current_height * 0.82)), margin=dict(t=36, b=18, l=12, r=12))
    st.plotly_chart(fig, width="stretch")

params = render_sidebar()
mobile_layout = bool(params.get("mobile_layout", False))

if not params["city"]:
    st.info("Enter a city in the sidebar to start.")
    st.stop()

with st.spinner("Loading climate dataset and live signals..."):
    try:
        payload = _load_bundle(params["city"], params["history_years"])
    except Exception as exc:
        st.error(f"Failed to load city data: {exc}")
        st.stop()

if params["retrain"]:
    with st.spinner("Retraining models for this city..."):
        try:
            retrain_models(city=params["city"], history_years=params["history_years"])
            st.success("Models retrained successfully.")
        except Exception as exc:
            st.error(f"Retraining failed: {exc}")

city_info = payload["city_info"]
weather = payload["weather"]
aqi = payload["aqi"]
climate_df = payload["climate_df"]
fallback_note = payload.get("used_fallback", False)

if climate_df.empty:
    st.error("Historical climate dataset is unavailable for this city.")
    st.stop()

scenario = predict_climate_scenario(
    year=params["year"],
    month=params["month"],
    co2_ppm=params["co2_ppm"],
    sea_level_mm=params["sea_level_mm"],
    renewable_share=params["renewable_share"],
    population_growth=params["population_growth"],
    land_use_change=params["land_use_change"],
)

temp_fc = forecast_time_series(climate_df, "temp_mean", params["forecast_days"])
rain_fc = forecast_time_series(climate_df, "precipitation", params["forecast_days"])
aqi_fc = forecast_time_series(climate_df, "aqi_proxy", params["forecast_days"])
metrics_df = get_model_metrics()

current = weather.get("current", {})
aqi_value = float(aqi.get("current_aqi", 0))
aqi_label, aqi_color, aqi_desc = Config.get_aqi_info(aqi_value)

summary = summarize_climate_frame(climate_df)
forecast_summary = {
    "horizon_days": params["forecast_days"],
    "temp_mean": round(float(temp_fc["prediction"].mean()), 2) if not temp_fc.empty else 0,
    "rain_mean": round(float(rain_fc["prediction"].mean()), 2) if not rain_fc.empty else 0,
    "aqi_mean": round(float(aqi_fc["prediction"].mean()), 2) if not aqi_fc.empty else 0,
}

st.markdown(
    f"""
        <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:12px; margin-bottom:10px;">
      <div>
                <h1 style="margin:0; font-size:{'1.6rem' if mobile_layout else '2.2rem'};">Climate Change Forecasting Lab</h1>
        <div style="color:#9ab8bc; font-size:0.9rem;">Exploratory analysis, time-series forecasting, ML prediction, and deep pattern recognition</div>
      </div>
            <div style="text-align:{'left' if mobile_layout else 'right'}; color:#9ab8bc; font-size:0.82rem;">
        {city_info['name']}, {city_info['country']}<br>
        Updated: {current_timestamp()}
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if fallback_note:
    st.warning("Historical archive fetch was unavailable. The app is currently using a synthetic fallback climate series for this session.")

wx_code = int(current.get("weathercode", 0) or 0)
wx_icon = get_weather_icon(wx_code)
wx_desc = get_weather_description(wx_code)

metric_cards = [
    ("Temp", "Current temp", _fmt_temp(current.get("temperature_2m"), params["unit"]), wx_desc),
    ("AQI", "Air quality", str(int(aqi_value)), aqi_label),
    ("Rain", "Avg rainfall", f"{summary['avg_precip']:.2f} mm", "Daily mean"),
    ("Anom", "Avg anomaly", f"{summary['avg_anomaly']:.2f} C", "Vs local baseline"),
    ("Risk", "Scenario risk", f"{scenario['climate_risk_score']:.1f}", risk_label(scenario["climate_risk_score"])),
    ("WX", "Weather", wx_icon, _fmt_temp(current.get("apparent_temperature"), params["unit"])),
]
_render_metric_grid(metric_cards, columns_per_row=2 if mobile_layout else 6)

tab_dash, tab_eda, tab_forecast, tab_models, tab_report = st.tabs(
    [
        "Dashboard",
        "EDA",
        "Forecasting",
        "Scenario and Models",
        "Report",
    ]
)

with tab_dash:
    section_header("Now", "Live Climate Snapshot")
    trend_df = get_temperature_trend_df(weather)

    if mobile_layout:
        if not trend_df.empty:
            _plot_chart(build_temperature_trend(trend_df, city_info["name"]), mobile_layout)
        _plot_chart(build_aqi_gauge(aqi_value), mobile_layout)
        aqi_badge(aqi_label, aqi_color, aqi_desc)
        _plot_chart(build_aqi_trend(aqi.get("hourly", {})), mobile_layout)
        _plot_chart(build_pollutant_radar(aqi.get("hourly", {})), mobile_layout)
    else:
        left, right = st.columns([1.2, 1])
        with left:
            if not trend_df.empty:
                _plot_chart(build_temperature_trend(trend_df, city_info["name"]), mobile_layout)
        with right:
            _plot_chart(build_aqi_gauge(aqi_value), mobile_layout)
            aqi_badge(aqi_label, aqi_color, aqi_desc)

        c_aqi, c_rad = st.columns([2, 1])
        with c_aqi:
            _plot_chart(build_aqi_trend(aqi.get("hourly", {})), mobile_layout)
        with c_rad:
            _plot_chart(build_pollutant_radar(aqi.get("hourly", {})), mobile_layout)

with tab_eda:
    section_header("EDA", "Exploratory Data Analysis")
    _plot_chart(build_long_term_trend(climate_df), mobile_layout)

    if mobile_layout:
        _plot_chart(build_seasonality_heatmap(climate_df), mobile_layout)
        _plot_chart(build_correlation_heatmap(climate_df), mobile_layout)
    else:
        eda_col1, eda_col2 = st.columns(2)
        with eda_col1:
            _plot_chart(build_seasonality_heatmap(climate_df), mobile_layout)
        with eda_col2:
            _plot_chart(build_correlation_heatmap(climate_df), mobile_layout)

    extremes = get_extreme_event_summary(climate_df)
    _plot_chart(build_extreme_events_chart(extremes), mobile_layout)

    st.dataframe(climate_df.tail(40), width="stretch", height=200 if mobile_layout else 260)

with tab_forecast:
    section_header("Forecast", f"{params['forecast_days']}-Day Time-Series Forecast")
    _plot_chart(build_forecast_plot(climate_df.tail(240), temp_fc, "temp_mean", "Temperature (C)"), mobile_layout)
    _plot_chart(build_forecast_plot(climate_df.tail(240), rain_fc, "precipitation", "Rainfall (mm/day)"), mobile_layout)
    _plot_chart(build_forecast_plot(climate_df.tail(240), aqi_fc, "aqi_proxy", "AQI Proxy"), mobile_layout)

    forecast_cards = [
        ("Temp", "Forecast temp avg", f"{forecast_summary['temp_mean']} C", ""),
        ("Rain", "Forecast rain avg", f"{forecast_summary['rain_mean']} mm", ""),
        ("AQI", "Forecast AQI avg", str(forecast_summary["aqi_mean"]), ""),
    ]
    _render_metric_grid(forecast_cards, columns_per_row=1 if mobile_layout else 3)

with tab_models:
    section_header("Scenario", "ML and Deep Climate Pattern Projection")
    prediction_result_card(
        anomaly=scenario["projected_anomaly_c"],
        absolute_temp=scenario["projected_temp_c"],
        confidence=scenario["confidence"],
        year=params["year"],
    )

    if mobile_layout:
        _plot_chart(build_scenario_result_chart(scenario), mobile_layout)
        _plot_chart(build_feature_importance(get_feature_importance()), mobile_layout)
    else:
        m1, m2 = st.columns([1.1, 1])
        with m1:
            _plot_chart(build_scenario_result_chart(scenario), mobile_layout)
        with m2:
            _plot_chart(build_feature_importance(get_feature_importance()), mobile_layout)

    _plot_chart(build_model_metrics_chart(metrics_df), mobile_layout)
    st.dataframe(metrics_df, width="stretch")

with tab_report:
    section_header("Report", "Export Climate Intelligence Summary")
    st.write("Generate a PDF including EDA findings, forecasts, scenario projection, and model metrics.")
    if st.button("Generate report", width="content"):
        report_bytes = generate_climate_report(
            city_info=city_info,
            climate_summary=summary,
            forecast_summary=forecast_summary,
            scenario_result=scenario,
            model_metrics=metrics_df.to_dict("records"),
        )
        st.download_button(
            label="Download PDF",
            data=report_bytes,
            file_name=f"climate_forecast_report_{city_info['name'].replace(' ', '_')}.pdf",
            mime="application/pdf",
            width="content",
        )

st.markdown(
    """
    <hr style="border-color:rgba(255,255,255,0.1)">
    <div style="text-align:center; color:#7f9ea3; font-size:0.78rem; padding-bottom:14px;">
      Open-Meteo data · Streamlit interface · ML + Time-Series + Neural pattern modeling
    </div>
    """,
    unsafe_allow_html=True,
)
