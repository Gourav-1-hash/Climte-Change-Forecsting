"""Sidebar controls for analysis and scenario configuration."""

from __future__ import annotations

import datetime as dt

import streamlit as st

from app.config import Config
from app.services.weather_service import detect_network_city


SCENARIO_DEFAULTS = {
    "Optimistic": {"co2": 430.0, "sea": 130.0, "renewable": 65.0, "pop": 0.8, "land": -5.0},
    "Moderate": {"co2": 520.0, "sea": 220.0, "renewable": 45.0, "pop": 1.2, "land": 5.0},
    "Worst Case": {"co2": 680.0, "sea": 360.0, "renewable": 25.0, "pop": 1.8, "land": 18.0},
}


def _init_sidebar_state() -> None:
    if "detected_default_city" not in st.session_state:
        detected = detect_network_city()
        st.session_state["detected_default_city"] = detected or Config.DEFAULT_CITY
        st.session_state["detected_city_source"] = "network" if detected else "config"

    if "city_input" not in st.session_state:
        st.session_state["city_input"] = st.session_state["detected_default_city"]

    if "scenario_name" not in st.session_state:
        st.session_state["scenario_name"] = "Moderate"


def render_sidebar() -> dict:
    """Render sidebar and return all selected controls in one dictionary."""
    _init_sidebar_state()

    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center; padding: 10px 0 8px 0;">
                <div style="font-size:2.1rem;">Climate Lab</div>
                <p style="margin:4px 0 0 0; color:#8f8f8f; font-size:0.8rem;">Forecasting and Risk Intelligence</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Location")
        if st.session_state.get("detected_city_source") == "network":
            st.caption(f"Auto-detected: {st.session_state['detected_default_city']}")

        city = st.text_input("City", key="city_input", placeholder="e.g. Delhi, London, Nairobi")

        st.markdown("### Data Window")
        history_years = st.slider("Historical years", min_value=5, max_value=35, value=Config.DEFAULT_HISTORY_YEARS, step=1)
        forecast_days = st.slider("Forecast horizon (days)", min_value=7, max_value=120, value=Config.DEFAULT_FORECAST_DAYS, step=1)

        st.markdown("### Scenario Controls")
        scenario_name = st.selectbox("Emissions scenario", options=list(SCENARIO_DEFAULTS.keys()), index=1)
        defaults = SCENARIO_DEFAULTS[scenario_name]

        projection_year = st.slider(
            "Projection year",
            min_value=dt.date.today().year,
            max_value=2100,
            value=min(dt.date.today().year + 15, 2100),
            step=1,
        )
        projection_month = st.select_slider(
            "Projection month",
            options=list(range(1, 13)),
            value=6,
            format_func=lambda x: dt.date(2000, x, 1).strftime("%B"),
        )

        co2_ppm = st.slider("CO2 concentration (ppm)", min_value=380.0, max_value=900.0, value=defaults["co2"], step=5.0)
        sea_level_mm = st.slider("Sea level rise (mm above 1950)", min_value=0.0, max_value=1400.0, value=defaults["sea"], step=5.0)
        renewable_share = st.slider("Renewable adoption (%)", min_value=0.0, max_value=100.0, value=defaults["renewable"], step=1.0)
        population_growth = st.slider("Population growth (%)", min_value=0.0, max_value=4.0, value=defaults["pop"], step=0.1)
        land_use_change = st.slider("Land-use change index", min_value=-20.0, max_value=30.0, value=defaults["land"], step=1.0)

        st.markdown("### Preferences")
        unit = st.radio("Temperature unit", ["Celsius", "Fahrenheit"], horizontal=True)
        retrain = st.button("Retrain models with current city", width="stretch")

        issues = Config.validate()
        if issues:
            for key, msg in issues.items():
                st.warning(f"{key}: {msg}")

        st.caption(f"Version {Config.APP_VERSION}")

    return {
        "city": (city or st.session_state.get("detected_default_city", Config.DEFAULT_CITY)).strip(),
        "history_years": history_years,
        "forecast_days": forecast_days,
        "scenario": scenario_name,
        "year": projection_year,
        "month": projection_month,
        "co2_ppm": co2_ppm,
        "sea_level_mm": sea_level_mm,
        "renewable_share": renewable_share,
        "population_growth": population_growth,
        "land_use_change": land_use_change,
        "unit": unit,
        "retrain": retrain,
    }
