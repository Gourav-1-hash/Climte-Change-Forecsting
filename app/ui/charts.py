"""Plotly chart builders used across EDA, forecasting, and scenario tabs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.config import Config


def _apply_base_layout(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=45, b=25, l=20, r=20),
        height=height,
        font=dict(color="#e6e6e6"),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
    )
    return fig


def build_temperature_trend(df: pd.DataFrame, city: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["temperature"],
            mode="lines",
            name="Temperature",
            line=dict(color="#5de2e7", width=2.4),
            fill="tozeroy",
            fillcolor="rgba(93,226,231,0.16)",
        )
    )
    fig.update_layout(title=f"Hourly Temperature Trend - {city}", yaxis_title="Temperature (C)")
    return _apply_base_layout(fig, height=320)


def build_aqi_gauge(aqi_value: float) -> go.Figure:
    label, color, _ = Config.get_aqi_info(aqi_value)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=aqi_value,
            title={"text": f"AQI - {label}"},
            number={"suffix": " AQI", "font": {"size": 34, "color": color}},
            gauge={
                "axis": {"range": [0, 500]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 50], "color": "rgba(0,228,0,0.2)"},
                    {"range": [50, 100], "color": "rgba(255,255,0,0.2)"},
                    {"range": [100, 150], "color": "rgba(255,126,0,0.2)"},
                    {"range": [150, 200], "color": "rgba(255,0,0,0.2)"},
                    {"range": [200, 300], "color": "rgba(143,63,151,0.2)"},
                    {"range": [300, 500], "color": "rgba(126,0,35,0.2)"},
                ],
            },
        )
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=260)
    return fig


def build_aqi_trend(hourly: dict) -> go.Figure:
    dates = pd.to_datetime(hourly.get("time", []), errors="coerce")
    values = [x if x is not None else np.nan for x in hourly.get("us_aqi", [])]
    fig = go.Figure(
        go.Scatter(
            x=dates,
            y=values,
            mode="lines+markers",
            line=dict(color="#f9a826", width=2),
            marker=dict(size=5),
            name="AQI",
        )
    )
    fig.update_layout(title="AQI Trend (72h)", yaxis_title="US AQI")
    return _apply_base_layout(fig, height=280)


def build_pollutant_radar(hourly: dict) -> go.Figure:
    def avg(key: str) -> float:
        vals = [v for v in hourly.get(key, []) if v is not None]
        return float(np.mean(vals)) if vals else 0.0

    labels = ["PM2.5", "PM10", "CO/100", "NO2", "Ozone"]
    values = [
        avg("pm2_5"),
        avg("pm10"),
        avg("carbon_monoxide") / 100,
        avg("nitrogen_dioxide"),
        avg("ozone"),
    ]

    fig = go.Figure(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=labels + [labels[0]],
            fill="toself",
            line=dict(color="#0ea5a8", width=2),
            fillcolor="rgba(14,165,168,0.22)",
        )
    )
    fig.update_layout(title="Pollutant Composition", polar=dict(bgcolor="rgba(0,0,0,0)"), showlegend=False, height=280)
    return fig


def build_feature_importance(importance_dict: dict) -> go.Figure:
    frame = pd.DataFrame({"feature": list(importance_dict.keys()), "importance": list(importance_dict.values())})
    frame = frame.sort_values("importance", ascending=True).tail(12)
    fig = px.bar(
        frame,
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale="Teal",
    )
    fig.update_layout(title="Top Feature Importance", coloraxis_showscale=False)
    return _apply_base_layout(fig, height=360)


def build_long_term_trend(climate_df: pd.DataFrame) -> go.Figure:
    yearly = climate_df.groupby("year", as_index=False).agg(temp_mean=("temp_mean", "mean"), precip_mean=("precipitation", "mean"))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yearly["year"], y=yearly["temp_mean"], mode="lines+markers", name="Temperature (C)", line=dict(color="#5de2e7", width=2.8)))
    fig.add_trace(go.Scatter(x=yearly["year"], y=yearly["precip_mean"], mode="lines", name="Precipitation (mm/day)", yaxis="y2", line=dict(color="#7bd389", width=2)))
    fig.update_layout(
        title="Long-term Climate Trend",
        xaxis_title="Year",
        yaxis_title="Temperature (C)",
        yaxis2=dict(title="Precipitation", overlaying="y", side="right", showgrid=False),
    )
    return _apply_base_layout(fig, height=350)


def build_seasonality_heatmap(climate_df: pd.DataFrame) -> go.Figure:
    pivot = climate_df.pivot_table(index="month", columns="year", values="temp_mean", aggfunc="mean")
    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="Turbo",
        labels={"x": "Year", "y": "Month", "color": "Temp (C)"},
    )
    fig.update_layout(title="Seasonality Heatmap (Monthly Mean Temperature)")
    return _apply_base_layout(fig, height=360)


def build_correlation_heatmap(climate_df: pd.DataFrame) -> go.Figure:
    cols = ["temp_mean", "temp_max", "temp_min", "precipitation", "windspeed", "co2_ppm", "sea_level_mm", "aqi_proxy"]
    corr = climate_df[cols].corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu", zmin=-1, zmax=1)
    fig.update_layout(title="Feature Correlation Matrix")
    return _apply_base_layout(fig, height=380)


def build_extreme_events_chart(summary_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=summary_df["year"], y=summary_df["extreme_heat_days"], name="Extreme Heat", marker_color="#f97316"))
    fig.add_trace(go.Bar(x=summary_df["year"], y=summary_df["extreme_rain_days"], name="Extreme Rain", marker_color="#38bdf8"))
    fig.update_layout(title="Extreme Events by Year", barmode="group", xaxis_title="Year", yaxis_title="Days")
    return _apply_base_layout(fig, height=340)


def build_forecast_plot(history_df: pd.DataFrame, forecast_df: pd.DataFrame, target_col: str, label: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history_df["date"],
            y=history_df[target_col],
            mode="lines",
            name="Historical",
            line=dict(color="#a1a1aa", width=1.6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["prediction"],
            mode="lines",
            name="Forecast",
            line=dict(color="#22d3ee", width=2.7),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pd.concat([forecast_df["date"], forecast_df["date"].iloc[::-1]]),
            y=pd.concat([forecast_df["upper"], forecast_df["lower"].iloc[::-1]]),
            fill="toself",
            fillcolor="rgba(34,211,238,0.18)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Confidence band",
        )
    )
    fig.update_layout(title=f"Forecast - {label}", yaxis_title=label)
    return _apply_base_layout(fig, height=340)


def build_model_metrics_chart(metrics_df: pd.DataFrame) -> go.Figure:
    frame = metrics_df.copy()
    frame["label"] = frame["target"] + " (R2)"
    fig = px.bar(frame, x="target", y="r2", color="target", text=frame["r2"].round(3))
    fig.update_layout(title="Model Accuracy (R2)", xaxis_title="Target", yaxis_title="R2 Score", showlegend=False)
    return _apply_base_layout(fig, height=300)


def build_scenario_result_chart(result: dict) -> go.Figure:
    labels = ["Temperature C", "Rainfall mm", "AQI", "Risk Score"]
    values = [
        result.get("projected_temp_c", 0),
        result.get("projected_rainfall_mm", 0),
        result.get("projected_aqi", 0),
        result.get("climate_risk_score", 0),
    ]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=["#5de2e7", "#7bd389", "#f59e0b", "#f43f5e"]))
    fig.update_layout(title="Scenario Output Summary")
    return _apply_base_layout(fig, height=300)
