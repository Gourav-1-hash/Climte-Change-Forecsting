"""General utility helpers used across the climate forecasting application."""

from __future__ import annotations

from datetime import datetime
from typing import Dict


def get_weather_icon(weathercode: int) -> str:
    if weathercode == 0:
        return "☀"
    if weathercode in (1, 2, 3):
        return "⛅"
    if weathercode in range(45, 50):
        return "🌫"
    if weathercode in range(51, 68):
        return "🌧"
    if weathercode in range(71, 78):
        return "❄"
    if weathercode in range(80, 87):
        return "🌦"
    if weathercode in range(95, 100):
        return "⛈"
    return "🌡"


def get_weather_description(weathercode: int) -> str:
    descriptions = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Rime fog",
        51: "Light drizzle",
        53: "Drizzle",
        55: "Dense drizzle",
        61: "Light rain",
        63: "Rain",
        65: "Heavy rain",
        71: "Light snowfall",
        73: "Snowfall",
        75: "Heavy snowfall",
        80: "Rain showers",
        95: "Thunderstorm",
    }
    return descriptions.get(weathercode, "Unknown weather")


def celsius_to_fahrenheit(celsius: float) -> float:
    return round(celsius * 9 / 5 + 32, 1)


def format_large_number(value: float) -> str:
    return f"{value:,.2f}"


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def current_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def risk_label(score: float) -> str:
    if score < 30:
        return "Low"
    if score < 60:
        return "Moderate"
    if score < 80:
        return "High"
    return "Severe"


def summarize_climate_frame(climate_df) -> Dict[str, float]:
    if climate_df is None or climate_df.empty:
        return {
            "avg_temp": 0.0,
            "avg_precip": 0.0,
            "avg_anomaly": 0.0,
            "extreme_heat_days": 0,
            "extreme_rain_days": 0,
        }

    return {
        "avg_temp": round(float(climate_df["temp_mean"].mean()), 2),
        "avg_precip": round(float(climate_df["precipitation"].mean()), 2),
        "avg_anomaly": round(float(climate_df["temp_anomaly"].mean()), 2),
        "extreme_heat_days": int(climate_df["is_extreme_heat"].sum()),
        "extreme_rain_days": int(climate_df["is_extreme_rain"].sum()),
    }
