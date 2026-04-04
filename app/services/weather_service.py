"""Data access and climate dataset construction utilities."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import requests

from app.config import Config

ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"


def _safe_json_get(url: str, params: Dict[str, Any], timeout: int = 20) -> Dict[str, Any]:
    """Execute a GET request and return parsed JSON or raise ConnectionError."""
    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        raise ConnectionError(f"Request failed for {url}: {exc}") from exc


def detect_network_city() -> Optional[str]:
    """Detect a default city from public IP providers."""
    providers = ["https://ipapi.co/json/", "https://ipwho.is/"]
    headers = {"User-Agent": "climate-forecasting-hub/2.0"}

    for url in providers:
        try:
            response = requests.get(url, headers=headers, timeout=4)
            response.raise_for_status()
            payload = response.json()

            if "ipwho.is" in url and payload.get("success") is False:
                continue

            city = str(payload.get("city", "")).strip()
            country = str(payload.get("country_name") or payload.get("country") or "").strip()
            if city:
                return f"{city}, {country}" if country else city
        except Exception:
            continue
    return None


def get_coordinates(city: str) -> Optional[Dict[str, Any]]:
    """Resolve city into geographic coordinates using Open-Meteo geocoding."""
    payload = _safe_json_get(
        Config.GEOCODING_API,
        {"name": city, "count": 1, "language": "en", "format": "json"},
        timeout=10,
    )
    results = payload.get("results", [])
    if not results:
        return None

    row = results[0]
    return {
        "name": row.get("name", city),
        "country": row.get("country", ""),
        "latitude": float(row["latitude"]),
        "longitude": float(row["longitude"]),
        "timezone": row.get("timezone", "auto"),
    }


def get_weather(lat: float, lon: float, timezone: str = "auto") -> Dict[str, Any]:
    """Fetch current and hourly weather forecast data."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": timezone,
        "current": "temperature_2m,apparent_temperature,relative_humidity_2m,windspeed_10m,weathercode,precipitation",
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,windspeed_10m",
        "forecast_days": 7,
    }
    return _safe_json_get(Config.WEATHER_API, params, timeout=15)


def get_aqi(lat: float, lon: float, timezone: str = "auto") -> Dict[str, Any]:
    """Fetch hourly AQI and pollutant data and compute latest AQI."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": timezone,
        "hourly": "us_aqi,pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,ozone",
        "forecast_days": 3,
    }
    payload = _safe_json_get(Config.AQI_API, params, timeout=15)
    hourly = payload.get("hourly", {})
    series = hourly.get("us_aqi", [])
    latest = next((value for value in reversed(series) if value is not None), 0)
    return {"current_aqi": float(latest or 0), "hourly": hourly}


def get_historical_daily(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    timezone: str = "auto",
) -> Dict[str, Any]:
    """Fetch historical daily weather series from Open-Meteo archive endpoint."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": timezone,
        "daily": (
            "temperature_2m_mean,temperature_2m_max,temperature_2m_min,"
            "precipitation_sum,windspeed_10m_max"
        ),
    }
    return _safe_json_get(ARCHIVE_API, params, timeout=30)


def _co2_proxy(year: pd.Series) -> pd.Series:
    """Generate a smooth CO2 curve used for scenario/feature modeling."""
    offset = year - 1950
    return 310 + 1.45 * offset + 0.0055 * (offset ** 2)


def _sea_level_proxy(year: pd.Series) -> pd.Series:
    """Generate a sea-level-rise proxy (mm) as an input feature."""
    offset = year - 1950
    return 2 + 1.7 * offset + 0.018 * (offset ** 2)


def _synthetic_daily_dataset(history_years: int) -> pd.DataFrame:
    """Return synthetic but physically-consistent climate daily data fallback."""
    today = date.today()
    start = today - timedelta(days=history_years * 365)
    idx = pd.date_range(start=start, end=today - timedelta(days=1), freq="D")
    n = len(idx)

    rng = np.random.default_rng(42)
    doy = idx.dayofyear.values
    years = idx.year.values

    season = 11 * np.sin(2 * np.pi * (doy / 365.25))
    trend = 0.03 * (years - years.min())
    temp_mean = 18 + season + trend + rng.normal(0, 1.3, n)
    temp_max = temp_mean + rng.normal(6.0, 1.0, n)
    temp_min = temp_mean - rng.normal(6.2, 1.0, n)

    precip = np.clip(rng.gamma(shape=1.8, scale=2.4, size=n) + 2.0 * np.sin(2 * np.pi * (doy / 365.25 + 0.2)), 0, None)
    wind = np.clip(rng.normal(18, 4.5, n), 4, None)

    df = pd.DataFrame(
        {
            "date": idx,
            "temp_mean": temp_mean,
            "temp_max": temp_max,
            "temp_min": temp_min,
            "precipitation": precip,
            "windspeed": wind,
        }
    )
    return enrich_climate_dataframe(df)


def enrich_climate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add climate engineered features used by EDA, ML, and forecasting flows."""
    if df.empty:
        return df

    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").reset_index(drop=True)

    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["dayofyear"] = data["date"].dt.dayofyear

    data["co2_ppm"] = _co2_proxy(data["year"]).round(2)
    data["sea_level_mm"] = _sea_level_proxy(data["year"]).round(2)

    baseline = float(data["temp_mean"].head(min(365, len(data))).mean())
    data["temp_anomaly"] = (data["temp_mean"] - baseline).round(3)

    data["temp_roll_30"] = data["temp_mean"].rolling(30, min_periods=5).mean()
    data["precip_roll_30"] = data["precipitation"].rolling(30, min_periods=5).mean()

    heat_threshold = float(data["temp_max"].quantile(0.9))
    rain_threshold = float(data["precipitation"].quantile(0.9))
    data["is_extreme_heat"] = (data["temp_max"] >= heat_threshold).astype(int)
    data["is_extreme_rain"] = (data["precipitation"] >= rain_threshold).astype(int)

    # AQI proxy derived from weather and emissions indicators for modeling exercises.
    aqi_proxy = (
        45
        + 0.32 * data["temp_mean"]
        + 0.022 * data["co2_ppm"]
        - 0.85 * data["precipitation"]
        - 0.28 * data["windspeed"]
    )
    data["aqi_proxy"] = np.clip(aqi_proxy, 15, 380).round(1)

    for col in ("temp_mean", "precipitation", "aqi_proxy"):
        data[f"{col}_lag1"] = data[col].shift(1)
        data[f"{col}_lag7"] = data[col].shift(7)

    data["sin_doy"] = np.sin(2 * np.pi * data["dayofyear"] / 365.25)
    data["cos_doy"] = np.cos(2 * np.pi * data["dayofyear"] / 365.25)

    return data


def build_analysis_dataframe(historical_payload: Dict[str, Any]) -> pd.DataFrame:
    """Transform archive API payload into a feature-rich climate dataframe."""
    daily = historical_payload.get("daily", {})
    if not daily or not daily.get("time"):
        return pd.DataFrame()

    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(daily.get("time", []), errors="coerce"),
            "temp_mean": daily.get("temperature_2m_mean", []),
            "temp_max": daily.get("temperature_2m_max", []),
            "temp_min": daily.get("temperature_2m_min", []),
            "precipitation": daily.get("precipitation_sum", []),
            "windspeed": daily.get("windspeed_10m_max", []),
        }
    )
    raw = raw.dropna(subset=["date"])
    raw = raw.sort_values("date").reset_index(drop=True)

    for col in ["temp_mean", "temp_max", "temp_min", "precipitation", "windspeed"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
        raw[col] = raw[col].interpolate(limit_direction="both")

    raw = raw.fillna(method="bfill").fillna(method="ffill")
    return enrich_climate_dataframe(raw)


def get_hourly_forecast_df(weather_payload: Dict[str, Any]) -> pd.DataFrame:
    """Convert weather hourly payload to a DataFrame used for forecast visuals."""
    hourly = weather_payload.get("hourly", {})
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(hourly.get("time", []), errors="coerce"),
            "temperature": hourly.get("temperature_2m", []),
            "humidity": hourly.get("relative_humidity_2m", []),
            "precipitation": hourly.get("precipitation", []),
            "windspeed": hourly.get("windspeed_10m", []),
        }
    )
    return df.dropna(subset=["time"])


def get_temperature_trend_df(weather_payload: Dict[str, Any]) -> pd.DataFrame:
    """Backward-compatible helper returning hourly temperature trend dataframe."""
    hourly = get_hourly_forecast_df(weather_payload)
    return hourly[["time", "temperature"]].dropna()


def get_extreme_event_summary(climate_df: pd.DataFrame) -> pd.DataFrame:
    """Return yearly counts of extreme heat and rainfall events."""
    if climate_df.empty:
        return pd.DataFrame(columns=["year", "extreme_heat_days", "extreme_rain_days"])

    summary = (
        climate_df.groupby("year", as_index=False)
        .agg(
            extreme_heat_days=("is_extreme_heat", "sum"),
            extreme_rain_days=("is_extreme_rain", "sum"),
        )
        .sort_values("year")
    )
    return summary


def get_city_climate_bundle(city: str, history_years: int = 15) -> Dict[str, Any]:
    """Load complete city dataset (historical, weather, AQI) with fallback safeguards."""
    city_info = get_coordinates(city)
    if not city_info:
        raise ValueError(f"City '{city}' not found.")

    lat = city_info["latitude"]
    lon = city_info["longitude"]
    tz = city_info.get("timezone", "auto")

    weather_payload = get_weather(lat, lon, tz)
    aqi_payload = get_aqi(lat, lon, tz)

    end_dt = date.today() - timedelta(days=1)
    start_dt = end_dt - timedelta(days=history_years * 365)

    try:
        hist_payload = get_historical_daily(
            lat=lat,
            lon=lon,
            start_date=start_dt.isoformat(),
            end_date=end_dt.isoformat(),
            timezone=tz,
        )
        climate_df = build_analysis_dataframe(hist_payload)
        used_fallback = climate_df.empty
    except Exception:
        climate_df = pd.DataFrame()
        used_fallback = True

    if used_fallback:
        climate_df = _synthetic_daily_dataset(history_years)

    return {
        "city_info": city_info,
        "weather": weather_payload,
        "aqi": aqi_payload,
        "climate_df": climate_df,
        "used_fallback": used_fallback,
    }
