"""Prediction and forecasting utilities for the climate app."""

from __future__ import annotations

import os
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from app.config import Config

_bundle = None
_scaler = None


def _load_or_train(city: str = "New Delhi", history_years: int = 15, retrain: bool = False):
    """Load model artifacts from disk or trigger model training when missing."""
    global _bundle, _scaler
    if _bundle is not None and _scaler is not None and not retrain:
        return

    if retrain or not os.path.exists(Config.MODEL_PATH) or not os.path.exists(Config.SCALER_PATH):
        from app.models.train_model import train_and_save

        _bundle, _scaler = train_and_save(city=city, history_years=history_years)
        return

    _bundle = joblib.load(Config.MODEL_PATH)
    _scaler = joblib.load(Config.SCALER_PATH)


def get_training_dataframe() -> pd.DataFrame:
    """Expose the training dataframe for EDA and summaries in the UI."""
    _load_or_train()
    frame = _bundle.get("training_frame")
    return frame.copy() if frame is not None else pd.DataFrame()


def get_model_metrics() -> pd.DataFrame:
    """Return model metrics as a dataframe for table/chart rendering."""
    _load_or_train()
    return pd.DataFrame(_bundle.get("metrics", []))


def retrain_models(city: str, history_years: int) -> pd.DataFrame:
    """Force a model refresh and return latest metrics."""
    _load_or_train(city=city, history_years=history_years, retrain=True)
    return get_model_metrics()


def _confidence_label(target_year: int, training_year_max: int) -> str:
    if target_year <= training_year_max:
        return "High"
    if target_year <= training_year_max + 7:
        return "Medium"
    return "Low"


def predict_climate_scenario(
    year: int,
    month: int,
    co2_ppm: float,
    sea_level_mm: float,
    renewable_share: float,
    population_growth: float,
    land_use_change: float,
) -> Dict[str, Any]:
    """Predict temperature, rainfall, AQI proxy, and deep climate risk signal."""
    _load_or_train()

    training_df = get_training_dataframe()
    if training_df.empty:
        raise ValueError("Training dataframe unavailable for scenario projection.")

    reference = training_df.iloc[-1].copy()
    dayofyear = int(max(1, min(365, (month - 1) * 30 + 15)))
    scenario_co2 = co2_ppm * (1 + (population_growth / 100.0) * 0.7) * (1 - (renewable_share / 100.0) * 0.35)
    scenario_sea = sea_level_mm * (1 + max(0.0, land_use_change) / 100.0 * 0.2)

    row = {
        "year": float(year),
        "month": float(month),
        "dayofyear": float(dayofyear),
        "co2_ppm": float(scenario_co2),
        "sea_level_mm": float(scenario_sea),
        "temp_mean_lag1": float(reference["temp_mean_lag1"]),
        "temp_mean_lag7": float(reference["temp_mean_lag7"]),
        "precipitation_lag1": float(reference["precipitation_lag1"]),
        "precipitation_lag7": float(reference["precipitation_lag7"]),
        "aqi_proxy_lag1": float(reference["aqi_proxy_lag1"]),
        "aqi_proxy_lag7": float(reference["aqi_proxy_lag7"]),
        "sin_doy": float(np.sin(2 * np.pi * dayofyear / 365.25)),
        "cos_doy": float(np.cos(2 * np.pi * dayofyear / 365.25)),
    }

    X = np.array([[row[col] for col in _bundle["feature_columns"]]], dtype=float)
    Xs = _scaler.transform(X)

    temp_pred = float(_bundle["temp_model"].predict(Xs)[0])
    rain_pred = float(max(0.0, _bundle["rain_model"].predict(Xs)[0]))
    aqi_pred = float(max(0.0, _bundle["aqi_model"].predict(Xs)[0]))
    deep_signal = float(_bundle["deep_pattern_model"].predict(Xs)[0])

    baseline_temp = float(training_df["temp_mean"].head(min(365, len(training_df))).mean())
    anomaly = temp_pred - baseline_temp

    risk_score = 50 + 16 * anomaly + 0.09 * aqi_pred + 0.4 * rain_pred - 0.35 * renewable_share
    risk_score = float(np.clip(risk_score, 0, 100))
    training_year_max = int(training_df["year"].max())

    return {
        "projected_temp_c": round(temp_pred, 2),
        "projected_rainfall_mm": round(rain_pred, 2),
        "projected_aqi": round(aqi_pred, 1),
        "projected_anomaly_c": round(anomaly, 2),
        "deep_pattern_signal": round(deep_signal, 3),
        "climate_risk_score": round(risk_score, 1),
        "confidence": _confidence_label(year, training_year_max),
    }


def predict_temperature(year: int, co2_ppm: float, sea_level_mm: float) -> Dict[str, Any]:
    """Compatibility wrapper for earlier UI usage."""
    result = predict_climate_scenario(
        year=year,
        month=6,
        co2_ppm=co2_ppm,
        sea_level_mm=sea_level_mm,
        renewable_share=35,
        population_growth=1.2,
        land_use_change=5,
    )
    return {
        "anomaly": result["projected_anomaly_c"],
        "absolute_approx": result["projected_temp_c"],
        "confidence": result["confidence"],
    }


def forecast_time_series(climate_df: pd.DataFrame, target_col: str, horizon_days: int = 30) -> pd.DataFrame:
    """Iterative autoregressive forecast with uncertainty bounds."""
    if climate_df.empty or target_col not in climate_df.columns:
        return pd.DataFrame(columns=["date", "prediction", "lower", "upper"])

    data = climate_df[["date", target_col]].copy().sort_values("date")
    data["date"] = pd.to_datetime(data["date"])
    data["lag1"] = data[target_col].shift(1)
    data["lag7"] = data[target_col].shift(7)
    data["roll7"] = data[target_col].rolling(7).mean().shift(1)
    data["doy"] = data["date"].dt.dayofyear
    data["sin_doy"] = np.sin(2 * np.pi * data["doy"] / 365.25)
    data["cos_doy"] = np.cos(2 * np.pi * data["doy"] / 365.25)
    data = data.dropna()

    if data.empty:
        return pd.DataFrame(columns=["date", "prediction", "lower", "upper"])

    feat_cols = ["lag1", "lag7", "roll7", "sin_doy", "cos_doy"]
    X = data[feat_cols].values
    y = data[target_col].values

    reg = LinearRegression()
    reg.fit(X, y)
    residuals = y - reg.predict(X)
    residual_std = float(np.std(residuals)) if len(residuals) > 1 else 0.5

    history_vals = list(climate_df.sort_values("date")[target_col].tail(30).values)
    current_date = pd.to_datetime(climate_df["date"].max())
    rows = []

    for _ in range(int(horizon_days)):
        current_date = current_date + pd.Timedelta(days=1)
        lag1 = history_vals[-1]
        lag7 = history_vals[-7] if len(history_vals) >= 7 else lag1
        roll7 = float(np.mean(history_vals[-7:])) if len(history_vals) >= 7 else lag1
        doy = current_date.dayofyear
        sin_doy = float(np.sin(2 * np.pi * doy / 365.25))
        cos_doy = float(np.cos(2 * np.pi * doy / 365.25))

        pred = float(reg.predict([[lag1, lag7, roll7, sin_doy, cos_doy]])[0])
        lower = pred - 1.64 * residual_std
        upper = pred + 1.64 * residual_std
        rows.append({"date": current_date, "prediction": pred, "lower": lower, "upper": upper})
        history_vals.append(pred)

    return pd.DataFrame(rows)


def get_feature_importance() -> Dict[str, float]:
    """Return feature importance values for the temperature model."""
    _load_or_train()
    model = _bundle["temp_model"]
    values = model.feature_importances_.tolist()
    return dict(zip(_bundle["feature_columns"], values))
