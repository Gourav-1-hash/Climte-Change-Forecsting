"""Train and persist multi-target climate ML models."""

from __future__ import annotations

import os
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from app.config import Config
from app.services.weather_service import get_city_climate_bundle


FEATURE_COLUMNS = [
    "year",
    "month",
    "dayofyear",
    "co2_ppm",
    "sea_level_mm",
    "temp_mean_lag1",
    "temp_mean_lag7",
    "precipitation_lag1",
    "precipitation_lag7",
    "aqi_proxy_lag1",
    "aqi_proxy_lag7",
    "sin_doy",
    "cos_doy",
]


def _clean_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing lag features and ensure numeric dtypes."""
    work = df.copy()
    required_cols = FEATURE_COLUMNS + ["temp_mean", "precipitation", "aqi_proxy", "temp_anomaly"]
    work = work.dropna(subset=required_cols)
    for col in required_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=required_cols)
    return work


def _evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return standard regression metrics for model monitoring."""
    return {
        "target": name,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_and_save(city: str = "New Delhi", history_years: int = 15) -> Tuple[Dict, StandardScaler]:
    """Train full climate model bundle and save artifacts for app inference."""
    bundle = get_city_climate_bundle(city=city, history_years=history_years)
    climate_df = _clean_training_frame(bundle["climate_df"])
    if climate_df.empty:
        raise ValueError("Training data is empty after preprocessing.")

    X = climate_df[FEATURE_COLUMNS].values
    y_temp = climate_df["temp_mean"].values
    y_rain = climate_df["precipitation"].values
    y_aqi = climate_df["aqi_proxy"].values
    y_anomaly = climate_df["temp_anomaly"].values

    X_train, X_test, y_temp_train, y_temp_test = train_test_split(
        X, y_temp, test_size=0.2, random_state=42
    )
    _, _, y_rain_train, y_rain_test = train_test_split(
        X, y_rain, test_size=0.2, random_state=42
    )
    _, _, y_aqi_train, y_aqi_test = train_test_split(
        X, y_aqi, test_size=0.2, random_state=42
    )
    _, _, y_anomaly_train, y_anomaly_test = train_test_split(
        X, y_anomaly, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    temp_model = RandomForestRegressor(
        n_estimators=140,
        max_depth=10,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1,
    )
    rain_model = GradientBoostingRegressor(random_state=42)
    aqi_model = RandomForestRegressor(
        n_estimators=110,
        max_depth=9,
        min_samples_split=3,
        random_state=42,
        n_jobs=-1,
    )
    deep_pattern_model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        learning_rate_init=0.002,
        random_state=42,
        max_iter=450,
    )

    temp_model.fit(X_train_s, y_temp_train)
    rain_model.fit(X_train_s, y_rain_train)
    aqi_model.fit(X_train_s, y_aqi_train)
    deep_pattern_model.fit(X_train_s, y_anomaly_train)

    temp_pred = temp_model.predict(X_test_s)
    rain_pred = rain_model.predict(X_test_s)
    aqi_pred = aqi_model.predict(X_test_s)
    deep_pred = deep_pattern_model.predict(X_test_s)

    metrics = [
        _evaluate("temperature", y_temp_test, temp_pred),
        _evaluate("rainfall", y_rain_test, rain_pred),
        _evaluate("aqi_proxy", y_aqi_test, aqi_pred),
        _evaluate("temp_anomaly_deep", y_anomaly_test, deep_pred),
    ]

    model_bundle = {
        "temp_model": temp_model,
        "rain_model": rain_model,
        "aqi_model": aqi_model,
        "deep_pattern_model": deep_pattern_model,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": metrics,
        "train_city": city,
        "history_years": history_years,
        "training_frame": climate_df,
    }

    os.makedirs(os.path.dirname(Config.MODEL_PATH), exist_ok=True)
    joblib.dump(model_bundle, Config.MODEL_PATH)
    joblib.dump(scaler, Config.SCALER_PATH)

    return model_bundle, scaler


if __name__ == "__main__":
    trained_bundle, _ = train_and_save()
    print("Training completed.")
    for metric in trained_bundle["metrics"]:
        print(
            f"- {metric['target']}: MAE={metric['mae']:.3f}, R2={metric['r2']:.3f}"
        )
