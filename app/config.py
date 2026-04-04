"""
config.py - Centralized configuration management
Loads environment variables securely via python-dotenv
"""

import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()


class Config:
    """Application configuration loaded from environment variables."""

    # --- OpenAI ---
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # --- Default Values ---
    DEFAULT_CITY: str = os.getenv("DEFAULT_CITY", "New Delhi")

    # --- Open-Meteo API (no key required) ---
    GEOCODING_API: str = "https://geocoding-api.open-meteo.com/v1/search"
    WEATHER_API: str = "https://api.open-meteo.com/v1/forecast"

    # --- Open-Meteo Air Quality API (no key required) ---
    AQI_API: str = "https://air-quality-api.open-meteo.com/v1/air-quality"

    # --- ML Model ---
    MODEL_PATH: str = os.path.join(os.path.dirname(__file__), "models", "climate_model.joblib")
    SCALER_PATH: str = os.path.join(os.path.dirname(__file__), "models", "scaler.joblib")

    # --- App Meta ---
    APP_TITLE: str = "Climate Change Forecasting Lab"
    APP_VERSION: str = "2.0.0"
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    MAX_CHAT_HISTORY: int = 20

    # --- Analysis defaults ---
    DEFAULT_HISTORY_YEARS: int = int(os.getenv("DEFAULT_HISTORY_YEARS", "15"))
    DEFAULT_FORECAST_DAYS: int = int(os.getenv("DEFAULT_FORECAST_DAYS", "30"))

    # --- AQI Thresholds ---
    AQI_LEVELS = {
        (0, 50): ("Good", "#00E400", "Air quality is satisfactory."),
        (51, 100): ("Moderate", "#FFFF00", "Acceptable; some pollutants may affect sensitive groups."),
        (101, 150): ("Unhealthy for Sensitive Groups", "#FF7E00", "Sensitive groups may experience health effects."),
        (151, 200): ("Unhealthy", "#FF0000", "Everyone may begin to experience health effects."),
        (201, 300): ("Very Unhealthy", "#8F3F97", "Health alerts; everyone may experience serious effects."),
        (301, 500): ("Hazardous", "#7E0023", "Emergency conditions. The entire population is affected."),
    }

    @classmethod
    def get_aqi_info(cls, aqi: float) -> tuple:
        """Return (label, color, description) for an AQI value."""
        for (lo, hi), info in cls.AQI_LEVELS.items():
            if lo <= aqi <= hi:
                return info
        return ("Hazardous", "#7E0023", "Extreme pollution level.")

    @classmethod
    def validate(cls) -> dict:
        """Return a dict of missing / misconfigured keys."""
        issues = {}
        if not cls.OPENAI_API_KEY or cls.OPENAI_API_KEY == "your_openai_api_key_here":
            issues["OPENAI_API_KEY"] = "Not set — AI Assistant will be unavailable."
        return issues
