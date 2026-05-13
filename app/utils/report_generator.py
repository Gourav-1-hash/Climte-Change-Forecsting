"""PDF report generation for climate forecasting outcomes."""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any, Dict, Iterable, List

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _table(data: List[List[str]]) -> Table:
    table = Table(data, colWidths=[2.4 * inch, 4.1 * inch])
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef6f7")),
                ("ROWBACKGROUNDS", (1, 0), (1, -1), [colors.white, colors.HexColor("#fafdfd")]),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d9e3e4")),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _metrics_rows(metrics: Iterable[Dict[str, Any]]) -> List[List[str]]:
    rows = []
    for metric in metrics:
        rows.append(
            [
                str(metric.get("target", "n/a")),
                f"MAE {metric.get('mae', 0):.3f}, R2 {metric.get('r2', 0):.3f}",
            ]
        )
    return rows or [["No metrics", "Unavailable"]]


def generate_climate_report(
    city_info: Dict[str, Any],
    climate_summary: Dict[str, Any],
    forecast_summary: Dict[str, Any],
    scenario_result: Dict[str, Any],
    model_metrics: Iterable[Dict[str, Any]] | None = None,
) -> bytes:
    """Generate a complete project report as PDF bytes."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.7 * inch,
        leftMargin=0.7 * inch,
        topMargin=0.8 * inch,
        bottomMargin=0.7 * inch,
    )

    styles = getSampleStyleSheet()
    title = ParagraphStyle("title", parent=styles["Title"], fontSize=21, textColor=colors.HexColor("#116466"), alignment=TA_CENTER)
    subtitle = ParagraphStyle("subtitle", parent=styles["Normal"], fontSize=10, textColor=colors.HexColor("#5f7d80"), alignment=TA_CENTER)
    section = ParagraphStyle("section", parent=styles["Heading2"], fontSize=12, textColor=colors.HexColor("#116466"), spaceBefore=10, spaceAfter=5)
    body = ParagraphStyle("body", parent=styles["Normal"], fontSize=9, leading=13)

    elements = []
    elements.append(Paragraph("Lab Report", title))
    elements.append(Paragraph(f"{city_info.get('name', 'Unknown')}, {city_info.get('country', '')}", subtitle))
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", subtitle))
    elements.append(Spacer(1, 10))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#d2e3e4")))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph("Location", section))
    elements.append(
        _table(
            [
                ["City", str(city_info.get("name", "n/a"))],
                ["Country", str(city_info.get("country", "n/a"))],
                ["Latitude", f"{city_info.get('latitude', 0):.4f}"],
                ["Longitude", f"{city_info.get('longitude', 0):.4f}"],
            ]
        )
    )

    elements.append(Paragraph("Exploratory Data Analysis Summary", section))
    elements.append(
        _table(
            [
                ["Average temperature", f"{climate_summary.get('avg_temp', 0)} C"],
                ["Average rainfall", f"{climate_summary.get('avg_precip', 0)} mm/day"],
                ["Average anomaly", f"{climate_summary.get('avg_anomaly', 0)} C"],
                ["Extreme heat days", str(climate_summary.get("extreme_heat_days", 0))],
                ["Extreme rain days", str(climate_summary.get("extreme_rain_days", 0))],
            ]
        )
    )

    elements.append(Paragraph("Forecast Summary", section))
    elements.append(
        _table(
            [
                ["Forecast horizon", str(forecast_summary.get("horizon_days", "n/a"))],
                ["Temperature forecast mean", f"{forecast_summary.get('temp_mean', 0)} C"],
                ["Rainfall forecast mean", f"{forecast_summary.get('rain_mean', 0)} mm/day"],
                ["AQI forecast mean", f"{forecast_summary.get('aqi_mean', 0)}"],
            ]
        )
    )

    elements.append(Paragraph("Scenario Projection", section))
    elements.append(
        _table(
            [
                ["Projected temperature", f"{scenario_result.get('projected_temp_c', 0)} C"],
                ["Projected temperature anomaly", f"{scenario_result.get('projected_anomaly_c', 0)} C"],
                ["Projected rainfall", f"{scenario_result.get('projected_rainfall_mm', 0)} mm/day"],
                ["Projected AQI", str(scenario_result.get("projected_aqi", 0))],
                ["Climate risk score", str(scenario_result.get("climate_risk_score", 0))],
                ["Model confidence", str(scenario_result.get("confidence", "n/a"))],
            ]
        )
    )

    elements.append(Paragraph("Model Metrics", section))
    elements.append(_table(_metrics_rows(model_metrics or [])))

    elements.append(Spacer(1, 10))
    disclaimer = (
        "This report combines open weather observations, proxy emissions indicators, "
        "machine learning predictions, autoregressive time-series forecasts, and a neural-network "
        "pattern model for educational analytics. Use domain and local authority guidance "
        "for policy or safety-critical decisions."
    )
    elements.append(Paragraph(disclaimer, body))

    doc.build(elements)
    return buffer.getvalue()
