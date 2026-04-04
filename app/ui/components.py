"""Reusable Streamlit UI components and shared CSS styling."""

from __future__ import annotations

import streamlit as st


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Source+Sans+3:wght@400;600&display=swap');

:root {
    --bg-1: #061421;
    --bg-2: #0f2430;
    --panel: rgba(255, 255, 255, 0.05);
    --border: rgba(255, 255, 255, 0.14);
    --text: #e6f2f3;
    --muted: #9ab8bc;
    --teal: #25c2c9;
    --sun: #f5a623;
}

.stApp {
    background:
        radial-gradient(1100px 500px at -5% -15%, rgba(37,194,201,0.14), transparent 60%),
        radial-gradient(900px 430px at 100% -20%, rgba(245,166,35,0.12), transparent 60%),
        linear-gradient(135deg, var(--bg-1), var(--bg-2));
    color: var(--text);
}

html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
}

h1, h2, h3, h4, h5 {
    font-family: 'Manrope', sans-serif !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(6,20,33,0.96), rgba(10,28,41,0.94));
    border-right: 1px solid rgba(37,194,201,0.2);
}

.metric-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 16px 14px;
    min-height: 120px;
}

.metric-icon {
    font-size: 1.35rem;
    margin-bottom: 3px;
}

.metric-label {
    font-size: 0.76rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
}

.metric-value {
    font-size: 1.7rem;
    font-family: 'Manrope', sans-serif;
    font-weight: 800;
    line-height: 1.2;
}

.metric-sub {
    font-size: 0.84rem;
    color: var(--muted);
}

.section-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 20px 0 8px 0;
    border-bottom: 1px solid rgba(37,194,201,0.28);
    padding-bottom: 6px;
}

.section-header-text {
    font-family: 'Manrope', sans-serif;
    font-size: 1.05rem;
    color: #c6f7f9;
    font-weight: 700;
}

.glass-panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 14px;
}

.chat-user, .chat-ai {
    border-radius: 12px;
    padding: 10px 13px;
    margin-top: 5px;
    margin-bottom: 8px;
    border: 1px solid var(--border);
}

.chat-user {
    background: rgba(37,194,201,0.16);
    margin-left: 28px;
}

.chat-ai {
    background: rgba(255,255,255,0.06);
    margin-right: 28px;
}

.chat-label-user, .chat-label-ai {
    color: var(--muted);
    font-size: 0.75rem;
}

.chat-label-user { text-align: right; }

.stButton > button {
    background: linear-gradient(135deg, #25c2c9, #17aeb5) !important;
    color: #031015 !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Manrope', sans-serif !important;
    font-weight: 700 !important;
}

div[data-testid="stDownloadButton"] button {
    background: linear-gradient(135deg, #f5a623, #f0891c) !important;
    color: #2d1b00 !important;
    border: none !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #66f0f4 !important;
    border-bottom-color: #66f0f4 !important;
}

#MainMenu, footer, header {
    visibility: hidden;
}
</style>
"""


def inject_css() -> None:
        st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def metric_card(icon: str, label: str, value: str, sub: str = "") -> None:
        sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
        st.markdown(
                f"""
                <div class="metric-card">
                        <div class="metric-icon">{icon}</div>
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{value}</div>
                        {sub_html}
                </div>
                """,
                unsafe_allow_html=True,
        )


def section_header(icon: str, title: str) -> None:
        st.markdown(
                f"""
                <div class="section-header">
                    <span>{icon}</span>
                    <span class="section-header-text">{title}</span>
                </div>
                """,
                unsafe_allow_html=True,
        )


def chat_bubble_user(message: str) -> None:
        st.markdown(f'<div class="chat-label-user">You</div><div class="chat-user">{message}</div>', unsafe_allow_html=True)


def chat_bubble_ai(message: str) -> None:
        st.markdown(f'<div class="chat-label-ai">ClimateAI</div><div class="chat-ai">{message}</div>', unsafe_allow_html=True)


def aqi_badge(label: str, color: str, description: str) -> None:
        st.markdown(
                f"""
                <div class="glass-panel" style="border-color:{color}77; background:{color}1f;">
                    <div style="font-weight:700; color:{color};">{label}</div>
                    <div style="color:#c4d8db; font-size:0.86rem;">{description}</div>
                </div>
                """,
                unsafe_allow_html=True,
        )


def prediction_result_card(anomaly: float, absolute_temp: float, confidence: str, year: int) -> None:
        sign = "+" if anomaly >= 0 else ""
        conf_color = "#4ade80" if confidence == "High" else "#f59e0b" if confidence == "Medium" else "#f87171"
        st.markdown(
                f"""
                <div class="glass-panel" style="text-align:center;">
                    <div style="text-transform:uppercase; letter-spacing:0.08em; color:#9ab8bc; font-size:0.76rem;">Projection for {year}</div>
                    <div style="font-family:Manrope,sans-serif; font-size:2.9rem; font-weight:800; color:#66f0f4;">{sign}{anomaly} C</div>
                    <div style="color:#c4d8db;">Estimated absolute temperature: <b>{absolute_temp} C</b></div>
                    <div style="display:inline-block; margin-top:8px; padding:4px 11px; border-radius:20px; background:{conf_color}22; border:1px solid {conf_color}66; color:{conf_color};">Confidence: {confidence}</div>
                </div>
                """,
                unsafe_allow_html=True,
        )
