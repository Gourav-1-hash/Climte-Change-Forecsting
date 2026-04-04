"""
services/ai_service.py
Manages conversation with OpenAI GPT for climate-related Q&A.
Compatible with openai >= 1.x and 2.x
"""

from typing import List, Dict
from app.config import Config

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False


# System prompt that keeps the AI on-topic
SYSTEM_PROMPT = """You are ClimateAI, an expert climate science assistant embedded in the 
Climate Intelligence Hub dashboard. You answer questions about:
- Climate change science, causes, and effects
- Temperature trends, global warming, and greenhouse gases
- Air quality, pollution, and health impacts
- Sea level rise and extreme weather events
- Environmental sustainability and green technology
- Climate policies and international agreements

Keep answers concise (2-4 paragraphs), accurate, and evidence-based.
Cite scientific consensus where relevant. If asked about something unrelated to climate 
or environment, politely redirect to your area of expertise.
"""


def get_ai_response(
    user_message: str,
    chat_history: List[Dict[str, str]],
) -> str:
    """
    Send a user message plus history to OpenAI and return the assistant reply.

    Args:
        user_message:  Latest user input.
        chat_history:  List of {"role": ..., "content": ...} dicts (recent messages).

    Returns:
        AI reply string, or an error message string.
    """
    if not _OPENAI_AVAILABLE:
        return "❌ OpenAI package is not installed. Run: pip install openai"

    if not Config.OPENAI_API_KEY or Config.OPENAI_API_KEY == "your_openai_api_key_here":
        return (
            "⚠️ OpenAI API key not configured. "
            "Please add your key to the `.env` file as `OPENAI_API_KEY=sk-...` "
            "and restart the app."
        )

    try:
        client = OpenAI(api_key=Config.OPENAI_API_KEY)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        # Add recent history (trim to avoid token limits)
        messages.extend(chat_history[-Config.MAX_CHAT_HISTORY:])
        messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model=Config.OPENAI_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=600,
        )
        return response.choices[0].message.content.strip()

    except Exception as exc:
        return f"❌ AI request failed: {exc}"
