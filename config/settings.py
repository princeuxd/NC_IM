"""Minimal configuration for the streamlined toolkit.

Only parameters that the current codebase still uses are kept.
• FRAME_INTERVAL_SEC  – seconds between extracted frames in video analysis.
• SENTIMENT_MODEL     – HuggingFace model id for sentiment scoring.
• OPENROUTER_API_KEY / GROQ_API_KEY – optional creds for LLM providers (if ever re-enabled).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv  # type: ignore

# Load variables from .env if present (shared with other config modules)
load_dotenv()


@dataclass(frozen=True)
class PipelineSettings:
    """Immutable container for runtime parameters."""

    # --- Frame extraction ------------------------------------------------
    frame_interval_sec: int = int(os.getenv("FRAME_INTERVAL_SEC", "5"))

    # --- Sentiment model --------------------------------------------------
    sentiment_model: str = os.getenv(
        "SENTIMENT_MODEL", "nlptown/bert-base-multilingual-uncased-sentiment"
    )

    # --- Optional LLM provider keys --------------------------------------
    openrouter_api_key: str | None = os.getenv("OPENROUTER_API_KEY")
    groq_api_key: str | None = os.getenv("GROQ_API_KEY")

    # Default chat models (kept for future use)
    openrouter_chat_model: str = os.getenv(
        "OPENROUTER_CHAT_MODEL", "google/gemini-2.0-flash-exp:free"
    )
    groq_chat_model: str = os.getenv(
        "GROQ_CHAT_MODEL", "llama3-8b-8192"
    )


# Singleton used by most callers
SETTINGS = PipelineSettings()


def update_from_kwargs(**overrides):  # type: ignore[override]
    """Return a new PipelineSettings with supplied overrides."""

    return PipelineSettings(
        frame_interval_sec=overrides.get("frame_interval_sec", SETTINGS.frame_interval_sec),
        sentiment_model=overrides.get("sentiment_model", SETTINGS.sentiment_model),
        openrouter_api_key=overrides.get("openrouter_api_key", SETTINGS.openrouter_api_key),
        groq_api_key=overrides.get("groq_api_key", SETTINGS.groq_api_key),
        openrouter_chat_model=overrides.get("openrouter_chat_model", SETTINGS.openrouter_chat_model),
        groq_chat_model=overrides.get("groq_chat_model", SETTINGS.groq_chat_model),
    ) 