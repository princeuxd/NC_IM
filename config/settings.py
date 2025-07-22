"""Minimal configuration for the streamlined toolkit.

Only parameters that the current codebase still uses are kept.
• FRAME_INTERVAL_SEC  – seconds between extracted frames in video analysis.
• SENTIMENT_MODEL     – HuggingFace model id for sentiment scoring.
• Multiple API keys per provider for automatic rotation when rate limits are hit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv  # type: ignore

# Load variables from .env if present (shared with other config modules)
load_dotenv()


def _get_multiple_keys(prefix: str) -> List[str]:
    """Extract multiple API keys from environment variables with numbered suffixes."""
    keys = []
    
    # First try the base key name (for backward compatibility)
    base_key = os.getenv(prefix)
    if base_key:
        keys.append(base_key)
    
    # Then try numbered keys (1, 2, 3, ...)
    counter = 1
    while True:
        key = os.getenv(f"{prefix}_{counter}")
        if key:
            keys.append(key)
            counter += 1
        else:
            break
    
    return keys


@dataclass(frozen=True)
class PipelineSettings:
    """Immutable container for runtime parameters."""

    # --- Frame extraction ------------------------------------------------
    frame_interval_sec: int = int(os.getenv("FRAME_INTERVAL_SEC", "5"))

    # --- Sentiment model --------------------------------------------------
    sentiment_model: str = os.getenv(
        "SENTIMENT_MODEL", "nlptown/bert-base-multilingual-uncased-sentiment"
    )

    # --- Multiple LLM provider keys for rotation -------------------------
    openrouter_api_keys: List[str] = None  # Will be populated in __post_init__
    groq_api_keys: List[str] = None
    gemini_api_keys: List[str] = None

    # --- Backward compatibility single key access -----------------------
    openrouter_api_key: str | None = None
    groq_api_key: str | None = None
    gemini_api_key: str | None = None

    # Default chat models per provider
    openrouter_chat_model: str = os.getenv(
        "OPENROUTER_CHAT_MODEL", "google/gemini-2.0-flash-exp:free"
    )
    groq_chat_model: str = os.getenv(
        "GROQ_CHAT_MODEL", "llama3-8b-8192"
    )
    gemini_chat_model: str = os.getenv(
        "GEMINI_CHAT_MODEL", "gemini-2.5-flash-lite"
    )

    def __post_init__(self):
        """Load multiple keys after initialization."""
        # Use object.__setattr__ because dataclass is frozen
        object.__setattr__(self, 'openrouter_api_keys', _get_multiple_keys("OPENROUTER_API_KEY"))
        object.__setattr__(self, 'groq_api_keys', _get_multiple_keys("GROQ_API_KEY"))
        object.__setattr__(self, 'gemini_api_keys', _get_multiple_keys("GEMINI_API_KEY"))
        
        # Set backward compatibility single keys (first key if available)
        object.__setattr__(self, 'openrouter_api_key', self.openrouter_api_keys[0] if self.openrouter_api_keys else None)
        object.__setattr__(self, 'groq_api_key', self.groq_api_keys[0] if self.groq_api_keys else None)
        object.__setattr__(self, 'gemini_api_key', self.gemini_api_keys[0] if self.gemini_api_keys else None)


# Singleton used by most callers
SETTINGS = PipelineSettings()


def update_from_kwargs(**overrides):  # type: ignore[override]
    """Return a new PipelineSettings with supplied overrides."""

    return PipelineSettings(
        frame_interval_sec=overrides.get("frame_interval_sec", SETTINGS.frame_interval_sec),
        sentiment_model=overrides.get("sentiment_model", SETTINGS.sentiment_model),
        openrouter_chat_model=overrides.get("openrouter_chat_model", SETTINGS.openrouter_chat_model),
        groq_chat_model=overrides.get("groq_chat_model", SETTINGS.groq_chat_model),
        gemini_chat_model=overrides.get("gemini_chat_model", SETTINGS.gemini_chat_model),
    ) 