"""Global pipeline configuration for video analysis pipeline.

This module centralizes tweakable parameters:
• FRAME_INTERVAL_SEC – seconds between extracted frames for object detection.
• PRODUCT_WINDOW_SEC – window (seconds) before/after a product appearance when correlating engagement metrics.
• OBJECT_DETECTION_MODEL – multimodal/vision model route (OpenRouter or Groq) used for detecting objects/logos in frames.
• SENTIMENT_MODEL – LLM route used for sentiment analysis of transcripts/comments.
• SUMMARY_MODEL – LLM route used for executive summaries.
• OPENROUTER_API_KEY / GROQ_API_KEY – credentials for chosen LLM provider.

All settings can be overridden via environment variables (e.g. in a .env file) so deployments & the Streamlit UI can tweak behaviour without code edits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv  # type: ignore

# Load variables from .env if present (shared with other config modules)
load_dotenv()


@dataclass(frozen=True)
class PipelineSettings:
    """Immutable container for pipeline-wide parameters."""

    # --- Frame / product timeline ------------------------------------------------
    frame_interval_sec: int = int(os.getenv("FRAME_INTERVAL_SEC", "5"))
    product_window_sec: int = int(os.getenv("PRODUCT_WINDOW_SEC", "120"))  # ± window

    # --- Model routes ------------------------------------------------------------
    # Free-tier routes (no credits required on OpenRouter)
    # Vision – Nous Hermes 2 (7B) with multimodal capability, marked free
    object_detection_model: str = os.getenv(
        "OBJECT_DETECTION_MODEL", "nousresearch/nous-hermes-2-vision-7b:free"
    )

    # Text sentiment – OpenHermes 2 (Mistral-7B) free variant
    sentiment_model: str = os.getenv(
        "SENTIMENT_MODEL", "teknium/openhermes-2-mistral-7b:free"
    )

    summary_model: str = os.getenv(
        "SUMMARY_MODEL", "google/gemma-3-27b-it:free"
    )

    # --- Provider credentials ----------------------------------------------------
    openrouter_api_key: str | None = os.getenv("OPENROUTER_API_KEY")
    groq_api_key: str | None = os.getenv("GROQ_API_KEY")


# Singleton used by most callers
SETTINGS = PipelineSettings()


def update_from_kwargs(**overrides):  # type: ignore[override]
    """Return a new PipelineSettings with selected fields overridden.

    Helpful inside Streamlit callbacks to reflect UI sliders/inputs without
    mutating the global immutable SETTINGS instance.
    """

    return PipelineSettings(
        frame_interval_sec=overrides.get("frame_interval_sec", SETTINGS.frame_interval_sec),
        product_window_sec=overrides.get("product_window_sec", SETTINGS.product_window_sec),
        object_detection_model=overrides.get(
            "object_detection_model", SETTINGS.object_detection_model
        ),
        sentiment_model=overrides.get("sentiment_model", SETTINGS.sentiment_model),
        summary_model=overrides.get("summary_model", SETTINGS.summary_model),
        openrouter_api_key=overrides.get("openrouter_api_key", SETTINGS.openrouter_api_key),
        groq_api_key=overrides.get("groq_api_key", SETTINGS.groq_api_key),
    ) 