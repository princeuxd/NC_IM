"""Vision LLM helper using OpenRouter multimodal model.

summarise_frames(frames) -> returns LLM summary of supplied JPEG frames.
Each *frame* is a tuple (timestamp_sec, Path).

Requires OPENROUTER_API_KEY in environment and a model that supports images
(default: google/gemini-pro-vision).
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import List, Tuple

from src.llms import get_client
from src.config.settings import SETTINGS
from src.prompts.vision_analysis import get_frame_analysis_prompt

logger = logging.getLogger(__name__)


def _b64(img_path: Path) -> str:
    return base64.b64encode(img_path.read_bytes()).decode()


def summarise_frames(
    frames: List[Tuple[float, Path]],
    *,
    prompt: str | None = None,
    model: str | None = None,
) -> str:
    """Send up to 16 frames to an OpenRouter vision model and return its reply."""

    # Check if any LLM provider keys are available
    if not (SETTINGS.openrouter_api_keys or SETTINGS.groq_api_keys or SETTINGS.gemini_api_keys):
        raise RuntimeError("No LLM API keys configured. Set OPENROUTER_API_KEY, GROQ_API_KEY, or GEMINI_API_KEY.")

    if not frames:
        return "No frames supplied."

    if model is None:
        model = SETTINGS.openrouter_chat_model  # should be vision-capable route

    if prompt is None:
        prompt = get_frame_analysis_prompt()

    # Take at most 16 frames (Gemini vision limit)
    frames = frames[:16]

    content = [{"type": "text", "text": prompt}]
    for ts, img in frames:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{_b64(img)}",
                    "detail": "low",
                },
            }  # type: ignore[arg-type]
        )
        content.append({"type": "text", "text": f"Timestamp: {ts:.1f}s"})

    # Use smart client for automatic key rotation and provider fallback
    from src.llms import get_smart_client
    
    try:
        client = get_smart_client()
        reply = client.chat(
            [{"role": "user", "content": content}],
            temperature=0.2,
            max_tokens=256,
        )
        return reply
    except Exception as e:
        logger.error("Vision analysis failed with all providers: %s", e)
        raise


__all__ = ["summarise_frames"] 