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

from llms import get_client
from config.settings import SETTINGS

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

    if not SETTINGS.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY not configured.")

    if not frames:
        return "No frames supplied."

    if model is None:
        model = SETTINGS.openrouter_chat_model  # should be vision-capable route

    if prompt is None:
        prompt = (
            "You are a video-analysis assistant. Based ONLY on the supplied frames "
            "(and provided timestamps), output a concise markdown report with the "
            "following sections:\n\n"
            "1. **Summary** – 3-5 bullet points describing what happens in the video.\n"
            "2. **Overall Sentiment** – one of Positive / Neutral / Negative with a short justification.\n"
            "3. **Category** – choose the single best fit from this list: "
            "{Lifestyle, Education, Technology, Gaming, Health, Finance, Entertainment, Travel, Food, Sports, News}.\n"
            "4. **Products / Brands Shown** – bullet list of distinct products, logos or brand names visible in the frames. If none, write 'None visible'."
        )

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

    client = get_client("openrouter", SETTINGS.openrouter_api_key)

    try:
        reply = client.chat(
            [{"role": "user", "content": content}],
            model=model,
            temperature=0.2,
            max_tokens=256,
        )
        return reply
    except Exception as e:
        # If we hit rate-limit or any OpenRouter specific error, fall back to Groq if key configured.
        if SETTINGS.groq_api_key:
            logger.warning("OpenRouter failed (%s). Falling back to Groq vision model…", e)

            # Groq multimodal limit: 5 images → rebuild content with first 5 imgs
            g_content = [content[0]]  # prompt text
            img_blocks = [blk for blk in content[1:] if blk.get("type") == "image_url"]
            txt_blocks = [blk for blk in content[1:] if blk.get("type") == "text"]

            for ib, tb in list(zip(img_blocks, txt_blocks))[:5]:
                g_content.extend([ib, tb])

            g_client = get_client("groq", SETTINGS.groq_api_key)
            return g_client.chat(
                [{"role": "user", "content": g_content}],
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                temperature=0.2,
                max_tokens=256,
            )
        raise


__all__ = ["summarise_frames"] 