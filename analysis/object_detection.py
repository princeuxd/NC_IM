"""Vision LLM-based object / product detection.

The helper functions here abstract away the choice between Groq and
OpenRouter. By default the pipeline uses **x-ai/grok-2-vision-1212**, a
multimodal model exposed through OpenRouter that accepts up to five images per
request.  The model is prompted to return a JSON list of detected objects with
optional bounding boxes and confidence scores.

Returned structure (per image):
    {
        "timestamp": 30.0,        # seconds in video
        "file": "frame_000005.jpg",
        "objects": [
            {
                "label": "Coca-Cola can",
                "confidence": 0.82,
                "bbox": [x1, y1, x2, y2]  # *optional* if model provides
            },
            ...
        ]
    }

If the selected model or provider fails, we fallback to an empty list so the
rest of the pipeline can continue gracefully.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import openai  # type: ignore

from config.settings import SETTINGS, PipelineSettings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider helpers
# ---------------------------------------------------------------------------


def _configure_client(settings: PipelineSettings = SETTINGS):
    """Set up OpenAI-compatible client for either OpenRouter or Groq."""
    from openai import OpenAI

    if settings.openrouter_api_key:
        client = OpenAI(
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/your-repo",
                "X-Title": "YouTube Video Analyzer"
            }
        )
        # Store client globally for use in detect_objects
        globals()['_openai_client'] = client
        return client
    
    if settings.groq_api_key:
        client = OpenAI(
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        globals()['_openai_client'] = client
        return client
    
    raise RuntimeError("No OPENROUTER_API_KEY or GROQ_API_KEY configured for vision LLM.")


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------


def _img_b64(path: Path) -> str:
    data = path.read_bytes()
    return base64.b64encode(data).decode()


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------


def detect_objects(
    frames: Sequence[Tuple[float, Path]],
    settings: PipelineSettings = SETTINGS,
) -> List[dict[str, Any]]:
    """Run object detection on extracted frames via multimodal LLM.

    For cost reasons we batch up to 5 images per call (OpenRouter limit for
    Grok vision). If *frames* >5 we loop.  This function intentionally keeps
    the conversation prompt terse to minimise tokens.
    """

    if not frames:
        return []

    _configure_client(settings)

    results: List[dict[str, Any]] = []

    BATCH = 5  # images per request supported by Grok vision

    system_prompt = (
        "You are a product/object detection assistant. Return ONLY valid JSON "
        "without any markdown.  For EACH image: {\"objects\": [..]}. If you "
        "cannot detect anything return an empty list."  # noqa: E501
    )

    for i in range(0, len(frames), BATCH):
        batch = frames[i : i + BATCH]
        # Build content array with images and text describing expected output
        content: List[Any] = []
        for ts, p in batch:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{_img_b64(p)}",
                        "detail": "low",
                    },
                }
            )
            # Provide timestamp context so model can include it in JSON
            content.append(
                {
                    "type": "text",
                    "text": f"Timestamp: {ts} seconds. Provide JSON key 'timestamp' with this exact value.",
                }
            )
        content.append(
            {
                "type": "text",
                "text": "For each of the images above, output a JSON object with keys 'timestamp', 'file', and 'objects'. The 'objects' value must be an array of {label, confidence}. Return an array containing one object per image in the same order they were provided. Return ONLY JSON.",
            }
        )

        try:
            # Use the properly configured client
            client = globals().get('_openai_client')
            if client:
                response = client.chat.completions.create(
                    model=settings.object_detection_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": content},
                    ],
                    temperature=0.2,
                    timeout=30  # Add timeout for better error handling
                )
            else:
                logger.error("No OpenAI client configured")
                continue

            # The OpenRouter shim or legacy OpenAI clients may return a plain
            # string instead of a ``ChatCompletion`` object.  Handle both.
            if hasattr(response, "choices"):
                txt = response.choices[0].message.content or ""  # type: ignore[index]
            else:
                # Assume the full JSON came back directly.
                txt = str(response)
            batch_json = json.loads(txt)
            # Ensure filepath exists in output; fill in when missing.
            for meta, (_, pth) in zip(batch_json, batch):
                meta.setdefault("file", pth.name)
            results.extend(batch_json)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Vision LLM detection failed on batch %d: %s", i // BATCH, exc)
            # Insert empty detections so downstream can continue.
            for ts, p in batch:
                results.append({"timestamp": ts, "file": p.name, "objects": []})

    logger.info("Detected objects on %d frames", len(results))
    return results


__all__ = ["detect_objects"] 