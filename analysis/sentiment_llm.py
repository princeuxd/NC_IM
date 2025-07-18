"""LLM-based sentiment analysis helpers (OpenRouter / Groq).

Replaces TextBlob polarity scoring with a more nuanced LLM judgment.  The
function batches inputs to minimise API calls and costs.  When no API key is
configured it falls back to the legacy TextBlob analyser so the pipeline keeps
working offline.
"""

from __future__ import annotations

import json
import logging
from typing import Any, List, Sequence

import openai  # type: ignore
from textblob import TextBlob  # fallback

from config.settings import SETTINGS, PipelineSettings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider config
# ---------------------------------------------------------------------------


def _configure_client(settings: PipelineSettings = SETTINGS):
    if settings.openrouter_api_key:
        openai.api_key = settings.openrouter_api_key  # type: ignore[attr-defined]
        openai.base_url = "https://openrouter.ai/api/v1"  # type: ignore[attr-defined]
        return True
    if settings.groq_api_key:
        openai.api_key = settings.groq_api_key  # type: ignore[attr-defined]
        openai.base_url = "https://api.groq.com/openai/v1"  # type: ignore[attr-defined]
        return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sentiment_scores(texts: Sequence[str], *, settings: PipelineSettings = SETTINGS) -> List[float]:
    """Return sentiment polarity scores using LLM.

    The output is a float between -1 (negative) and 1 (positive). If the model
    returns a categorical label we map: positive->0.7, neutral->0, negative->-0.7.
    """

    if not _configure_client(settings):
        # Fallback to TextBlob
        return [TextBlob(t).sentiment.polarity for t in texts]  # type: ignore[attr-defined]

    BATCH = 20
    scores: List[float] = []

    system = (
        "You are a sentiment analysis assistant. For EACH user input text, "
        "respond with ONLY a JSON number in the range -1.0 to 1.0 where -1 is "
        "very negative, 0 is neutral, 1 is very positive. Return an array with "
        "the same length as the provided list. Do not include any extra keys."  # noqa: E501
    )

    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]
        try:
            content = json.dumps(batch)
            response = openai.chat.completions.create(  # type: ignore[attr-defined]
                model=settings.sentiment_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": content},
                ],
                temperature=0,
            )
            arr = json.loads(response.choices[0].message.content)  # type: ignore[index]
            scores.extend([float(x) for x in arr])
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("LLM sentiment failed (%s). Falling back to TextBlob for this batch.", exc)
            scores.extend([TextBlob(t).sentiment.polarity for t in batch])  # type: ignore[attr-defined]

    return scores


def analyze_comment_sentiment_llm(comments: List[dict[str, Any]]):
    """Attach 'sentiment' key to each comment using LLM."""

    texts = [c.get("text", "") for c in comments]
    scores = sentiment_scores(texts)
    for c, s in zip(comments, scores):
        c["sentiment_llm"] = s
    return comments


__all__ = ["sentiment_scores", "analyze_comment_sentiment_llm"] 