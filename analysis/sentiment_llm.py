"""LLM-based sentiment analysis helpers (OpenRouter / Groq).

Replaces TextBlob polarity scoring with a more nuanced LLM judgment.  The
function batches inputs to minimise API calls and costs.  When no API key is
configured it falls back to the legacy TextBlob analyser so the pipeline keeps
working offline.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any, List, Sequence

# Hugging Face pipeline at runtime (no external API)
from transformers import pipeline  # type: ignore

from config.settings import SETTINGS

# No external settings required – we run locally.

logger = logging.getLogger(__name__)

# Maximum number of input texts to send in a single LLM request – keeping
# this fairly small mitigates long context lengths and reduces latency.
BATCH_SIZE: int = 32


@lru_cache(maxsize=2)
def _sentiment_pipeline(model_id: str | None = None):
    """Lazily load and cache the HuggingFace sentiment-analysis pipeline."""
    if model_id is None:
        model_id = SETTINGS.sentiment_model
    return pipeline("sentiment-analysis", model=model_id, tokenizer=model_id, device=-1)  # type: ignore


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sentiment_scores(texts: Sequence[str]) -> List[float]:
    """Return polarity scores (-1 = negative .. 1 = positive) using a local HF model."""

    if not texts:
        return []

    # Run the pipeline in one go – HF handles internal batching.
    pipe = _sentiment_pipeline()
    results = pipe(list(texts), batch_size=BATCH_SIZE, truncation=True)

    scores: List[float] = []
    for res in results:
        label = res["label"].lower()
        conf = float(res["score"])
        if "1 star" in label:
            scores.append(-1.0 * conf)
        elif "2 stars" in label:
            scores.append(-0.5 * conf)
        elif "3 stars" in label:
            scores.append(0.0)
        elif "4 stars" in label:
            scores.append(0.5 * conf)
        elif "5 stars" in label:
            scores.append(1.0 * conf)
        else:  # unknown label
            scores.append(0.0)

    return scores


def analyze_comment_sentiment_llm(comments: List[dict[str, Any]]) -> List[dict[str, Any]]:
    """Attach 'sentiment' key to each comment using LLM."""

    texts = [c.get("textDisplay", "") or c.get("text", "") for c in comments]
    scores = sentiment_scores(texts)
    for c, s in zip(comments, scores):
        c["sentiment"] = s  # Use standard 'sentiment' field name
    return comments


__all__ = ["sentiment_scores", "analyze_comment_sentiment_llm"] 