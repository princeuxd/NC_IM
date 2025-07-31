"""Unified sentiment scoring utilities using a local HuggingFace model.

The model and other defaults are controlled by *config.settings* so they
can be tweaked via environment variables without changing code.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import List, Sequence

from transformers import pipeline  # type: ignore

from src.config.settings import SETTINGS

logger = logging.getLogger(__name__)

# Maximum items per pipeline batch – HF will chunk internally anyway but we
# keep this modest to limit memory usage when very large comment lists are
# analysed.
BATCH_SIZE = 32


@lru_cache(maxsize=2)
def _hf_pipeline(model_id: str | None = None):
    """Lazy-load and cache the HuggingFace *sentiment-analysis* pipeline."""

    if model_id is None:
        model_id = SETTINGS.sentiment_model
    logger.info("Loading sentiment model: %s", model_id)
    return pipeline("sentiment-analysis", model=model_id, tokenizer=model_id, device=-1)  # type: ignore


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def sentiment_scores(texts: Sequence[str]) -> List[float]:
    """Return polarity scores in the range [-1, 1] for *texts*."""

    if not texts:
        return []

    pipe = _hf_pipeline()
    results = pipe(list(texts), batch_size=BATCH_SIZE, truncation=True)

    scores: List[float] = []
    for res in results:
        label = res["label"].lower()
        conf = float(res["score"])
        # Map label → signed score according to common 1-5 star scheme.
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
        else:
            scores.append(0.0)

    return scores


__all__ = ["sentiment_scores"] 