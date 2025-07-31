"""Comment fetching & sentiment analysis.

This module centralises all comment-related utilities so that the
Streamlit UI, the CLI pipeline and any future automation can use a
single entry-point.

Public API
==========
fetch_comments()      – get comments via YouTube Data API (public key *or* OAuth service)
attach_sentiment()    – run LLM-based sentiment scoring and return enriched list
fetch_and_analyze()   – helper that performs both steps and writes JSON if *out_path* given

The sentiment engine comes from :pymod:`analysis.sentiment_llm`, which in
turn uses the generic :pymod:`llms` layer – so provider choice and models
are controlled by ``config.settings``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

from googleapiclient.discovery import Resource  # type: ignore

from src.analysis.sentiment import sentiment_scores

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fetch helpers (copied from old analysis.core)
# ---------------------------------------------------------------------------


def fetch_comments(
    service: Resource,
    video_id: str,
    *,
    max_pages: int = 10,
    order: str = "relevance",
) -> List[Dict[str, Any]]:
    """Fetch top-level comments with graceful fallback when API-key forbids relevance order."""

    comments: List[Dict[str, Any]] = []
    next_token = None
    for _ in range(max_pages):
        req = service.commentThreads().list(  # type: ignore[attr-defined]
            part="snippet",
            videoId=video_id,
            pageToken=next_token,
            maxResults=100,
            order=order,
            textFormat="plainText",
        )
        try:
            resp = req.execute()
        except Exception as exc:  # pragma: no cover
            if order == "relevance" and "insufficientPermissions" in str(exc):
                logger.info("order='relevance' not allowed without OAuth – retrying with 'time'")
                return fetch_comments(service, video_id, max_pages=max_pages, order="time")
            logger.warning("Comment fetch failed (%s). Returning what we have.", exc)
            break

        for item in resp.get("items", []):
            snip = item["snippet"]["topLevelComment"]["snippet"]
            comments.append(
                {
                    "author": snip.get("authorDisplayName"),
                    "text": snip.get("textDisplay"),
                    "likeCount": snip.get("likeCount"),
                    "publishedAt": snip.get("publishedAt"),
                }
            )
        next_token = resp.get("nextPageToken")
        if not next_token:
            break

    logger.info("Fetched %d comments", len(comments))
    return comments


# ---------------------------------------------------------------------------
# Sentiment helpers
# ---------------------------------------------------------------------------

def attach_sentiment(comments: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach *sentiment* float to each comment dict and return a new list."""

    if not comments:
        return []

    texts = [c.get("textDisplay") or c.get("text", "") for c in comments]
    scores = sentiment_scores(texts)
    enriched: List[Dict[str, Any]] = []
    for c, score in zip(comments, scores):
        new_c = dict(c)  # shallow copy – don't mutate caller's list
        new_c["sentiment"] = score
        enriched.append(new_c)
    return enriched


# ---------------------------------------------------------------------------
# Combined convenience function
# ---------------------------------------------------------------------------

def fetch_and_analyze(
    service: Resource,
    video_id: str,
    *,
    out_path: Path | None = None,
    **fetch_kwargs: Any,
) -> List[Dict[str, Any]]:
    """Fetch comments then enrich with sentiment; optionally write JSON."""

    comments = fetch_comments(service, video_id, **fetch_kwargs)
    enriched = attach_sentiment(comments)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(enriched, indent=2, ensure_ascii=False))
        logger.info("Saved %d comments with sentiment to %s", len(enriched), out_path)

    return enriched


__all__ = [
    "fetch_comments",
    "attach_sentiment",
    "fetch_and_analyze",
] 