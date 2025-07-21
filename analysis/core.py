"""Advanced analysis utilities for YouTube videos.

Notes:
- Whisper/torch used for offline transcription.
- TextBlob for quick sentiment scoring (polarity -1..1).
- Google Cloud Video Intelligence (logo detection) optional; requires GCP creds.
- Functions are modular so you can mix & match.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, List, cast, Dict

import langdetect  # type: ignore
from analysis.sentiment_llm import sentiment_scores

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Transcription via Whisper
# ---------------------------------------------------------------------------

def transcribe_audio(audio_path: Path | str) -> List[dict[str, Any]]:
    """Return Whisper transcript segments.

    Each segment dict: {"start": float, "end": float, "text": str}
    """
    try:
        import whisper  # type: ignore
        if not hasattr(whisper, "load_model"):
            raise ImportError  # fall through to openai-whisper
    except ImportError:
        try:
            import openai_whisper as whisper  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "openai-whisper not installed; run `pip install -U openai-whisper`"
            ) from exc

    import os, certifi, ssl
    # Ensure proper root certificates for HTTPS downloads (fixes SSL CERT_VERIFY_FAILED)
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())

    try:
        model = whisper.load_model("base")  # type: ignore[attr-defined]
        result = model.transcribe(str(audio_path), word_timestamps=False)
        segments = result.get("segments", [])
        return cast(List[Dict[str, Any]], segments)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Whisper transcription skipped (%s)", exc)
        return cast(List[Dict[str, Any]], [])

# ---------------------------------------------------------------------------
# Sentiment helpers
# ---------------------------------------------------------------------------

def sentiment(text: str) -> float:
    """Return polarity score (-1 negative .. 1 positive)."""
    if not text.strip():
        return 0.0
    return sentiment_scores([text])[0]


def analyze_transcript_sentiment(segments: List[dict[str, Any]]) -> List[dict[str, Any]]:
    """Add sentiment per segment."""
    out = []
    for seg in segments:
        score = sentiment(seg["text"])
        out.append({**seg, "sentiment": score})
    return out

# ---------------------------------------------------------------------------
# Comment fetching
# ---------------------------------------------------------------------------

from googleapiclient.discovery import Resource  # type: ignore

def fetch_comments(
    service: Resource,
    video_id: str,
    max_pages: int = 10,
    *,
    order: str = "relevance",
) -> List[dict[str, Any]]:
    """Fetch top-level comments.

    When using an **API key** the YouTube Data API does **not** allow
    ``order="relevance"`` and returns HTTP 403 ``insufficientPermissions``.
    In that case we transparently retry with ``order="time"`` so the caller
    still gets comments instead of an empty list.
    """
    comments = []
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
        except Exception as exc:  # pylint: disable=broad-except
            # If relevance ordering is not permitted with an API key, retry once
            # with the chronological ordering which *is* allowed.
            if order == "relevance" and "insufficientPermissions" in str(exc):
                logger.info(
                    "order='relevance' not allowed without OAuth; retrying with order='time'"
                )
                return fetch_comments(
                    service,
                    video_id,
                    max_pages=max_pages,
                    order="time",
                )
            logger.warning("Comment fetch failed (%s). Returning what we have.", exc)
            break
        for item in resp.get("items", []):
            snip = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "author": snip.get("authorDisplayName"),
                "text": snip.get("textDisplay"),
                "likeCount": snip.get("likeCount"),
                "publishedAt": snip.get("publishedAt"),
            })
        next_token = resp.get("nextPageToken")
        if not next_token:
            break
    logger.info("Fetched %d comments", len(comments))
    return comments


def analyze_comment_sentiment(comments: List[dict[str, Any]]):
    """Attach language + polarity to each comment.

    Handles cases where the comment text is empty or langdetect fails by
    gracefully falling back to empty strings / None so that downstream code
    never crashes (e.g. Streamlit app)."""
    for c in comments:
        txt = c.get("text") or ""
        lang = ""
        if txt.strip():
            try:
                # langdetect raises when the text has too few features (< 3 chars).
                lang = langdetect.detect(txt)
            except langdetect.lang_detect_exception.LangDetectException:  # type: ignore[attr-defined]
                lang = ""
        c["lang"] = lang
        c["sentiment"] = sentiment(txt) if txt.strip() else 0.0
    return comments

# ---------------------------------------------------------------------------
# Product / logo detection via Google Cloud
# ---------------------------------------------------------------------------

def detect_logos(video_path: Path | str):
    """Run Cloud Video Intelligence logo recognition.

    Returns list of dicts: {"entity": str, "start": seconds, "end": seconds}
    """
    try:
        from google.cloud import videointelligence_v1 as vi  # type: ignore
    except ImportError as exc:
        raise RuntimeError("google-cloud-videointelligence not installed.") from exc

    client = vi.VideoIntelligenceServiceClient()
    features = [vi.Feature.LOGO_RECOGNITION]
    with Path(video_path).open("rb") as f:
        operation = client.annotate_video(
            request={
                "features": features,
                "input_content": f.read(),
            }
        )
    logger.info("Processing video for logo detection ... (could take minutes)")
    result = operation.result(timeout=600)  # type: ignore[attr-defined]
    logo_segments: list[dict[str, Any]] = []
    for annotation in result.annotation_results[0].logo_recognition_annotations:  # type: ignore[attr-defined]
        desc = annotation.entity.description
        for track in annotation.tracks:
            start = track.segment.start_time_offset.total_seconds()
            end = track.segment.end_time_offset.total_seconds()
            logo_segments.append({"entity": desc, "start": start, "end": end})
    return logo_segments

# ---------------------------------------------------------------------------
# Correlation helpers
# ---------------------------------------------------------------------------

def correlate_retention(logo_segments: List[dict[str, Any]], retention_rows: list[list[Any]]):
    """Join logo timestamps with audience retention percentage.

    retention_rows: rows from Analytics API where first col is elapsedVideoTimeRatio (0..1),
    second col audienceWatchRatio.
    """
    # Convert retention rows to dict elapsed->ratio
    retention_map = {float(r[0]): float(r[1]) for r in retention_rows}
    enriched = []
    for seg in logo_segments:
        midpoint = (seg["start"] + seg["end"]) / 2
        # approximate elapsed ratio by dividing midpoint by video length unknown – caller should supply length.
        elapsed_ratio = None
        if logo_segments and retention_rows:
            # can't compute properly without duration, placeholder
            elapsed_ratio = None
        ratio = retention_map.get(elapsed_ratio) if elapsed_ratio is not None else None
        enriched.append({**seg, "audienceWatchRatio": ratio})
    return enriched

# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False)) 