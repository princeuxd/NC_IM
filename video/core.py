"""Video processing and data fetching utilities.

This module offers high-level helpers to:
• Resolve a YouTube video URL to its ID.
• Fetch video metadata (public) and analytics (owner OAuth).
• Download the video file with yt-dlp.
• Extract a mono WAV audio track via ffmpeg for downstream transcription.
• Save fetched JSON data to disk.

Dependencies:
    yt-dlp, ffmpeg (system binary) or ffmpeg-python, google-api-python-client
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import tempfile
import urllib.parse
from datetime import date
from pathlib import Path
from typing import Any, Optional, List, Tuple
from PIL import Image  # type: ignore

from googleapiclient.discovery import Resource  # type: ignore
from auth import get_public_service

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------
_YT_VIDEO_REGEX = re.compile(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})")

def extract_video_id(url: str) -> str:
    """Return the 11-char YouTube video ID from any standard URL."""
    m = _YT_VIDEO_REGEX.search(url)
    if m:
        return m.group(1)
    # fallback: parse query string
    parsed = urllib.parse.urlparse(url)
    qs = urllib.parse.parse_qs(parsed.query)
    if "v" in qs:
        return qs["v"][0]
    raise ValueError(f"Unable to parse video id from URL: {url}")

# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_video_metadata(service: Resource, video_id: str) -> dict[str, Any]:
    """Fetch full public metadata for a single video."""
    logger.debug("Fetching metadata for video %s", video_id)
    response = (
        service.videos()  # type: ignore[attr-defined]
        .list(
            id=video_id,
            part="snippet,statistics,contentDetails,topicDetails,status,liveStreamingDetails",
            maxResults=1,
        )
        .execute()
    )
    items = response.get("items", [])
    if not items:
        raise ValueError(f"Video {video_id} not found or not accessible")
    return items[0]


def fetch_video_analytics(
    analytics: Resource,
    video_id: str,
    channel_id: str,
    start_date: date,
    end_date: date,
) -> list[list[Any]]:
    """Return audience retention curve & key metrics (owner only)."""
    logger.debug("Fetching analytics for video %s", video_id)
    try:
        # When requesting the elapsedVideoTimeRatio dimension, YouTube Analytics
        # only supports the audience retention metrics (audienceWatchRatio and
        # relativeRetentionPerformance). Other metrics like views or likes
        # must be queried separately. See official table of supported reports:
        # https://developers.google.com/youtube/analytics/v2/available_reports
        response = analytics.reports().query(  # type: ignore[attr-defined]
            ids=f"channel=={channel_id}",
            metrics="audienceWatchRatio",
            dimensions="elapsedVideoTimeRatio",
            filters=f"video=={video_id}",
            startDate=start_date.isoformat(),
            endDate=end_date.isoformat(),
        ).execute()
        return response.get("rows", [])
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Analytics fetch failed (%s). Proceeding without analytics.", exc)
        return []

# ---------------------------------------------------------------------------
# Aggregate metrics helpers
# ---------------------------------------------------------------------------


def fetch_video_metrics(
    analytics: Resource,
    video_id: str,
    channel_id: str,
    start_date: date,
    end_date: date,
) -> list[list[Any]]:
    """Fetch total views, likes, comments, etc. for the video over the period.

    This uses a *different* query than audience retention because dimensions
    cannot be combined arbitrarily (see official docs).
    """
    logger.debug("Fetching summary metrics for video %s", video_id)
    metrics = "views,likes,comments,estimatedMinutesWatched,averageViewDuration"
    try:
        response = analytics.reports().query(  # type: ignore[attr-defined]
            ids=f"channel=={channel_id}",
            metrics=metrics,
            filters=f"video=={video_id}",
            startDate=start_date.isoformat(),
            endDate=end_date.isoformat(),
        ).execute()
        return response.get("rows", [])
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Metric fetch failed (%s). Proceeding without summary metrics.", exc)
        return []

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def save_json(data: Any, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    logger.info("Saved JSON to %s", out_path)

# ---------------------------------------------------------------------------
# Video download & audio extraction
# ---------------------------------------------------------------------------

def download_video(url: str, output_dir: Path | str = "downloads") -> Path:
    """Download video via yt-dlp. Returns the path to the MP4 file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    # template: videoid.mp4
    video_id = extract_video_id(url)
    output_path = output_dir / f"{video_id}.mp4"
    cmd = [
        "yt-dlp",
        "-f",
        "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
        "--merge-output-format",
        "mp4",
        "-o",
        str(output_path),
        url,
    ]
    logger.info("Downloading video %s to %s", url, output_path)
    subprocess.run(cmd, check=True)
    return output_path


def extract_audio(video_file: Path | str, wav_path: Optional[Path | str] = None) -> Path:
    """Extract mono 16-kHz WAV audio using ffmpeg."""
    video_file = Path(video_file)
    if wav_path is None:
        wav_path = video_file.with_suffix(".wav")
    wav_path = Path(wav_path)

    if shutil.which("ffmpeg") is None:
        logger.warning("ffmpeg binary not found in PATH; skipping audio extraction. Install ffmpeg to enable.")
        return wav_path  # return intended path even if not created

    cmd = [
        "ffmpeg",
        "-y",  # overwrite
        "-i",
        str(video_file),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(wav_path),
    ]
    logger.info("Extracting audio to %s", wav_path)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return wav_path

# ---------------------------------------------------------------------------
# Frame extraction helper for vision models
# ---------------------------------------------------------------------------


def extract_frames(
    video_file: Path | str,
    out_dir: Path | str,
    every_sec: int = 5,
    limit: Optional[int] = None,
) -> List[Tuple[float, Path]]:
    """Extract JPEG frames every ``every_sec`` seconds.

    Returns a list of tuples ``(timestamp_sec, frame_path)``. If *limit* is
    provided, extraction stops after that many frames (useful for quick demos).
    Requires ``ffmpeg`` in PATH.
    """
    video_file = Path(video_file)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if shutil.which("ffmpeg") is None:
        logger.warning("ffmpeg not found – skipping frame extraction.")
        return []

    # Use ffmpeg to sample frames at regular interval.
    # -vf fps=1/N yields one frame per N seconds.
    pattern = out_dir / "frame_%06d.jpg"
    cmd = [
        "ffmpeg",
        "-y",  # overwrite existing
        "-i",
        str(video_file),
        "-vf",
        f"fps=1/{every_sec}",
        "-q:v",
        "2",  # quality (low number = high quality)
        str(pattern),
    ]
    logger.info("Extracting frames every %ds from %s", every_sec, video_file)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    frames = sorted(out_dir.glob("frame_*.jpg"))
    if limit is not None:
        frames = frames[:limit]

    # Build (timestamp, path) list; ts = index * interval seconds.
    result: List[Tuple[float, Path]] = [
        (idx * every_sec, p) for idx, p in enumerate(frames)
    ]
    logger.info("Extracted %d frames to %s", len(result), out_dir)
    return result

# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def process_video(
    url: str,
    public_service: Resource,
    analytics_service: Optional[Resource] = None,
    channel_id: Optional[str] = None,
    output_dir: Path | str = "data",
):
    """High-level convenience wrapper:

    1. Resolve ID & fetch metadata (public).
    2. Optionally fetch owner analytics (if analytics_service provided).
    3. Download MP4 and extract WAV.
    4. Persist everything under output_dir/{video_id}/ ...
    """
    output_dir = Path(output_dir)
    vid = extract_video_id(url)
    # Avoid double nesting if caller already specifies a directory named vid
    if Path(output_dir).name == vid:
        video_folder = Path(output_dir)
    else:
        video_folder = Path(output_dir) / vid
    video_folder.mkdir(parents=True, exist_ok=True)

    # 1. Metadata
    meta = fetch_video_metadata(public_service, vid)
    save_json(meta, video_folder / "metadata.json")

    # 2. Analytics (if allowed)
    if analytics_service and channel_id:
        today = date.today()
        ana_rows = fetch_video_analytics(
            analytics_service, vid, channel_id, start_date=today.replace(year=today.year - 1), end_date=today
        )
        save_json(ana_rows, video_folder / "analytics.json")

        # Fetch additional aggregate metrics (views, likes, etc.)
        summary_rows = fetch_video_metrics(
            analytics_service,
            vid,
            channel_id,
            start_date=today.replace(year=today.year - 1),
            end_date=today,
        )
        save_json(summary_rows, video_folder / "metrics_summary.json")

    # 3. Download video + audio
    mp4_path = download_video(url, output_dir=video_folder)
    extract_audio(mp4_path, video_folder / "audio.wav")

    logger.info("Finished processing %s", url) 