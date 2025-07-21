"""Frame extraction & object detection helpers.

This module wraps the existing `video.extract_frames` and
`analysis.object_detection.detect_objects` utilities into a single cohesive
API so other parts of the codebase only need to import from here.

Public API
==========
extract_frames()            – thin passthrough to video.extract_frames
analyze_frames()            – extract selected frames *and* run detection
                             in one call; saves detections JSON if requested.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Tuple

from config.settings import SETTINGS, PipelineSettings
import re, subprocess, shutil, urllib.parse
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------

def extract_frames(
    video_file: Path | str,
    out_dir: Path | str,
    *,
    every_sec: int | None = None,
    limit: int | None = None,
    settings: PipelineSettings = SETTINGS,
):
    """Passthrough to the original extract_frames function.

    If *every_sec* is None we default to settings.frame_interval_sec.
    """

    if every_sec is None:
        every_sec = settings.frame_interval_sec

    return _ffmpeg_extract_frames(video_file, out_dir, every_sec=every_sec, limit=limit)


# --- ID helper & download -------------------------------------------------
_YT_REGEX = re.compile(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})")

def extract_video_id(url: str) -> str:
    m = _YT_REGEX.search(url)
    if m:
        return m.group(1)
    parsed = urllib.parse.urlparse(url)
    qs = urllib.parse.parse_qs(parsed.query)
    if "v" in qs:
        return qs["v"][0]
    raise ValueError(f"Unable to parse video id from URL: {url}")


def download_video(url: str, output_dir: Path | str = "downloads") -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vid = extract_video_id(url)
    out_path = output_dir / f"{vid}.mp4"
    cmd = [
        "yt-dlp",
        "-f",
        "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
        "--merge-output-format",
        "mp4",
        "-o",
        str(out_path),
        url,
    ]
    logger.info("Downloading video %s -> %s", url, out_path)
    subprocess.run(cmd, check=True)
    return out_path


# ---------------------------------------------------------------------------
# Combined helper
# ---------------------------------------------------------------------------

def analyze_frames(
    video_file: Path | str,
    out_dir: Path,
    *,
    every_sec: int | None = None,
    limit: int | None = None,
    settings: PipelineSettings = SETTINGS,
    save_json_path: Path | None = None,
):
    """Extract frames then run vision detection.

    Returns list of detection dicts. Optionally writes JSON when
    *save_json_path* supplied (defaults to ``out_dir / 'detections.json'``).
    """

    frames: List[Tuple[float, Path]] = extract_frames(
        video_file,
        out_dir,
        every_sec=every_sec,
        limit=limit,
        settings=settings,
    )

    if not frames:
        logger.warning("No frames extracted from %s", video_file)
        return []

    # Vision AI step removed – we simply return frame metadata.
    detections = [
        {"timestamp": ts, "file": p.name} for ts, p in frames
    ]

    if save_json_path is None:
        save_json_path = Path(out_dir) / "frames.json"

    save_json_path.parent.mkdir(parents=True, exist_ok=True)
    save_json_path.write_text(json.dumps(detections, indent=2, ensure_ascii=False))
    logger.info("Saved %d frame entries to %s", len(detections), save_json_path)

    return detections


# ffmpeg-based frame extraction (copied from former video.core)

def _ffmpeg_extract_frames(
    video_file: Path | str,
    out_dir: Path | str,
    every_sec: int,
    limit: Optional[int] = None,
) -> List[Tuple[float, Path]]:
    video_file = Path(video_file)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if shutil.which("ffmpeg") is None:
        logger.warning("ffmpeg not found – skipping frame extraction.")
        return []

    pattern = out_dir / "frame_%06d.jpg"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_file),
        "-vf",
        f"fps=1/{every_sec}",
        "-q:v",
        "2",
        str(pattern),
    ]
    logger.info("Extracting frames every %ds from %s", every_sec, video_file)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    frames = sorted(out_dir.glob("frame_*.jpg"))
    if limit is not None:
        frames = frames[:limit]

    return [(idx * every_sec, p) for idx, p in enumerate(frames)]


__all__ = [
    "extract_frames",
    "analyze_frames",
    "extract_video_id",
    "download_video",
] 