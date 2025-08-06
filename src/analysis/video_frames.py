"""Frame extraction & object detection helpers.

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

from src.config.settings import SETTINGS, PipelineSettings
import os
import re, subprocess, shutil, urllib.parse
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Duration and auto-quality helpers
# ---------------------------------------------------------------------------

def parse_iso_duration_to_minutes(iso_duration: str) -> int:
    """Convert ISO-8601 duration (PT#H#M#S) to total minutes."""
    if not iso_duration:
        return 0
    
    pattern = re.compile(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?")
    match = pattern.match(iso_duration)
    if not match:
        return 0
    
    hours, minutes, seconds = (int(x) if x else 0 for x in match.groups())
    total_minutes = hours * 60 + minutes + (seconds / 60)
    return int(round(total_minutes))


def get_video_duration_from_url(url: str) -> int:
    """Get video duration in minutes from YouTube URL using yt-dlp."""
    try:
        vid = extract_video_id(url)
        cmd = ["yt-dlp", "--print", "duration", f"https://www.youtube.com/watch?v={vid}"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration_seconds = float(result.stdout.strip())
        return int(round(duration_seconds / 60))
    except Exception as e:
        logger.warning(f"Failed to get video duration: {e}")
        return 0


def auto_select_video_quality(duration_minutes: int) -> str:
    """Automatically select video quality based on duration.
    
    Rules:
    - 0-5 minutes: high quality
    - 5-15 minutes: medium quality  
    - 15+ minutes: low quality (small)
    """
    if duration_minutes <= 5:
        return "best"
    elif duration_minutes <= 15:
        return "medium"
    else:
        return "small"


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
_YT_REGEX = re.compile(r"(?:v=|youtu\.be/|/shorts/)([A-Za-z0-9_-]{11})")

def extract_video_id(url: str) -> str:
    m = _YT_REGEX.search(url)
    if m:
        return m.group(1)
    parsed = urllib.parse.urlparse(url)
    qs = urllib.parse.parse_qs(parsed.query)
    if "v" in qs:
        return qs["v"][0]
    raise ValueError(f"Unable to parse video id from URL: {url}")


def download_video(url: str, output_dir: Path | str = "downloads", quality: str = "best") -> Path:
    """Download video with configurable quality settings to manage file size.
    
    Args:
        url: YouTube video URL
        output_dir: Directory to save the video
        quality: Quality setting - options:
            - "best": Best quality (largest files) - current default
            - "medium": Good balance of quality/size (720p max)
            - "small": Smaller files (480p max)
            - "tiny": Smallest files (360p max, audio-only fallback)
            - "audio": Audio only (for transcription-focused workflows)
    
    Returns:
        Path to downloaded video file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vid = extract_video_id(url)
    
    # Configure format selection based on quality preference
    if quality == "best":
        # Current behavior - best quality available
        format_selector = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4"
        ext = "mp4"
    elif quality == "medium":
        # Max 720p, prefer smaller sizes
        format_selector = "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/mp4"
        ext = "mp4"
    elif quality == "small":
        # Max 480p for smaller files
        format_selector = "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480][ext=mp4]/mp4"
        ext = "mp4"
    elif quality == "tiny":
        # Max 360p, very small files
        format_selector = "bestvideo[height<=360][ext=mp4]+bestaudio[ext=m4a]/best[height<=360][ext=mp4]/worst[ext=mp4]/mp4"
        ext = "mp4"
    elif quality == "audio":
        # Audio only - smallest possible size for transcription workflows
        format_selector = "bestaudio[ext=m4a]/bestaudio"
        ext = "m4a"
    else:
        raise ValueError(f"Invalid quality setting: {quality}. Use: best, medium, small, tiny, or audio")
    
    out_path = output_dir / f"{vid}.{ext}"
    
    # Build command - only use merge-output-format for video+audio combinations
    # Supply a realistic User-Agent and referer to reduce the chance of HTTP 403 errors
    user_agent = os.getenv("YT_USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:118.0) Gecko/20100101 Firefox/118.0")
    cmd = [
        "yt-dlp",
        "--user-agent", user_agent,
        "--referer", url,
        "-f", format_selector,
    ]
    
    if quality != "audio":
        # Only add merge format for video downloads that combine video+audio
        cmd.extend(["--merge-output-format", ext])
    
    cmd.extend(["-o", str(out_path), url])
    
    logger.info("Downloading video %s -> %s (quality: %s)", url, out_path, quality)
    
    # Check if file already exists and is valid
    if out_path.exists() and out_path.stat().st_size > 0:
        logger.info(f"Video file already exists: {out_path}")
        return out_path
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if out_path.exists() and out_path.stat().st_size > 0:
            logger.info(f"Successfully downloaded video: {out_path} ({out_path.stat().st_size / (1024*1024):.1f} MB)")
            return out_path
        else:
            raise RuntimeError(f"Download completed but file not found or empty: {out_path}")
            
    except subprocess.CalledProcessError as e:
        logger.error(f"yt-dlp failed with command: {' '.join(cmd)}")
        logger.error(f"Exit code: {e.returncode}")
        if e.stderr:
            logger.error(f"stderr: {e.stderr}")
        
        # Try again with a simpler progressive MP4 if DASH split formats
        # are blocked (common cause of 403). This is attempted exactly once.
        logger.info("Retrying with single-stream MP4 fallback …")
        fallback_cmd = [
            "yt-dlp",
            "--user-agent", user_agent,
            "--referer", url,
            "-f", "best[ext=mp4]/mp4",
            "-o", str(out_path),
            url,
        ]
        try:
            subprocess.run(fallback_cmd, check=True, capture_output=True, text=True)
            if out_path.exists() and out_path.stat().st_size > 0:
                logger.info(f"Fallback download successful: {out_path}")
                return out_path
            else:
                raise RuntimeError(f"Fallback download completed but file not found: {out_path}")
        except subprocess.CalledProcessError as fallback_error:
            logger.error(f"Fallback download also failed: {fallback_error}")
        
        # Re-raise the original exception with more context
        raise RuntimeError(f"Video download failed for {url}. Original error: {e}") from e


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
    "parse_iso_duration_to_minutes",
    "get_video_duration_from_url", 
    "auto_select_video_quality",
] 