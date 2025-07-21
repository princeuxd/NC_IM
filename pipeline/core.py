from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Callable
import isodate  # type: ignore

from analysis import (
    analyze_comment_sentiment,
    analyze_transcript_sentiment,
    save as save_json,
    transcribe_audio,
)
from analysis.sentiment_llm import analyze_comment_sentiment_llm
from analysis.summarizer import generate_summary
from analysis.object_detection import detect_objects
from auth import get_oauth_service, get_public_service
from config.settings import SETTINGS
from video import (
    extract_video_id,
    fetch_and_save_video_metadata,
    download_video,
    extract_audio,
    extract_frames,
)
from analysis.analytics_helpers import (
    fetch_engagement_timeseries,
    correlate_products_with_engagement,
)

logger = logging.getLogger(__name__)


def run_pipeline(
    url: str,
    output_dir: Path,
    public_api_key: str | None = None,
    client_secrets_file: str | None = None,
    token_file: str | None = None,
    num_frames_for_summary: int = 10,
    progress_callback: Callable[[str], None] | None = None,
):
    """Run the full analysis pipeline, with progress updates."""

    def _update_status(message: str):
        logger.info(message)
        if progress_callback:
            progress_callback(message)

    vid = extract_video_id(url)
    video_folder = output_dir / vid
    video_folder.mkdir(parents=True, exist_ok=True)

    # --- Authentication ---
    _update_status("🔐 Authenticating...")
    pub_service = get_public_service(public_api_key) if public_api_key else None
    oauth_service, analytics_service, channel_id = None, None, None
    if client_secrets_file and Path(client_secrets_file).exists() and token_file:
        try:
            from googleapiclient.discovery import build

            oauth_service = get_oauth_service(client_secrets_file, token_file)
            me = oauth_service.channels().list(part="id", mine=True).execute()
            channel_id = me["items"][0]["id"]
            analytics_service = build(
                "youtubeAnalytics", "v2", credentials=oauth_service._http.credentials
            )
            _update_status("🔓 OAuth successful.")
        except Exception as e:
            logger.warning(f"OAuth failed, proceeding with public access only: {e}")
            oauth_service = analytics_service = None

    service = oauth_service or pub_service
    if not service:
        raise ConnectionError("No valid YouTube service object. Check API keys/OAuth.")

    # --- Video Processing ---
    _update_status("🎬 Fetching video metadata...")
    fetch_and_save_video_metadata(
        service,
        vid,
        video_folder,
        analytics_service=analytics_service,
        channel_id=channel_id,
    )

    _update_status("🔽 Downloading video...")
    mp4_path = download_video(url, video_folder)

    _update_status("🔊 Extracting audio...")
    wav_path = extract_audio(mp4_path, video_folder / "audio.wav")

    # --- Analysis ---
    if wav_path.exists():
        _update_status("✍️ Transcribing audio...")
        segments = transcribe_audio(wav_path)
        segments_sent = analyze_transcript_sentiment(segments)
        save_json(segments_sent, video_folder / "transcript_sentiment.json")

    _update_status("💬 Analyzing comments...")
    from analysis import fetch_comments
    comments = fetch_comments(service, vid)
    if SETTINGS.openrouter_api_key or SETTINGS.groq_api_key:
        comments_sent = analyze_comment_sentiment_llm(comments)
    else:
        comments_sent = analyze_comment_sentiment(comments)
    save_json(comments_sent, video_folder / "comments_sentiment.json")

    # --- Advanced Analytics (if OAuth is available) ---
    if analytics_service and channel_id:
        _update_status("📈 Correlating engagement data...")
        frames = extract_frames(mp4_path, video_folder / "frames")
        if frames:
            detections = detect_objects(frames)
            save_json(detections, video_folder / "product_frames.json")

            today = date.today()
            start = today.replace(year=today.year - 1)
            ts_rows = fetch_engagement_timeseries(
                analytics_service, vid, channel_id, start_date=start, end_date=today
            )
            if ts_rows:
                enriched = correlate_products_with_engagement(detections, ts_rows)
                save_json(enriched, video_folder / "product_impact.json")

    # --- Summary ---
    _update_status("🖼️ Extracting frames for summary...")
    # Get video duration to sample frames evenly
    metadata_path = video_folder / "metadata.json"
    duration_sec = 0
    if metadata_path.exists():
        import json
        metadata = json.loads(metadata_path.read_text())
        duration_iso = metadata.get("contentDetails", {}).get("duration")
        if duration_iso:
            duration_sec = isodate.parse_duration(duration_iso).total_seconds()

    # Determine frame extraction interval
    frame_interval = (
        int(duration_sec / (num_frames_for_summary + 1))
        if duration_sec > 0 and num_frames_for_summary > 0
        else 0
    )
    
    summary_frame_paths = []
    if frame_interval > 0:
        summary_frames = extract_frames(
            mp4_path,
            video_folder / "summary_frames",
            every_sec=frame_interval,
            limit=num_frames_for_summary,
        )
        summary_frame_paths = [p for _, p in summary_frames]

    # --- Summary ---
    _update_status("📝 Generating executive summary...")
    generate_summary(video_folder, summary_frame_paths)

    _update_status("✅ Pipeline complete!")
    return video_folder
