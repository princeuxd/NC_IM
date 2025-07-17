#!/usr/bin/env python3
"""End-to-end pipeline: download video, fetch metadata, owner analytics (if possible),
transcribe, sentiment, logo detection, correlation.

Usage examples:
    python fetch_and_analyze.py --url https://youtu.be/7lCDEYXw3mM \
        --api-key $YT_API_KEY \
        --client-secrets-file client_secret.json \
        --token-file oauth_token.json

If OAuth flags are omitted the script runs in public-only mode (skips analytics).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Make sibling top-level packages importable when this file is executed as a
# script (``python pipeline/fetch_and_analyze.py``). Python normally puts the
# *containing* directory (``pipeline/``) on ``sys.path`` which means sibling
# packages like ``analysis`` are not visible.  We prepend the parent directory
# to ``sys.path`` at runtime so that ``import analysis`` etc. work without the
# user needing to set PYTHONPATH or use ``python -m``.
# ---------------------------------------------------------------------------

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------

import argparse
import logging

from analysis import (  # type: ignore[attr-defined]
    analyze_comment_sentiment,
    analyze_transcript_sentiment,
    detect_logos,
    fetch_comments,
    save as save_json,
    transcribe_audio,
)
from config import parse_args as parse_base
from video import process_video, extract_video_id  # type: ignore[attr-defined]
from auth import get_oauth_service, get_public_service  # type: ignore[attr-defined]

logging.basicConfig(level=logging.INFO)


def parse_cli():
    base = parse_base()  # type: ignore[misc]
    parser = argparse.ArgumentParser(add_help=True, description="Fetch + analyze YouTube video")
    parser.add_argument("--url", required=True, help="YouTube video URL")
    parser.add_argument("--output", default="reports", help="Output directory root")
    args = parser.parse_args(namespace=base)
    return args


def main():
    args = parse_cli()  # type: ignore[arg-type]

    pub_service = None
    if args.yt_api_key:  # type: ignore[truthy-bool]
        try:
            pub_service = get_public_service(args.yt_api_key)  # type: ignore[misc]
        except Exception as exc:  # pylint: disable=broad-except
            logging.warning("Public API key failed (%s). Will rely on OAuth if available.", exc)

    oauth_service = None
    analytics_service = None
    channel_id = None
    if args.client_secrets_file and Path(args.client_secrets_file).exists():
        try:
            from googleapiclient.discovery import build  # lazy import to avoid cost if no oauth

            oauth_service = get_oauth_service(args.client_secrets_file, args.token_file)

            # obtain channel id of owner via YouTube Data API
            me = oauth_service.channels().list(part="id", mine=True).execute()  # type: ignore[attr-defined]
            channel_id = me["items"][0]["id"]

            # Build separate YouTube Analytics v2 service for in-depth metrics
            analytics_service = build("youtubeAnalytics", "v2", credentials=oauth_service._http.credentials)  # type: ignore[attr-defined,protected-access]
        except Exception as exc:  # pylint: disable=broad-except
            logging.warning("OAuth not available (%s). Proceeding public-only.", exc)
            oauth_service = None
            analytics_service = None

    # Ensure we have *some* Data API service for public metadata/comments.
    if pub_service is None and oauth_service is not None:
        pub_service = oauth_service

    if pub_service is None:
        logging.error("No usable YouTube Data API client – supply --api-key or OAuth files.")
        sys.exit(1)

    vid = extract_video_id(args.url)  # type: ignore[attr-defined]
    base_dir = Path(args.output)  # type: ignore[attr-defined]

    process_video(  # type: ignore[attr-defined]
        args.url,  # type: ignore[attr-defined]
        public_service=pub_service,
        analytics_service=analytics_service,
        channel_id=channel_id,
        output_dir=base_dir,  # video_utils will handle subfolder creation
    )

    root = base_dir / vid

    # ------------ Transcript & sentiment ------------
    wav_path = root / "audio.wav"
    if wav_path.exists():
        segments = transcribe_audio(wav_path)  # type: ignore[arg-type]
        segments_sent = analyze_transcript_sentiment(segments)
        save_json(segments_sent, root / "transcript_sentiment.json")

    # ------------ Comments sentiment ---------------
    comments = fetch_comments(pub_service, vid)  # type: ignore[arg-type]
    comments_sent = analyze_comment_sentiment(comments)
    save_json(comments_sent, root / "comments_sentiment.json")

    # ------------ Logo / product detection ---------
    mp4_path = root / f"{vid}.mp4"
    if mp4_path.exists():
        try:
            logo_segments = detect_logos(mp4_path)
            save_json(logo_segments, root / "logo_segments.json")
        except Exception as exc:  # pylint: disable=broad-except
            logging.warning("Logo detection skipped (%s)", exc)

    logging.info("All done. Results in %s", root)


if __name__ == "__main__":
    main() 