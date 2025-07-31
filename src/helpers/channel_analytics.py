"""Business-logic helper for multi-video *channel* analytics.

Thin façade around `analysis.channel_analysis.ChannelAnalysisService` so that UI
code can call a single function without worrying about class setup.
"""
from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Dict, Any

from src.config.settings import SETTINGS
from src.analysis.channel_analysis import ChannelAnalysisService
from src.youtube.public import get_service as get_public_service

logger = logging.getLogger(__name__)

REPORTS_DIR = Path("data/reports/channel_analysis")


def analyze_channel(channel_id: str, *, num_videos: int = 10) -> Dict[str, Any]:
    """Run full channel-level analysis.

    Parameters
    ----------
    channel_id : str
        YouTube Channel ID (starting with ``UC``) or other accepted identifier
        for `ChannelAnalysisService.extract_channel_id`.
    num_videos : int, default 10
        Maximum number of recent uploads to include in the analysis.

    Returns
    -------
    Dict[str, Any]
        A result dict mirroring the structure returned by
        ``ChannelAnalysisService.process_channel_videos`` plus a ``collective``
        entry that contains LLM summary information when generation succeeds.
    """

    api_key = os.getenv("YT_API_KEY") or SETTINGS.youtube_api_key
    if not api_key:
        raise RuntimeError("YouTube API key not configured (env YT_API_KEY or config.settings.youtube_api_key)")

    svc = ChannelAnalysisService(api_key)

    # Resolve channel title via public API (for nicer filenames / summaries)
    try:
        pub_svc = get_public_service(api_key)
        ch_resp = pub_svc.channels().list(part="snippet", id=channel_id).execute()  # type: ignore[attr-defined]
        channel_title = ch_resp["items"][0]["snippet"]["title"] if ch_resp.get("items") else channel_id
    except Exception:
        channel_title = channel_id

    logger.info("Starting channel analysis for %s (%s) – %d videos", channel_title, channel_id, num_videos)

    results = svc.process_channel_videos(channel_id, channel_title, max_videos=num_videos)

    # Run collective (cross-video) analysis using LLM
    try:
        collective = svc.generate_collective_analysis(channel_id, channel_title, results["output_dir"])
        results["collective"] = collective
    except Exception as e:
        logger.warning("Collective LLM analysis failed: %s", e)
        results["collective"] = {"error": str(e)}

    return results


__all__ = ["analyze_channel"]
