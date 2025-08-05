"""Business-logic helpers for single-video analytics.

This helper orchestrates comprehensive audio, visual, comment, and statistic analysis for a
single YouTube video, using the same detailed approach as channel analytics.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from src.analysis import audio as audio_mod
from src.analysis import video_frames as vf_mod
from src.analysis import video_vision as vision_mod
from src.analysis import comments as comments_mod
from src.analysis.video_frames import download_video, get_video_duration_from_url, auto_select_video_quality
from src.analysis.audio import extract_audio, transcribe as transcribe_audio
from src.config.settings import SETTINGS
from src.youtube import public as yt_public
from src.youtube.oauth import get_service as get_oauth_service
from src.auth.manager import list_token_files, TOKENS_DIR
from src.llms import get_smart_client
from src.prompts.audio_analysis import get_enhanced_audio_analysis_prompt
from src.prompts.video_summary import get_comprehensive_video_summary_prompt
from src.prompts.comments_analysis import get_comments_summary_prompt

logger = logging.getLogger(__name__)

# Default root where analysis artefacts are written
REPORTS_DIR = Path("data/reports/video_analysis")
DEFAULT_CLIENT_SECRET = Path("client_secret.json")


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _json_dump(data: Any, path: Path) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    logger.info("Saved JSON → %s", path)


def _get_oauth_service_for_video(video_id: str):
    """Try to get OAuth service for enhanced analytics, return None if not available."""
    try:
        # Check if client_secret.json exists or environment variables are set
        if not DEFAULT_CLIENT_SECRET.exists():
            # Try to create from environment variables
            from src.auth.manager import create_temp_client_secret_file
            temp_client_secret = create_temp_client_secret_file()
            if not temp_client_secret:
                logger.info("No OAuth client configuration available (missing client_secret.json and environment variables)")
                return None
            client_secret_path = temp_client_secret
        else:
            client_secret_path = DEFAULT_CLIENT_SECRET
            
        # Check if we have any valid OAuth tokens available
        token_files = list_token_files()
        if not token_files:
            logger.info("No OAuth tokens found in data/tokens/ directory")
            return None
            
        # Use the first valid OAuth token we find
        for token_file in token_files:
            try:
                oauth_service = get_oauth_service(client_secret_path, token_file)
                # Test if the service works by making a simple call
                me = oauth_service.channels().list(part="id", mine=True).execute()
                if me.get("items"):
                    logger.info(f"Using OAuth token: {token_file.name}")
                    return oauth_service
            except Exception as e:
                logger.debug(f"OAuth token {token_file} failed: {e}")
                continue
                
        logger.info("No valid OAuth tokens found")
        return None
    except Exception as e:
        logger.debug(f"OAuth service detection failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_recent_videos(channel_id: str, *, max_videos: int = 20) -> List[Dict[str, Any]]:
    """Return list of ``{"video_id", "title"}`` for *channel_id* (public API)."""

    api_key = os.getenv("YT_API_KEY") or SETTINGS.youtube_api_key
    if not api_key:
        raise RuntimeError("YouTube API key not configured (set YT_API_KEY env var or config.settings.youtube_api_key)")

    svc = yt_public.get_service(api_key)

    ch_resp = (
        svc.channels()
        .list(part="contentDetails", id=channel_id)
        .execute()
    )
    if not ch_resp.get("items"):
        raise RuntimeError(f"Channel not found: {channel_id}")

    uploads_pl = ch_resp["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

    vids: List[Dict[str, Any]] = []
    next_token = None
    while len(vids) < max_videos:
        pl_resp = (
            svc.playlistItems()
            .list(part="contentDetails,snippet", playlistId=uploads_pl, maxResults=50, pageToken=next_token)
            .execute()
        )
        for item in pl_resp.get("items", []):
            vids.append(
                {
                    "video_id": item["contentDetails"]["videoId"],
                    "title": item["snippet"]["title"],
                }
            )
            if len(vids) >= max_videos:
                break
        next_token = pl_resp.get("nextPageToken")
        if not next_token:
            break
    return vids


def analyze_video(
    video_id: str,
    *,
    output_base: Path | str | None = None,
    frame_interval_sec: int | None = None,
) -> Dict[str, Any]:
    """Run comprehensive audio, visual, comment and stats analysis for *video_id*.
    
    Uses the same detailed analysis approach as channel analytics.
    """

    output_base = Path(output_base or REPORTS_DIR) / video_id
    _ensure_dir(output_base)

    api_key = os.getenv("YT_API_KEY") or SETTINGS.youtube_api_key
    if not api_key:
        raise RuntimeError("YouTube API key not configured (set YT_API_KEY env var or config.settings.youtube_api_key)")

    yt_service = yt_public.get_service(api_key)
    oauth_service = _get_oauth_service_for_video(video_id)
    
    # Log OAuth status for debugging
    if oauth_service:
        logger.info(f"OAuth service available for video {video_id}")
    else:
        logger.info(f"No OAuth service available for video {video_id}")

    # Get video info first
    video_resp = yt_service.videos().list(part="snippet,statistics,contentDetails", id=video_id).execute()
    if not video_resp.get("items"):
        raise RuntimeError(f"Video not found: {video_id}")
    
    video_info = video_resp["items"][0]
    video_title = video_info["snippet"]["title"]

    result = {
        "video_id": video_id,
        "title": video_title,
        "success": False,
        "output_dir": output_base,
        "audio_analysis": None,
        "video_analysis": None,
        "comments_analysis": None,
        "statistics": None,
        "oauth_analytics": None,
        "summary_path": None,
        "analysis_json_path": None,
        "error": None
    }

    try:
        # ------------------------------------------------------------------
        # 1. Download video (respect auto quality based on duration)
        # ------------------------------------------------------------------
        url = f"https://www.youtube.com/watch?v={video_id}"
        duration_minutes = get_video_duration_from_url(url)
        quality = auto_select_video_quality(duration_minutes)
        video_path = download_video(url, output_dir=output_base, quality=quality)

        # ------------------------------------------------------------------
        # 2. Enhanced Audio Analysis (same as channel analytics)
        # ------------------------------------------------------------------
        audio_wav = extract_audio(video_path, output_base / f"{video_id}.wav")
        segments = transcribe_audio(audio_wav)
        full_transcript = ""
        
        if segments:
            full_transcript = "\n".join(s.get("text", "") for s in segments)
            
            # Enhanced audio analysis with LLM (same prompt as channel analytics)
            audio_analysis = None
            if SETTINGS.openrouter_api_keys or SETTINGS.groq_api_keys or SETTINGS.gemini_api_keys:
                try:
                    client = get_smart_client()
                    
                    enhanced_audio_prompt = get_enhanced_audio_analysis_prompt(full_transcript)

                    audio_analysis = client.chat(
                        [{"role": "user", "content": enhanced_audio_prompt}],
                        temperature=0.3,
                        max_tokens=2000,
                    )
                except Exception as e:
                    logger.warning(f"Enhanced audio analysis failed: {e}")
                    audio_analysis = "Enhanced audio analysis failed"

        # Save audio data with sentiment
        audio_data = audio_mod.analyze_audio(audio_wav, out_path=output_base / f"{video_id}_audio.json")
        result["audio_analysis"] = audio_analysis

        # ------------------------------------------------------------------
        # 3. Video frames → vision analysis (same as channel analytics)
        # ------------------------------------------------------------------
        frames_dir = output_base / "frames"
        frames: List[Tuple[float, Path]] = vf_mod.extract_frames(
            video_path,
            frames_dir,
            every_sec=frame_interval_sec or SETTINGS.frame_interval_sec,
            limit=32,
        )
        _json_dump([{"timestamp": ts, "file": p.name} for ts, p in frames], output_base / f"{video_id}_frames.json")

        vision_analysis = ""
        if frames:
            try:
                vision_analysis = vision_mod.summarise_frames(frames)
                (output_base / f"{video_id}_vision_summary.md").write_text(vision_analysis, encoding="utf-8")
            except Exception as e:
                logger.warning("Vision analysis failed: %s", e)
                vision_analysis = "Vision analysis failed"

        result["video_analysis"] = vision_analysis

        # ------------------------------------------------------------------
        # 4. Comments → fetch + sentiment (same as channel analytics)
        # ------------------------------------------------------------------
        try:
            comments_data = comments_mod.fetch_and_analyze(
                yt_service, video_id, out_path=output_base / f"{video_id}_comments.json"
            )
            
            # Enhanced comment analysis using centralized prompts
            if comments_data:
                try:
                    # Generate comprehensive comments analysis using LLM
                    if SETTINGS.openrouter_api_keys or SETTINGS.groq_api_keys or SETTINGS.gemini_api_keys:
                        client = get_smart_client()
                        comments_prompt = get_comments_summary_prompt(comments_data)
                        
                        comments_analysis = client.chat(
                            [{"role": "user", "content": comments_prompt}],
                            temperature=0.3,
                            max_tokens=1000,
                        )
                        result["comments_analysis"] = comments_analysis
                    else:
                        # Fallback to basic summary if no LLM available
                        sentiments = [c.get('sentiment', 0) for c in comments_data if 'sentiment' in c]
                        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
                        positive_count = sum(1 for s in sentiments if s > 0.1)
                        negative_count = sum(1 for s in sentiments if s < -0.1)
                        
                        comments_summary = f"""Comments Analysis Summary:
- Total Comments: {len(comments_data)}
- Average Sentiment: {avg_sentiment:.2f}
- Positive Comments: {positive_count}
- Negative Comments: {negative_count}
- Neutral Comments: {len(comments_data) - positive_count - negative_count}
"""
                        result["comments_analysis"] = comments_summary
                except Exception as e:
                    logger.warning(f"Enhanced comments analysis failed: {e}")
                    result["comments_analysis"] = f"Comments analysis failed: {e}"
            else:
                result["comments_analysis"] = "No comments found or analysis failed"
                
        except Exception as e:
            logger.warning("Comment analysis failed: %s", e)
            result["comments_analysis"] = f"Comment analysis failed: {e}"

        # ------------------------------------------------------------------
        # 5. Statistics (Public + OAuth if available)
        # ------------------------------------------------------------------
        # Public stats already fetched
        stats = video_info
        _json_dump(stats, output_base / f"{video_id}_stats.json")
        
        # OAuth analytics if available (use All Time period)
        oauth_analytics = None
        if oauth_service:
            try:
                from src.youtube.analytics import get_comprehensive_video_analytics
                from datetime import datetime
                
                channel_id = video_info["snippet"]["channelId"]
                
                # Calculate days_back for "All Time" from video publish date
                published_at = video_info["snippet"].get("publishedAt")
                if published_at:
                    created_date = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                    days_back = (datetime.now(created_date.tzinfo) - created_date).days
                else:
                    days_back = 3650  # fallback to 10 years if published_at missing
                
                oauth_analytics = get_comprehensive_video_analytics(oauth_service, video_id, channel_id, days_back=days_back)
                if oauth_analytics:
                    _json_dump(oauth_analytics, output_base / f"{video_id}_oauth_analytics.json")
            except Exception as e:
                logger.warning(f"OAuth analytics failed: {e}")
                oauth_analytics = {"error": str(e)}
        
        result["statistics"] = stats
        result["oauth_analytics"] = oauth_analytics

        # ------------------------------------------------------------------
        # 6. Generate comprehensive summary (same as channel analytics)
        # ------------------------------------------------------------------
        try:
            if SETTINGS.openrouter_api_keys or SETTINGS.groq_api_keys or SETTINGS.gemini_api_keys:
                client = get_smart_client()
                
                summary_prompt = get_comprehensive_video_summary_prompt(
                    video_title,
                    audio_analysis,
                    vision_analysis,
                    result.get('comments_analysis'),
                    stats
                )

                summary = client.chat(
                    [{"role": "user", "content": summary_prompt}],
                    temperature=0.3,
                    max_tokens=3000,
                )
                
                summary_path = output_base / f"{video_id}_summary.md"
                summary_path.write_text(summary, encoding="utf-8")
                result["summary_path"] = summary_path
                
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")

        # ------------------------------------------------------------------
        # 7. Create combined analysis JSON for further LLM processing
        # ------------------------------------------------------------------
        analysis_data = {
            "video_id": video_id,
            "title": video_title,
            "audio_analysis": audio_analysis,
            "video_analysis": vision_analysis,
            "comments_analysis": result.get("comments_analysis"),
            "statistics": stats,
            "oauth_analytics": oauth_analytics,
            "transcript": full_transcript,
            "frames_count": len(frames),
            "comments_count": len(comments_data) if comments_data else 0,
        }
        
        analysis_json_path = output_base / f"{video_id}_analysis.json"
        _json_dump(analysis_data, analysis_json_path)
        result["analysis_json_path"] = analysis_json_path

        result["success"] = True
        logger.info(f"Video analysis completed successfully for {video_id}")

    except Exception as e:
        logger.error(f"Video analysis failed for {video_id}: {e}")
        result["error"] = str(e)

    return result


__all__ = [
    "fetch_recent_videos",
    "analyze_video",
]