import os
import json
import time
from pathlib import Path

import streamlit as st
import pandas as pd
from youtube.oauth import get_service as get_oauth_service

from youtube.public import get_service as get_public_service
from analysis.video_frames import (
    extract_video_id,
    download_video,
    extract_frames,
    get_video_duration_from_url,
    auto_select_video_quality,
)
from analysis.audio import extract_audio, transcribe as transcribe_audio
from llms import get_client
from config.settings import SETTINGS

from dotenv import load_dotenv  # type: ignore
import logging
import warnings
from auth.manager import (
    list_token_files as _list_token_files,
    get_creator_details,
    validate_client_secret,
    onboard_creator,
    remove_creator as _remove_creator,
    refresh_creator_token,
    validate_env_oauth_config,
    create_temp_client_secret_file,
)
from analysis.video_vision import summarise_frames
from analysis.azure_video_indexer import AzureVideoIndexer, AzureVideoIndexerError, format_insights_summary, upload_and_analyze

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress FP16 warning from Whisper
warnings.filterwarnings(
    "ignore", message="FP16 is not supported on CPU; using FP32 instead"
)

# Suppress googleapiclient discovery_cache warning
logging.getLogger("googleapiclient.discovery_cache").setLevel(logging.WARNING)

# Suppress object detection warnings (they're handled gracefully in the code)
logging.getLogger("analysis.object_detection").setLevel(logging.ERROR)

load_dotenv()

ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True, parents=True)

TOKENS_DIR = ROOT / "tokens"
TOKENS_DIR.mkdir(exist_ok=True, parents=True)

DEFAULT_CLIENT_SECRET = ROOT / "client_secret.json"


# ---------------------------------------------
# Environment status section removed as per user request
# (Key lookups kept for potential debug use)
yt_key = os.getenv("YT_API_KEY")
openrouter_key = os.getenv("OPENROUTER_API_KEY")
groq_key = os.getenv("GROQ_API_KEY")
# ---------------------------------------------

# Optional debug section (moved to expander)
with st.expander("🔧 Debug Info", expanded=False):
    st.code(f"Root: {ROOT}")
    st.code(
        f"Keys: YT={'✓' if yt_key else '✗'} OR={'✓' if openrouter_key else '✗'} GQ={'✓' if groq_key else '✗'}"
    )
    
    # Rate limit reset button
    if st.button("🔄 Reset Rate Limits", help="Clear all rate limit timers for LLM providers"):
        try:
            from llms.key_manager import key_manager
            key_manager.clear_rate_limits()
            st.success("✅ Rate limits cleared for all providers!")
            st.rerun()  # Refresh the page to show updated status
        except Exception as e:
            st.error(f"❌ Failed to reset rate limits: {e}")
# ---------------------------------------------


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
# Duplicated helper functions have been replaced by imports from ``auth.manager``


def get_channel_stats(service, channel_id: str) -> dict:
    """Fetch channel statistics and snippet information."""
    try:
        response = (
            service.channels().list(part="snippet,statistics", id=channel_id).execute()
        )

        if response["items"]:
            channel = response["items"][0]

            # Get the best available thumbnail
            thumbnails = channel["snippet"].get("thumbnails", {})
            thumbnail_url = None

            # Try thumbnails in order of preference (highest quality first)
            for size in ["high", "medium", "default"]:
                if size in thumbnails:
                    raw_url = thumbnails[size]["url"]

                    # Clean up Google's channel thumbnail URL parameters that can cause issues
                    if "yt3.ggpht.com" in raw_url:
                        try:
                            # Remove problematic parameters and use a simpler format
                            base_url = raw_url.split("=")[0]
                            thumbnail_url = f"{base_url}=s240-c-k-c0x00ffffff-no-rj"
                        except Exception:
                            thumbnail_url = raw_url
                    else:
                        thumbnail_url = raw_url

                    break

            return {
                "title": channel["snippet"]["title"],
                "description": channel["snippet"]["description"],
                "thumbnail": thumbnail_url,
                "subscriber_count": int(
                    channel["statistics"].get("subscriberCount", 0)
                ),
                "video_count": int(channel["statistics"].get("videoCount", 0)),
                "view_count": int(channel["statistics"].get("viewCount", 0)),
                "published_at": channel["snippet"]["publishedAt"],
                "custom_url": channel["snippet"].get("customUrl", ""),
            }
    except Exception as e:
        logger.error(f"Failed to fetch channel stats: {e}")
    return {}


def display_channel_stats(channel_stats: dict):
    """Display channel statistics in the main content area."""
    if not channel_stats:
        return

    with st.expander("📺 Channel Information", expanded=True):
        # Channel thumbnail and name
        col_thumb, col_info = st.columns([1, 3])

        with col_thumb:
            thumbnail_url = channel_stats.get("thumbnail")
            if thumbnail_url:
                try:
                    # Ensure HTTPS for better compatibility
                    if thumbnail_url.startswith("http://"):
                        thumbnail_url = thumbnail_url.replace("http://", "https://", 1)

                    st.image(thumbnail_url, width=120, caption="Channel Avatar")
                except Exception:
                    st.markdown("🖼️ **Channel Avatar**")
            else:
                st.info("No thumbnail available")

        with col_info:
            st.markdown(f"**{channel_stats.get('title', 'Unknown Channel')}**")
            if channel_stats.get("custom_url"):
                st.markdown(f"@{channel_stats['custom_url']}")

            # Description preview
            if channel_stats.get("description"):
                description = channel_stats["description"]
                preview = (
                    description[:150] + "..." if len(description) > 150 else description
                )
                st.caption(f"_{preview}_")

        # Statistics header
        st.markdown("**📊 Channel Statistics**")

    # Format numbers nicely
    def format_number(num):
        if num >= 1_000_000:
            return f"{num/1_000_000:.1f}M"
        elif num >= 1_000:
            return f"{num/1_000:.1f}K"
        return str(num)

    subscribers = format_number(channel_stats.get("subscriber_count", 0))
    videos = format_number(channel_stats.get("video_count", 0))
    views = format_number(channel_stats.get("view_count", 0))

    # Create metrics in a compact layout
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("👥 Subscribers", subscribers)
    with col2:
        st.metric("🎬 Videos", videos)
    with col3:
        st.metric("👀 Total Views", views)
    with col4:
        if channel_stats.get("video_count", 0) > 0:
            avg_views = (
                channel_stats.get("view_count", 0) // channel_stats["video_count"]
            )
            st.metric("📈 Avg Views", format_number(avg_views))

    # Channel age
    from datetime import datetime

    try:
        published_at = channel_stats.get("published_at")
        if published_at:
            created_date = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
            years_old = (datetime.now(created_date.tzinfo) - created_date).days // 365
            if years_old > 0:
                st.caption(f"**📅 Channel Age:** {years_old} years")
            else:
                days_old = (datetime.now(created_date.tzinfo) - created_date).days
                st.caption(f"**📅 Channel Age:** {days_old} days")
    except Exception as e:
        logger.error(f"Failed to parse channel age: {e}")

# ---------------------------------------------------------------------------
# Audio Analyzer section
# ---------------------------------------------------------------------------


def audio_analyzer_section():
    st.title("🎤 Audio Analyzer")

    url = st.text_input(
        "YouTube video URL",
        placeholder="https://youtu.be/abc123XYZ",
        key="audio_url",
    )

    # Automatically use audio quality for audio analysis
    quality = "audio"
    st.info("ℹ️ **Auto Quality**: Using audio-only download for optimal transcription performance.")

    if st.button("Run Audio Analysis", key="run_audio"):
        if not url.strip():
            st.error("Please enter a YouTube URL")
            return

        vid = extract_video_id(url)
        out_dir = REPORTS_DIR / vid
        out_dir.mkdir(parents=True, exist_ok=True)

        with st.spinner("Downloading video..."):
            try:
                mp4_path = download_video(url, out_dir, quality=quality)
            except Exception as e:
                st.error(f"Download failed: {e}")
                return

        with st.spinner("Extracting audio track..."):
            wav_path = extract_audio(mp4_path, out_dir / "audio.wav")
            if not wav_path.exists():
                st.error("Failed to extract audio (ffmpeg missing?)")
                return

        with st.spinner("Transcribing audio …"):
            segments = transcribe_audio(wav_path)

        if not segments:
            st.warning("No transcript returned.")
            return

        # Combine transcript text (simple concatenation)
        full_text = "\n".join(s.get("text", "") for s in segments)

        # Check if any LLM provider keys are available
        if not (SETTINGS.openrouter_api_keys or SETTINGS.groq_api_keys or SETTINGS.gemini_api_keys):
            st.warning("No LLM API keys configured – cannot generate summary.")
            st.info("Configure `OPENROUTER_API_KEY`, `GROQ_API_KEY`, or `GEMINI_API_KEY` environment variables")
            return

        with st.spinner("Generating summary via LLM …"):
            from llms import get_smart_client
            
            client = get_smart_client()
            prompt_msg = (
                "You are a podcast/video-transcript analyst. Based on the raw transcript, "
                "return a concise markdown report with the following sections:\n\n"
                "1. **Summary** – 3-5 bullet points describing the main ideas.\n"
                "2. **Overall Sentiment** – Positive / Neutral / Negative with one-sentence justification.\n"
                "3. **Category** – choose best fit from {Lifestyle, Education, Technology, Gaming, Health, Finance, Entertainment, Travel, Food, Sports, News}.\n"
                "4. **Products / Brands Mentioned** – bullet list of product names or brand references found in the transcript. If none, write 'None mentioned'."
            )

            try:
                summary = client.chat(
                    [
                        {"role": "system", "content": prompt_msg},
                        {"role": "user", "content": full_text[:12000]},
                    ],
                    temperature=0.3,
                    max_tokens=512,
                )
            except Exception as e:
                st.error(f"All LLM providers failed: {e}")
                return

        st.subheader("📝 Video Summary")
        st.markdown(summary)


# Video Analyzer section (frame → vision LLM summary)
def video_analyzer_section():
    st.title("🖼️ Video Frame Analyzer")

    url = st.text_input(
        "YouTube video URL",
        placeholder="https://youtu.be/abc123XYZ",
        key="video_url",
    )

    # Auto-select quality based on video duration
    if url.strip():
        duration_minutes = get_video_duration_from_url(url)
        quality = auto_select_video_quality(duration_minutes)
        if duration_minutes > 0:
            st.info(f"ℹ️ **Auto Quality**: Video duration ~{duration_minutes} min → Using '{quality}' quality")
        else:
            quality = "small"  # fallback
            st.warning("⚠️ Could not determine video duration, using 'small' quality as fallback")
    else:
        quality = "small"  # default when no URL provided

    every_sec = st.slider("Frame interval (seconds)", 1, 30, 5)
    max_frames = st.slider("Max frames to send", 4, 16, 8)

    if st.button("Summarise Video", key="run_video"):
        # Check if any LLM provider keys are available
        if not (SETTINGS.openrouter_api_keys or SETTINGS.groq_api_keys or SETTINGS.gemini_api_keys):
            st.error("No LLM API keys configured in environment.")
            st.info("Configure `OPENROUTER_API_KEY`, `GROQ_API_KEY`, or `GEMINI_API_KEY` environment variables")
            return

        if not url.strip():
            st.error("Please enter a YouTube URL")
            return

        vid = extract_video_id(url)
        out_dir = REPORTS_DIR / vid
        out_dir.mkdir(parents=True, exist_ok=True)

        with st.spinner("Downloading video …"):
            try:
                mp4_path = download_video(url, out_dir, quality=quality)
            except Exception as e:
                st.error(f"Download failed: {e}")
                return

        with st.spinner("Extracting frames …"):
            frames = extract_frames(mp4_path, out_dir / "frames", every_sec=every_sec)
            if not frames:
                st.error("No frames extracted – ffmpeg missing?")
                return

            frames = frames[:max_frames]

        with st.spinner("Generating vision summary …"):
            try:
                summary = summarise_frames(frames)
            except Exception as e:
                st.error(f"Vision LLM call failed: {e}")
                return

        st.subheader("📄 Frame-based Summary")
        st.markdown(summary)

        with st.expander("Preview Frames"):
            st.image([str(p) for _, p in frames], width=160)


# ---------------------------------------------------------------------------
# OAuth Enhancement System
# ---------------------------------------------------------------------------


def detect_oauth_capabilities():
    """Detect available OAuth credentials and their capabilities."""
    token_files = _list_token_files()
    oauth_info = {
        "available": len(token_files) > 0,
        "channels": [],
        "analytics_access": False,
        "private_access": False,
    }

    for token_file in token_files:
        try:
            # Test OAuth service
            oauth_service = get_oauth_service(DEFAULT_CLIENT_SECRET, token_file)

            # Get channel info
            me = (
                oauth_service.channels()
                .list(part="id,snippet,statistics", mine=True)
                .execute()
            )
            if me["items"]:
                channel_info = me["items"][0]
                channel_id = channel_info["id"]

                # Test analytics access
                analytics_available = False
                try:
                    from googleapiclient.discovery import build

                    analytics_service = build(
                        "youtubeAnalytics",
                        "v2",
                        credentials=oauth_service._http.credentials,
                    )
                    # Test with a simple query
                    from datetime import date, timedelta

                    end_date = date.today()
                    start_date = end_date - timedelta(days=30)

                    test_query = (
                        analytics_service.reports()
                        .query(
                            ids=f"channel=={channel_id}",
                            startDate=start_date.strftime("%Y-%m-%d"),
                            endDate=end_date.strftime("%Y-%m-%d"),
                            metrics="views",
                            maxResults=1,
                        )
                        .execute()
                    )
                    analytics_available = True
                except Exception:
                    analytics_available = False

                oauth_info["channels"].append(
                    {
                        "id": channel_id,
                        "title": channel_info["snippet"]["title"],
                        "token_file": token_file,
                        "analytics_access": analytics_available,
                        "subscriber_count": int(
                            channel_info["statistics"].get("subscriberCount", 0)
                        ),
                        "video_count": int(
                            channel_info["statistics"].get("videoCount", 0)
                        ),
                    }
                )

                if analytics_available:
                    oauth_info["analytics_access"] = True
                    oauth_info["private_access"] = True

        except Exception as e:
            logger.error(f"OAuth detection failed for {token_file}: {e}")

    return oauth_info


def get_enhanced_service(public_api_key, video_id=None):
    """Get the best available service (OAuth if available, otherwise public)."""
    oauth_info = detect_oauth_capabilities()

    # Build public service as fallback
    public_service = get_public_service(public_api_key)

    if not oauth_info["available"]:
        return {
            "public_service": public_service,
            "oauth_service": None,
            "analytics_service": None,
            "channel_id": None,
            "access_level": "public",
            "oauth_info": oauth_info,
            "video_owned": False,
        }

    # Check if video belongs to any of our OAuth channels
    video_owned = False
    target_channel = None

    if video_id:
        try:
            video_response = (
                public_service.videos().list(part="snippet", id=video_id).execute()
            )
            if video_response["items"]:
                video_channel_id = video_response["items"][0]["snippet"]["channelId"]
                # Check if we have OAuth for this video's channel
                for channel in oauth_info["channels"]:
                    if channel["id"] == video_channel_id:
                        target_channel = channel
                        video_owned = True
                        break
        except Exception as e:
            logger.error(f"Failed to check video ownership: {e}")

    # If no video match, use first available OAuth for enhanced comment access
    if not target_channel and oauth_info["channels"]:
        target_channel = oauth_info["channels"][0]
        video_owned = False  # We have OAuth but don't own this video

    if target_channel:
        try:
            oauth_service = get_oauth_service(
                DEFAULT_CLIENT_SECRET, target_channel["token_file"]
            )

            # Build analytics service only if we own the video
            analytics_service = None
            if video_owned and target_channel["analytics_access"]:
                try:
                    from googleapiclient.discovery import build

                    analytics_service = build(
                        "youtubeAnalytics",
                        "v2",
                        credentials=oauth_service._http.credentials,
                    )
                except Exception as e:
                    logger.error(f"Failed to build analytics service: {e}")

            # Determine access level
            if video_owned and target_channel["analytics_access"]:
                access_level = "oauth_full"  # Full access with analytics
            elif video_owned:
                access_level = "oauth_basic"  # Own video but no analytics
            else:
                access_level = "oauth_enhanced"  # OAuth for comments but not our video

            return {
                "public_service": public_service,
                "oauth_service": oauth_service,
                "analytics_service": analytics_service,
                "channel_id": target_channel["id"],
                "access_level": access_level,
                "oauth_info": oauth_info,
                "channel_info": target_channel,
                "video_owned": video_owned,
            }
        except Exception as e:
            logger.error(f"OAuth service creation failed: {e}")

    # Fallback to public only
    return {
        "public_service": public_service,
        "oauth_service": None,
        "analytics_service": None,
        "channel_id": None,
        "access_level": "public",
        "oauth_info": oauth_info,
        "video_owned": False,
    }


def display_access_level(service_info):
    """Display the current access level and capabilities."""
    access_level = service_info["access_level"]
    oauth_info = service_info["oauth_info"]
    video_owned = service_info.get("video_owned", False)

    if access_level == "oauth_full":
        st.success(
            "🔓 **Full OAuth Access**: Your video - complete analytics and insights available"
        )
        channel_info = service_info.get("channel_info", {})
        st.info(
            f"📊 Your Channel: **{channel_info.get('title', 'Unknown')}** ({channel_info.get('subscriber_count', 0):,} subscribers)"
        )
    elif access_level == "oauth_basic":
        st.warning(
            "🔐 **Basic OAuth Access**: Your video - limited analytics available"
        )
        channel_info = service_info.get("channel_info", {})
        st.info(f"📊 Your Channel: **{channel_info.get('title', 'Unknown')}**")
    elif access_level == "oauth_enhanced":
        st.info(
            "🔓 **Enhanced Access**: OAuth available for comments, public data for video"
        )
        channel_info = service_info.get("channel_info", {})
        st.info(
            f"📊 OAuth Channel: **{channel_info.get('title', 'Unknown')}** (not video owner)"
        )
    else:
        st.info("🔒 **Public Access**: Limited to public data only")
        if oauth_info["available"]:
            st.info(
                f"💡 **{len(oauth_info['channels'])} OAuth channels available** for enhanced comment access"
            )


def get_enhanced_analytics(service_info, video_id, days_back=30):
    """Get enhanced analytics data when OAuth is available and video is owned."""
    # Only get analytics if we own the video and have analytics access
    if not service_info["video_owned"] or not service_info["analytics_service"]:
        return None

    try:
        from youtube.analytics import get_comprehensive_video_analytics
        
        analytics_service = service_info["analytics_service"]
        channel_id = service_info["channel_id"]

        # Get comprehensive video analytics using the dedicated function
        comprehensive_data = get_comprehensive_video_analytics(
            analytics_service, video_id, channel_id, days_back=days_back
        )
        
        return comprehensive_data

    except Exception as e:
        logger.warning(f"Enhanced analytics failed: {e}")
        return None


def get_legacy_enhanced_analytics(service_info, video_id, days_back=30):
    """Fallback method for enhanced analytics (legacy implementation)."""
    # Only get analytics if we own the video and have analytics access
    if not service_info["video_owned"] or not service_info["analytics_service"]:
        return None

    try:
        from datetime import date, timedelta

        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)

        analytics_service = service_info["analytics_service"]
        channel_id = service_info["channel_id"]

        # Get basic video analytics
        video_analytics = (
            analytics_service.reports()
            .query(
                ids=f"channel=={channel_id}",
                startDate=start_date.strftime("%Y-%m-%d"),
                endDate=end_date.strftime("%Y-%m-%d"),
                metrics="views,estimatedMinutesWatched,averageViewDuration,subscribersGained,likes,comments",
                dimensions="video",
                filters=f"video=={video_id}",
                maxResults=1,
            )
            .execute()
        )

        # Get audience retention data
        retention_data = None
        try:
            retention_data = (
                analytics_service.reports()
                .query(
                    ids=f"channel=={channel_id}",
                    startDate=start_date.strftime("%Y-%m-%d"),
                    endDate=end_date.strftime("%Y-%m-%d"),
                    metrics="audienceWatchRatio",
                    dimensions="elapsedVideoTimeRatio",
                    filters=f"video=={video_id}",
                    maxResults=100,
                )
                .execute()
            )
        except Exception as e:
            logger.warning(f"Retention data failed: {e}")

        # Get traffic sources
        traffic_sources = None
        try:
            traffic_sources = (
                analytics_service.reports()
                .query(
                    ids=f"channel=={channel_id}",
                    startDate=start_date.strftime("%Y-%m-%d"),
                    endDate=end_date.strftime("%Y-%m-%d"),
                    metrics="views",
                    dimensions="insightTrafficSourceType",
                    filters=f"video=={video_id}",
                    maxResults=10,
                )
                .execute()
            )
        except Exception as e:
            logger.warning(f"Traffic sources failed: {e}")

        # Geography breakdown (top countries)
        geography_data = None
        try:
            geography_data = (
                analytics_service.reports()
                .query(
                    ids=f"channel=={channel_id}",
                    startDate=start_date.strftime("%Y-%m-%d"),
                    endDate=end_date.strftime("%Y-%m-%d"),
                    metrics="views",
                    dimensions="country",
                    filters=f"video=={video_id}",
                    maxResults=250,
                )
                .execute()
            )
        except Exception as e:
            logger.warning(f"Geography breakdown failed: {e}")

        # Demographics: age & gender
        demographics_data = None
        try:
            demographics_data = (
                analytics_service.reports()
                .query(
                    ids=f"channel=={channel_id}",
                    startDate=start_date.strftime("%Y-%m-%d"),
                    endDate=end_date.strftime("%Y-%m-%d"),
                    metrics="viewerPercentage",
                    dimensions="ageGroup,gender",
                    filters=f"video=={video_id}",
                    maxResults=200,
                )
                .execute()
            )
        except Exception as e:
            logger.warning(f"Demographics data failed: {e}")

        return {
            "video_analytics": video_analytics,
            "retention_data": retention_data,
            "traffic_sources": traffic_sources,
            "geography_data": geography_data,
            "demographics_data": demographics_data,
            "period": f"{start_date} to {end_date}",
        }

    except Exception as e:
        logger.error(f"Enhanced analytics failed: {e}")
        return None


def display_enhanced_analytics(analytics_data, video_title, service_info=None, video_id=None, channel_id=None, days_back=28):
    """Display comprehensive enhanced analytics data in a beautiful format."""
    if not analytics_data:
        return

    st.subheader("📊 Comprehensive Video Analytics (OAuth)")
    
    # Create tabs for different analytics sections
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📈 Overview", "📊 Audience Retention", "🌍 Demographics", 
        "🗺️ Geography", "💰 Monetization", "📅 Time Series", "🚀 Engagement",
        "👥 Views by Subscriber Status"
    ])
    
    with tab1:
        display_overview_metrics(analytics_data)
    
    with tab2:
        display_audience_retention(analytics_data)
    
    with tab3:
        display_demographics_analytics(analytics_data)
    
    with tab4:
        display_geography_analytics(analytics_data)
    
    with tab5:
        display_monetization_analytics(analytics_data)
    
    with tab6:
        display_time_series_analytics(analytics_data)
    
    with tab7:
        display_engagement_analytics(analytics_data)
    
    with tab8:
        # Use passed-in service_info, video_id, channel_id, days_back
        try:
            oauth_service = service_info.get("oauth_service") if service_info else None
            if oauth_service and video_id and channel_id:
                from youtube.analytics import video_subscriber_status_breakdown
                sub_status_data = video_subscriber_status_breakdown(
                    oauth_service, video_id, channel_id, days_back=days_back
                )
                rows = sub_status_data.get("rows", [])
                if rows:
                    st.markdown("### 👥 Views by Subscriber Status")
                    table = []
                    for row in rows:
                        status = row[0].capitalize() if row[0] else "Unknown"
                        views = int(row[1]) if len(row) > 1 else 0
                        watch_time = int(row[2]) if len(row) > 2 else 0
                        avg_view_duration = int(row[3]) if len(row) > 3 else 0
                        mins, secs = divmod(avg_view_duration, 60)
                        table.append({
                            "Status": status,
                            "Views": f"{views:,}",
                            "Watch Time (min)": f"{watch_time:,}",
                            "Avg View Duration": f"{mins}m {secs}s"
                        })
                    st.table(table)
                else:
                    st.info("ℹ️ Subscriber status breakdown (views, watch time, avg view duration) is not available for this video or period. This requires sufficient data and OAuth access.")
            else:
                st.info("ℹ️ Subscriber status breakdown is only available for your own videos with OAuth access.")
        except Exception as e:
            st.info(f"ℹ️ Could not fetch subscriber status breakdown: {e}")


def display_overview_metrics(analytics_data):
    """Display overview metrics and key performance indicators."""
    
    # Summary metrics
    summary_data = analytics_data.get("summary_metrics", {})
    engagement_data = analytics_data.get("engagement_metrics", {})
    impressions_data = analytics_data.get("impressions", {})
    
    if summary_data.get("rows"):
        row = summary_data["rows"][0]
        
        st.markdown("### 🎯 Key Performance Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            views = int(row[1]) if len(row) > 1 else 0
            st.metric("👀 Views", f"{views:,}")
        with col2:
            watch_time = int(row[2]) if len(row) > 2 else 0
            st.metric("⏱️ Watch Time", f"{watch_time:,} min")
        with col3:
            avg_duration = int(row[3]) if len(row) > 3 else 0
            st.metric("🎯 Avg Duration", f"{avg_duration:,} sec")
        with col4:
            likes = int(row[4]) if len(row) > 4 else 0
            st.metric("👍 Likes", f"{likes:,}")
        with col5:
            comments = int(row[5]) if len(row) > 5 else 0
            st.metric("💬 Comments", f"{comments:,}")
    
    # Enhanced engagement metrics
    if engagement_data.get("rows"):
        eng_row = engagement_data["rows"][0]
        st.markdown("### 📊 Engagement Breakdown")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            shares = int(eng_row[4]) if len(eng_row) > 4 else 0
            st.metric("🔄 Shares", f"{shares:,}")
        with col2:
            subs_gained = int(eng_row[5]) if len(eng_row) > 5 else 0
            st.metric("🔔 Subs Gained", f"+{subs_gained:,}")
        with col3:
            playlist_adds = int(eng_row[7]) if len(eng_row) > 7 else 0
            st.metric("📋 Playlist Adds", f"{playlist_adds:,}")
        with col4:
            saves = int(eng_row[8]) if len(eng_row) > 8 else 0
            st.metric("💾 Saves", f"{saves:,}")
        with col5:
            # Calculate engagement rate
            total_views = int(eng_row[0]) if len(eng_row) > 0 else 0
            total_likes = int(eng_row[1]) if len(eng_row) > 1 else 0
            total_comments = int(eng_row[3]) if len(eng_row) > 3 else 0
            
            if total_views > 0:
                engagement_rate = ((total_likes + total_comments + shares) / total_views) * 100
                st.metric("📈 Engagement Rate", f"{engagement_rate:.2f}%")
            else:
                st.metric("📈 Engagement Rate", "0.00%")
        with col6:
            # Like to view ratio
            if total_views > 0:
                like_ratio = (total_likes / total_views) * 100
                st.metric("👍 Like Rate", f"{like_ratio:.2f}%")
            else:
                st.metric("👍 Like Rate", "0.00%")
    
    # Impressions data
    if impressions_data.get("rows") and not impressions_data.get("error"):
        imp_row = impressions_data["rows"][0]
        st.markdown("### 👁️ Impressions & Discovery")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            impressions = int(imp_row[0]) if len(imp_row) > 0 else 0
            st.metric("👁️ Impressions", f"{impressions:,}")
        with col2:
            ctr = float(imp_row[1]) if len(imp_row) > 1 else 0
            st.metric("🎯 Click-through Rate", f"{ctr:.2f}%")
        with col3:
            unique_viewers = int(imp_row[2]) if len(imp_row) > 2 else 0
            st.metric("👤 Unique Viewers", f"{unique_viewers:,}")


def display_audience_retention(analytics_data):
    """Display audience retention analytics with interactive charts."""
    
    retention_data = analytics_data.get("audience_retention", [])
    
    if retention_data and not isinstance(retention_data, dict):
        st.markdown("### 📊 Audience Retention Curve")
        
        # Convert retention data to chart format
        import pandas as pd
        
        if retention_data:
            time_points = []
            retention_rates = []
            
            for row in retention_data:
                if len(row) >= 2:
                    time_points.append(float(row[0]) * 100)  # Convert to percentage
                    retention_rates.append(float(row[1]) * 100)  # Convert to percentage
            
            if time_points and retention_rates:
                df = pd.DataFrame({
                    'Video Progress (%)': time_points,
                    'Audience Retention (%)': retention_rates
                })
                
                st.line_chart(df.set_index('Video Progress (%)'))
                
                # Key insights
                st.markdown("### 🔍 Retention Insights")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_retention = sum(retention_rates) / len(retention_rates)
                    st.metric("📊 Average Retention", f"{avg_retention:.1f}%")
                
                with col2:
                    max_retention = max(retention_rates)
                    max_point = time_points[retention_rates.index(max_retention)]
                    st.metric("🎯 Peak Retention", f"{max_retention:.1f}% at {max_point:.0f}%")
                
                with col3:
                    min_retention = min(retention_rates)
                    min_point = time_points[retention_rates.index(min_retention)]
                    st.metric("📉 Lowest Retention", f"{min_retention:.1f}% at {min_point:.0f}%")
                
                # Identify key moments
                st.markdown("### 🎬 Key Moments Analysis")
                
                # Find spikes (increases)
                spikes = []
                dips = []
                
                for i in range(1, len(retention_rates)):
                    change = retention_rates[i] - retention_rates[i-1]
                    if change > 5:  # Significant spike
                        spikes.append((time_points[i], retention_rates[i], change))
                    elif change < -5:  # Significant dip
                        dips.append((time_points[i], retention_rates[i], abs(change)))
                
                col1, col2 = st.columns(2)
                with col1:
                    if spikes:
                        st.markdown("**📈 Retention Spikes (Rewatched/Shared moments):**")
                        for time_point, retention, change in spikes[:3]:
                            st.write(f"• {time_point:.0f}% mark: +{change:.1f}% retention boost")
                    else:
                        st.info("No significant retention spikes detected")
                
                with col2:
                    if dips:
                        st.markdown("**📉 Retention Dips (Drop-off points):**")
                        for time_point, retention, change in dips[:3]:
                            st.write(f"• {time_point:.0f}% mark: -{change:.1f}% drop")
                    else:
                        st.info("No significant retention dips detected")
        else:
            st.info("No retention data available for the selected period")
    else:
        st.info("Audience retention data not available - requires video ownership and sufficient views (100+ views)")


def display_demographics_analytics(analytics_data):
    """Display demographic breakdown of the audience."""
    
    demographics_data = analytics_data.get("demographics", [])
    
    if demographics_data and not isinstance(demographics_data, dict):
        st.markdown("### 👥 Audience Demographics")
        
        import pandas as pd
        
        # Process demographics data
        age_gender_data = []
        
        for row in demographics_data:
            if len(row) >= 3:
                age_group = row[0]
                gender = row[1]
                percentage = float(row[2])
                age_gender_data.append({
                    'Age Group': age_group,
                    'Gender': gender,
                    'Percentage': percentage
                })
        
        if age_gender_data:
            df = pd.DataFrame(age_gender_data)
            
            # Age distribution
            age_totals = df.groupby('Age Group')['Percentage'].sum().reset_index()
            age_totals = age_totals.sort_values('Percentage', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 Age Distribution**")
                st.bar_chart(age_totals.set_index('Age Group'))
            
            with col2:
                st.markdown("**⚧ Gender Distribution**")
                gender_totals = df.groupby('Gender')['Percentage'].sum().reset_index()
                st.bar_chart(gender_totals.set_index('Gender'))
            
            # Top demographics
            st.markdown("### 🎯 Top Demographics")
            df_sorted = df.sort_values('Percentage', ascending=False)
            
            for i, row in df_sorted.head(5).iterrows():
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write(f"**{row['Age Group']} - {row['Gender']}**")
                with col2:
                    st.write(f"{row['Percentage']:.1f}%")
                with col3:
                    # Create a simple progress bar
                    progress = row['Percentage'] / df['Percentage'].max()
                    st.progress(progress)
        else:
            st.info("No demographic data available")
    else:
        st.info("Demographics data not available - requires sufficient views and channel permissions")


def display_geography_analytics(analytics_data):
    """Display geographic distribution of viewers."""
    
    geography_data = analytics_data.get("geography", [])
    
    if geography_data and not isinstance(geography_data, dict):
        st.markdown("### 🗺️ Geographic Distribution")
        
        import pandas as pd
        
        # Process geography data
        country_data = []
        
        for row in geography_data:
            if len(row) >= 2:
                country_code = row[0]
                views = int(row[1])
                
                # Map country codes to names (basic mapping)
                country_names = {
                    'US': 'United States', 'GB': 'United Kingdom', 'CA': 'Canada',
                    'AU': 'Australia', 'DE': 'Germany', 'FR': 'France', 'IN': 'India',
                    'JP': 'Japan', 'BR': 'Brazil', 'MX': 'Mexico', 'IT': 'Italy',
                    'ES': 'Spain', 'RU': 'Russia', 'KR': 'South Korea', 'NL': 'Netherlands'
                }
                
                country_name = country_names.get(country_code, country_code)
                country_data.append({
                    'Country': country_name,
                    'Country Code': country_code,
                    'Views': views
                })
        
        if country_data:
            df = pd.DataFrame(country_data)
            df = df.sort_values('Views', ascending=False)
            
            # Calculate percentages
            total_views = df['Views'].sum()
            df['Percentage'] = (df['Views'] / total_views * 100).round(1)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🌍 Top 10 Countries by Views**")
                top_countries = df.head(10)
                st.bar_chart(top_countries.set_index('Country')['Views'])
            
            with col2:
                st.markdown("**📊 Geographic Breakdown**")
                for i, row in df.head(10).iterrows():
                    col_country, col_views, col_percent = st.columns([2, 1, 1])
                    with col_country:
                        st.write(f"🌍 **{row['Country']}**")
                    with col_views:
                        st.write(f"{row['Views']:,}")
                    with col_percent:
                        st.write(f"{row['Percentage']}%")
            
            # Geographic insights
            st.markdown("### 🌐 Geographic Insights")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                top_country = df.iloc[0]
                st.metric("🥇 Top Country", top_country['Country'])
                st.caption(f"{top_country['Views']:,} views ({top_country['Percentage']}%)")
            
            with col2:
                countries_count = len(df)
                st.metric("🌍 Countries Reached", f"{countries_count}")
            
            with col3:
                top_5_percentage = df.head(5)['Percentage'].sum()
                st.metric("🎯 Top 5 Countries", f"{top_5_percentage:.1f}%")
        else:
            st.info("No geographic data available")
    else:
        st.info("Geographic data not available - requires sufficient views")


def display_monetization_analytics(analytics_data):
    """Display monetization and revenue analytics."""
    
    monetization_data = analytics_data.get("monetization", {})
    
    if monetization_data.get("rows") and not monetization_data.get("error"):
        st.markdown("### 💰 Monetization Analytics")
        
        row = monetization_data["rows"][0]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            estimated_revenue = float(row[0]) if len(row) > 0 else 0
            st.metric("💵 Estimated Revenue", f"${estimated_revenue:.2f}")
        
        with col2:
            ad_revenue = float(row[1]) if len(row) > 1 else 0
            st.metric("📺 Ad Revenue", f"${ad_revenue:.2f}")
        
        with col3:
            cpm = float(row[4]) if len(row) > 4 else 0
            st.metric("📊 CPM", f"${cpm:.2f}")
        
        with col4:
            playback_cpm = float(row[5]) if len(row) > 5 else 0
            st.metric("▶️ Playback CPM", f"${playback_cpm:.2f}")
        
        # Revenue breakdown
        if estimated_revenue > 0:
            st.markdown("### 💹 Revenue Breakdown")
            
            red_revenue = float(row[2]) if len(row) > 2 else 0
            gross_revenue = float(row[3]) if len(row) > 3 else 0
            
            revenue_data = {
                'Ad Revenue': ad_revenue,
                'YouTube Premium Revenue': red_revenue,
                'Other Revenue': max(0, gross_revenue - ad_revenue - red_revenue)
            }
            
            import pandas as pd
            df = pd.DataFrame(list(revenue_data.items()), columns=['Revenue Type', 'Amount'])
            df = df[df['Amount'] > 0]  # Only show non-zero revenues
            
            if not df.empty:
                st.bar_chart(df.set_index('Revenue Type'))
            
            # Performance indicators
            st.markdown("### 📈 Performance Indicators")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Revenue per 1000 views
                summary_data = analytics_data.get("summary_metrics", {})
                if summary_data.get("rows"):
                    views = int(summary_data["rows"][0][1])
                    rpm = (estimated_revenue / views * 1000) if views > 0 else 0
                    st.metric("💰 RPM (Revenue per 1000 views)", f"${rpm:.2f}")
            
            with col2:
                impressions_data = analytics_data.get("impressions", {})
                if impressions_data.get("rows") and not impressions_data.get("error"):
                    impressions = int(impressions_data["rows"][0][0])
                    impression_cpm = float(row[6]) if len(row) > 6 else 0
                    st.metric("👁️ Impression CPM", f"${impression_cpm:.2f}")
            
            with col3:
                # Ad revenue percentage
                ad_percentage = (ad_revenue / estimated_revenue * 100) if estimated_revenue > 0 else 0
                st.metric("📺 Ad Revenue %", f"{ad_percentage:.1f}%")
    else:
        st.info("💰 Monetization data not available - requires monetized channel and sufficient revenue")


def display_time_series_analytics(analytics_data):
    """Display time series data with interactive charts."""
    
    time_series_data = analytics_data.get("time_series", {})
    
    if time_series_data.get("rows"):
        st.markdown("### 📅 Performance Over Time")
        
        import pandas as pd
        
        # Process time series data
        dates = []
        views = []
        likes = []
        subscribers = []
        watch_time = []
        shares = []
        comments = []
        
        for row in time_series_data["rows"]:
            if len(row) >= 6:
                dates.append(row[0])  # Date
                views.append(int(row[1]))  # Views
                likes.append(int(row[2]))  # Likes
                subscribers.append(int(row[3]))  # Subscribers gained
                watch_time.append(int(row[4]))  # Watch time
                shares.append(int(row[5]))  # Shares
                comments.append(int(row[6]) if len(row) > 6 else 0)  # Comments
        
        if dates:
            df = pd.DataFrame({
                'Date': pd.to_datetime(dates),
                'Views': views,
                'Likes': likes,
                'Subscribers Gained': subscribers,
                'Watch Time (min)': watch_time,
                'Shares': shares,
                'Comments': comments
            })
            df = df.set_index('Date')
            
            # Display charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📈 Views Over Time**")
                st.line_chart(df[['Views']])
                
                st.markdown("**👍 Likes Over Time**")
                st.line_chart(df[['Likes']])
            
            with col2:
                st.markdown("**🔔 Subscribers Gained Over Time**")
                st.line_chart(df[['Subscribers Gained']])
                
                st.markdown("**⏱️ Watch Time Over Time**")
                st.line_chart(df[['Watch Time (min)']])
            
            # Combined engagement chart
            st.markdown("**📊 Engagement Metrics Over Time**")
            engagement_df = df[['Likes', 'Shares', 'Comments']]
            st.line_chart(engagement_df)
            
            # Summary statistics
            st.markdown("### 📊 Time Series Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_views = sum(views)
                st.metric("📈 Total Views", f"{total_views:,}")
                
                peak_views_day = df.loc[df['Views'].idxmax()].name.strftime('%Y-%m-%d')
                st.caption(f"Peak: {peak_views_day}")
            
            with col2:
                total_likes = sum(likes)
                st.metric("👍 Total Likes", f"{total_likes:,}")
                
                avg_likes = total_likes / len(likes) if likes else 0
                st.caption(f"Avg/day: {avg_likes:.1f}")
            
            with col3:
                total_subs = sum(subscribers)
                st.metric("🔔 Subscribers Gained", f"+{total_subs:,}")
                
                best_sub_day = max(subscribers) if subscribers else 0
                st.caption(f"Best day: +{best_sub_day}")
            
            with col4:
                total_watch_time = sum(watch_time)
                st.metric("⏱️ Total Watch Time", f"{total_watch_time:,} min")
                
                hours = total_watch_time / 60
                st.caption(f"{hours:.1f} hours")
        else:
            st.info("No time series data available")
    else:
        st.info("Time series data not available - requires video ownership")


def display_engagement_analytics(analytics_data):
    """Display detailed engagement analytics and metrics."""
    
    st.markdown("### 🚀 Engagement Analytics")
    
    # Traffic sources
    traffic_data = analytics_data.get("traffic_sources", [])
    engagement_data = analytics_data.get("engagement_metrics", {})
    
    if traffic_data and not isinstance(traffic_data, dict):
        st.markdown("### 🚦 Traffic Sources")
        
        import pandas as pd
        
        traffic_list = []
        for row in traffic_data:
            if len(row) >= 2:
                source_type = row[0]
                views = int(row[1])
                
                # Friendly source names
                source_names = {
                    'PLAYLIST': '📋 Playlists',
                    'SEARCH': '🔍 YouTube Search',
                    'SUGGESTED_VIDEO': '💡 Suggested Videos',
                    'BROWSE': '🏠 Browse Features',
                    'CHANNEL': '📺 Channel Page',
                    'EXTERNAL': '🌐 External Sources',
                    'DIRECT': '🔗 Direct Links',
                    'NOTIFICATION': '🔔 Notifications'
                }
                
                friendly_name = source_names.get(source_type, source_type)
                traffic_list.append({'Source': friendly_name, 'Views': views})
        
        if traffic_list:
            df = pd.DataFrame(traffic_list)
            df = df.sort_values('Views', ascending=False)
            
            total_traffic_views = df['Views'].sum()
            df['Percentage'] = (df['Views'] / total_traffic_views * 100).round(1)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.bar_chart(df.set_index('Source')['Views'])
            
            with col2:
                st.markdown("**Traffic Source Breakdown:**")
                for i, row in df.iterrows():
                    st.write(f"**{row['Source']}**: {row['Views']:,} views ({row['Percentage']}%)")
    
    # Engagement summary
    if engagement_data.get("rows"):
        row = engagement_data["rows"][0]
        
        st.markdown("### 💫 Engagement Summary")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_views = int(row[0]) if len(row) > 0 else 0
            st.metric("👀 Total Views", f"{total_views:,}")
        
        with col2:
            total_likes = int(row[1]) if len(row) > 1 else 0
            dislikes = int(row[2]) if len(row) > 2 else 0
            like_ratio = (total_likes / (total_likes + dislikes) * 100) if (total_likes + dislikes) > 0 else 0
            st.metric("👍 Like Ratio", f"{like_ratio:.1f}%")
        
        with col3:
            comments = int(row[3]) if len(row) > 3 else 0
            comment_rate = (comments / total_views * 100) if total_views > 0 else 0
            st.metric("💬 Comment Rate", f"{comment_rate:.2f}%")
        
        with col4:
            shares = int(row[4]) if len(row) > 4 else 0
            share_rate = (shares / total_views * 100) if total_views > 0 else 0
            st.metric("🔄 Share Rate", f"{share_rate:.2f}%")
        
        with col5:
            subs_gained = int(row[5]) if len(row) > 5 else 0
            sub_rate = (subs_gained / total_views * 100) if total_views > 0 else 0
            st.metric("🔔 Sub Rate", f"{sub_rate:.2f}%")
        
        # Additional engagement metrics
        st.markdown("### 📊 Additional Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            playlist_adds = int(row[7]) if len(row) > 7 else 0
            st.metric("📋 Playlist Additions", f"{playlist_adds:,}")
        
        with col2:
            saves = int(row[8]) if len(row) > 8 else 0
            st.metric("💾 Saves", f"{saves:,}")
        
        with col3:
            # Calculate overall engagement score
            if total_views > 0:
                engagement_score = ((total_likes + comments + shares + playlist_adds + saves) / total_views * 100)
                st.metric("🌟 Engagement Score", f"{engagement_score:.2f}%")
            else:
                st.metric("🌟 Engagement Score", "0.00%")


def display_legacy_enhanced_analytics(analytics_data, video_title):
    """Legacy display function for backwards compatibility."""
    if not analytics_data:
        return

    st.subheader("📊 Enhanced Analytics (OAuth)")

    # Video performance metrics
    video_data = analytics_data.get("video_analytics", {})
    if video_data.get("rows"):
        row = video_data["rows"][0]

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.metric("📈 Total Views", f"{int(row[1]):,}")
        with col2:
            st.metric("⏱️ Watch Time", f"{int(row[2]):,} min")
        with col3:
            st.metric("🎯 Avg Duration", f"{int(row[3]):,} sec")
        with col4:
            st.metric("👥 Subscribers", f"+{int(row[4]):,}")
        with col5:
            st.metric("👍 Likes", f"{int(row[5]):,}")
        with col6:
            st.metric("💬 Comments", f"{int(row[6]):,}")

    # Audience retention
    retention_data = analytics_data.get("retention_data", {})
    if retention_data and retention_data.get("rows"):
        st.subheader("🎯 Audience Retention")

        # Create retention chart
        ratios = []
        retention_rates = []

        for row in retention_data["rows"]:
            ratios.append(float(row[0]) * 100)  # Convert to percentage
            retention_rates.append(float(row[1]) * 100)

        if ratios and retention_rates:
            try:
                import matplotlib.pyplot as plt
                import io

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(ratios, retention_rates, "b-", linewidth=2)
                ax.set_xlabel("Video Progress (%)")
                ax.set_ylabel("Audience Retention (%)")
                ax.set_title(f"Audience Retention - {video_title}")
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, 100)

                # Save plot to bytes
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")
                img_buffer.seek(0)

                st.image(
                    img_buffer,
                    caption="Audience retention shows how well your content keeps viewers engaged",
                )
                plt.close()
            except ImportError:
                st.error("Matplotlib not available for retention chart")
                # Show data as table instead
                retention_table = []
                for ratio, rate in zip(
                    ratios[:10], retention_rates[:10]
                ):  # Show first 10 points
                    retention_table.append(
                        {"Progress": f"{ratio:.1f}%", "Retention": f"{rate:.1f}%"}
                    )
                st.table(retention_table)
            except Exception as e:
                st.error(f"Chart generation failed: {e}")

    # Traffic sources
    traffic_data = analytics_data["traffic_sources"]
    if traffic_data and traffic_data.get("rows"):
        st.subheader("🚀 Traffic Sources")

        traffic_sources = []
        for row in traffic_data["rows"]:
            source_type = row[0]
            views = int(row[1])

            # Map internal names to friendly names
            source_map = {
                "YT_SEARCH": "YouTube Search",
                "SUGGESTED_VIDEO": "Suggested Videos",
                "EXTERNAL_URL": "External Links",
                "BROWSE_FEATURES": "Browse Features",
                "NOTIFICATION": "Notifications",
                "DIRECT_OR_UNKNOWN": "Direct/Unknown",
                "PLAYLIST": "Playlists",
                "CHANNEL": "Channel Pages",
                "SUBSCRIBER": "Subscribers",
            }

            friendly_name = source_map.get(source_type, source_type)
            traffic_sources.append({"Source": friendly_name, "Views": f"{views:,}"})

        # Display as table
        if traffic_sources:
            st.table(traffic_sources)
        else:
            st.info("No traffic source data available")

    # Geography breakdown
    geo = analytics_data.get("geography_data")
    if geo and geo.get("rows"):
        st.subheader("🌍 Top Countries")
        rows = sorted(geo["rows"], key=lambda r: int(r[1]), reverse=True)[:10]
        table = [{"Country": r[0], "Views": f"{int(r[1]):,}"} for r in rows]
        st.table(table)

    # Demographics breakdown
    demo = analytics_data.get("demographics_data")
    if demo and demo.get("rows"):
        st.subheader("👥 Audience Demographics")

        # Build age->gender map
        age_map = {}
        for age, gender, perc in demo["rows"]:
            if age not in age_map:
                age_map[age] = {"Male %": 0.0, "Female %": 0.0, "Other %": 0.0}
            key = "Male %" if gender.lower() == "male" else "Female %" if gender.lower() == "female" else "Other %"
            age_map[age][key] = round(float(perc), 2)

        demo_table = []
        for age_group in sorted(age_map.keys()):
            demo_table.append({
                "Age Group": age_group,
                "Male %": f"{age_map[age_group].get('male', 0):.1f}%",
                "Female %": f"{age_map[age_group].get('female', 0):.1f}%"
            })
        
        st.table(demo_table)

    st.caption(f"📅 Data period: {analytics_data['period']}")

    # Show what data is available
    data_available = []
    if video_data.get("rows"):
        data_available.append("✅ Video Performance")
    if retention_data and retention_data.get("rows"):
        data_available.append("✅ Audience Retention")
    if traffic_data and traffic_data.get("rows"):
        data_available.append("✅ Traffic Sources")
    if geo and geo.get("rows"):
        data_available.append("✅ Geography")
    if demo and demo.get("rows"):
        data_available.append("✅ Demographics")

    if data_available:
        st.caption(f"**Available data:** {', '.join(data_available)}")
    else:
        st.info("Limited analytics data available for this video")


def get_enhanced_comments(service_info, video_id):
    """Get enhanced comment data with OAuth capabilities."""
    # Try OAuth first if available, then fallback to public
    oauth_service = service_info.get("oauth_service")
    public_service = service_info["public_service"]

    # First try with OAuth if available
    if oauth_service:
        try:
            # Try the full OAuth request with replies first
            comments = []
            request = oauth_service.commentThreads().list(
                part="snippet,replies",
                videoId=video_id,
                maxResults=100,
                order="relevance",
            )

            while request and len(comments) < 500:  # Limit to prevent timeout
                response = request.execute()

                for item in response["items"]:
                    top_comment = item["snippet"]["topLevelComment"]["snippet"]
                    comment_data = {
                        "author": top_comment["authorDisplayName"],
                        "authorChannelId": top_comment.get("authorChannelId", {}).get(
                            "value"
                        ),
                        "textDisplay": top_comment["textDisplay"],
                        "likeCount": top_comment["likeCount"],
                        "publishedAt": top_comment["publishedAt"],
                        "updatedAt": top_comment["updatedAt"],
                        "totalReplyCount": item["snippet"]["totalReplyCount"],
                    }

                    # Add reply data if available
                    if "replies" in item:
                        comment_data["replies"] = []
                        for reply in item["replies"]["comments"]:
                            reply_data = reply["snippet"]
                            comment_data["replies"].append(
                                {
                                    "author": reply_data["authorDisplayName"],
                                    "textDisplay": reply_data["textDisplay"],
                                    "likeCount": reply_data["likeCount"],
                                    "publishedAt": reply_data["publishedAt"],
                                }
                            )

                    comments.append(comment_data)

                request = oauth_service.commentThreads().list_next(request, response)

            logger.info(
                f"OAuth: Successfully fetched {len(comments)} comments with full data"
            )
            return comments

        except Exception as e:
            logger.warning(f"OAuth comment fetching with replies failed: {e}")

            # Try OAuth without replies as fallback
            try:
                comments = []
                request = oauth_service.commentThreads().list(
                    part="snippet",  # Only snippet, no replies
                    videoId=video_id,
                    maxResults=100,
                    order="time",  # Use time order as fallback
                )

                response = request.execute()
                for item in response.get("items", []):
                    top_comment = item["snippet"]["topLevelComment"]["snippet"]
                    comment_data = {
                        "author": top_comment["authorDisplayName"],
                        "authorChannelId": top_comment.get("authorChannelId", {}).get(
                            "value"
                        ),
                        "textDisplay": top_comment["textDisplay"],
                        "likeCount": top_comment["likeCount"],
                        "publishedAt": top_comment["publishedAt"],
                        "updatedAt": top_comment["updatedAt"],
                        "totalReplyCount": item["snippet"]["totalReplyCount"],
                    }
                    comments.append(comment_data)

                logger.info(
                    f"OAuth: Successfully fetched {len(comments)} comments without replies"
                )
                return comments

            except Exception as e2:
                logger.warning(
                    f"OAuth comment fetching (basic) failed: {e2}, falling back to public API"
                )

    # Fallback to public API
    try:
        from analysis.comments import fetch_comments

        public_comments = fetch_comments(public_service, video_id)

        # Standardize field names - convert 'text' to 'textDisplay' for consistency
        standardized_comments = []
        for comment in public_comments:
            standardized_comment = {
                "author": comment.get("author", "Unknown"),
                "textDisplay": comment.get(
                    "text", ""
                ),  # Convert 'text' to 'textDisplay'
                "likeCount": comment.get("likeCount", 0),
                "publishedAt": comment.get("publishedAt", ""),
                "totalReplyCount": 0,  # Public API doesn't include reply count
            }
            standardized_comments.append(standardized_comment)

        logger.info(
            f"Public API: Successfully fetched {len(standardized_comments)} comments"
        )
        return standardized_comments

    except Exception as e:
        logger.error(f"Public comment fetching failed: {e}")
        return []


# ---------------------------------------------------------------------------
# UI section: only Creator Onboarding remains
# ---------------------------------------------------------------------------


def onboarding_section():
    """Modern creator onboarding interface with comprehensive management features."""

    # Page header with stats
    col_header, col_stats = st.columns([3, 1])
    with col_header:
        st.title("🎯 Creator Management Hub")
        st.markdown(
            "Manage YouTube creator OAuth credentials and monitor authentication status"
        )

    with col_stats:
        token_files = _list_token_files()
        st.metric("🔑 Active Creators", len(token_files))

    # Tab navigation for better organization
    tab_creators, tab_onboard = st.tabs(["👥 Manage Creators", "➕ Add New Creator"])

    # =================== CREATORS MANAGEMENT TAB ===================
    with tab_creators:
        if not token_files:
            st.info(
                "🚀 **Get started** by adding your first creator in the 'Add New Creator' tab!"
            )
        else:
            st.subheader("Active Creator Accounts")

            # Batch operations
            col_batch1, col_batch2, col_batch3 = st.columns([1, 1, 2])
            with col_batch1:
                if st.button("🔄 Refresh All", help="Refresh all expired tokens"):
                    refreshed_count = 0
                    for tf in token_files:
                        details = get_creator_details(tf)
                        if not details["is_valid"]:
                            if refresh_creator_token(details["channel_id"]):
                                refreshed_count += 1

                    if refreshed_count > 0:
                        st.success(f"Refreshed {refreshed_count} creator token(s)")
                        st.rerun()
                else:
                    st.info("No tokens needed refreshing")

            with col_batch2:
                if st.button("📊 Export List", help="Export creator list to JSON"):
                    creator_list = []
                    for tf in token_files:
                        details = get_creator_details(tf)
                        creator_list.append(
                            {
                                "channel_id": details["channel_id"],
                                "title": details["title"],
                                "is_valid": details["is_valid"],
                                "last_checked": details["last_checked"],
                            }
                        )

                    import json

                    json_str = json.dumps(creator_list, indent=2)
                    st.download_button(
                        "💾 Download creators.json",
                        json_str,
                        "creators.json",
                        "application/json",
                    )

            st.divider()

            # Creator cards with enhanced UI
            for tf in token_files:
                details = get_creator_details(tf)

                # Status indicator styling
                status_color = "🟢" if details["is_valid"] else "🔴"
                status_text = "Active" if details["is_valid"] else "Invalid/Expired"

                with st.container():
                    # Creator card header
                    col_avatar, col_info, col_stats, col_actions = st.columns(
                        [1, 3, 2, 2]
                    )

                    with col_avatar:
                        if details.get("thumbnail_url"):
                            try:
                                st.image(details["thumbnail_url"], width=60)
                            except:
                                st.markdown("👤")
                        else:
                            st.markdown("👤")

                    with col_info:
                        st.markdown(f"**{details['title']}**")
                        st.caption(f"{status_color} {status_text}")
                        st.caption(f"ID: `{details['channel_id']}`")

                    with col_stats:
                        if details["is_valid"]:

                            def format_number(num):
                                if num >= 1_000_000:
                                    return f"{num/1_000_000:.1f}M"
                                elif num >= 1_000:
                                    return f"{num/1_000:.1f}K"
                                return str(num)

                            st.metric(
                                "👥 Subscribers",
                                format_number(details["subscriber_count"]),
                            )
                            st.caption(
                                f"📹 {format_number(details['video_count'])} videos"
                            )
                        else:
                            st.markdown("⚠️ **Token Invalid**")
                            if "error" in details:
                                st.caption(f"Error: {details['error']}")

                    with col_actions:
                        action_col1, action_col2 = st.columns(2)

                        with action_col1:
                            if not details["is_valid"]:
                                if st.button(
                                    "🔄",
                                    key=f"refresh_{details['channel_id']}",
                                    help="Refresh token",
                                ):
                                    if refresh_creator_token(details["channel_id"]):
                                        st.success("Token refreshed!")
                                        st.rerun()
                                    else:
                                        st.error("Failed to refresh token")
                            else:
                                st.button("✅", disabled=True, help="Token is valid")

                        with action_col2:
                            if st.button(
                                "🗑️",
                                key=f"remove_{details['channel_id']}",
                                help="Remove creator",
                            ):
                                # Confirmation dialog using session state
                                confirm_key = f"confirm_remove_{details['channel_id']}"
                                if confirm_key not in st.session_state:
                                    st.session_state[confirm_key] = False

                                if not st.session_state[confirm_key]:
                                    st.session_state[confirm_key] = True
                                    st.warning(
                                        f"⚠️ Click again to confirm removal of **{details['title']}**"
                                    )
                                else:
                                    if _remove_creator(details["channel_id"]):
                                        st.success(f"✅ Removed {details['title']}")
                                        del st.session_state[confirm_key]
                                        st.rerun()
                                    else:
                                        st.error("Failed to remove creator")

                    st.divider()

    # =================== ONBOARDING TAB ===================
    with tab_onboard:
        st.subheader("🚀 Add New Creator")
        st.markdown("Connect a YouTube creator account using OAuth 2.0 authentication")

        # Step 1: Environment Variable Configuration
        st.markdown("### Step 1: OAuth Configuration")
        
        # Import the new environment validation function
        from auth.manager import validate_env_oauth_config, create_temp_client_secret_file

        # Check environment variables
        env_config = validate_env_oauth_config()
        
        col_config, col_validation = st.columns([2, 1])
        
        with col_config:
            st.markdown("**Required Environment Variables:**")
            
            # Check if environment variables are set
            oauth_client_id = os.getenv("OAUTH_CLIENT_ID")
            oauth_client_secret = os.getenv("OAUTH_CLIENT_SECRET") 
            oauth_project_id = os.getenv("OAUTH_PROJECT_ID")
            
            # Display current status
            id_status = "✅" if oauth_client_id else "❌"
            secret_status = "✅" if oauth_client_secret else "❌"
            project_status = "✅" if oauth_project_id else "⚠️"
            
            st.code(f"""
{id_status} OAUTH_CLIENT_ID={'Set' if oauth_client_id else 'Missing'}
{secret_status} OAUTH_CLIENT_SECRET={'Set' if oauth_client_secret else 'Missing'}
{project_status} OAUTH_PROJECT_ID={'Set' if oauth_project_id else 'Optional (will use default)'}
            """)
            
            if not env_config["valid"]:
                st.error("❌ Please set the required environment variables in your `.env` file:")
                st.code("""
# Add these to your .env file:
OAUTH_CLIENT_ID=your_client_id_here
OAUTH_CLIENT_SECRET=your_client_secret_here
OAUTH_PROJECT_ID=your_project_id_here  # Optional
                """)
                
                with st.expander("🔧 How to get these values"):
                    st.markdown("""
                    1. Go to [Google Cloud Console](https://console.cloud.google.com/)
                    2. Create or select a project
                    3. Enable the YouTube Data API v3 and YouTube Analytics API
                    4. Go to **Credentials** → **Create Credentials** → **OAuth 2.0 Client IDs**
                    5. Choose **Desktop application** as the application type
                    6. Download the JSON file and extract:
                       - `client_id` → `OAUTH_CLIENT_ID`
                       - `client_secret` → `OAUTH_CLIENT_SECRET`
                       - `project_id` → `OAUTH_PROJECT_ID`
                    """)

        # Validation and status display
        with col_validation:
            if env_config["valid"]:
                st.success("✅ OAuth Config Valid")
                st.caption(f"🏗️ Project: {env_config.get('project_id', 'Default')}")
                st.caption(f"👤 Client ID: ...{env_config.get('client_id', '')[-8:]}")
                st.caption(f"🔧 Type: {env_config.get('type', 'installed')}")
            else:
                st.error("❌ Configuration Invalid")
                st.caption(env_config.get("error", "Unknown error"))

        # Show detailed validation info in expander
        if env_config["valid"]:
            with st.expander("🔍 OAuth Configuration Details"):
                st.code(f"""
Project ID: {env_config.get('project_id', 'N/A')}
Client ID: {env_config.get('client_id', 'N/A')[:20]}...
Type: {env_config.get('type', 'N/A')}
Source: Environment Variables
                """)

        # Step 2: OAuth Flow
        st.markdown("### Step 2: OAuth Authentication")

        if not env_config["valid"]:
            st.warning("⚠️ Please configure OAuth environment variables to proceed")
            return

        # Enhanced OAuth button with preview
        col_oauth, col_info = st.columns([1, 2])

        with col_oauth:
            oauth_btn = st.button(
                "🔐 Start OAuth Flow",
                key="oauth_start",
                type="primary",
                help="Opens Google OAuth consent screen in a new tab",
            )

        # Persist onboarding state between reruns so the flow continues
        if oauth_btn and not st.session_state.get("oauth_flow_active"):
            st.session_state["oauth_flow_active"] = True

        with col_info:
            st.info(
                """
            **What happens next:**
            1. 🌐 Google OAuth page opens in new tab
            2. 📋 Select your YouTube channel
            3. ✅ Grant required permissions
            4. 🎉 Channel gets added to your dashboard
            """
            )

        if st.session_state.get("oauth_flow_active"):
            with st.spinner(
                "🔄 Initiating OAuth flow... Please complete authentication in the new browser tab."
            ):
                try:
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Simulate progress for better UX
                    import time

                    status_text.text("⏳ Creating OAuth configuration...")
                    progress_bar.progress(25)
                    time.sleep(0.5)

                    # Create temporary client secret file from environment variables
                    temp_client_secret = create_temp_client_secret_file()
                    if not temp_client_secret:
                        raise Exception("Failed to create OAuth configuration from environment variables")

                    status_text.text("🔐 Opening OAuth consent screen...")
                    progress_bar.progress(50)

                    # Actual OAuth call using temporary file
                    _token_path, cid, title = onboard_creator(temp_client_secret)

                    progress_bar.progress(75)
                    status_text.text("📊 Fetching channel information...")
                    time.sleep(0.5)

                    progress_bar.progress(100)
                    status_text.text("✅ Success!")

                    # Clean up temporary file
                    if temp_client_secret.exists():
                        temp_client_secret.unlink()

                    # Success celebration
                    st.balloons()
                    st.success(
                        f"""
                    🎉 **Successfully onboarded!**
                    
                    **Channel:** {title}  
                    **ID:** `{cid}`  
                    **Token:** `{_token_path.name}`
                    """
                    )

                    # Flow complete – clear flag
                    st.session_state.pop("oauth_flow_active", None)

                    # Auto-switch to creators tab
                    st.info(
                        "💡 Switch to the 'Manage Creators' tab to see your new creator account!"
                    )

                except Exception as exc:
                    st.error(
                        f"""
                    ❌ **OAuth onboarding failed**
                    
                    **Error:** {str(exc)}
                    
                    **Troubleshooting:**
                    - Ensure your OAuth environment variables are correctly set
                    - Check that you have the required YouTube channel permissions
                    - Verify your Google Cloud Console OAuth setup
                    """
                    )

                    # Show detailed error in expander for debugging
                    with st.expander("🔧 Debug Information"):
                        st.code(
                            f"Error Type: {type(exc).__name__}\nError Message: {str(exc)}"
                        )

                    # Clean up temporary file on error
                    temp_client_secret = TOKENS_DIR / "_temp_client_secret.json"
                    if temp_client_secret.exists():
                        temp_client_secret.unlink()

                    # Clear flag on error to allow retry
                    st.session_state.pop("oauth_flow_active", None)


# ---------------------------------------------------------------------------
# Video Statistics section – OAuth-aware analytics & public fallback
# ---------------------------------------------------------------------------

def video_statistics_section():
    """Display rich video statistics with OAuth upgrades when available."""

    st.title("📊 Video Statistics")

    url = st.text_input(
        "YouTube video URL",
        placeholder="https://youtu.be/abc123XYZ",
        key="stats_url",
    )

    # Add time period selector
    col1, col2 = st.columns([3, 1])
    with col2:
        period_options = {
            7: 7,
            14: 14,
            28: 28,
            60: 60,
            90: 90,
            "All Time": None  # Will be set dynamically
        }
        period_labels = [f"{k} days" if isinstance(k, int) else k for k in period_options.keys()]
        selected_period = st.selectbox(
            "📅 Analytics Period",
            options=list(period_options.keys()),
            index=2,  # Default to 28 days
            format_func=lambda x: f"{x} days" if isinstance(x, int) else x
        )
        days_back = period_options[selected_period]

    if st.button("Get Statistics", key="run_stats"):
        if not url.strip():
            st.error("Please enter a YouTube URL")
            return

        # Extract video ID
        try:
            vid = extract_video_id(url)
        except Exception as exc:
            st.error(f"Failed to extract video ID: {exc}")
            return

        # We need a Data API key for public fallback
        if not yt_key:
            st.error("YT_API_KEY not configured – please set the env variable for public access")
            return

        # Get best-available services (public and/or OAuth)
        service_info = get_enhanced_service(yt_key, vid)

        # Inform user about access capabilities
        display_access_level(service_info)

        # Choose whichever service is available for public data fetches
        svc = service_info.get("oauth_service") or service_info["public_service"]

        # ------------------------------------------------------------------
        # Fetch basic video metadata & statistics
        # ------------------------------------------------------------------
        try:
            v_resp = (
                svc.videos()
                .list(
                    part="snippet,statistics,contentDetails",
                    id=vid,
                )
                .execute()
            )
            if not v_resp["items"]:
                st.warning("No video found – please check the URL")
                return
            v_item = v_resp["items"][0]
        except Exception as exc:
            st.error(f"Failed to fetch video details: {exc}")
            return

        snippet = v_item["snippet"]
        stats = v_item.get("statistics", {})
        content = v_item.get("contentDetails", {})

        # If 'All Time' is selected, calculate days_back from video publishedAt
        if selected_period == "All Time":
            published_at = snippet.get("publishedAt")
            from datetime import datetime
            if published_at:
                created_date = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                days_back = (datetime.now(created_date.tzinfo) - created_date).days
            else:
                days_back = 3650  # fallback to 10 years if published_at missing

        # Friendly formatting helpers
        def fmt_num(val: str | int | None):
            try:
                num = int(val or 0)
            except Exception:
                return "0"
            if num >= 1_000_000:
                return f"{num/1_000_000:.1f}M"
            if num >= 1_000:
                return f"{num/1_000:.1f}K"
            return f"{num:,}"

        import re

        def parse_iso_duration(iso: str) -> str:
            """Convert ISO-8601 duration (PT#H#M#S) to HH:MM:SS string."""
            if not iso:
                return "00:00:00"

            pattern = re.compile(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?")
            m = pattern.match(iso)
            if not m:
                return iso  # Fallback to raw string if parsing fails

            h, mnt, s = (int(x) if x else 0 for x in m.groups())
            return f"{h:02d}:{mnt:02d}:{s:02d}"

        # ------------------------------------------------------------------
        # Layout: thumbnail & basic info
        # ------------------------------------------------------------------
        col_thumb, col_meta = st.columns([1, 3])
        with col_thumb:
            thumb_url = (
                snippet.get("thumbnails", {})
                .get("high", {})
                .get("url")
                or snippet.get("thumbnails", {})
                .get("default", {})
                .get("url")
            )
            if thumb_url:
                st.image(thumb_url, width=320)
        with col_meta:
            st.markdown(f"### {snippet.get('title', 'Unknown Title')}")
            st.caption(f"Published: {snippet.get('publishedAt', 'N/A')[:10]}")
            st.caption(f"Duration: {parse_iso_duration(content.get('duration', ''))}")
            if snippet.get("tags"):
                st.caption("🏷️ " + ", ".join(snippet["tags"][:10]))

        # ------------------------------------------------------------------
        # Metrics grid
        # ------------------------------------------------------------------
        view_count = int(stats.get("viewCount", 0))
        like_count = int(stats.get("likeCount", 0))
        comment_count = int(stats.get("commentCount", 0))
        favorite_count = int(stats.get("favoriteCount", 0))
        
        views = fmt_num(view_count)
        likes = fmt_num(like_count)
        comments_cnt = fmt_num(comment_count)
        favorites = fmt_num(favorite_count)

        # Main metrics
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("👀 Views", views)
        mcol2.metric("👍 Likes", likes)
        mcol3.metric("💬 Comments", comments_cnt)
        mcol4.metric("⭐ Favourites", favorites)
        
        # Calculate and display engagement ratios
        st.markdown("### 📊 Engagement Ratios")
        ratio_col1, ratio_col2, ratio_col3, ratio_col4 = st.columns(4)
        
        with ratio_col1:
            like_ratio = (like_count / view_count * 100) if view_count > 0 else 0
            st.metric("👍 Like Rate", f"{like_ratio:.2f}%")
        
        with ratio_col2:
            comment_ratio = (comment_count / view_count * 100) if view_count > 0 else 0
            st.metric("💬 Comment Rate", f"{comment_ratio:.2f}%")
        
        with ratio_col3:
            engagement_rate = ((like_count + comment_count) / view_count * 100) if view_count > 0 else 0
            st.metric("📈 Engagement Rate", f"{engagement_rate:.2f}%")
        
        with ratio_col4:
            # Views per day (approximate based on publish date)
            try:
                from datetime import datetime
                publish_date = datetime.fromisoformat(snippet.get('publishedAt', '').replace('Z', '+00:00'))
                days_since_publish = (datetime.now(publish_date.tzinfo) - publish_date).days
                views_per_day = view_count / max(days_since_publish, 1)
                st.metric("📅 Views/Day", f"{views_per_day:,.0f}")
            except:
                st.metric("📅 Views/Day", "N/A")

        # ------------------------------------------------------------------
        # Channel statistics (public)
        # ------------------------------------------------------------------
        channel_id = snippet.get("channelId")
        if channel_id:
            ch_stats = get_channel_stats(svc, channel_id)
            display_channel_stats(ch_stats)

        # ------------------------------------------------------------------
        # OAuth-powered enhanced analytics (if available & owned)
        # ------------------------------------------------------------------
        analytics_data = get_enhanced_analytics(service_info, vid, days_back)
        if analytics_data:
            display_enhanced_analytics(analytics_data, snippet.get("title", "Video"), service_info, vid, channel_id, days_back)
        else:
            st.info("Only public metrics available – OAuth analytics not accessible for this video.")


# ---------------------------------------------------------------------------
# Channel Statistics section
# ---------------------------------------------------------------------------

def channel_statistics_section():
    """Complete channel analytics: Public data + OAuth extras, now with tabbed UI."""
    
    st.title("📊 Channel Analytics (OAuth Enhanced)")
    
    # Check for OAuth credentials
    oauth_info = detect_oauth_capabilities()
    if not oauth_info["available"]:
        st.warning("🔐 **OAuth Required**: Use 'Creator Onboarding' to authenticate first.")
        return
    
    # Channel selection
    channels = oauth_info["channels"]
    if len(channels) > 1:
        channel_names = [f"{ch['title']} ({ch['subscriber_count']:,} subs)" for ch in channels]
        selected_idx = st.selectbox("📺 Select Channel", range(len(channels)), format_func=lambda i: channel_names[i])
        selected_channel = channels[selected_idx]
    else:
        selected_channel = channels[0]
    
    # Initialize services
    try:
        oauth_service = get_oauth_service(DEFAULT_CLIENT_SECRET, selected_channel["token_file"])
        public_service = oauth_service  # Same credentials for Data API
    except Exception as e:
        st.error(f"❌ Service initialization failed: {e}")
        return
    
    channel_id = selected_channel["id"]
    
    # Period selection
    st.subheader("📅 Analysis Period")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        period_options = {
            "Last 7 days": 7,
            "Last 14 days": 14,
            "Last 30 days": 30,
            "Last 60 days": 60,
            "Last 90 days": 90,
            "Last 6 months": 180,
            "Last year": 365,
            "All Time": None  # Will be set dynamically
        }
        
        selected_period = st.selectbox(
            "Select time period for analytics",
            options=list(period_options.keys()),
            index=2,  # Default to "Last 30 days"
            help="Choose the time range for your analytics data"
        )
        
        days_back = period_options[selected_period]
        
        # If 'All Time' is selected, calculate days_back from channel published_at
        if selected_period == "All Time":
            published_at = selected_channel.get("published_at") or selected_channel.get("snippet", {}).get("publishedAt")
            from datetime import datetime
            if published_at:
                created_date = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                days_back = (datetime.now(created_date.tzinfo) - created_date).days
            else:
                days_back = 3650  # fallback to 10 years if published_at missing
    
    with col2:
        st.metric("📊 Analysis Period", selected_period)
    
    # Fetch comprehensive data
    with st.spinner(f"🔍 Loading analytics for {selected_period.lower()}..."):
        from analytics_helpers import get_full_channel_analytics
        analytics_data = get_full_channel_analytics(oauth_service, public_service, channel_id, days_back=days_back)
    
    if analytics_data.get("error"):
        st.error(f"❌ {analytics_data['error']}")
        return
    
    # Channel overview always at the top
    display_public_channel_overview(analytics_data)
    
    # Tabbed analytics layout
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📈 Engagement Analysis",
        "📅 Upload Patterns",
        "🏆 Top Content",
        "👥 Audience Insights (OAuth, All Time)",
        "🚦 Traffic Sources",
        "💰 Monetization",
        "📈 Growth Trends (OAuth, All Time)",
        "👥 Views by Subscriber Status"
    ])
    
    with tab1:
        display_oauth_enhanced_engagement(analytics_data, selected_period)
    with tab2:
        display_public_upload_patterns(analytics_data)
    with tab3:
        display_public_top_content(analytics_data)
    with tab4:
        display_oauth_audience_insights(analytics_data, "All Time")
    with tab5:
        # Extract and display only the traffic sources table from audience insights (OAuth)
        oauth_data = analytics_data.get("oauth", {})
        traffic = oauth_data.get("traffic_sources", {})
        if traffic.get("rows"):
            st.subheader("🚀 Traffic Sources (OAuth)")
            total_views = sum(int(row[1]) for row in traffic["rows"])
            source_names = {
                "YT_SEARCH": "YouTube Search",
                "SUGGESTED_VIDEO": "Suggested Videos",
                "EXTERNAL_URL": "External Links",
                "BROWSE_FEATURES": "Browse Features",
                "NOTIFICATION": "Notifications",
                "DIRECT_OR_UNKNOWN": "Direct/Unknown",
                "PLAYLIST": "Playlists",
                "CHANNEL": "Channel Pages",
                "SUBSCRIBER": "Subscribers"
            }
            traffic_table = []
            for row in traffic["rows"]:
                source = row[0]
                views = row[1]
                friendly_name = source_names.get(source, source)
                view_count = int(views)
                percentage = (view_count / total_views * 100) if total_views > 0 else 0
                traffic_table.append({
                    "Traffic Source": friendly_name,
                    "Views": f"{view_count:,}",
                    "Percentage": f"{percentage:.1f}%"
                })
            st.table(traffic_table)
        else:
            st.info("No OAuth traffic source data available.")
    with tab6:
        display_oauth_revenue_metrics(analytics_data, selected_period)
    with tab7:
        display_oauth_growth_trends(analytics_data, "All Time")
    with tab8:
        # Channel-level subscriber status breakdown (OAuth only)
        try:
            if oauth_service and channel_id:
                from youtube.analytics import channel_subscriber_status_breakdown
                sub_status_data = channel_subscriber_status_breakdown(
                    oauth_service, channel_id, days_back=days_back
                )
                rows = sub_status_data.get("rows", [])
                if rows:
                    st.markdown("### 👥 Views by Subscriber Status")
                    table = []
                    for row in rows:
                        status = row[0].capitalize() if row[0] else "Unknown"
                        views = int(row[1]) if len(row) > 1 else 0
                        watch_time = int(row[2]) if len(row) > 2 else 0
                        avg_view_duration = int(row[3]) if len(row) > 3 else 0
                        mins, secs = divmod(avg_view_duration, 60)
                        table.append({
                            "Status": status,
                            "Views": f"{views:,}",
                            "Watch Time (min)": f"{watch_time:,}",
                            "Avg View Duration": f"{mins}m {secs}s"
                        })
                    st.table(table)
                else:
                    st.info("ℹ️ Subscriber status breakdown (views, watch time, avg view duration) is not available for this channel or period. This requires sufficient data and OAuth access.")
            else:
                st.info("ℹ️ Subscriber status breakdown is only available for your own channels with OAuth access.")
        except Exception as e:
            st.info(f"ℹ️ Could not fetch subscriber status breakdown: {e}")


def display_oauth_enhanced_engagement(analytics_data, period="Last 30 days"):
    """Enhanced engagement section with OAuth data."""
    
    # First show public engagement analysis
    display_public_engagement_analysis(analytics_data, None)
    
    # Add OAuth enhancements
    oauth_data = analytics_data.get("oauth", {})
    if not oauth_data or oauth_data.get("error"):
        return
    
    st.subheader(f"🔒 Enhanced Engagement Metrics (OAuth) - {period}")
    
    col1, col2, col3 = st.columns(3)
    
    # Impressions data
    impressions = oauth_data.get("impressions", {})
    if impressions.get("rows"):
        row = impressions["rows"][0]
        with col1:
            total_impressions = int(row[0]) if len(row) > 0 else 0
            st.metric("👁️ Impressions", f"{total_impressions:,}")
        with col2:
            ctr = float(row[1]) if len(row) > 1 else 0
            st.metric("🎯 Click-Through Rate", f"{ctr:.2f}%")
        with col3:
            unique_viewers = int(row[2]) if len(row) > 2 else 0
            st.metric("👤 Unique Viewers", f"{unique_viewers:,}")
    else:
        st.info("ℹ️ Impressions, Click-Through Rate, and Unique Viewers data are not available for this channel or period. This is common for new or low-activity channels, or if YouTube has not made these metrics available.")
    
    # Engagement breakdown
    engagement = oauth_data.get("engagement_breakdown", {})
    if engagement.get("rows"):
        row = engagement["rows"][0]
        st.markdown("**📊 Detailed Engagement Breakdown**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            likes = int(row[1]) if len(row) > 1 else 0
            st.metric("👍 Likes", f"{likes:,}")
        with col2:
            shares = int(row[4]) if len(row) > 4 else 0
            st.metric("🔄 Shares", f"{shares:,}")
        with col3:
            saves = int(row[5]) if len(row) > 5 else 0
            st.metric("💾 Saves", f"{saves:,}")
        with col4:
            playlist_adds = int(row[7]) if len(row) > 7 else 0
            st.metric("📝 Playlist Adds", f"{playlist_adds:,}")


def display_oauth_audience_insights(analytics_data, period="Last 30 days"):
    """Display OAuth-only audience demographics and geography."""
    
    oauth_data = analytics_data.get("oauth", {})
    if not oauth_data or oauth_data.get("error"):
        return
    
    st.header(f"👥 Audience Insights (OAuth) - {period}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Demographics
        demo = oauth_data.get("demographics", {})
        if demo.get("rows"):
            st.subheader("📊 Demographics")
            age_gender_map = {}
            for age, gender, percentage in demo["rows"]:
                if age not in age_gender_map:
                    age_gender_map[age] = {}
                age_gender_map[age][gender] = round(float(percentage), 1)
            
            demo_table = []
            for age_group in sorted(age_gender_map.keys()):
                row_data = {"Age Group": age_group}
                row_data.update({f"{gender.title()} %": f"{pct}%" for gender, pct in age_gender_map[age_group].items()})
                demo_table.append(row_data)
            
            st.table(demo_table)
        else:
            st.info("Demographics data not available")
    
    with col2:
        # Geography
        geo = oauth_data.get("geography", {})
        if geo.get("rows"):
            st.subheader("🌍 Top Countries")
            geo_rows = sorted(geo["rows"], key=lambda x: int(x[1]), reverse=True)[:10]
            geo_table = []
            for row in geo_rows:
                country, views, watch_time = row[:3]  # Only take the first three columns
                geo_table.append({
                    "Country": country,
                    "Views": f"{int(views):,}",
                    "Watch Time": f"{int(watch_time):,} min"
                })
            st.table(geo_table)
        else:
            st.info("Geographic data not available")
    
    # Traffic Sources (full width)
    traffic = oauth_data.get("traffic_sources", {})
    if traffic.get("rows"):
        st.subheader("🚀 Traffic Sources")
        total_views = sum(int(row[1]) for row in traffic["rows"])
        
        # Friendly source name mapping
        source_names = {
            "YT_SEARCH": "YouTube Search",
            "SUGGESTED_VIDEO": "Suggested Videos",
            "EXTERNAL_URL": "External Links",
            "BROWSE_FEATURES": "Browse Features",
            "NOTIFICATION": "Notifications",
            "DIRECT_OR_UNKNOWN": "Direct/Unknown",
            "PLAYLIST": "Playlists",
            "CHANNEL": "Channel Pages",
            "SUBSCRIBER": "Subscribers"
        }
        
        traffic_table = []
        for row in traffic["rows"]:
            source = row[0]
            views = row[1]
            friendly_name = source_names.get(source, source)
            view_count = int(views)
            percentage = (view_count / total_views * 100) if total_views > 0 else 0
            
            traffic_table.append({
                "Traffic Source": friendly_name,
                "Views": f"{view_count:,}",
                "Percentage": f"{percentage:.1f}%"
            })
        
        st.table(traffic_table)


def display_oauth_revenue_metrics(analytics_data, period="Last 30 days"):
    """Display monetization and revenue data."""
    
    oauth_data = analytics_data.get("oauth", {})
    if not oauth_data or oauth_data.get("error"):
        return
    
    monetization = oauth_data.get("monetization", {})
    if monetization.get("error") or not monetization.get("rows"):
        st.header("💰 Monetization")
        st.info("💡 Monetization data not available - channel may not be monetized or data access restricted")
        return
    
    st.header(f"💰 Revenue Analytics (OAuth) - {period}")
    
    row = monetization["rows"][0]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        revenue = float(row[0]) if row[0] else 0
        st.metric("💵 Est. Revenue", f"${revenue:.2f}")
    
    with col2:
        ad_revenue = float(row[1]) if len(row) > 1 and row[1] else 0
        st.metric("📺 Ad Revenue", f"${ad_revenue:.2f}")
    
    with col3:
        cpm = float(row[4]) if len(row) > 4 and row[4] else 0
        st.metric("📊 CPM", f"${cpm:.2f}")
    
    with col4:
        playback_cpm = float(row[5]) if len(row) > 5 and row[5] else 0
        st.metric("▶️ Playback CPM", f"${playback_cpm:.2f}")
    
    st.caption(f"📅 Revenue data for {period.lower()}")


def display_oauth_growth_trends(analytics_data, period="Last 30 days"):
    """Display growth trends and subscriber analytics."""
    
    oauth_data = analytics_data.get("oauth", {})
    if not oauth_data or oauth_data.get("error"):
        return
    
    growth = oauth_data.get("growth_metrics", {})
    if not growth.get("rows"):
        return
    
    st.header(f"📈 Growth Trends (OAuth) - {period}")
    
    # Process growth data
    import pandas as pd
    
    growth_data = []
    for row in growth["rows"]:
        if len(row) >= 6:
            growth_data.append({
                "Date": row[0],
                "Views": int(row[1]),
                "Subs Gained": int(row[2]),
                "Subs Lost": int(row[3]),
                "Watch Time": int(row[4]),
                "Net Subscribers": int(row[2]) - int(row[3])
            })
    
    if growth_data:
        df = pd.DataFrame(growth_data)
        df["Date"] = pd.to_datetime(df["Date"])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Daily Views")
            st.line_chart(df.set_index("Date")["Views"])
        
        with col2:
            st.subheader("👥 Net Subscriber Growth")
            st.line_chart(df.set_index("Date")["Net Subscribers"])
        
        # Summary metrics
        total_views = df["Views"].sum()
        net_subs = df["Net Subscribers"].sum()
        avg_daily_views = df["Views"].mean()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Total Views", f"{total_views:,}")
        with col2:
            st.metric("👥 Net Subscribers", f"{net_subs:+,}")
        with col3:
            st.metric("📈 Avg Daily Views", f"{avg_daily_views:.0f}")


# ---------------------------------------------------------------------------
# Public Channel Analysis section
# ---------------------------------------------------------------------------

def public_channel_analysis_section():
    """Analyze any public YouTube channel using only the Data API, with a modern tabbed UI."""
    
    st.title("🌐 Public Channel Analysis")
    st.markdown("Analyze **any YouTube channel** using only public data - no OAuth required!")
    
    # Check if we have API key
    if not yt_key:
        st.error("❌ **YouTube Data API key required**")
        st.info("💡 Set the `YT_API_KEY` environment variable to enable public channel analysis")
        return
    
    # Input section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        channel_id = st.text_input(
            "🔗 Enter YouTube Channel ID (starts with 'UC')",
            placeholder="UCxIJaCMEptJjxmmQgGFsnCg",
            help="Find the 24-character channel ID in the channel URL – it always starts with 'UC'."
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
        analyze_button = st.button("🔍 Analyze Channel", type="primary")
    
    # Fixed analysis period (last 30 days)
    recent_days = 30

    if analyze_button and channel_id.strip():
        with st.spinner("🔍 Analyzing channel... This may take a moment"):
            try:
                from youtube.public import get_comprehensive_channel_data, get_channel_recent_performance
                
                # Get public service
                public_service = get_public_service(yt_key)
                
                # Get comprehensive data
                channel_data = get_comprehensive_channel_data(public_service, channel_id.strip())
                
                if channel_data.get("error"):
                    st.error(f"❌ {channel_data['error']}")
                    st.info("💡 Ensure the channel ID is correct and the channel is public.")
                    return
                
                # Get recent performance
                channel_id = channel_data["channel_info"]["id"]
                recent_performance = get_channel_recent_performance(public_service, channel_id, recent_days)
                
                # Verify correct channel was found
                found_channel = channel_data["channel_info"]
                st.success(f"✅ **Found Channel**: {found_channel['snippet']['title']} (ID: {found_channel['id']})")
                
                # Tabbed analytics layout
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📊 Overview",
                    "📈 Engagement Analysis",
                    "📅 Upload Patterns",
                    "🏆 Top Content"
                ])
                with tab1:
                    display_public_channel_overview(channel_data)
                with tab2:
                    display_public_engagement_analysis(channel_data, recent_performance)
                with tab3:
                    display_public_upload_patterns(channel_data)
                with tab4:
                    display_public_top_content(channel_data)
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.info("💡 Make sure the channel URL is valid and publicly accessible")


def display_public_channel_overview(channel_data):
    """Display comprehensive channel overview from public data."""
    
    channel_info = channel_data["channel_info"]
    snippet = channel_info["snippet"]
    statistics = channel_info["statistics"]
    
    st.header("📊 Channel Overview")
    
    # Channel header with thumbnail
    col1, col2 = st.columns([1, 4])
    
    with col1:
        thumbnail_url = snippet.get("thumbnails", {}).get("high", {}).get("url")
        if thumbnail_url:
            st.image(thumbnail_url, width=120)
    
    with col2:
        st.markdown(f"## {snippet['title']}")
        if snippet.get("customUrl"):
            st.markdown(f"**@{snippet['customUrl']}**")
        
        # Channel description preview
        description = snippet.get("description", "")
        if description:
            preview = description[:300] + "..." if len(description) > 300 else description
            st.markdown(f"*{preview}*")
    
    # Key statistics
    st.subheader("📈 Channel Statistics")
    
    subscribers = int(statistics.get("subscriberCount", 0))
    videos = int(statistics.get("videoCount", 0))
    views = int(statistics.get("viewCount", 0))
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("👥 Subscribers", f"{subscribers:,}")
    with col2:
        st.metric("🎬 Videos", f"{videos:,}")
    with col3:
        st.metric("👀 Total Views", f"{views:,}")
    with col4:
        if videos > 0:
            avg_views = views // videos
            st.metric("📊 Avg Views/Video", f"{avg_views:,}")
    with col5:
        # Calculate channel age
        published_at = snippet.get("publishedAt")
        if published_at:
            from datetime import datetime
            created_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
            years_old = (datetime.now(created_date.tzinfo) - created_date).days // 365
            st.metric("📅 Channel Age", f"{years_old} years")
    
    # Channel insights
    if subscribers < 1000:
        st.info("🌱 **Growing Channel**: Building towards monetization threshold (1K subscribers)")
    elif subscribers < 10000:
        st.success("🚀 **Established Channel**: Good subscriber base with growth potential")
    elif subscribers < 100000:
        st.success("⭐ **Popular Channel**: Strong subscriber base and engagement")
    else:
        st.success("🏆 **Major Channel**: Large, established audience")


def display_public_engagement_analysis(channel_data, recent_performance):
    """Display engagement metrics and recent performance analysis."""
    
    st.header("📊 Engagement Analysis")
    
    engagement_data = channel_data.get("engagement_analysis", {})
    
    if not engagement_data:
        st.info("No engagement data available")
        return
    
    # Core engagement metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Overall Performance")
        
        total_videos = engagement_data.get("total_videos_analyzed", 0)
        total_views = engagement_data.get("total_views", 0)
        total_likes = engagement_data.get("total_likes", 0)
        total_comments = engagement_data.get("total_comments", 0)
        avg_engagement = engagement_data.get("avg_engagement_rate", 0)
        
        st.metric("📹 Videos Analyzed", f"{total_videos}")
        st.metric("👀 Total Views", f"{total_views:,}")
        st.metric("🔥 Avg Engagement Rate", f"{avg_engagement}%")
        st.metric("👍 Like-to-View Ratio", f"{engagement_data.get('like_to_view_ratio', 0)}%")
        
        # Performance tier
        tier = engagement_data.get("performance_tier", "Unknown")
        tier_colors = {"High": "🟢", "Medium": "🟡", "Growing": "🔵"}
        st.markdown(f"**Performance Tier**: {tier_colors.get(tier, '⚪')} {tier}")
        
    with col2:
        st.subheader("📈 Averages & Consistency")
        
        avg_views = engagement_data.get("avg_views_per_video", 0)
        avg_likes = engagement_data.get("avg_likes_per_video", 0)
        avg_comments = engagement_data.get("avg_comments_per_video", 0)
        consistency = engagement_data.get("consistency_score", 0)
        
        st.metric("📊 Avg Views per Video", f"{avg_views:,.0f}")
        st.metric("👍 Avg Likes per Video", f"{avg_likes:.1f}")
        st.metric("💬 Avg Comments per Video", f"{avg_comments:.1f}")
        
        # Consistency score interpretation
        if consistency > 0.7:
            consistency_label = "🟢 Very Consistent"
        elif consistency > 0.5:
            consistency_label = "🟡 Moderately Consistent"
        else:
            consistency_label = "🔴 Variable Performance"
        
        st.metric("📊 Consistency Score", f"{consistency:.2f}")
        st.markdown(f"**{consistency_label}**")
    
    # Recent performance
    if recent_performance and not recent_performance.get("error"):
        st.subheader(f"⏰ Recent Performance ({recent_performance.get('period_days', 30)} days)")
        
        recent_videos = recent_performance.get("videos_count", 0)
        recent_views = recent_performance.get("total_views", 0)
        recent_avg = recent_performance.get("avg_views_per_video", 0)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📹 Recent Videos", f"{recent_videos}")
        with col2:
            st.metric("👀 Recent Views", f"{recent_views:,}")
        with col3:
            st.metric("📊 Recent Avg Views", f"{recent_avg:,.0f}")
        
        # Compare with overall average
        overall_avg = engagement_data.get("avg_views_per_video", 0)
        if overall_avg > 0 and recent_avg > 0:
            performance_change = ((recent_avg - overall_avg) / overall_avg) * 100
            if abs(performance_change) > 5:
                trend_emoji = "📈" if performance_change > 0 else "📉"
                st.markdown(f"**Trend**: {trend_emoji} {performance_change:+.1f}% vs overall average")
    
    # Best performing video
    best_video = engagement_data.get("best_performing_video", {})
    if best_video:
        st.subheader("🏆 Best Performing Video")
        st.markdown(f"**{best_video.get('title', 'Unknown')}**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("👀 Views", f"{best_video.get('views', 0):,}")
        with col2:
            st.metric("👍 Likes", f"{best_video.get('likes', 0):,}")
        with col3:
            video_id = best_video.get('video_id')
            if video_id:
                st.markdown(f"[🔗 Watch Video](https://youtube.com/watch?v={video_id})")


def display_public_upload_patterns(channel_data):
    """Display upload frequency and timing analysis."""
    
    st.header("📅 Upload Patterns & Consistency")
    
    upload_data = channel_data.get("upload_patterns", {})
    
    if not upload_data:
        st.info("No upload pattern data available")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("⏰ Upload Frequency")
        
        total_videos = upload_data.get("total_videos", 0)
        avg_days = upload_data.get("avg_days_between_uploads", 0)
        consistency = upload_data.get("upload_consistency", "Unknown")
        
        st.metric("📹 Videos Analyzed", total_videos)
        st.metric("📅 Avg Days Between Uploads", f"{avg_days:.1f}")
        
        consistency_colors = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
        st.markdown(f"**Consistency**: {consistency_colors.get(consistency, '⚪')} {consistency}")
        
        # Upload frequency insights
        if avg_days < 7:
            st.success("🚀 Very active - uploads multiple times per week")
        elif avg_days < 14:
            st.info("📈 Good activity - weekly uploads")
        elif avg_days < 30:
            st.warning("📉 Moderate activity - bi-weekly uploads")
        else:
            st.error("🐌 Low activity - monthly or less frequent uploads")
    
    with col2:
        st.subheader("📊 Optimal Upload Times")
        
        best_day = upload_data.get("most_common_upload_day")
        best_hour = upload_data.get("most_common_upload_hour")
        
        if best_day:
            st.metric("📅 Most Common Day", best_day)
        if best_hour is not None:
            st.metric("🕐 Most Common Hour", f"{best_hour}:00")
        
        # Day distribution
        day_distribution = upload_data.get("day_distribution", {})
        if day_distribution:
            st.markdown("**Day Distribution:**")
            for day, count in sorted(day_distribution.items(), key=lambda x: x[1], reverse=True):
                st.markdown(f"- {day}: {count} videos")
    
    with col3:
        st.subheader("🎬 Content Length")
        
        avg_duration = upload_data.get("avg_video_duration_seconds", 0)
        
        if avg_duration > 0:
            minutes = int(avg_duration // 60)
            seconds = int(avg_duration % 60)
            st.metric("⏱️ Avg Video Length", f"{minutes}m {seconds}s")
            
            # Duration insights
            if avg_duration < 300:  # 5 minutes
                st.info("⚡ Short-form content focus")
            elif avg_duration < 900:  # 15 minutes
                st.success("🎯 Optimal length for engagement")
            elif avg_duration < 1800:  # 30 minutes
                st.info("📺 Long-form content")
            else:
                st.warning("🎭 Very long content - ensure high retention")
        
        # Latest upload
        latest_upload = upload_data.get("latest_upload")
        if latest_upload:
            from datetime import datetime
            latest_date = datetime.fromisoformat(latest_upload.replace('Z', '+00:00'))
            days_ago = (datetime.now(latest_date.tzinfo) - latest_date).days
            st.metric("📅 Latest Upload", f"{days_ago} days ago")


def display_public_top_content(channel_data):
    """Display top performing videos and playlists."""
    
    st.header("🏆 Top Performing Content")
    
    popular_videos = channel_data.get("popular_videos", [])
    playlists = channel_data.get("playlists", [])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📹 Most Popular Videos")
        
        if popular_videos:
            top_videos_data = []
            for i, video in enumerate(popular_videos[:10]):
                title = video["snippet"]["title"]
                views = int(video["statistics"].get("viewCount", 0))
                likes = int(video["statistics"].get("likeCount", 0))
                published = video["snippet"]["publishedAt"][:10]  # Date only
                
                # Shorten title for display
                short_title = title[:40] + "..." if len(title) > 40 else title
                
                top_videos_data.append({
                    "#": i + 1,
                    "Title": short_title,
                    "Views": f"{views:,}",
                    "Likes": f"{likes:,}",
                    "Published": published
                })
            
            st.table(top_videos_data)
            
            # Quick stats
            total_top_views = sum(int(v["statistics"].get("viewCount", 0)) for v in popular_videos[:10])
            st.caption(f"📊 Top 10 videos: {total_top_views:,} total views")
        else:
            st.info("No video data available")
    
    with col2:
        st.subheader("📝 Channel Playlists")
        
        if playlists:
            playlist_data = []
            for playlist in playlists[:10]:
                title = playlist["snippet"]["title"]
                video_count = playlist["contentDetails"]["itemCount"]
                
                # Shorten title for display
                short_title = title[:40] + "..." if len(title) > 40 else title
                
                playlist_data.append({
                    "Playlist": short_title,
                    "Videos": video_count
                })
            
            st.table(playlist_data)
            
            total_playlists = len(playlists)
            total_playlist_videos = sum(p["contentDetails"]["itemCount"] for p in playlists)
            st.caption(f"📊 {total_playlists} playlists with {total_playlist_videos} total videos")
        else:
            st.info("No public playlists found")
    
    # Content recommendations
    if popular_videos:
        st.subheader("💡 Content Strategy Recommendations")
        
        # Analyze top videos for patterns
        top_5_titles = [v["snippet"]["title"].lower() for v in popular_videos[:5]]
        
        recommendations = []
        
        # Check for tutorial content
        if any('tutorial' in title or 'how' in title for title in top_5_titles):
            recommendations.append("🎯 **Tutorial content performs well** - Consider more educational videos")
        
        # Check for project showcases
        if any('project' in title or 'build' in title for title in top_5_titles):
            recommendations.append("🛠️ **Project showcases are popular** - Share more development processes")
        
        # Check for tech content
        if any('code' in title or 'programming' in title for title in top_5_titles):
            recommendations.append("💻 **Tech content resonates** - Expand programming tutorials and reviews")
        
        # Views analysis
        top_video_views = int(popular_videos[0]["statistics"].get("viewCount", 0))
        avg_views = sum(int(v["statistics"].get("viewCount", 0)) for v in popular_videos[:5]) / 5
        
        if top_video_views > avg_views * 3:
            recommendations.append(f"⭐ **Viral potential identified** - Analyze what made your top video special")
        
        if recommendations:
            for rec in recommendations:
                st.markdown(rec)
        else:
            st.markdown("📈 **Keep creating consistent content** - More data needed for specific recommendations")


# ---------------------------------------------------------------------------
# Comments Analyzer section
# ---------------------------------------------------------------------------

def comments_analyzer_section():
    """Analyze YouTube video comments for sentiment, opinions, and themes."""
    
    st.title("💬 Comments Analyzer")
    st.markdown("Analyze viewer sentiment and opinions from YouTube video comments using AI")

    url = st.text_input(
        "YouTube video URL",
        placeholder="https://youtu.be/abc123XYZ",
        key="comments_url",
    )

    if st.button("Analyze Comments", key="run_comments"):
        if not url.strip():
            st.error("Please enter a YouTube URL")
            return

        # Extract video ID
        try:
            vid = extract_video_id(url)
        except Exception as exc:
            st.error(f"Failed to extract video ID: {exc}")
            return

        # Check for API access
        if not yt_key:
            st.error("YT_API_KEY not configured – please set the environment variable")
            return

        # Get enhanced services (OAuth + public)
        service_info = get_enhanced_service(yt_key, vid)
        display_access_level(service_info)

        # Fetch comments
        with st.spinner("🔍 Fetching all comments..."):
            comments = get_enhanced_comments(service_info, vid)
            
            if not comments:
                st.warning("No comments found or comments are disabled for this video")
                return

            st.info(f"ℹ️ Found {len(comments)} comments - analyzing all of them!")

        # Analyze with LLM
        with st.spinner("🧠 Analyzing sentiment and themes..."):
            try:
                analysis_results = analyze_comments_with_llm(comments)
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                return

        # Display results
        display_simple_comment_analysis(analysis_results, vid)


def analyze_comments_with_llm(comments):
    """Analyze comments using LLM for sentiment and themes."""
    
    # Check if LLM is available
    if not (SETTINGS.openrouter_api_keys or SETTINGS.groq_api_keys or SETTINGS.gemini_api_keys):
        st.error("❌ **LLM API key required**")
        st.info("Set `OPENROUTER_API_KEY`, `GROQ_API_KEY`, or `GEMINI_API_KEY` environment variable")
        return None

    # Get smart LLM client with automatic fallback
    from llms import get_smart_client
    try:
        client = get_smart_client()
    except Exception as e:
        st.error(f"Failed to initialize LLM client: {e}")
        return None

    # Prepare comments for analysis - use all comments but optimize for token limits
    total_comments = len(comments)
    
    # Sort comments by likes to prioritize most engaging ones for analysis
    sorted_comments = sorted(comments, key=lambda x: x.get('likeCount', 0), reverse=True)
    
    # Take up to 100 most engaging comments for detailed analysis
    comments_to_analyze = sorted_comments[:100] if len(sorted_comments) > 100 else sorted_comments
    
    comment_texts = []
    for i, comment in enumerate(comments_to_analyze):
        text = comment.get('textDisplay', '') or comment.get('text', '')
        likes = comment.get('likeCount', 0)
        # Truncate very long comments but keep meaningful content
        truncated_text = text[:200] + "..." if len(text) > 200 else text
        comment_texts.append(f"{i+1}. [{likes} likes] {truncated_text}")

    prompt = f"""
Analyze these YouTube video comments ({len(comments_to_analyze)} most engaging out of {total_comments} total) and provide a comprehensive summary:

**Comments:**
{chr(10).join(comment_texts)}

**Provide analysis in this format:**

## Overall Sentiment
[Positive/Negative/Mixed] - Brief explanation

## Main Themes
- Theme 1: Description
- Theme 2: Description  
- Theme 3: Description

## Top Positive Points
- Point 1
- Point 2
- Point 3

## Top Concerns/Criticisms
- Concern 1
- Concern 2
- Concern 3

## Creator Recommendations
- Recommendation 1
- Recommendation 2
- Recommendation 3

Keep it concise but thorough. Focus on actionable insights for the creator.
"""

    try:
        response = client.chat(
            [
                {"role": "system", "content": "You are an expert at analyzing social media engagement and audience sentiment. Focus on providing actionable insights for content creators."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000,
        )
        
        return {
            "summary": response,
            "comments": comments,  # Return all comments
            "total_analyzed": len(comments_to_analyze),
            "total_comments": total_comments
        }
        
    except Exception as e:
        st.error(f"LLM analysis failed: {e}")
        return None


def display_simple_comment_analysis(analysis_results, video_id):
    """Display simplified comment analysis results."""
    
    if not analysis_results:
        st.warning("No analysis results to display")
        return

    comments = analysis_results["comments"]
    total_analyzed = analysis_results["total_analyzed"]
    total_comments = analysis_results["total_comments"]
    
    # Basic stats
    st.header("📊 Analysis Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📝 Total Comments", total_comments)
    with col2:
        if total_analyzed == total_comments:
            st.metric("🔍 Analyzed", "All comments")
        else:
            st.metric("🔍 Analyzed", f"{total_analyzed} (top by engagement)")
    with col3:
        # Most liked comment
        most_liked = max(comments, key=lambda x: x.get('likeCount', 0), default={})
        st.metric("🔥 Top Likes", most_liked.get('likeCount', 0))

    # Creator Authenticity Score
    st.header("🎯 Creator Authenticity Score")
    authenticity_data = calculate_creator_authenticity(comments, analysis_results)
    display_authenticity_score(authenticity_data)

    # AI Analysis Results
    st.header("🤖 AI Analysis")
    
    # Add note about analysis approach if we had to prioritize comments
    if total_analyzed < total_comments:
        st.info(f"📊 **Analysis Strategy**: With {total_comments} comments, we analyzed the top {total_analyzed} most engaging comments (by likes) to provide the most representative insights while staying within AI processing limits.")
    
    st.markdown(analysis_results["summary"])

    # Show sample comments
    st.header("💬 Sample Comments")
    
    # Show top 5 most liked comments
    most_liked_comments = sorted(comments, key=lambda x: x.get('likeCount', 0), reverse=True)[:5]
    
    for i, comment in enumerate(most_liked_comments, 1):
        with st.expander(f"#{i} - 👍 {comment.get('likeCount', 0)} likes"):
            st.markdown(f"**{comment.get('author', 'Unknown')}**")
            st.write(comment.get('textDisplay', ''))
            st.caption(f"Published: {comment.get('publishedAt', 'Unknown')[:10]}")

    # Simple export
    st.header("💾 Export")
    if st.button("📊 Export Comments to CSV"):
        import pandas as pd
        df = pd.DataFrame([
            {
                "Author": c.get('author', ''),
                "Comment": c.get('textDisplay', ''),
                "Likes": c.get('likeCount', 0),
                "Published": c.get('publishedAt', ''),
            }
            for c in comments
        ])
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            csv,
            f"comments_{video_id}.csv",
            "text/csv"
        )

 



# ---------------------------------------------------------------------------
# App entry point – single page
# ---------------------------------------------------------------------------


def main():
    st.set_page_config(page_title="YouTube Creator OAuth Manager", layout="wide")
    tab_onboard, tab_audio, tab_video, tab_azure_vi, tab_stats, tab_channel, tab_public, tab_comments = st.tabs([
        "Creator Onboarding",
        "Audio Analyzer", 
        "Video Analyzer",
        "☁️ Azure Video Indexer",
        "Video Statistics",
        "Channel Analytics",
        "Public Channel Analysis",
        "Comments Analyzer",
    ])

    with tab_onboard:
        onboarding_section()

    with tab_audio:
        audio_analyzer_section()

    with tab_video:
        video_analyzer_section()

    with tab_azure_vi:
        azure_video_indexer_section()

    with tab_stats:
        video_statistics_section()
        
    with tab_channel:
        channel_statistics_section()
        
    with tab_public:
        public_channel_analysis_section()
        
    with tab_comments:
        comments_analyzer_section()


def calculate_creator_authenticity(comments, analysis_results):
    """Calculate creator authenticity percentage using LLM analysis of comments and engagement patterns."""
    
    if not comments:
        return {"score": 0, "breakdown": {}, "insights": []}
    
    # Check if LLM is available
    if not (SETTINGS.openrouter_api_keys or SETTINGS.groq_api_keys or SETTINGS.gemini_api_keys):
        st.warning("⚠️ **LLM required for authenticity analysis** - Set API keys to enable")
        return {"score": 0, "breakdown": {}, "insights": ["LLM API key required"]}

    # Get smart LLM client with automatic fallback
    from llms import get_smart_client
    try:
        client = get_smart_client()
    except Exception as e:
        return {"score": 0, "breakdown": {}, "insights": [f"Failed to initialize LLM client: {e}"]}

    # Prepare data for LLM analysis
    total_comments = len(comments)
    total_likes = sum(c.get('likeCount', 0) for c in comments)
    avg_comment_length = sum(len(c.get('textDisplay', '')) for c in comments) / len(comments)
    comments_with_replies = sum(1 for c in comments if c.get('totalReplyCount', 0) > 0)
    
    # Prepare ALL comments for analysis (truncate if too long for token limits)
    comment_sample = []
    for i, comment in enumerate(comments):
        text = comment.get('textDisplay', '')
        likes = comment.get('likeCount', 0)
        replies = comment.get('totalReplyCount', 0)
        # Truncate very long comments to fit more into token limits
        truncated_text = text[:80] + "..." if len(text) > 80 else text
        comment_sample.append(f"{i+1}. [{likes}❤️ {replies}💬] {truncated_text}")

    # Join all comments but respect token limits (estimate ~4 chars per token)
    all_comments_text = chr(10).join(comment_sample)
    
    # If the text is too long, truncate to fit in prompt (keep roughly 8000 chars for comments)
    if len(all_comments_text) > 8000:
        # Take comments up to the character limit
        truncated_comments = []
        current_length = 0
        for comment_line in comment_sample:
            if current_length + len(comment_line) > 8000:
                break
            truncated_comments.append(comment_line)
            current_length += len(comment_line)
        
        all_comments_text = chr(10).join(truncated_comments)
        comments_shown = len(truncated_comments)
        truncation_note = f"(Showing first {comments_shown} comments due to length limits)"
    else:
        comments_shown = len(comments)
        truncation_note = ""

    prompt = f"""
Analyze this YouTube video's comment section to determine the creator's AUTHENTICITY LEVEL (0-100%).

**Comment Data:**
- Total comments: {total_comments}
- Total likes: {total_likes}
- Average comment length: {avg_comment_length:.1f} characters
- Comments with replies: {comments_with_replies}

**ALL Comments Analysis {truncation_note}:**
{all_comments_text}

**Analyze these factors and provide a score:**

1. **Comment Quality (0-100)**: Are comments genuine, thoughtful, and diverse?
2. **Audience Loyalty (0-100)**: Do comments show real engagement and community?

**Look for:**
- Bot indicators: repetitive text, emoji spam, very short comments
- Authentic engagement: varied lengths, thoughtful responses, questions
- Community signs: conversations, personal stories, specific feedback
- Loyalty indicators: recurring themes, inside jokes, personal connections

**Respond EXACTLY in this format:**
AUTHENTICITY_SCORE: [0-100]
COMMENT_QUALITY: [0-100]
AUDIENCE_LOYALTY: [0-100]
LEVEL: [Highly Authentic/Authentic/Moderately Authentic/Low Authenticity]
INSIGHTS: [3-5 specific observations about authenticity patterns]
"""

    try:
        response = client.chat(
            [
                {"role": "system", "content": "You are an expert at detecting authentic vs fake social media engagement. Analyze comment patterns to assess creator authenticity."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=800,
        )
        
        # Parse LLM response (handle both plain and markdown formats)
        score = 0
        comment_quality = 0
        audience_loyalty = 0
        level = "Unknown"
        insights = []
        
        for line in response.strip().split('\n'):
            # Clean line of markdown formatting
            clean_line = line.strip().replace('**', '').replace('*', '')
            
            if 'AUTHENTICITY_SCORE:' in clean_line:
                try:
                    score = float(clean_line.split(':')[1].strip())
                except (ValueError, IndexError):
                    pass
            elif 'COMMENT_QUALITY:' in clean_line:
                try:
                    comment_quality = float(clean_line.split(':')[1].strip())
                except (ValueError, IndexError):
                    pass
            elif 'AUDIENCE_LOYALTY:' in clean_line:
                try:
                    audience_loyalty = float(clean_line.split(':')[1].strip())
                except (ValueError, IndexError):
                    pass
            elif 'LEVEL:' in clean_line:
                try:
                    level = clean_line.split(':')[1].strip()
                except IndexError:
                    pass
            elif 'INSIGHTS:' in clean_line:
                # For insights, collect numbered points from subsequent lines
                insights_text = clean_line.split(':', 1)[1].strip()
                if insights_text:
                    insights = [insight.strip() for insight in insights_text.split(';') if insight.strip()]
        
        # If no insights were found in the main line, try to extract from numbered points
        if not insights:
            insights = []
            for line in response.strip().split('\n'):
                clean_line = line.strip()
                # Look for numbered insights (1. 2. 3. etc.)
                if clean_line and (clean_line.startswith(('1.', '2.', '3.', '4.', '5.'))):
                    insight_text = clean_line[2:].strip()  # Remove "1. " etc.
                    if insight_text and len(insight_text) > 10:  # Only meaningful insights
                        insights.append(insight_text)
        
        # Set level emoji
        level_emoji = "🌟" if "Highly" in level else "✅" if level == "Authentic" else "⚠️" if "Moderately" in level else "🔴"
        
        return {
            "score": round(score, 1),
            "level": level,
            "level_emoji": level_emoji,
            "breakdown": {
                "Comment Quality": round(comment_quality, 1),
                "Audience Loyalty": round(audience_loyalty, 1)
            },
            "insights": insights,
            "metrics": {
                "total_comments": total_comments,
                "avg_comment_length": round(avg_comment_length, 1),
                "total_engagement": total_likes,
                "comments_with_replies": comments_with_replies
            }
        }
        
    except Exception as e:
        st.error(f"Authenticity analysis failed: {e}")
        return {"score": 0, "breakdown": {}, "insights": [f"Analysis failed: {e}"]}


def display_authenticity_score(authenticity_data):
    """Display the creator authenticity score with breakdown."""
    
    if not authenticity_data or authenticity_data["score"] == 0:
        st.warning("Unable to calculate authenticity score - insufficient comment data")
        return
    
    score = authenticity_data["score"]
    level = authenticity_data["level"]
    level_emoji = authenticity_data["level_emoji"]
    breakdown = authenticity_data["breakdown"]
    insights = authenticity_data["insights"]
    metrics = authenticity_data["metrics"]
    
    # Main authenticity score display
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.metric(
            "🎯 Authenticity Score", 
            f"{score}%",
            help="Based on comment quality and audience loyalty analysis"
        )
    
    with col2:
        st.metric("📊 Level", f"{level_emoji} {level}")
    
    with col3:
        st.metric("💬 Comments Analyzed", metrics["total_comments"])
    
    # Score breakdown
    with st.expander("📊 Score Breakdown", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Comment Quality")
            st.progress(breakdown["Comment Quality"] / 100)
            st.caption(f"Score: {breakdown['Comment Quality']}%")
            st.write("*Analyzes comment genuineness, diversity, and thoughtfulness*")
        
        with col2:
            st.subheader("Audience Loyalty")
            st.progress(breakdown["Audience Loyalty"] / 100)
            st.caption(f"Score: {breakdown['Audience Loyalty']}%")
            st.write("*Evaluates engagement patterns and community interaction*")
    
    # Insights
    if insights:
        st.subheader("💡 Authenticity Insights")
        for insight in insights:
            st.write(insight)
    
    # Detailed metrics
    with st.expander("🔍 Detailed Metrics", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Comment Analysis:**")
            st.write(f"• Average comment length: {metrics['avg_comment_length']} characters")
            st.write(f"• Total engagement (likes): {metrics['total_engagement']}")
        
        with col2:
            st.write("**Community Indicators:**")
            st.write(f"• Comments with replies: {metrics['comments_with_replies']}")
            st.write(f"• Analysis powered by AI")


# ---------------------------------------------------------------------------
# Azure Video Indexer section
# ---------------------------------------------------------------------------


def azure_video_indexer_section():
    st.title("☁️ Azure Video Indexer")
    
    # Configuration status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        subscription_key_status = "✅" if SETTINGS.azure_vi_subscription_key else "❌"
        st.metric("Subscription Key", subscription_key_status)
        if SETTINGS.azure_vi_subscription_key:
            st.caption(f"Key: ...{SETTINGS.azure_vi_subscription_key[-8:]}")
    
    with col2:
        account_id_status = "✅" if SETTINGS.azure_vi_account_id else "❌"
        st.metric("Account ID", account_id_status)
        if SETTINGS.azure_vi_account_id:
            st.caption(f"ID: {SETTINGS.azure_vi_account_id}")
    
    with col3:
        st.metric("Location", SETTINGS.azure_vi_location)
    
    # Test connection button
    if SETTINGS.azure_vi_subscription_key and SETTINGS.azure_vi_account_id:
        if st.button("🔍 Test API Connection", help="Test if your credentials are working"):
            try:
                vi = AzureVideoIndexer()
                with st.spinner("Testing connection..."):
                    # Try to get an access token
                    token = vi._get_access_token()
                    st.success("✅ Connection successful! Your credentials are working.")
                    st.info(f"Access token obtained (expires in ~1 hour)")
            except AzureVideoIndexerError as e:
                st.error(f"❌ Connection failed: {e}")
                st.write("**Common issues:**")
                st.write("- Make sure you're using an ARM-based Azure Video Indexer account")
                st.write("- Verify your subscription key is from the Developer Portal")
                st.write("- Check that your Account ID matches your Azure resource")
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")
    
    # Configuration help
    if not SETTINGS.azure_vi_subscription_key or not SETTINGS.azure_vi_account_id:
        with st.expander("🔧 Configuration Help", expanded=True):
            st.warning("Azure Video Indexer configuration is incomplete. Please set the following environment variables:")
            st.code("""
# Required environment variables:
AZURE_VI_SUBSCRIPTION_KEY=your_api_subscription_key_here
AZURE_VI_ACCOUNT_ID=your_account_id_here

# Optional (defaults to 'eastus'):
AZURE_VI_LOCATION=eastus
            """)
            st.info("**Step-by-step setup instructions:**")
            
            st.write("**1. Create an ARM-based Azure Video Indexer Account:**")
            st.write("   - Go to [Azure Portal](https://portal.azure.com)")
            st.write("   - Create a new 'Azure AI Video Indexer' resource")
            st.write("   - Note down your Account ID and Location from the resource overview")
            
            st.write("**2. Get API Subscription Key:**")
            st.write("   - Visit [Azure Video Indexer Developer Portal](https://api-portal.videoindexer.ai/)")
            st.write("   - Sign in with your Azure account (same one used for Azure Portal)")
            st.write("   - Click 'APIs' in the top menu")
            st.write("   - Subscribe to the API (if not already subscribed)")
            st.write("   - Go to 'Profile' in the top-right corner")
            st.write("   - Click 'Subscriptions' to see your subscription keys")
            st.write("   - Copy either the 'Primary key' or 'Secondary key'")
            
            st.write("**3. Configure Environment Variables:**")
            st.write("   - Add the keys to your `.env` file or environment variables")
            st.write("   - Restart your Streamlit app after adding the variables")
            
            st.error("**Important:** Classic Video Indexer accounts are deprecated. You must use an ARM-based account created through Azure Portal.")
        return
    
    # Main interface
    st.write("Upload a video to Azure Video Indexer for comprehensive AI analysis including:")
    st.write("• **Audio**: Transcription, speaker identification, sentiment analysis")
    st.write("• **Video**: Face detection, object recognition, scene analysis")
    st.write("• **Content**: Topic extraction, keyword identification, content moderation")
    
    # Video input methods
    input_method = st.radio(
        "Choose input method:",
        ["YouTube URL", "Upload File"],
        horizontal=True
    )
    
    video_path = None
    video_name = None
    
    if input_method == "YouTube URL":
        url = st.text_input(
            "YouTube video URL",
            placeholder="https://youtu.be/abc123XYZ",
            key="azure_vi_url",
        )
        
        if url.strip():
            try:
                vid = extract_video_id(url)
                video_name = f"youtube_{vid}"
                
                # Quality selection for Azure VI
                quality = st.selectbox(
                    "Video Quality (affects processing time & cost)",
                    ["best", "medium", "small", "audio"],
                    index=1,  # Default to medium
                    help="Lower quality = faster processing & lower cost"
                )
                
                if st.button("Download & Prepare Video", key="download_for_azure"):
                    with st.spinner("Downloading video..."):
                        try:
                            out_dir = REPORTS_DIR / vid
                            out_dir.mkdir(parents=True, exist_ok=True)
                            video_path = download_video(url, out_dir, quality=quality)
                            st.success(f"✅ Video downloaded: {video_path.name}")
                            st.session_state['azure_vi_video_path'] = str(video_path)
                            st.session_state['azure_vi_video_name'] = video_name
                        except Exception as e:
                            st.error(f"❌ Download failed: {e}")
                            return
            except ValueError as e:
                st.error(f"❌ Invalid YouTube URL: {e}")
    
    else:  # Upload File
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
            key="azure_vi_upload"
        )
        
        if uploaded_file is not None:
            video_name = uploaded_file.name.split('.')[0]
            
            # Save uploaded file temporarily
            temp_dir = REPORTS_DIR / "temp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            video_path = temp_dir / uploaded_file.name
            
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            st.session_state['azure_vi_video_path'] = str(video_path)
            st.session_state['azure_vi_video_name'] = video_name
    
    # Use video from session state if available
    if 'azure_vi_video_path' in st.session_state:
        video_path = Path(st.session_state['azure_vi_video_path'])
        video_name = st.session_state['azure_vi_video_name']
    
    # Analysis options
    if video_path and video_path.exists():
        st.write("---")
        st.subheader("📤 Upload to Azure Video Indexer")
        
        col1, col2 = st.columns(2)
        
        with col1:
            language = st.selectbox(
                "Video Language",
                ["English", "Spanish", "French", "German", "Italian", "Portuguese", "Chinese", "Japanese"],
                key="azure_vi_language"
            )
        
        with col2:
            wait_for_completion = st.checkbox(
                "Wait for processing to complete",
                value=True,
                help="If unchecked, will upload and return immediately"
            )
        
        # Processing options
        with st.expander("⚙️ Advanced Options", expanded=False):
            timeout_minutes = st.slider(
                "Processing timeout (minutes)",
                min_value=5,
                max_value=60,
                value=30,
                help="Maximum time to wait for processing"
            )
            
            privacy = st.selectbox(
                "Privacy Setting",
                ["Private", "Public"],
                help="Private videos are only visible to your account"
            )
        
        # Upload button
        if st.button("🚀 Upload & Analyze with Azure Video Indexer", type="primary"):
            try:
                vi = AzureVideoIndexer()
                # Check for existing processed video
                st.info("Checking for existing processed video...")
                existing_id = None
                try:
                    videos = vi.list_videos()
                    if isinstance(videos, list):
                        for v in videos:
                            if v.get('name') == video_name and v.get('state') == 'Processed':
                                existing_id = v['id']
                                break
                    else:
                        st.warning(f"Azure Video Indexer returned an error when listing videos: {videos}")
                except Exception as e:
                    st.warning(f"Could not check for existing videos: {e}")
                if existing_id:
                    st.success("This video has already been processed. Fetching insights...")
                    try:
                        insights = vi.get_video_insights(existing_id)
                        if isinstance(insights, dict):
                            # Try to use summarizedInsights if present
                            if 'summarizedInsights' in insights:
                                si = insights['summarizedInsights']
                                st.markdown(f"**Video Name:** {si.get('name', 'N/A')}")
                                st.markdown(f"**Duration:** {si.get('duration', {}).get('time', 'N/A')}")
                                if si.get('keywords'):
                                    st.markdown("**Keywords:** " + ", ".join([kw.get('name', '') for kw in si['keywords']]))
                                if si.get('faces'):
                                    st.markdown("**Faces Detected:** " + ", ".join([face.get('name', 'Unknown') for face in si['faces']]))
                                if si.get('brands'):
                                    st.markdown("**Brands Detected:** " + ", ".join([brand.get('name', '') for brand in si['brands']]))
                                if si.get('topics'):
                                    st.markdown("**Topics:** " + ", ".join([topic.get('name', '') for topic in si['topics']]))
                                if si.get('sentiments'):
                                    sentiments = si['sentiments']
                                    pos = sum(1 for s in sentiments if s.get('sentimentType') == 'Positive')
                                    neg = sum(1 for s in sentiments if s.get('sentimentType') == 'Negative')
                                    neu = len(sentiments) - pos - neg
                                    st.markdown(f"**Sentiment:** Positive: {pos}, Neutral: {neu}, Negative: {neg}")
                                if si.get('transcript'):
                                    st.markdown("**Transcript (first 5 segments):**")
                                    for t in si['transcript'][:5]:
                                        st.write(f"- {t.get('text', '')}")
                                # Add more fields as needed
                                if not any([si.get('keywords'), si.get('faces'), si.get('brands'), si.get('topics'), si.get('sentiments'), si.get('transcript')]):
                                    st.info("No detailed insights available. Here is the raw summary:")
                                    st.write(si)
                            else:
                                summary = format_insights_summary(insights)
                                st.markdown(summary)
                        else:
                            st.error(f"Azure Video Indexer returned an error: {insights}")
                            return
                    except Exception as e:
                        st.error(f"Failed to fetch insights for existing video: {e}")
                        st.write(insights)
                        return
                # If not found, proceed with upload and analysis
                with st.spinner("Uploading video to Azure Video Indexer..."):
                    try:
                        video_id = vi.upload_video(
                            video_path,
                            video_name=video_name,
                            language=language,
                            privacy=privacy
                        )
                    except AzureVideoIndexerError as e:
                        if 'ALREADY_EXISTS' in str(e):
                            st.error("This video content was already uploaded and is blocked by Azure's duplicate prevention. Please wait before re-uploading or use a different video.")
                            return
                        else:
                            st.error(f"❌ Azure Video Indexer error: {e}")
                            return
                st.success(f"✅ Video uploaded successfully! Video ID: `{video_id}`")
                if wait_for_completion:
                    with st.spinner(f"Processing video (this may take up to {timeout_minutes} minutes)..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        try:
                            start_time = time.time()
                            timeout_seconds = timeout_minutes * 60
                            while time.time() - start_time < timeout_seconds:
                                status = vi.get_video_status(video_id)
                                state = status.get('state', 'Unknown')
                                elapsed = time.time() - start_time
                                progress = min(elapsed / timeout_seconds, 0.95)
                                progress_bar.progress(progress)
                                status_text.text(f"Status: {state} ({elapsed:.0f}s elapsed)")
                                if state == 'Processed':
                                    progress_bar.progress(1.0)
                                    status_text.text("✅ Processing completed!")
                                    break
                                elif state == 'Failed':
                                    st.error("❌ Processing failed!")
                                    return
                                time.sleep(10)
                            insights = vi.get_video_insights(video_id)
                            if isinstance(insights, dict):
                                if 'summarizedInsights' in insights:
                                    si = insights['summarizedInsights']
                                    st.markdown(f"**Video Name:** {si.get('name', 'N/A')}")
                                    st.markdown(f"**Duration:** {si.get('duration', {}).get('time', 'N/A')}")
                                    if si.get('keywords'):
                                        st.markdown("**Keywords:** " + ", ".join([kw.get('name', '') for kw in si['keywords']]))
                                    if si.get('faces'):
                                        st.markdown("**Faces Detected:** " + ", ".join([face.get('name', 'Unknown') for face in si['faces']]))
                                    if si.get('brands'):
                                        st.markdown("**Brands Detected:** " + ", ".join([brand.get('name', '') for brand in si['brands']]))
                                    if si.get('topics'):
                                        st.markdown("**Topics:** " + ", ".join([topic.get('name', '') for topic in si['topics']]))
                                    if si.get('sentiments'):
                                        sentiments = si['sentiments']
                                        pos = sum(1 for s in sentiments if s.get('sentimentType') == 'Positive')
                                        neg = sum(1 for s in sentiments if s.get('sentimentType') == 'Negative')
                                        neu = len(sentiments) - pos - neg
                                        st.markdown(f"**Sentiment:** Positive: {pos}, Neutral: {neu}, Negative: {neg}")
                                    if si.get('transcript'):
                                        st.markdown("**Transcript (first 5 segments):**")
                                        for t in si['transcript'][:5]:
                                            st.write(f"- {t.get('text', '')}")
                                    if not any([si.get('keywords'), si.get('faces'), si.get('brands'), si.get('topics'), si.get('sentiments'), si.get('transcript')]):
                                        st.info("No detailed insights available. Here is the raw summary:")
                                        st.write(si)
                                else:
                                    summary = format_insights_summary(insights)
                                    st.markdown(summary)
                            else:
                                st.error(f"Azure Video Indexer returned an error: {insights}")
                        except AzureVideoIndexerError as e:
                            st.error(f"❌ Processing error: {e}")
                        except Exception as e:
                            st.error(f"❌ Unexpected error: {e}")
                else:
                    st.info(f"📤 Video uploaded with ID: `{video_id}`. Check back later for results.")
            except AzureVideoIndexerError as e:
                st.error(f"❌ Azure Video Indexer error: {e}")
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
