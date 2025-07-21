import os
import json
from pathlib import Path

import streamlit as st
from youtube.oauth import get_service as get_oauth_service

from youtube.public import get_service as get_public_service
from analysis.video_frames import (
    extract_video_id,
    download_video,
    extract_frames,
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
)
from analysis.video_vision import summarise_frames

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

    if st.button("Run Audio Analysis", key="run_audio"):
        if not url.strip():
            st.error("Please enter a YouTube URL")
            return

        vid = extract_video_id(url)
        out_dir = REPORTS_DIR / vid
        out_dir.mkdir(parents=True, exist_ok=True)

        with st.spinner("Downloading video..."):
            try:
                mp4_path = download_video(url, out_dir)
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

        if not SETTINGS.openrouter_api_key:
            st.warning("OpenRouter API key not configured – cannot generate summary.")
            return

        with st.spinner("Generating summary via LLM …"):
            client = get_client("openrouter", SETTINGS.openrouter_api_key)
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
                    model=SETTINGS.openrouter_chat_model,
                    temperature=0.3,
                    max_tokens=512,
                )
            except Exception as e:
                # Fallback to Groq if key present
                if SETTINGS.groq_api_key:
                    g_client = get_client("groq", SETTINGS.groq_api_key)
                    summary = g_client.chat(
                        [
                            {"role": "system", "content": prompt_msg},
                            {"role": "user", "content": full_text[:12000]},
                        ],
                        temperature=0.3,
                        max_tokens=512,
                    )
                else:
                    st.error(f"LLM call failed: {e}")
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

    every_sec = st.slider("Frame interval (seconds)", 1, 30, 5)
    max_frames = st.slider("Max frames to send", 4, 16, 8)

    if st.button("Summarise Video", key="run_video"):
        if not SETTINGS.openrouter_api_key:
            st.error("OPENROUTER_API_KEY not configured in environment.")
            return

        if not url.strip():
            st.error("Please enter a YouTube URL")
            return

        vid = extract_video_id(url)
        out_dir = REPORTS_DIR / vid
        out_dir.mkdir(parents=True, exist_ok=True)

        with st.spinner("Downloading video …"):
            try:
                mp4_path = download_video(url, out_dir)
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


def display_enhanced_analytics(analytics_data, video_title):
    """Display enhanced analytics data in a beautiful format."""
    if not analytics_data:
        return

    st.subheader("📊 Enhanced Analytics (OAuth)")

    # Video performance metrics
    video_data = analytics_data["video_analytics"]
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
    retention_data = analytics_data["retention_data"]
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
            row = {"Age Group": age_group}
            row.update(age_map[age_group])
            demo_table.append(row)

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
        from analysis import fetch_comments

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

        # Step 1: Client Secret Upload/Validation
        st.markdown("### Step 1: Client Secret Configuration")

        col_upload, col_validation = st.columns([2, 1])

        with col_upload:
            uploaded_cs = st.file_uploader(
                "📄 Upload client_secret.json",
                type="json",
                key="client_secret_uploader",
                help="🔐 Your Google Cloud Console OAuth 2.0 client secret file. Leave empty to use default file in project root.",
                label_visibility="visible",
            )

        # Determine client secret path and validate
        if uploaded_cs is not None:
            tmp_cs_path = ROOT / "uploaded_client_secret.json"  # store outside tokens to avoid being treated as a credential file
            tmp_cs_path.write_bytes(uploaded_cs.read())
            client_secret_path = tmp_cs_path
        else:
            client_secret_path = DEFAULT_CLIENT_SECRET

        # Validation and status display
        validation_result = validate_client_secret(client_secret_path)

        with col_validation:
            if validation_result["valid"]:
                st.success("✅ Valid Client Secret")
                st.caption(f"📂 {client_secret_path.name}")
                if validation_result.get("project_id"):
                    st.caption(f"🏗️ Project: {validation_result['project_id']}")
                st.caption(f"🔧 Type: {validation_result.get('type', 'unknown')}")
            else:
                st.error("❌ Invalid/Missing")
                st.caption(validation_result["error"])

        # Show detailed validation info in expander
        if validation_result["valid"]:
            with st.expander("🔍 Client Secret Details"):
                st.code(
                    f"""
Project ID: {validation_result.get('project_id', 'N/A')}
Client ID: {validation_result.get('client_id', 'N/A')[:20]}...
Type: {validation_result.get('type', 'N/A')}
File: {client_secret_path}
                """
                )

        # Step 2: OAuth Flow
        st.markdown("### Step 2: OAuth Authentication")

        if not validation_result["valid"]:
            st.warning("⚠️ Please provide a valid client_secret.json file to proceed")
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

        if oauth_btn:
            with st.spinner(
                "🔄 Initiating OAuth flow... Please complete authentication in the new browser tab."
            ):
                try:
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Simulate progress for better UX
                    import time

                    status_text.text("⏳ Opening OAuth consent screen...")
                    progress_bar.progress(25)
                    time.sleep(0.5)

                    status_text.text("🔐 Waiting for authentication...")
                    progress_bar.progress(50)

                    # Actual OAuth call
                    _token_path, cid, title = onboard_creator(client_secret_path)

                    progress_bar.progress(75)
                    status_text.text("📊 Fetching channel information...")
                    time.sleep(0.5)

                    progress_bar.progress(100)
                    status_text.text("✅ Success!")

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
                    - Ensure your client_secret.json is valid
                    - Check that you have the required YouTube channel permissions
                    - Verify your Google Cloud Console OAuth setup
                    """
                    )

                    # Show detailed error in expander for debugging
                    with st.expander("🔧 Debug Information"):
                        st.code(
                            f"Error Type: {type(exc).__name__}\nError Message: {str(exc)}"
                        )


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
        views = fmt_num(stats.get("viewCount"))
        likes = fmt_num(stats.get("likeCount"))
        comments_cnt = fmt_num(stats.get("commentCount"))
        favorites = fmt_num(stats.get("favoriteCount"))

        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("👀 Views", views)
        mcol2.metric("👍 Likes", likes)
        mcol3.metric("💬 Comments", comments_cnt)
        mcol4.metric("⭐ Favourites", favorites)

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
        analytics_data = get_enhanced_analytics(service_info, vid)
        if analytics_data:
            display_enhanced_analytics(analytics_data, snippet.get("title", "Video"))
        else:
            st.info("Only public metrics available – OAuth analytics not accessible for this video.")


# ---------------------------------------------------------------------------
# App entry point – single page
# ---------------------------------------------------------------------------


def main():
    st.set_page_config(page_title="YouTube Creator OAuth Manager", layout="wide")
    tab_onboard, tab_audio, tab_video, tab_stats = st.tabs([
        "Creator Onboarding",
        "Audio Analyzer",
        "Video Analyzer",
        "Video Statistics",
    ])

    with tab_onboard:
        onboarding_section()

    with tab_audio:
        audio_analyzer_section()

    with tab_video:
        video_analyzer_section()

    with tab_stats:
        video_statistics_section()


if __name__ == "__main__":
    main()
