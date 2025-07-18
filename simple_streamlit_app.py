import os
from pathlib import Path
from typing import List

import streamlit as st

from auth import get_public_service, get_oauth_service
from video import extract_video_id, process_video, extract_frames
from analysis import transcribe_audio, analyze_transcript_sentiment, save as save_json
from analysis.summarizer import generate_summary
from config.settings import SETTINGS, update_from_kwargs
import shutil
from dotenv import load_dotenv  # type: ignore
import logging
import warnings

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress FP16 warning from Whisper
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

load_dotenv()

ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True, parents=True)

TOKENS_DIR = ROOT / "tokens"
TOKENS_DIR.mkdir(exist_ok=True, parents=True)

DEFAULT_CLIENT_SECRET = ROOT / "client_secret.json"


# ---------------------------------------------------------------------------
# Helper utilities (copied / trimmed from full app)
# ---------------------------------------------------------------------------


def _list_token_files():
    """Return sorted list of saved OAuth credential JSON files."""
    return sorted(TOKENS_DIR.glob("*.json"))


def _channel_info_from_token(token_file: Path):
    """Return (channel_id, channel_title) for a stored token file."""
    try:
        svc = get_oauth_service(DEFAULT_CLIENT_SECRET, token_file)
        resp = (
            svc.channels()  # type: ignore[attr-defined]
            .list(part="snippet", mine=True)
            .execute()
        )
        item = resp["items"][0]
        return item["id"], item["snippet"]["title"]
    except Exception:  # pylint: disable=broad-except
        return token_file.stem, "<unknown>"


def get_channel_stats(service, channel_id: str) -> dict:
    """Fetch channel statistics and snippet information."""
    try:
        response = service.channels().list(
            part="snippet,statistics",
            id=channel_id
        ).execute()
        
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
                            base_url = raw_url.split('=')[0]
                            thumbnail_url = f"{base_url}=s240-c-k-c0x00ffffff-no-rj"
                        except Exception as e:
                            thumbnail_url = raw_url
                    else:
                        thumbnail_url = raw_url
                    
                    break
            
            return {
                "title": channel["snippet"]["title"],
                "description": channel["snippet"]["description"],
                "thumbnail": thumbnail_url,
                "subscriber_count": int(channel["statistics"].get("subscriberCount", 0)),
                "video_count": int(channel["statistics"].get("videoCount", 0)),
                "view_count": int(channel["statistics"].get("viewCount", 0)),
                "published_at": channel["snippet"]["publishedAt"],
                "custom_url": channel["snippet"].get("customUrl", ""),
            }
    except Exception as e:
        logger.error(f"Failed to fetch channel stats: {e}")
    return {}


def display_channel_stats(channel_stats: dict):
    """Display channel statistics in the sidebar."""
    if not channel_stats:
        return
        
    st.sidebar.markdown("---")
    st.sidebar.subheader("📺 Channel Info")
    
    # Channel thumbnail and name
    thumbnail_url = channel_stats.get("thumbnail")
    if thumbnail_url:
        try:
            # Ensure HTTPS for better compatibility
            if thumbnail_url.startswith('http://'):
                thumbnail_url = thumbnail_url.replace('http://', 'https://', 1)
            
            st.sidebar.image(thumbnail_url, width=120, caption="Channel Avatar")
        except Exception as e:
            st.sidebar.markdown("🖼️ **Channel Avatar**")
    else:
        st.sidebar.info("No thumbnail available")
    
    st.sidebar.markdown(f"**{channel_stats.get('title', 'Unknown Channel')}**")
    if channel_stats.get("custom_url"):
        st.sidebar.markdown(f"@{channel_stats['custom_url']}")
    
    # Statistics
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📊 Channel Statistics**")
    
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
    
    # Create metrics in a more compact layout
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("👥 Subscribers", subscribers)
        st.metric("🎬 Videos", videos)
    with col2:
        st.metric("👀 Total Views", views)
        
        # Calculate average views per video
        if channel_stats.get("video_count", 0) > 0:
            avg_views = channel_stats.get("view_count", 0) // channel_stats["video_count"]
            st.metric("📈 Avg Views", format_number(avg_views))
    
    # Channel age
    from datetime import datetime
    try:
        published_at = channel_stats.get("published_at")
        if published_at:
            created_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
            years_old = (datetime.now(created_date.tzinfo) - created_date).days // 365
            if years_old > 0:
                st.sidebar.markdown(f"**📅 Channel Age:** {years_old} years")
            else:
                days_old = (datetime.now(created_date.tzinfo) - created_date).days
                st.sidebar.markdown(f"**📅 Channel Age:** {days_old} days")
    except Exception as e:
        logger.error(f"Failed to parse channel age: {e}")
    
    # Description preview
    if channel_stats.get("description"):
        st.sidebar.markdown("---")
        st.sidebar.markdown("**📝 About**")
        description = channel_stats["description"]
        preview = description[:150] + "..." if len(description) > 150 else description
        st.sidebar.markdown(f"_{preview}_")


# ---------------------------------------------------------------------------
# UI sections
# ---------------------------------------------------------------------------


def onboarding_section():
    st.header("🔑 Creator OAuth onboarding")

    # Existing creators table
    token_files = _list_token_files()
    if token_files:
        data = []
        for tf in token_files:
            cid, title = _channel_info_from_token(tf)
            data.append({"Channel": title, "Channel ID": cid, "Token file": tf.name})
        st.table(data)
    else:
        st.info("No creators onboarded yet.")

    st.divider()
    st.subheader("Add a new creator")

    uploaded_cs = st.file_uploader(
        "Client secret JSON (leave empty to use default client_secret.json)",
        type="json",
    )

    if uploaded_cs is not None:
        tmp_cs_path = TOKENS_DIR / "uploaded_client_secret.json"
        tmp_cs_path.write_bytes(uploaded_cs.read())
        client_secret_path = tmp_cs_path
    else:
        client_secret_path = DEFAULT_CLIENT_SECRET

    if not client_secret_path.exists():
        st.error("client_secret.json not found. Provide it via file uploader or place it in project root.")
        return

    if st.button("Start OAuth flow", key="oauth_start", type="primary"):
        with st.spinner("Opening Google OAuth consent screen… please complete authentication in the new tab."):
            temp_token = TOKENS_DIR / "_temp_creds.json"
            try:
                svc = get_oauth_service(client_secret_path, temp_token)
                me = svc.channels().list(part="id,snippet", mine=True).execute()  # type: ignore[attr-defined]
                info = me["items"][0]
                cid = info["id"]
                title = info["snippet"]["title"]
                final_token_path = TOKENS_DIR / f"{cid}.json"
                temp_token.replace(final_token_path)
                st.success(f"Onboarded channel '{title}' (ID: {cid})")
            except Exception as exc:  # pylint: disable=broad-except
                st.error(f"OAuth onboarding failed: {exc}")
                temp_token.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def summarize_section():
    st.title("🎬 YouTube Video Summarizer")
    st.markdown("Enter a YouTube URL and get a concise summary based on the audio transcript and key video frames. Ideal for non-technical users.")

    st.sidebar.markdown("### 🔍 Analysis Tools")
    st.sidebar.info("Channel info will appear here when you analyze a video")

    url = st.text_input("YouTube video URL", placeholder="https://youtu.be/abc123XYZ")
    api_key = st.text_input("YouTube Data API key", value=os.getenv("YT_API_KEY", ""), type="password")
    
    # Check for OpenRouter key in environment
    env_openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_key = st.text_input(
        "OpenRouter API key (optional)", 
        value=env_openrouter_key,
        type="password",
        help="Leave empty to use OPENROUTER_API_KEY from .env file"
    )
    


    if st.button("Summarize video", key="run_summary", type="primary"):
        if not url or not api_key:
            st.error("Please provide both the video URL and a YouTube Data API key.")
            st.stop()

        vid = extract_video_id(url)
        out_dir = REPORTS_DIR / vid
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build service
        service = get_public_service(api_key)
        
        # Get video metadata to extract channel ID
        with st.spinner("Fetching video information..."):
            try:
                video_response = service.videos().list(part="snippet", id=vid).execute()
                if video_response["items"]:
                    channel_id = video_response["items"][0]["snippet"]["channelId"]
                    video_snippet = video_response["items"][0]["snippet"]
                    
                    # Fetch and display channel stats in sidebar
                    channel_stats = get_channel_stats(service, channel_id)
                    
                    display_channel_stats(channel_stats)
                    
                    # Show video information
                    video_title = video_snippet["title"]
                    
                    # Get video thumbnail with fallback
                    video_thumbnails = video_snippet.get("thumbnails", {})
                    video_thumbnail = None
                    for size in ["high", "medium", "default"]:
                        if size in video_thumbnails:
                            video_thumbnail = video_thumbnails[size]["url"]
                            break
                    
                    # Display video info with thumbnail
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if video_thumbnail:
                            try:
                                # Ensure HTTPS for better compatibility
                                if video_thumbnail.startswith('http://'):
                                    video_thumbnail = video_thumbnail.replace('http://', 'https://', 1)
                                
                                st.image(video_thumbnail, width=150, caption="Video")
                            except Exception as e:
                                st.error(f"Failed to load video thumbnail: {e}")
                        else:
                            st.info("No video thumbnail available")
                    
                    with col2:
                        st.subheader(f"🎥 {video_title}")
                        st.markdown(f"**Channel:** {video_snippet['channelTitle']}")
                        
                        # Show channel thumbnail in main area
                        channel_thumbnail = channel_stats.get("thumbnail")
                        if channel_thumbnail:
                            try:
                                # Ensure HTTPS for better compatibility
                                if channel_thumbnail.startswith('http://'):
                                    channel_thumbnail = channel_thumbnail.replace('http://', 'https://', 1)
                                
                                st.image(channel_thumbnail, width=60, caption="Channel")
                            except Exception as e:
                                st.write("🖼️ Channel thumbnail unavailable")
                        
                        # Format publish date
                        from datetime import datetime
                        try:
                            pub_date = datetime.fromisoformat(video_snippet["publishedAt"].replace('Z', '+00:00'))
                            st.markdown(f"**Published:** {pub_date.strftime('%B %d, %Y')}")
                        except Exception as e:
                            logger.error(f"Failed to parse publish date: {e}")
                else:
                    st.error("Video not found or is private")
                    return
            except Exception as e:
                st.error(f"Failed to fetch video information: {e}")
                return
        
        # Process video
        with st.spinner("Processing video (downloading, transcribing, extracting frames)..."):
            # Download video and extract audio
            mp4_path = out_dir / f"{vid}.mp4"
            wav_path = out_dir / "audio.wav"
            process_video(url, service, analytics_service=None, channel_id=None, output_dir=out_dir)

        if wav_path.exists():
            with st.spinner("Transcribing audio…"):
                segments = transcribe_audio(wav_path)  # type: ignore
                segments_sent = analyze_transcript_sentiment(segments)
                save_json(segments_sent, out_dir / "transcript_sentiment.json")
        else:
            st.warning("Audio extraction failed – summary might be less accurate.")

        mp4_path = out_dir / f"{vid}.mp4"
        preview_frames: List[tuple[float, Path]] = []
        if mp4_path.exists():
            with st.spinner("Extracting preview frames…"):
                preview_frames = extract_frames(mp4_path, out_dir / "frames_preview", every_sec=10)[:8]  # type: ignore

        # Update settings with provided keys
        if openrouter_key:
            run_settings = update_from_kwargs(openrouter_api_key=openrouter_key)
        else:
            run_settings = SETTINGS
        
        # Generate summary with updated settings
        with st.spinner("Generating summary..."):
            summary_md = generate_summary(out_dir, settings=run_settings)
        
        st.subheader("Executive Summary")
        if summary_md:
            if summary_md.lstrip().startswith("<!"):
                st.error("LLM provider returned an HTML error page – please verify your OpenRouter API key and model route.")
            else:
                st.markdown(summary_md)
        else:
            if openrouter_key:
                st.error("Summary generation failed despite providing OpenRouter API key. Check the logs for details.")
            else:
                st.info("Summary generation skipped (no OpenRouter API key provided).")

        if preview_frames:
            st.subheader("Sample Frames")
            cols = st.columns(len(preview_frames))
            for col, (ts, img_path) in zip(cols, preview_frames):
                with col:
                    st.image(str(img_path), caption=f"{ts:.0f}s", use_container_width=True)


# ---------------------------------------------------------------------------
# Main layout with tabs
# ---------------------------------------------------------------------------


TAB_ONBOARD, TAB_SUMMARIZE = st.tabs(["Creator Onboarding", "Summarize Video"])

with TAB_ONBOARD:
    onboarding_section()

with TAB_SUMMARIZE:
    summarize_section() 