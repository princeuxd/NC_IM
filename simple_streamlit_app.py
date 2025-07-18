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


# Show environment status
st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Environment Status")

# Check API keys
yt_key = os.getenv("YT_API_KEY")
openrouter_key = os.getenv("OPENROUTER_API_KEY")
groq_key = os.getenv("GROQ_API_KEY")

if yt_key:
    st.sidebar.success("✅ YouTube API")
else:
    st.sidebar.warning("⚠️ YouTube API missing")

if openrouter_key:
    st.sidebar.success("✅ OpenRouter API")
else:
    st.sidebar.warning("⚠️ OpenRouter API missing")

if groq_key:
    st.sidebar.success("✅ Groq API")
else:
    st.sidebar.info("ℹ️ Groq API (optional)")

# .env file check
env_file = ROOT / ".env"
if env_file.exists():
    st.sidebar.success("✅ .env file")
else:
    st.sidebar.error("❌ .env file missing")

# Optional debug section
if st.sidebar.checkbox("Debug"):
    st.sidebar.code(f"Root: {ROOT}")
    st.sidebar.code(f"Keys: YT={'✓' if yt_key else '✗'} OR={'✓' if openrouter_key else '✗'} GQ={'✓' if groq_key else '✗'}")

st.sidebar.markdown("---")


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


def comments_analyzer_section():
    st.title("💬 Comments Analyzer")
    st.markdown("Analyze YouTube video comments for sentiment, engagement patterns, and audience insights.")

    st.sidebar.markdown("### 📊 Comments Analysis")
    st.sidebar.info("Deep dive into comment sentiment and engagement patterns")

    # Input section
    col1, col2 = st.columns([2, 1])
    with col1:
        url = st.text_input("YouTube video URL", placeholder="https://youtu.be/abc123XYZ", key="comments_url")
    with col2:
        api_key = st.text_input("YouTube API key", value=os.getenv("YT_API_KEY", ""), type="password", key="comments_api_key")

    # OpenRouter API for advanced sentiment analysis
    openrouter_key = st.text_input("OpenRouter API key (for advanced sentiment)", value=os.getenv("OPENROUTER_API_KEY", ""), type="password", key="comments_openrouter")

    if st.button("Analyze Comments", key="analyze_comments", type="primary"):
        if not url or not api_key:
            st.error("Please provide both video URL and YouTube API key.")
            return

        vid = extract_video_id(url)
        out_dir = REPORTS_DIR / vid
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build service
        service = get_public_service(api_key)

        # Fetch video info
        with st.spinner("Fetching video information..."):
            try:
                video_response = service.videos().list(part="snippet,statistics", id=vid).execute()
                if not video_response["items"]:
                    st.error("Video not found or is private")
                    return
                
                video_info = video_response["items"][0]
                video_snippet = video_info["snippet"]
                video_stats = video_info["statistics"]
                
                # Display video info
                st.subheader(f"📺 {video_snippet['title']}")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("👀 Views", f"{int(video_stats.get('viewCount', 0)):,}")
                with col2:
                    st.metric("👍 Likes", f"{int(video_stats.get('likeCount', 0)):,}")
                with col3:
                    st.metric("💬 Comments", f"{int(video_stats.get('commentCount', 0)):,}")
                
            except Exception as e:
                st.error(f"Failed to fetch video info: {e}")
                return

        # Fetch and analyze comments
        with st.spinner("Fetching and analyzing comments..."):
            try:
                from analysis import fetch_comments, analyze_comment_sentiment, save as save_json
                from analysis.sentiment_llm import analyze_comment_sentiment_llm
                
                # Fetch comments
                comments = fetch_comments(service, vid)
                
                if not comments:
                    st.warning("No comments found for this video.")
                    return
                
                # Analyze sentiment
                if openrouter_key:
                    run_settings = update_from_kwargs(openrouter_api_key=openrouter_key)
                    comments_sent = analyze_comment_sentiment_llm(comments)
                else:
                    comments_sent = analyze_comment_sentiment(comments)
                
                # Save results
                save_json(comments_sent, out_dir / "comments_sentiment.json")
                
                # Display results
                st.subheader("📊 Comment Analysis Results")
                
                # Sentiment overview
                sentiments = [c.get('sentiment', 0) for c in comments_sent if 'sentiment' in c]
                if sentiments:
                    avg_sentiment = sum(sentiments) / len(sentiments)
                    positive = sum(1 for s in sentiments if s > 0.1)
                    negative = sum(1 for s in sentiments if s < -0.1)
                    neutral = len(sentiments) - positive - negative
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("📈 Avg Sentiment", f"{avg_sentiment:.2f}")
                    with col2:
                        st.metric("😊 Positive", f"{positive}")
                    with col3:
                        st.metric("😐 Neutral", f"{neutral}")
                    with col4:
                        st.metric("😟 Negative", f"{negative}")
                
                # Sample comments
                st.subheader("🔍 Sample Comments with Sentiment")
                for i, comment in enumerate(comments_sent[:10]):
                    sentiment = comment.get('sentiment', 0)
                    if sentiment > 0.1:
                        emoji = "😊"
                        color = "green"
                    elif sentiment < -0.1:
                        emoji = "😟"
                        color = "red"
                    else:
                        emoji = "😐"
                        color = "gray"
                    
                    with st.expander(f"{emoji} Comment {i+1} (Sentiment: {sentiment:.2f})"):
                        st.markdown(f"**Author:** {comment.get('author', 'Unknown')}")
                        st.markdown(f"**Likes:** {comment.get('likeCount', 0)}")
                        st.markdown(f"**Text:** {comment.get('textDisplay', 'No text')}")
                        st.markdown(f"**Sentiment:** :{color}[{sentiment:.2f}]")
                
                st.success(f"✅ Analyzed {len(comments_sent)} comments successfully!")
                
            except Exception as e:
                st.error(f"Comment analysis failed: {e}")
                

def video_audio_analyzer_section():
    st.title("🎬 Video + Audio Analyzer")
    st.markdown("Generate comprehensive summary based on video content, audio transcription, and AI analysis.")

    st.sidebar.markdown("### 🎥 Video Analysis")
    st.sidebar.info("Generate executive summary from video and audio content")

    # Input section
    col1, col2 = st.columns([2, 1])
    with col1:
        url = st.text_input("YouTube video URL", placeholder="https://youtu.be/abc123XYZ", key="video_url")
    with col2:
        api_key = st.text_input("YouTube API key", value=os.getenv("YT_API_KEY", ""), type="password", key="video_api_key")

    # OpenRouter API for advanced analysis
    openrouter_key = st.text_input("OpenRouter API key (for AI summary)", value=os.getenv("OPENROUTER_API_KEY", ""), type="password", key="video_openrouter")

    if st.button("Generate Analysis & Summary", key="analyze_video", type="primary"):
        if not url or not api_key:
            st.error("Please provide both video URL and YouTube API key.")
            return

        vid = extract_video_id(url)
        out_dir = REPORTS_DIR / vid
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build service
        service = get_public_service(api_key)

        # Get video info and display
        with st.spinner("Fetching video information..."):
            try:
                video_response = service.videos().list(part="snippet,statistics", id=vid).execute()
                if not video_response["items"]:
                    st.error("Video not found or is private")
                    return
                
                video_info = video_response["items"][0]
                video_snippet = video_info["snippet"]
                video_stats = video_info["statistics"]
                channel_id = video_snippet["channelId"]
                
                # Fetch and display channel stats
                channel_stats = get_channel_stats(service, channel_id)
                display_channel_stats(channel_stats)
                
                # Display video info
                st.subheader(f"🎬 {video_snippet['title']}")
                
                # Video thumbnail and stats
                video_thumbnails = video_snippet.get("thumbnails", {})
                video_thumbnail = None
                for size in ["high", "medium", "default"]:
                    if size in video_thumbnails:
                        video_thumbnail = video_thumbnails[size]["url"]
                        break
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    if video_thumbnail:
                        # Ensure HTTPS for better compatibility
                        if video_thumbnail.startswith('http://'):
                            video_thumbnail = video_thumbnail.replace('http://', 'https://', 1)
                        st.image(video_thumbnail, width=200, caption="Video Thumbnail")
                
                with col2:
                    st.markdown(f"**Channel:** {video_snippet['channelTitle']}")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("👀 Views", f"{int(video_stats.get('viewCount', 0)):,}")
                    with col_b:
                        st.metric("👍 Likes", f"{int(video_stats.get('likeCount', 0)):,}")
                    with col_c:
                        st.metric("💬 Comments", f"{int(video_stats.get('commentCount', 0)):,}")
                    
                    # Format publish date
                    from datetime import datetime
                    try:
                        pub_date = datetime.fromisoformat(video_snippet["publishedAt"].replace('Z', '+00:00'))
                        st.markdown(f"**Published:** {pub_date.strftime('%B %d, %Y')}")
                    except Exception as e:
                        pass
                
            except Exception as e:
                st.error(f"Failed to fetch video info: {e}")
                return

        # Process video for comprehensive analysis
        with st.spinner("Processing video (downloading, transcribing, analyzing)..."):
            try:
                # Download video and extract audio
                process_video(url, service, analytics_service=None, channel_id=None, output_dir=out_dir)
                
                # Transcribe audio
                wav_path = out_dir / "audio.wav"
                if wav_path.exists():
                    from analysis import transcribe_audio, analyze_transcript_sentiment, save as save_json
                    segments = transcribe_audio(wav_path)
                    segments_sent = analyze_transcript_sentiment(segments)
                    save_json(segments_sent, out_dir / "transcript_sentiment.json")
                
                # Analyze comments
                from analysis import fetch_comments, analyze_comment_sentiment
                comments = fetch_comments(service, vid)
                if comments:
                    comments_sent = analyze_comment_sentiment(comments)
                    save_json(comments_sent, out_dir / "comments_sentiment.json")
                
                # Extract preview frames
                mp4_path = out_dir / f"{vid}.mp4"
                if mp4_path.exists():
                    preview_frames = extract_frames(mp4_path, out_dir / "frames_preview", every_sec=10)[:8]
                
                st.success("✅ Video processing completed!")
                
            except Exception as e:
                st.error(f"Video processing failed: {e}")
                return

        # Generate AI Summary
        if openrouter_key:
            with st.spinner("Generating AI summary..."):
                try:
                    run_settings = update_from_kwargs(openrouter_api_key=openrouter_key)
                    summary_md = generate_summary(out_dir, settings=run_settings)
                    
                    st.subheader("📋 Executive Summary")
                    if summary_md:
                        if summary_md.lstrip().startswith("<!"):
                            st.error("LLM provider returned an HTML error page – please verify your OpenRouter API key.")
                        else:
                            st.markdown(summary_md)
                    else:
                        st.error("Summary generation failed. Check your OpenRouter API key.")
                        
                except Exception as e:
                    st.error(f"Summary generation failed: {e}")
        else:
            st.info("💡 Add OpenRouter API key to generate AI summary")

        # Show preview frames
        if 'preview_frames' in locals() and preview_frames:
            st.subheader("🖼️ Key Video Frames")
            cols = st.columns(min(len(preview_frames), 4))
            for i, (timestamp, frame_path) in enumerate(preview_frames):
                with cols[i % 4]:
                    st.image(str(frame_path), caption=f"{timestamp:.1f}s", use_container_width=True)
                    
        st.success("🎉 Analysis complete! Check the sidebar for channel insights.")


def frame_analyzer_section():
    st.title("🖼️ Frame-by-Frame Analyzer")
    st.markdown("Detailed analysis of video frames with synchronized audio transcription and sentiment analysis.")

    st.sidebar.markdown("### 🔍 Frame Analysis")
    st.sidebar.info("Deep analysis of video frames with synchronized audio and transcript")

    # Input section
    col1, col2 = st.columns([2, 1])
    with col1:
        url = st.text_input("YouTube video URL", placeholder="https://youtu.be/abc123XYZ", key="frame_url")
    with col2:
        api_key = st.text_input("YouTube API key", value=os.getenv("YT_API_KEY", ""), type="password", key="frame_api_key")

    # OpenRouter API for vision analysis
    openrouter_key = st.text_input("OpenRouter API key (for vision analysis)", value=os.getenv("OPENROUTER_API_KEY", ""), type="password", key="frame_openrouter")

    # Analysis settings
    st.subheader("⚙️ Analysis Settings")
    col1, col2, col3 = st.columns(3)
    with col1:
        frame_interval = st.slider("Frame extraction interval (seconds)", 1, 30, 5)
    with col2:
        max_frames = st.slider("Maximum frames to analyze", 5, 50, 20)
    with col3:
        show_transcript = st.checkbox("Show detailed transcript", value=True)

    if st.button("Analyze Frames & Audio", key="analyze_frames", type="primary"):
        if not url or not api_key:
            st.error("Please provide both video URL and YouTube API key.")
            return

        vid = extract_video_id(url)
        out_dir = REPORTS_DIR / vid
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build service
        service = get_public_service(api_key)

        # Get and display video info
        with st.spinner("Fetching video information..."):
            try:
                video_response = service.videos().list(part="snippet,statistics", id=vid).execute()
                if not video_response["items"]:
                    st.error("Video not found or is private")
                    return
                
                video_info = video_response["items"][0]
                video_snippet = video_info["snippet"]
                video_stats = video_info["statistics"]
                
                st.subheader(f"🎬 {video_snippet['title']}")
                
                # Show basic stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("👀 Views", f"{int(video_stats.get('viewCount', 0)):,}")
                with col2:
                    st.metric("👍 Likes", f"{int(video_stats.get('likeCount', 0)):,}")
                with col3:
                    st.metric("💬 Comments", f"{int(video_stats.get('commentCount', 0)):,}")
                    
            except Exception as e:
                st.error(f"Failed to fetch video info: {e}")
                return

        # Process video and extract frames
        with st.spinner("Processing video and extracting frames..."):
            try:
                process_video(url, service, analytics_service=None, channel_id=None, output_dir=out_dir)
                
                # Extract frames
                mp4_path = out_dir / f"{vid}.mp4"
                frames_dir = out_dir / "frames_analysis"
                frames_dir.mkdir(exist_ok=True)
                
                if mp4_path.exists():
                    frames = extract_frames(mp4_path, frames_dir, every_sec=frame_interval)
                    frames = frames[:max_frames]  # Limit number of frames
                    st.success(f"✅ Extracted {len(frames)} frames for analysis")
                else:
                    st.error("Video file not found.")
                    return
                    
            except Exception as e:
                st.error(f"Video processing failed: {e}")
                return

        # Transcribe audio for detailed analysis
        transcript_segments = []
        wav_path = out_dir / "audio.wav"
        if wav_path.exists():
            with st.spinner("Transcribing and analyzing audio..."):
                try:
                    from analysis import transcribe_audio, analyze_transcript_sentiment, save as save_json
                    segments = transcribe_audio(wav_path)
                    transcript_segments = analyze_transcript_sentiment(segments)
                    save_json(transcript_segments, out_dir / "transcript_sentiment.json")
                    
                    if show_transcript:
                        # Show transcript overview
                        st.subheader("🎤 Audio Transcript Overview")
                        sentiments = [s.get('sentiment', 0) for s in transcript_segments if 'sentiment' in s]
                        if sentiments:
                            avg_sentiment = sum(sentiments) / len(sentiments)
                            positive = sum(1 for s in sentiments if s > 0.1)
                            negative = sum(1 for s in sentiments if s < -0.1)
                            neutral = len(sentiments) - positive - negative
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("📈 Avg Sentiment", f"{avg_sentiment:.2f}")
                            with col2:
                                st.metric("😊 Positive", f"{positive}")
                            with col3:
                                st.metric("😐 Neutral", f"{neutral}")
                            with col4:
                                st.metric("😟 Negative", f"{negative}")
                        
                        # Show detailed transcript segments
                        st.subheader("📝 Detailed Transcript Segments")
                        for i, segment in enumerate(transcript_segments):
                            start_time = segment.get('start', 0)
                            end_time = segment.get('end', 0)
                            text = segment.get('text', '')
                            sentiment = segment.get('sentiment', 0)
                            
                            if sentiment > 0.1:
                                emoji = "😊"
                                color = "green"
                            elif sentiment < -0.1:
                                emoji = "😟"
                                color = "red"
                            else:
                                emoji = "😐"
                                color = "gray"
                            
                            with st.expander(f"{emoji} {start_time:.1f}s - {end_time:.1f}s (Sentiment: {sentiment:.2f})"):
                                st.markdown(f"**Text:** {text}")
                                st.markdown(f"**Sentiment:** :{color}[{sentiment:.2f}]")
                                st.markdown(f"**Duration:** {end_time - start_time:.1f} seconds")
                        
                        st.success(f"✅ Transcribed and analyzed {len(transcript_segments)} segments!")
                        
                except Exception as e:
                    st.warning(f"Audio transcription failed: {e}")

        # Frame-by-frame analysis with vision AI
        if frames:
            st.subheader("🔍 Frame-by-Frame Visual Analysis")
            
            if openrouter_key:
                with st.spinner("Analyzing frames with AI vision..."):
                    try:
                        from analysis.object_detection import detect_objects
                        run_settings = update_from_kwargs(openrouter_api_key=openrouter_key)
                        
                        for i, (timestamp, frame_path) in enumerate(frames):
                            # Find corresponding audio segment
                            audio_text = ""
                            audio_sentiment = 0
                            for segment in transcript_segments:
                                if segment.get('start', 0) <= timestamp <= segment.get('end', 0):
                                    audio_text = segment.get('text', '')
                                    audio_sentiment = segment.get('sentiment', 0)
                                    break
                            
                            # Analyze frame with vision AI
                            with st.expander(f"🖼️ Frame {i+1} - {timestamp:.1f}s"):
                                col1, col2 = st.columns([1, 2])
                                
                                with col1:
                                    st.image(str(frame_path), caption=f"Frame at {timestamp:.1f}s")
                                
                                with col2:
                                    st.markdown(f"**⏱️ Timestamp:** {timestamp:.1f} seconds")
                                    
                                    # Show synchronized audio
                                    if audio_text:
                                        sentiment_emoji = "😊" if audio_sentiment > 0.1 else "😟" if audio_sentiment < -0.1 else "😐"
                                        st.markdown(f"**🎤 Audio:** {audio_text}")
                                        st.markdown(f"**😊 Sentiment:** {sentiment_emoji} {audio_sentiment:.2f}")
                                    else:
                                        st.markdown("**🎤 Audio:** No audio at this timestamp")
                                    
                                    # Vision AI object detection
                                    try:
                                        single_frame_list = [(timestamp, frame_path)]
                                        detections = detect_objects(single_frame_list, settings=run_settings)
                                        if detections and detections[0].get('objects'):
                                            st.markdown("**🔍 Detected Objects:**")
                                            for obj in detections[0]['objects']:
                                                label = obj.get('label', 'Unknown object')
                                                confidence = obj.get('confidence', 0)
                                                st.markdown(f"- {label} (confidence: {confidence:.2f})")
                                        else:
                                            st.markdown("**🔍 Detected Objects:** None detected")
                                    except Exception as e:
                                        st.markdown(f"**🔍 Object Detection:** Failed ({str(e)})")
                        
                        st.success(f"✅ Analyzed {len(frames)} frames with AI vision!")
                        
                    except Exception as e:
                        st.error(f"Frame analysis failed: {e}")
            else:
                st.warning("💡 Add OpenRouter API key to enable AI vision analysis")
                
                # Show frames without AI analysis
                for i, (timestamp, frame_path) in enumerate(frames):
                    audio_text = ""
                    audio_sentiment = 0
                    for segment in transcript_segments:
                        if segment.get('start', 0) <= timestamp <= segment.get('end', 0):
                            audio_text = segment.get('text', '')
                            audio_sentiment = segment.get('sentiment', 0)
                            break
                    
                    with st.expander(f"🖼️ Frame {i+1} - {timestamp:.1f}s"):
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.image(str(frame_path), caption=f"Frame at {timestamp:.1f}s")
                        
                        with col2:
                            st.markdown(f"**⏱️ Timestamp:** {timestamp:.1f} seconds")
                            if audio_text:
                                sentiment_emoji = "😊" if audio_sentiment > 0.1 else "😟" if audio_sentiment < -0.1 else "😐"
                                st.markdown(f"**🎤 Audio:** {audio_text}")
                                st.markdown(f"**😊 Sentiment:** {sentiment_emoji} {audio_sentiment:.2f}")
                            else:
                                st.markdown("**🎤 Audio:** No audio at this timestamp")
                            st.info("🔍 Enable AI vision analysis with OpenRouter API key")
        else:
            st.warning("No frames to analyze.")


# ---------------------------------------------------------------------------
# Main layout with tabs
# ---------------------------------------------------------------------------


TAB_ONBOARD, TAB_COMMENTS, TAB_VIDEO_AUDIO, TAB_FRAMES = st.tabs([
    "Creator Onboarding", 
    "Comments Analyzer", 
    "Video + Audio Analyzer", 
    "Frame-by-Frame Analyzer"
])

with TAB_ONBOARD:
    onboarding_section()

with TAB_COMMENTS:
    comments_analyzer_section()

with TAB_VIDEO_AUDIO:
    video_audio_analyzer_section()

with TAB_FRAMES:
    frame_analyzer_section() 