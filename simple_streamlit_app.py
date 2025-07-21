import os
from pathlib import Path
from typing import List

import streamlit as st

from auth import get_public_service, get_oauth_service
from video import extract_video_id, download_video, extract_frames
from analysis import transcribe_audio, analyze_transcript_sentiment, save as save_json
from analysis.summarizer import generate_summary
from config.settings import SETTINGS, update_from_kwargs
import shutil
from dotenv import load_dotenv  # type: ignore
import logging
import warnings
from pipeline.core import run_pipeline
from auth.manager import (
    list_token_files as _list_token_files,
    get_creator_details,
    validate_client_secret,
    onboard_creator,
    remove_creator as _remove_creator,
    refresh_creator_token,
)

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
    st.code(f"Keys: YT={'✓' if yt_key else '✗'} OR={'✓' if openrouter_key else '✗'} GQ={'✓' if groq_key else '✗'}")
# ---------------------------------------------


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
# Duplicated helper functions have been replaced by imports from ``auth.manager``


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
            if thumbnail_url.startswith('http://'):
                thumbnail_url = thumbnail_url.replace('http://', 'https://', 1)
            
                    st.image(thumbnail_url, width=120, caption="Channel Avatar")
        except Exception as e:
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
                preview = description[:150] + "..." if len(description) > 150 else description
                st.caption(f"_{preview}_")
    
    # Statistics
        st.markdown("**�� Channel Statistics**")
    
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
                    st.caption(f"**📅 Channel Age:** {years_old} years")
            else:
                days_old = (datetime.now(created_date.tzinfo) - created_date).days
                    st.caption(f"**📅 Channel Age:** {days_old} days")
    except Exception as e:
        logger.error(f"Failed to parse channel age: {e}")


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
        "private_access": False
    }
    
    for token_file in token_files:
        try:
            # Test OAuth service
            oauth_service = get_oauth_service(DEFAULT_CLIENT_SECRET, token_file)
            
            # Get channel info
            me = oauth_service.channels().list(part="id,snippet,statistics", mine=True).execute()
            if me["items"]:
                channel_info = me["items"][0]
                channel_id = channel_info["id"]
                
                # Test analytics access
                analytics_available = False
                try:
                    from googleapiclient.discovery import build
                    analytics_service = build("youtubeAnalytics", "v2", credentials=oauth_service._http.credentials)
                    # Test with a simple query
                    from datetime import date, timedelta
                    end_date = date.today()
                    start_date = end_date - timedelta(days=30)
                    
                    test_query = analytics_service.reports().query(
                        ids=f"channel=={channel_id}",
                        startDate=start_date.strftime('%Y-%m-%d'),
                        endDate=end_date.strftime('%Y-%m-%d'),
                        metrics="views",
                        maxResults=1
                    ).execute()
                    analytics_available = True
                except Exception:
                    analytics_available = False
                
                oauth_info["channels"].append({
                    "id": channel_id,
                    "title": channel_info["snippet"]["title"],
                    "token_file": token_file,
                    "analytics_access": analytics_available,
                    "subscriber_count": int(channel_info["statistics"].get("subscriberCount", 0)),
                    "video_count": int(channel_info["statistics"].get("videoCount", 0))
                })
                
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
            "video_owned": False
        }
    
    # Check if video belongs to any of our OAuth channels
    video_owned = False
    target_channel = None
    
    if video_id:
        try:
            video_response = public_service.videos().list(part="snippet", id=video_id).execute()
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
            oauth_service = get_oauth_service(DEFAULT_CLIENT_SECRET, target_channel["token_file"])
            
            # Build analytics service only if we own the video
            analytics_service = None
            if video_owned and target_channel["analytics_access"]:
                try:
                    from googleapiclient.discovery import build
                    analytics_service = build("youtubeAnalytics", "v2", credentials=oauth_service._http.credentials)
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
                "video_owned": video_owned
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
        "video_owned": False
    }

def display_access_level(service_info):
    """Display the current access level and capabilities."""
    access_level = service_info["access_level"]
    oauth_info = service_info["oauth_info"]
    video_owned = service_info.get("video_owned", False)
    
    if access_level == "oauth_full":
        st.success("🔓 **Full OAuth Access**: Your video - complete analytics and insights available")
        channel_info = service_info.get("channel_info", {})
        st.info(f"📊 Your Channel: **{channel_info.get('title', 'Unknown')}** ({channel_info.get('subscriber_count', 0):,} subscribers)")
    elif access_level == "oauth_basic":
        st.warning("🔐 **Basic OAuth Access**: Your video - limited analytics available")
        channel_info = service_info.get("channel_info", {})
        st.info(f"📊 Your Channel: **{channel_info.get('title', 'Unknown')}**")
    elif access_level == "oauth_enhanced":
        st.info("🔓 **Enhanced Access**: OAuth available for comments, public data for video")
        channel_info = service_info.get("channel_info", {})
        st.info(f"📊 OAuth Channel: **{channel_info.get('title', 'Unknown')}** (not video owner)")
    else:
        st.info("🔒 **Public Access**: Limited to public data only")
        if oauth_info["available"]:
            st.info(f"💡 **{len(oauth_info['channels'])} OAuth channels available** for enhanced comment access")

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
        video_analytics = analytics_service.reports().query(
            ids=f"channel=={channel_id}",
            startDate=start_date.strftime('%Y-%m-%d'),
            endDate=end_date.strftime('%Y-%m-%d'),
            metrics="views,estimatedMinutesWatched,averageViewDuration,subscribersGained",
            dimensions="video",
            filters=f"video=={video_id}",
            maxResults=1
        ).execute()
        
        # Get audience retention data
        retention_data = None
        try:
            retention_data = analytics_service.reports().query(
                ids=f"channel=={channel_id}",
                startDate=start_date.strftime('%Y-%m-%d'),
                endDate=end_date.strftime('%Y-%m-%d'),
                metrics="audienceWatchRatio",
                dimensions="elapsedVideoTimeRatio", 
                filters=f"video=={video_id}",
                maxResults=100
            ).execute()
        except Exception as e:
            logger.warning(f"Retention data failed: {e}")
        
        # Get traffic sources (simplified)
        traffic_sources = None
        try:
            traffic_sources = analytics_service.reports().query(
                ids=f"channel=={channel_id}",
                startDate=start_date.strftime('%Y-%m-%d'),
                endDate=end_date.strftime('%Y-%m-%d'),
                metrics="views",
                dimensions="insightTrafficSourceType",
                filters=f"video=={video_id}",
                maxResults=10
            ).execute()
        except Exception as e:
            logger.warning(f"Traffic sources failed: {e}")
        
        return {
            "video_analytics": video_analytics,
            "retention_data": retention_data,
            "traffic_sources": traffic_sources,
            "period": f"{start_date} to {end_date}"
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
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📈 Total Views", f"{int(row[1]):,}")
        with col2:
            st.metric("⏱️ Watch Time", f"{int(row[2]):,} min")
        with col3:
            st.metric("🎯 Avg Duration", f"{int(row[3]):,} sec")
        with col4:
            st.metric("📊 Subscribers", f"+{int(row[4]):,}")
    
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
                ax.plot(ratios, retention_rates, 'b-', linewidth=2)
                ax.set_xlabel('Video Progress (%)')
                ax.set_ylabel('Audience Retention (%)')
                ax.set_title(f'Audience Retention - {video_title}')
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, 100)
                
                # Save plot to bytes
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                img_buffer.seek(0)
                
                st.image(img_buffer, caption="Audience retention shows how well your content keeps viewers engaged")
                plt.close()
            except ImportError:
                st.error("Matplotlib not available for retention chart")
                # Show data as table instead
                retention_table = []
                for ratio, rate in zip(ratios[:10], retention_rates[:10]):  # Show first 10 points
                    retention_table.append({"Progress": f"{ratio:.1f}%", "Retention": f"{rate:.1f}%"})
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
                "SUBSCRIBER": "Subscribers"
            }
            
            friendly_name = source_map.get(source_type, source_type)
            traffic_sources.append({"Source": friendly_name, "Views": f"{views:,}"})
        
        # Display as table
        if traffic_sources:
            st.table(traffic_sources)
        else:
            st.info("No traffic source data available")
    
    st.caption(f"📅 Data period: {analytics_data['period']}")
    
    # Show what data is available
    data_available = []
    if video_data.get("rows"):
        data_available.append("✅ Video Performance")
    if retention_data and retention_data.get("rows"):
        data_available.append("✅ Audience Retention")
    if traffic_data and traffic_data.get("rows"):
        data_available.append("✅ Traffic Sources")
    
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
                order="relevance"
            )
            
            while request and len(comments) < 500:  # Limit to prevent timeout
                response = request.execute()
                
                for item in response["items"]:
                    top_comment = item["snippet"]["topLevelComment"]["snippet"]
                    comment_data = {
                        "author": top_comment["authorDisplayName"],
                        "authorChannelId": top_comment.get("authorChannelId", {}).get("value"),
                        "textDisplay": top_comment["textDisplay"],
                        "likeCount": top_comment["likeCount"],
                        "publishedAt": top_comment["publishedAt"],
                        "updatedAt": top_comment["updatedAt"],
                        "totalReplyCount": item["snippet"]["totalReplyCount"]
                    }
                    
                    # Add reply data if available
                    if "replies" in item:
                        comment_data["replies"] = []
                        for reply in item["replies"]["comments"]:
                            reply_data = reply["snippet"]
                            comment_data["replies"].append({
                                "author": reply_data["authorDisplayName"],
                                "textDisplay": reply_data["textDisplay"],
                                "likeCount": reply_data["likeCount"],
                                "publishedAt": reply_data["publishedAt"]
                            })
                    
                    comments.append(comment_data)
                
                request = oauth_service.commentThreads().list_next(request, response)
            
            logger.info(f"OAuth: Successfully fetched {len(comments)} comments with full data")
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
                    order="time"  # Use time order as fallback
                )
                
                response = request.execute()
                for item in response.get("items", []):
                    top_comment = item["snippet"]["topLevelComment"]["snippet"]
                    comment_data = {
                        "author": top_comment["authorDisplayName"],
                        "authorChannelId": top_comment.get("authorChannelId", {}).get("value"),
                        "textDisplay": top_comment["textDisplay"],
                        "likeCount": top_comment["likeCount"],
                        "publishedAt": top_comment["publishedAt"],
                        "updatedAt": top_comment["updatedAt"],
                        "totalReplyCount": item["snippet"]["totalReplyCount"]
                    }
                    comments.append(comment_data)
                
                logger.info(f"OAuth: Successfully fetched {len(comments)} comments without replies")
                return comments
                
            except Exception as e2:
                logger.warning(f"OAuth comment fetching (basic) failed: {e2}, falling back to public API")
    
    # Fallback to public API
    try:
        from analysis import fetch_comments
        public_comments = fetch_comments(public_service, video_id)
        
        # Standardize field names - convert 'text' to 'textDisplay' for consistency
        standardized_comments = []
        for comment in public_comments:
            standardized_comment = {
                "author": comment.get("author", "Unknown"),
                "textDisplay": comment.get("text", ""),  # Convert 'text' to 'textDisplay'
                "likeCount": comment.get("likeCount", 0),
                "publishedAt": comment.get("publishedAt", ""),
                "totalReplyCount": 0  # Public API doesn't include reply count
            }
            standardized_comments.append(standardized_comment)
        
        logger.info(f"Public API: Successfully fetched {len(standardized_comments)} comments")
        return standardized_comments
        
    except Exception as e:
        logger.error(f"Public comment fetching failed: {e}")
        return []


# ---------------------------------------------------------------------------
# UI sections
# ---------------------------------------------------------------------------


def onboarding_section():
    """Modern creator onboarding interface with comprehensive management features."""

    # Page header with stats
    col_header, col_stats = st.columns([3, 1])
    with col_header:
        st.title("🎯 Creator Management Hub")
        st.markdown("Manage YouTube creator OAuth credentials and monitor authentication status")
    
    with col_stats:
    token_files = _list_token_files()
        st.metric("🔑 Active Creators", len(token_files))
    
    # Tab navigation for better organization
    tab_creators, tab_onboard = st.tabs(["👥 Manage Creators", "➕ Add New Creator"])
    
    # =================== CREATORS MANAGEMENT TAB ===================
    with tab_creators:
        if not token_files:
            st.info("🚀 **Get started** by adding your first creator in the 'Add New Creator' tab!")
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
                        creator_list.append({
                            "channel_id": details["channel_id"],
                            "title": details["title"],
                            "is_valid": details["is_valid"],
                            "last_checked": details["last_checked"]
                        })
                    
                    import json
                    json_str = json.dumps(creator_list, indent=2)
                    st.download_button(
                        "💾 Download creators.json",
                        json_str,
                        "creators.json",
                        "application/json"
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
                    col_avatar, col_info, col_stats, col_actions = st.columns([1, 3, 2, 2])
                    
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
                            
                            st.metric("👥 Subscribers", format_number(details["subscriber_count"]))
                            st.caption(f"📹 {format_number(details['video_count'])} videos")
                        else:
                            st.markdown("⚠️ **Token Invalid**")
                            if "error" in details:
                                st.caption(f"Error: {details['error']}")
                    
                    with col_actions:
                        action_col1, action_col2 = st.columns(2)
                        
                        with action_col1:
                            if not details["is_valid"]:
                                if st.button("🔄", key=f"refresh_{details['channel_id']}", help="Refresh token"):
                                    if refresh_creator_token(details["channel_id"]):
                                        st.success("Token refreshed!")
                                        st.rerun()
                                    else:
                                        st.error("Failed to refresh token")
                            else:
                                st.button("✅", disabled=True, help="Token is valid")
                        
                        with action_col2:
                            if st.button("🗑️", key=f"remove_{details['channel_id']}", help="Remove creator"):
                                # Confirmation dialog using session state
                                confirm_key = f"confirm_remove_{details['channel_id']}"
                                if confirm_key not in st.session_state:
                                    st.session_state[confirm_key] = False
                                
                                if not st.session_state[confirm_key]:
                                    st.session_state[confirm_key] = True
                                    st.warning(f"⚠️ Click again to confirm removal of **{details['title']}**")
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
                label_visibility="visible"
            )
                
        # Determine client secret path and validate
        if uploaded_cs is not None:
            tmp_cs_path = TOKENS_DIR / "uploaded_client_secret.json"
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
                st.code(f"""
Project ID: {validation_result.get('project_id', 'N/A')}
Client ID: {validation_result.get('client_id', 'N/A')[:20]}...
Type: {validation_result.get('type', 'N/A')}
File: {client_secret_path}
                """)
                
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
                help="Opens Google OAuth consent screen in a new tab"
            )
        
        with col_info:
            st.info("""
            **What happens next:**
            1. 🌐 Google OAuth page opens in new tab
            2. 📋 Select your YouTube channel
            3. ✅ Grant required permissions
            4. 🎉 Channel gets added to your dashboard
            """)
        
        if oauth_btn:
            with st.spinner("🔄 Initiating OAuth flow... Please complete authentication in the new browser tab."):
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
                    st.success(f"""
                    🎉 **Successfully onboarded!**
                    
                    **Channel:** {title}  
                    **ID:** `{cid}`  
                    **Token:** `{_token_path.name}`
                    """)
                
                    # Auto-switch to creators tab
                    st.info("💡 Switch to the 'Manage Creators' tab to see your new creator account!")
                    
                except Exception as exc:
                    st.error(f"""
                    ❌ **OAuth onboarding failed**
                    
                    **Error:** {str(exc)}
                    
                    **Troubleshooting:**
                    - Ensure your client_secret.json is valid
                    - Check that you have the required YouTube channel permissions
                    - Verify your Google Cloud Console OAuth setup
                    """)
                    
                    # Show detailed error in expander for debugging
                    with st.expander("🔧 Debug Information"):
                        st.code(f"Error Type: {type(exc).__name__}\nError Message: {str(exc)}")


def video_audio_analyzer_section():
    st.title("🎬 Video + Audio Analyzer")
    st.markdown("Generate a comprehensive analysis and summary of a YouTube video.")

    # --- Inputs ---
    url = st.text_input("YouTube video URL", placeholder="https://youtu.be/abc123XYZ", key="video_url")
    api_key = st.text_input("YouTube API key", value=os.getenv("YT_API_KEY", ""), type="password", key="video_api_key")
    openrouter_key = st.text_input("OpenRouter API key (required for summary)", value=os.getenv("OPENROUTER_API_KEY", ""), type="password", key="video_openrouter")
    
    # New slider for frame count
    num_frames = st.slider("Number of frames for summary (more frames = more detail & cost)", 0, 20, 5, key="num_frames")

    if st.button("🚀 Generate Full Analysis", key="analyze_video", type="primary"):
        if not url or not api_key:
            st.error("Please provide both a video URL and a YouTube API key.")
            return

        vid = extract_video_id(url)
        out_dir = REPORTS_DIR / vid

        # --- Progress Display ---
        with st.status("🚀 Kicking off analysis...", expanded=True) as status:
            try:
                # Use a dedicated callback to update the status
                def progress_callback(message: str):
                    status.update(label=message)

                # Find the best available token file if multiple exist
                token_files = _list_token_files()
                token_file = token_files[0] if token_files else None

                # Run the full pipeline
                video_folder = run_pipeline(
                    url=url,
                    output_dir=REPORTS_DIR,
                    public_api_key=api_key,
                    client_secrets_file=str(DEFAULT_CLIENT_SECRET),
                    token_file=str(token_file) if token_file else None,
                    num_frames_for_summary=num_frames,
                    progress_callback=progress_callback,
                )
                status.update(label="✅ Analysis Complete!", state="complete", expanded=False)

            except Exception as e:
                status.update(label=f"🚨 Pipeline Failed: {e}", state="error")
                return

        # --- Display Results ---
        st.subheader("🎬 Analysis Results")
        
        # Display summary
        summary_path = video_folder / "summary.md"
        if summary_path.exists():
            st.markdown("### 📋 Executive Summary")
            summary_md = summary_path.read_text()
            st.markdown(summary_md)
        else:
            st.warning("Summary could not be generated.")

        # Display preview of frames used in summary
        summary_frames_dir = video_folder / "summary_frames"
        if summary_frames_dir.exists():
            st.markdown("### 🖼️ Frames Analyzed in Summary")
            frames = sorted(summary_frames_dir.glob("*.jpg"))
            if frames:
                st.image([str(f) for f in frames], width=150)
            else:
                st.info("No frames were analyzed for this summary.")
        
        # Display sentiment analysis from comments
        comments_path = video_folder / "comments_sentiment.json"
        if comments_path.exists():
            st.markdown("### 💬 Comment Sentiment")
            import json
            comments = json.loads(comments_path.read_text())
            sentiments = [c.get('sentiment', 0) for c in comments]
            if sentiments:
                avg_sentiment = sum(sentiments) / len(sentiments)
                positive = sum(1 for s in sentiments if s > 0.1)
                negative = sum(1 for s in sentiments if s < -0.1)
                neutral = len(sentiments) - positive - negative
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Avg Sentiment", f"{avg_sentiment:.2f}")
                col2.metric("Positive", f"{positive}")
                col3.metric("Neutral", f"{neutral}")
                col4.metric("Negative", f"{negative}")


def frame_analyzer_section():
    st.title("🖼️ Frame-by-Frame Analyzer")
    st.markdown("Detailed analysis with OAuth-enhanced transcript and comment data when available.")

    st.info("💡 OAuth enhancement automatically detected for deeper insights")

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

        # Get enhanced service with OAuth detection
        service_info = get_enhanced_service(api_key, vid)
        display_access_level(service_info)

        # Get and display video info
        with st.spinner("Fetching video information..."):
            try:
                service = service_info["public_service"]
                video_response = service.videos().list(part="snippet,statistics", id=vid).execute()
                if not video_response["items"]:
                    st.error("Video not found or is private")
                    return
                
                video_info = video_response["items"][0]
                video_snippet = video_info["snippet"]
                video_stats = video_info["statistics"]
                
                st.subheader(f"🎬 {video_snippet['title']}")
                
                # Enhanced analytics if available
                analytics_data = get_enhanced_analytics(service_info, vid)
                if analytics_data:
                    display_enhanced_analytics(analytics_data, video_snippet['title'])
                else:
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
                # process_video(url, service_info["public_service"], analytics_service=service_info["analytics_service"], 
                #             channel_id=service_info.get("channel_id"), output_dir=out_dir)
                mp4_path = download_video(url, out_dir)
                
                # Extract frames
                # mp4_path = out_dir / f"{vid}.mp4" - This is now returned by download_video
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
            
        # Show access level summary
        access_level = service_info["access_level"]
        if access_level == "oauth_full":
            st.success("🎉 Analysis complete with full OAuth access! Enhanced transcript and analytics available.")
        elif access_level == "oauth_enhanced":
            st.info("🎉 Analysis complete with enhanced OAuth access for detailed insights!")
        else:
            st.info("🎉 Analysis complete with public API access!")


# ---------------------------------------------------------------------------
# Main layout with tabs
# ---------------------------------------------------------------------------


TAB_ONBOARD, TAB_VIDEO_AUDIO, TAB_FRAMES = st.tabs([
    "Creator Onboarding", 
    "Video + Audio Analyzer", 
    "Frame-by-Frame Analyzer"
])

with TAB_ONBOARD:
    onboarding_section()

with TAB_VIDEO_AUDIO:
    video_audio_analyzer_section()

with TAB_FRAMES:
    frame_analyzer_section() 