"""
Channel Analysis Business Logic Module

This module contains all the business logic for the Final Channel Stat section,
separated from the Streamlit UI code for better maintainability and testability.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone
from collections import Counter

import pandas as pd

from src.youtube.oauth import get_service as get_oauth_service
from src.youtube.public import get_service as get_public_service, extract_channel_id_from_url
from src.analysis.video_frames import (
    extract_video_id,
    download_video,
    extract_frames,
    get_video_duration_from_url,
    auto_select_video_quality,
)
from src.analysis.audio import extract_audio, transcribe as transcribe_audio
from src.analysis.video_vision import summarise_frames
from src.llms import get_smart_client
from src.prompts.audio_analysis import get_enhanced_audio_analysis_prompt
from src.prompts.video_summary import get_channel_collective_analysis_prompt
from src.config.settings import SETTINGS
from src.auth.manager import list_token_files as _list_token_files

logger = logging.getLogger(__name__)

# Constants
ROOT = Path(__file__).resolve().parent.parent.parent 
REPORTS_DIR = ROOT / "data" / "reports"
REPORTS_DIR.mkdir(exist_ok=True, parents=True)

TOKENS_DIR = ROOT / "data" / "tokens"
TOKENS_DIR.mkdir(exist_ok=True, parents=True)

DEFAULT_CLIENT_SECRET = ROOT / "client_secret.json"


class ChannelAnalysisService:
    """Service class for channel analysis business logic."""
    
    def __init__(self, yt_api_key: str, enable_brand_analysis: bool = False):
        self.yt_api_key = yt_api_key
        self.enable_brand_analysis = enable_brand_analysis
        self.oauth_info = self._detect_oauth_capabilities()
        
        # Initialize brand-focused service if enabled
        self.brand_service = None
        if enable_brand_analysis:
            try:
                from src.analysis.brand_focused_channel_analysis import BrandFocusedChannelAnalysisService
                self.brand_service = BrandFocusedChannelAnalysisService(yt_api_key)
                logger.info("Brand-focused analysis enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize brand analysis: {e}")
                self.enable_brand_analysis = False
    
    def _detect_oauth_capabilities(self) -> Dict:
        """Detect available OAuth credentials and their capabilities."""
        token_files = _list_token_files()
        logger.info(f"Found {len(token_files)} OAuth token files: {[tf.name for tf in token_files]}")
        
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
                    logger.info(f"Found OAuth channel: {channel_info['snippet']['title']} ({channel_id})")

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
                        logger.info(f"Analytics access confirmed for {channel_id}")
                    except Exception as e:
                        analytics_available = False
                        logger.warning(f"Analytics access failed for {channel_id}: {e}")

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
    
    def extract_channel_id(self, channel_input: str) -> Optional[str]:
        """Extract channel ID from various input formats."""
        if not channel_input.strip():
            return None
            
        # Direct channel ID
        if channel_input.startswith("UC") and len(channel_input) == 24:
            return channel_input
        
        # Try to extract from URL
        try:
            return extract_channel_id_from_url(channel_input)
        except Exception:
            return None
    
    def get_service_for_channel(self, channel_id: str) -> Tuple[object, str]:
        """Get the appropriate service (OAuth or public) for a channel."""
        has_oauth = any(ch["id"] == channel_id for ch in self.oauth_info.get("channels", []))
        
        if has_oauth:
            matching_channel = next(ch for ch in self.oauth_info["channels"] if ch["id"] == channel_id)
            try:
                service = get_oauth_service(DEFAULT_CLIENT_SECRET, matching_channel["token_file"])
                return service, "oauth"
            except Exception as e:
                logger.error(f"OAuth service failed: {e}")
                raise
        else:
            try:
                service = get_public_service(self.yt_api_key)
                return service, "public"
            except Exception as e:
                logger.error(f"Public service failed: {e}")
                raise
    
    def get_channel_info(self, service: object, channel_id: str) -> Dict:
        """Fetch channel information and statistics."""
        try:
            channel_response = service.channels().list(
                part="snippet,statistics",
                id=channel_id
            ).execute()
            
            if not channel_response["items"]:
                raise ValueError("Channel not found")
            
            return channel_response["items"][0]
        except Exception as e:
            logger.error(f"Failed to fetch channel info: {e}")
            raise
    
    def get_channel_videos(self, service: object, channel_id: str, max_videos: int = 50) -> List[Dict]:
        """Fetch videos from a channel."""
        try:
            # Get uploads playlist ID
            channel_response = service.channels().list(
                part="contentDetails",
                id=channel_id
            ).execute()
            
            if not channel_response["items"]:
                return []
            
            uploads_playlist_id = channel_response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
            
            # Get videos from uploads playlist
            videos = []
            next_page_token = None
            
            while len(videos) < max_videos:
                playlist_response = service.playlistItems().list(
                    part="snippet",
                    playlistId=uploads_playlist_id,
                    maxResults=min(50, max_videos - len(videos)),
                    pageToken=next_page_token
                ).execute()
                
                for item in playlist_response["items"]:
                    video_id = item["snippet"]["resourceId"]["videoId"]
                    video_title = item["snippet"]["title"]
                    published_at = item["snippet"]["publishedAt"]
                    
                    videos.append({
                        "video_id": video_id,
                        "title": video_title,
                        "published_at": published_at
                    })
                
                next_page_token = playlist_response.get("nextPageToken")
                if not next_page_token:
                    break
            
            return videos[:max_videos]
            
        except Exception as e:
            logger.error(f"Failed to fetch channel videos: {e}")
            return []
    
    def process_single_video(self, video_id: str, video_title: str, output_base_dir: Path) -> Dict:
        """Process a single video completely - download, analyze audio+video, generate summary, cleanup."""
        
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        video_dir = output_base_dir / video_id
        video_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if video has already been processed successfully
        summary_file = video_dir / f"{video_id}_summary.md"
        json_file = video_dir / f"{video_id}_data.json"
        
        if summary_file.exists() and json_file.exists():
            logger.info(f"Video {video_id} already processed, skipping...")
            return {
                "video_id": video_id,
                "title": video_title,
                "success": True,
                "audio_analysis": "Already processed",
                "video_analysis": "Already processed",
                "structured_data": None,
                "error": None,
                "skipped": True
            }
        
        result = {
            "video_id": video_id,
            "title": video_title,
            "success": False,
            "audio_analysis": None,
            "video_analysis": None,
            "structured_data": None,
            "error": None,
            "skipped": False
        }
        
        try:
            # Step 1: Download video with appropriate quality
            duration_minutes = get_video_duration_from_url(video_url)
            quality = auto_select_video_quality(duration_minutes)
            
            mp4_path = download_video(video_url, video_dir, quality=quality)
            
            # Step 2: Enhanced Audio Analysis
            # Extract audio
            wav_path = extract_audio(mp4_path, video_dir / "audio.wav")
            
            # Transcribe
            segments = transcribe_audio(wav_path)
            full_transcript = ""
            if segments:
                full_transcript = "\n".join(s.get("text", "") for s in segments)
                
                # Enhanced audio analysis with LLM
                if SETTINGS.openrouter_api_keys or SETTINGS.groq_api_keys or SETTINGS.gemini_api_keys:
                    client = get_smart_client()
                    
                    # Add video context to the transcript for analysis
                    transcript_with_context = f"Video Title: {video_title}\nVideo Duration: ~{duration_minutes} minutes\n\nTranscript:\n{full_transcript}"
                    enhanced_audio_prompt = get_enhanced_audio_analysis_prompt(transcript_with_context)
                    
                    try:
                        audio_summary = client.chat(
                            [
                                {"role": "system", "content": "You are an expert video content analyst. Provide detailed, structured analysis following the exact format requested."},
                                {"role": "user", "content": enhanced_audio_prompt},
                            ],
                            temperature=0.3,
                            max_tokens=1500,
                        )
                        result["audio_analysis"] = audio_summary
                    except Exception as e:
                        logger.warning(f"Enhanced audio analysis failed: {e}")
                        # Fallback to basic analysis
                        basic_prompt = (
                            "Analyze this video transcript and provide: 1) Content summary, 2) Sentiment, 3) Content type, 4) Any products/brands mentioned."
                        )
                        result["audio_analysis"] = client.chat(
                            [{"role": "system", "content": basic_prompt}, {"role": "user", "content": full_transcript[:8000]}],
                            temperature=0.3, max_tokens=800,
                        )
            
            # Step 3: Enhanced Video Analysis (frames)
            frames = extract_frames(mp4_path, video_dir / "frames", every_sec=10)  # Every 10 seconds for better coverage
            if frames:
                frames = frames[:12]  # Increase to 12 frames for better analysis
                
                if SETTINGS.openrouter_api_keys or SETTINGS.groq_api_keys or SETTINGS.gemini_api_keys:
                    enhanced_video_prompt = f"""Analyze these video frames and provide insights about:

## VISUAL CONTENT ANALYSIS
- Production quality (lighting, camera work, editing)
- Setting and environment (home, studio, outdoor, etc.)
- Visual elements and graphics
- Product placement visibility
- Text overlays or graphics

## VISUAL PRODUCTS AND BRANDING
Identify any products, logos, or branding visible in the frames:
- Product names or brands visible
- Approximate time segments where they appear
- How prominently they're featured
- Context (being used, displayed, mentioned)

## PRODUCTION STYLE ASSESSMENT
- Professional vs. amateur production
- Editing sophistication
- Visual engagement techniques
- Brand consistency
- Overall aesthetic quality

Video Title: {video_title}
Number of frames analyzed: {len(frames)}
Frame interval: ~10 seconds apart"""

                    try:
                        # Use the summarise_frames function properly
                        video_summary = summarise_frames(frames, prompt=enhanced_video_prompt)
                        result["video_analysis"] = video_summary
                    except Exception as e:
                        logger.warning(f"Enhanced video analysis failed: {e}")
                        # Fallback to basic frame analysis
                        try:
                            result["video_analysis"] = summarise_frames(frames)
                        except Exception as e2:
                            logger.warning(f"Basic video analysis also failed: {e2}")
                            result["video_analysis"] = f"Frame analysis failed for {len(frames)} frames. Error: {str(e)}"
            
            # Step 4: Generate Structured Data Summary
            if result.get("audio_analysis"):
                structured_analysis_prompt = f"""Based on the following video analysis, extract structured data in JSON format.

Audio Analysis:
{result.get('audio_analysis', '')}

Video Analysis:
{result.get('video_analysis', '')}

Extract the following information and return ONLY valid JSON (no markdown, no explanations):

{{
    "content_type": {{"primary": "Educational", "secondary": ["Technology"]}},
    "voice_style": {{"pace": "medium", "tone": "casual", "language_complexity": "moderate", "personality_traits": ["enthusiastic", "informative"]}},
    "content_quality": {{"script_quality": "semi-scripted", "information_density": "high", "educational_value": "high", "entertainment_value": "medium"}},
    "sentiment": {{"overall": "positive", "energy_level": "high", "emotional_tone": "enthusiastic"}},
    "authenticity": {{"score": 8, "reasoning": "Natural delivery with genuine reactions"}},
    "products_mentioned": [
        {{"name": "Product Name", "context": "review", "timestamp_segment": "2:30-4:15", "sponsored": false}}
    ],
    "key_moments": [
        {{"description": "Main topic introduction", "timestamp_segment": "0:00-1:30", "importance": "high"}}
    ],
    "production_quality": {{"visual_quality": "high", "audio_quality": "good", "editing_sophistication": "moderate"}},
    "engagement_techniques": ["direct address", "call-to-action"]
}}

Return only the JSON object above, filled with actual data from the analysis:"""

                try:
                    if SETTINGS.openrouter_api_keys or SETTINGS.groq_api_keys or SETTINGS.gemini_api_keys:
                        client = get_smart_client()
                        structured_response = client.chat(
                            [{"role": "system", "content": "You are a data extraction specialist. Return only valid JSON without any markdown formatting or additional text."}, 
                             {"role": "user", "content": structured_analysis_prompt}],
                            temperature=0.1, max_tokens=1000,
                        )
                        
                        # Clean the response to ensure it's valid JSON
                        cleaned_response = structured_response.strip()
                        
                        # Remove markdown code blocks if present
                        if cleaned_response.startswith("```json"):
                            cleaned_response = cleaned_response[7:]
                        if cleaned_response.startswith("```"):
                            cleaned_response = cleaned_response[3:]
                        if cleaned_response.endswith("```"):
                            cleaned_response = cleaned_response[:-3]
                        
                        cleaned_response = cleaned_response.strip()
                        
                        # Try to parse JSON
                        try:
                            structured_data = json.loads(cleaned_response)
                            result["structured_data"] = structured_data
                            logger.info(f"Successfully parsed structured data for {video_id}")
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse structured data JSON for {video_id}: {e}")
                            logger.debug(f"Raw response: {cleaned_response[:500]}...")
                            
                            # Create fallback structured data from analysis text
                            fallback_data = {
                                "content_type": {"primary": "Unknown", "secondary": []},
                                "voice_style": {"pace": "unknown", "tone": "unknown", "language_complexity": "unknown", "personality_traits": []},
                                "content_quality": {"script_quality": "unknown", "information_density": "unknown", "educational_value": "unknown", "entertainment_value": "unknown"},
                                "sentiment": {"overall": "neutral", "energy_level": "unknown", "emotional_tone": "unknown"},
                                "authenticity": {"score": 5, "reasoning": "Could not determine from analysis"},
                                "products_mentioned": [],
                                "key_moments": [],
                                "production_quality": {"visual_quality": "unknown", "audio_quality": "unknown", "editing_sophistication": "unknown"},
                                "engagement_techniques": []
                            }
                            result["structured_data"] = fallback_data
                            logger.info(f"Using fallback structured data for {video_id}")
                            
                except Exception as e:
                    logger.warning(f"Structured data extraction failed: {e}")
                    # Create minimal fallback data
                    result["structured_data"] = {
                        "content_type": {"primary": "Unknown", "secondary": []},
                        "voice_style": {"pace": "unknown", "tone": "unknown", "language_complexity": "unknown", "personality_traits": []},
                        "content_quality": {"script_quality": "unknown", "information_density": "unknown", "educational_value": "unknown", "entertainment_value": "unknown"},
                        "sentiment": {"overall": "neutral", "energy_level": "unknown", "emotional_tone": "unknown"},
                        "authenticity": {"score": 5, "reasoning": "Analysis failed"},
                        "products_mentioned": [],
                        "key_moments": [],
                        "production_quality": {"visual_quality": "unknown", "audio_quality": "unknown", "editing_sophistication": "unknown"},
                        "engagement_techniques": []
                    }
            
            # Step 5: Fetch video statistics and OAuth analytics
            # First get video info to determine which channel it belongs to
            public_service = get_public_service(self.yt_api_key)
            video_response = public_service.videos().list(part="snippet,statistics,contentDetails", id=video_id).execute()
            
            if not video_response.get("items"):
                logger.warning(f"Video {video_id} not found")
                video_info = None
                service = public_service
                access_type = "public"
            else:
                video_info = video_response["items"][0]
                video_channel_id = video_info["snippet"]["channelId"]
                
                # Check if we have OAuth for this video's channel
                logger.info(f"Video {video_id} belongs to channel {video_channel_id}")
                logger.info(f"OAuth info: {self.oauth_info}")
                
                try:
                    service, access_type = self.get_service_for_channel(video_channel_id)
                    logger.info(f"Service type for {video_channel_id}: {access_type}")
                except Exception as e:
                    logger.warning(f"Failed to get service for channel {video_channel_id}: {e}")
                    service = public_service
                    access_type = "public"
            
            # Save video statistics and fetch OAuth analytics if available
            if video_info:
                # Save public statistics
                stats_file = video_dir / f"{video_id}_stats.json"
                with open(stats_file, 'w', encoding='utf-8') as f:
                    json.dump(video_info, f, indent=2, ensure_ascii=False)
                
                # OAuth analytics if available
                logger.info(f"Checking OAuth analytics: access_type={access_type}, analytics_access={self.oauth_info.get('analytics_access')}")
                
                if access_type == "oauth" and self.oauth_info.get("analytics_access"):
                    logger.info(f"Attempting to fetch OAuth analytics for {video_id}")
                    try:
                        from src.youtube.analytics import get_comprehensive_video_analytics
                        
                        channel_id = video_info["snippet"]["channelId"]
                        
                        # Calculate days_back for "All Time" from video publish date
                        published_at = video_info["snippet"].get("publishedAt")
                        if published_at:
                            try:
                                created_date = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                                days_back = (datetime.now(created_date.tzinfo) - created_date).days
                            except Exception:
                                days_back = 3650  # fallback if date parsing fails
                        else:
                            days_back = 3650  # fallback to 10 years if published_at missing
                        
                        logger.info(f"Fetching OAuth analytics for {video_id} with days_back={days_back}")
                        oauth_analytics = get_comprehensive_video_analytics(service, video_id, channel_id, days_back=days_back)
                        
                        if oauth_analytics:
                            oauth_file = video_dir / f"{video_id}_oauth_analytics.json"
                            with open(oauth_file, 'w', encoding='utf-8') as f:
                                json.dump(oauth_analytics, f, indent=2, ensure_ascii=False)
                            logger.info(f"OAuth analytics saved for {video_id}")
                        else:
                            logger.warning(f"OAuth analytics returned empty for {video_id}")
                    except Exception as e:
                        logger.error(f"OAuth analytics failed for {video_id}: {e}")
                        import traceback
                        logger.error(f"Full traceback: {traceback.format_exc()}")
                        # Save error info
                        oauth_error_file = video_dir / f"{video_id}_oauth_analytics.json"
                        with open(oauth_error_file, 'w', encoding='utf-8') as f:
                            json.dump({"error": str(e)}, f, indent=2, ensure_ascii=False)
                else:
                    logger.info(f"OAuth analytics not available for {video_id} (access_type: {access_type}, analytics_access: {self.oauth_info.get('analytics_access')})")

            # Step 6: Save comprehensive summary files
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Markdown summary
            summary_content = f"""# Comprehensive Analysis: {video_title}

**Video ID:** {video_id}
**URL:** {video_url}
**Duration:** ~{duration_minutes} minutes
**Processing Date:** {timestamp}

---

{result.get('audio_analysis', 'No audio analysis available')}

---

## 🖼️ Visual Analysis
{result.get('video_analysis', 'No video analysis available')}

---

## 📊 Technical Details
- **Frames Analyzed:** {len(frames) if frames else 0}
- **Audio Quality:** {quality}
- **Processing Time:** {timestamp}

---
*Generated by NC_IM Final Channel Stat Analyzer*
"""
            
            summary_file = video_dir / f"{video_id}_summary.md"
            summary_file.write_text(summary_content, encoding='utf-8')
            
            # Save structured data as JSON
            if result.get("structured_data"):
                json_file = video_dir / f"{video_id}_data.json"
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "video_id": video_id,
                        "title": video_title,
                        "url": video_url,
                        "duration_minutes": duration_minutes,
                        "processing_date": timestamp,
                        "analysis": result["structured_data"]
                    }, f, indent=2, ensure_ascii=False)
            
            # Step 7: Cleanup downloaded files
            try:
                if mp4_path.exists():
                    mp4_path.unlink()
                if wav_path.exists():
                    wav_path.unlink()
                # Keep frames for visual reference
            except Exception as cleanup_error:
                logger.warning(f"Cleanup failed for {video_id}: {cleanup_error}")
            
            result["success"] = True
            return result
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Failed to process video {video_id}: {e}")
            return result
    
    def generate_collective_analysis(self, channel_id: str, channel_title: str, output_dir: Path) -> Dict:
        """Generate comprehensive collective analysis from all video summaries."""
        
        # Find all JSON data files - check both regular and brand analysis directories
        json_files = []
        
        # Check brand analysis directory first (more comprehensive data)
        brand_analysis_dir = REPORTS_DIR / "brand_analysis" / channel_id
        if brand_analysis_dir.exists():
            for video_dir in brand_analysis_dir.iterdir():
                if video_dir.is_dir():
                    brand_file = video_dir / f"{video_dir.name}_brand_analysis.json"
                    if brand_file.exists():
                        json_files.append(brand_file)
        
        # Check regular analysis directory if no brand analysis found
        if not json_files:
            regular_analysis_dir = REPORTS_DIR / "channel_analysis" / channel_id
            if regular_analysis_dir.exists():
                for video_dir in regular_analysis_dir.iterdir():
                    if video_dir.is_dir():
                        regular_file = video_dir / f"{video_dir.name}_data.json"
                        if regular_file.exists():
                            json_files.append(regular_file)
        
        # Also check the provided output_dir as fallback
        if not json_files and output_dir.exists():
            for video_dir in output_dir.iterdir():
                if video_dir.is_dir():
                    brand_file = video_dir / f"{video_dir.name}_brand_analysis.json"
                    regular_file = video_dir / f"{video_dir.name}_data.json"
                    
                    if brand_file.exists():
                        json_files.append(brand_file)
                    elif regular_file.exists():
                        json_files.append(regular_file)
        
        if not json_files:
            # Provide helpful debugging info
            brand_dir_exists = brand_analysis_dir.exists() if 'brand_analysis_dir' in locals() else False
            regular_dir_exists = regular_analysis_dir.exists() if 'regular_analysis_dir' in locals() else False
            return {
                "error": f"No analysis data found for channel {channel_id}",
                "debug_info": {
                    "brand_analysis_dir": str(brand_analysis_dir) if 'brand_analysis_dir' in locals() else "Not checked",
                    "brand_dir_exists": brand_dir_exists,
                    "regular_analysis_dir": str(regular_analysis_dir) if 'regular_analysis_dir' in locals() else "Not checked", 
                    "regular_dir_exists": regular_dir_exists,
                    "output_dir": str(output_dir),
                    "output_dir_exists": output_dir.exists()
                }
            }
        
        # Load all video data
        video_analyses = []
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    video_analyses.append(data)
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
        
        if not video_analyses:
            return {"error": "No valid analysis data found"}
        
        # Generate collective insights
        try:
            if SETTINGS.openrouter_api_keys or SETTINGS.groq_api_keys or SETTINGS.gemini_api_keys:
                client = get_smart_client()
                
                # Prepare detailed summary data for LLM analysis
                video_summaries = []
                total_duration = 0
                content_types = []
                authenticity_scores = []
                products_count = 0
                brand_safety_scores = []
                influence_scores = []
                community_insights = []
                is_brand_analysis = False
                
                for i, video in enumerate(video_analyses, 1):
                    # Handle both regular analysis and brand analysis data structures
                    if 'creator_profile' in video:
                        # Brand analysis data structure
                        is_brand_analysis = True
                        duration = video.get("duration_minutes", 0)
                        total_duration += duration
                        
                        # Get brand-specific data
                        creator_profile = video.get('creator_profile', {})
                        content_category = video.get('content_category', 'Unknown')
                        content_types.append(content_category)
                        
                        # Brand analysis scores
                        if isinstance(creator_profile, dict):
                            auth_score = creator_profile.get('overall_authenticity', 50) / 10  # Convert to /10 scale
                            brand_safety = creator_profile.get('brand_safety_score', 50)
                            influence = creator_profile.get('audience_influence_power', 50)
                        else:
                            auth_score = 5.0  # Default
                            brand_safety = 50.0
                            influence = 50.0
                        
                        authenticity_scores.append(auth_score)
                        brand_safety_scores.append(brand_safety)
                        influence_scores.append(influence)
                        
                        # Comments insights
                        comments_analysis = video.get('comments_analysis', {})
                        if isinstance(comments_analysis, dict):
                            insights = comments_analysis.get('community_insights', [])
                            community_insights.extend(insights[:2])  # Top 2 insights per video
                        
                        # Create brand analysis summary
                        subcategories = video.get('content_subcategories', [])
                        subcats_str = ', '.join(subcategories[:3]) if subcategories else 'None'
                        
                        summary = f"""
**Video {i}: {video.get('title', 'Unknown')}**
- Duration: {duration} minutes
- Content Category: {content_category}
- Subcategories: {subcats_str}
- Authenticity Score: {auth_score:.1f}/10
- Brand Safety Score: {brand_safety:.0f}/100
- Audience Influence: {influence:.0f}/100
- Community Authenticity: {comments_analysis.get('community_authenticity', 0):.0f}/100
- Brand Receptivity: {comments_analysis.get('brand_receptivity', 0):.0f}/100
- Key Brand Insights: {video.get('transcript_excerpt', 'No transcript available')[:200]}..."""
                    
                    else:
                        # Regular analysis data structure
                        analysis = video.get("analysis", {})
                        duration = video.get("duration_minutes", 0)
                        total_duration += duration
                        
                        # Extract data for metrics
                        primary_type = analysis.get('content_type', {}).get('primary', 'Unknown')
                        content_types.append(primary_type)
                        
                        auth_score = analysis.get('authenticity', {}).get('score', 5)
                        if isinstance(auth_score, (int, float)):
                            authenticity_scores.append(auth_score)
                        
                        products_mentioned = analysis.get('products_mentioned', [])
                        products_count += len(products_mentioned)
                        
                        # Create regular analysis summary
                        summary = f"""
**Video {i}: {video.get('title', 'Unknown')}**
- Duration: {duration} minutes
- Content Type: {primary_type}
- Secondary Types: {', '.join(analysis.get('content_type', {}).get('secondary', []))}
- Voice Style: {analysis.get('voice_style', {}).get('tone', 'Unknown')} tone, {analysis.get('voice_style', {}).get('pace', 'Unknown')} pace
- Language Complexity: {analysis.get('voice_style', {}).get('language_complexity', 'Unknown')}
- Sentiment: {analysis.get('sentiment', {}).get('overall', 'Unknown')} ({analysis.get('sentiment', {}).get('energy_level', 'Unknown')} energy)
- Authenticity Score: {auth_score}/10 - {analysis.get('authenticity', {}).get('reasoning', 'No reasoning provided')}
- Products Mentioned: {len(products_mentioned)} items
- Production Quality: Visual - {analysis.get('production_quality', {}).get('visual_quality', 'Unknown')}, Audio - {analysis.get('production_quality', {}).get('audio_quality', 'Unknown')}
- Key Engagement Techniques: {', '.join(analysis.get('engagement_techniques', []))}
"""
                    video_summaries.append(summary)
                
                # Calculate metrics
                avg_authenticity = sum(authenticity_scores) / len(authenticity_scores) if authenticity_scores else 0
                content_type_distribution = Counter(content_types)
                
                # Calculate brand-specific metrics if available
                brand_metrics_text = ""
                if is_brand_analysis and brand_safety_scores:
                    avg_brand_safety = sum(brand_safety_scores) / len(brand_safety_scores)
                    avg_influence = sum(influence_scores) / len(influence_scores)
                    brand_metrics_text = f"""
- Brand Safety Score: {avg_brand_safety:.0f}/100
- Audience Influence Power: {avg_influence:.0f}/100
- Community Insights: {len(community_insights)} insights collected"""
                
                # Create comprehensive prompt
                analysis_type = "Brand Partnership Analysis" if is_brand_analysis else "Standard Content Analysis"
                
                collective_prompt = get_channel_collective_analysis_prompt(
                    channel_title,
                    video_analyses,
                    total_duration,
                    avg_authenticity,
                    products_count,
                    content_type_distribution
                )

                try:
                    system_prompt = "You are a comprehensive YouTube and creator analysis expert." + (
                        " You specialize in brand partnership evaluation, creator commercial viability, and audience authenticity assessment for marketing decision-making." if is_brand_analysis 
                        else " You provide detailed content strategy analysis and performance insights."
                    ) + " Provide thorough, detailed analytical insights with specific examples and data-driven observations. Focus on what IS rather than what SHOULD BE. Be comprehensive but focused on analysis, not recommendations."
                    
                    collective_analysis = client.chat(
                        [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": collective_prompt},
                        ],
                        temperature=0.3,
                        max_tokens=3000,  # Increased from 1000 to prevent truncation
                    )
                    
                    # Save enhanced collective analysis
                    collective_file = output_dir / f"COLLECTIVE_ANALYSIS_{channel_id}.md"
                    collective_content = f"""# 📺 Comprehensive Channel Analysis Report
## {channel_title}

---

**📋 ANALYSIS OVERVIEW**
- **Channel ID:** `{channel_id}`
- **Videos Analyzed:** {len(video_analyses)}
- **Total Content Duration:** {total_duration} minutes ({total_duration/60:.1f} hours)
- **Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Average Authenticity Score:** {avg_authenticity:.1f}/10

---

{collective_analysis}

---

# 📈 DETAILED METRICS & DATA

## Content Type Breakdown
{chr(10).join([f"- **{content_type}:** {count} videos ({count/len(video_analyses)*100:.1f}%)" for content_type, count in content_type_distribution.most_common()])}

## Video Portfolio
{chr(10).join([f"**{i}.** {v.get('title', 'Unknown Title')} ({v.get('duration_minutes', 0)} min)" for i, v in enumerate(video_analyses, 1)])}

## Key Statistics
- **📊 Total Videos:** {len(video_analyses)}
- **⏱️ Average Duration:** {total_duration/len(video_analyses):.1f} minutes
- **🏆 Authenticity Score:** {avg_authenticity:.1f}/10{f'''
- **🛡️ Brand Safety Score:** {avg_brand_safety:.0f}/100
- **📈 Audience Influence Power:** {avg_influence:.0f}/100
- **💬 Community Insights:** {len(community_insights)} collected''' if is_brand_analysis and brand_safety_scores else ''}
- **🛍️ Products Mentioned:** {products_count} across all videos
- **🎯 Primary Content Type:** {content_type_distribution.most_common(1)[0][0] if content_type_distribution else 'Unknown'}

## Analysis Methodology
- **Data Sources:** Audio transcripts + Visual frame analysis + Structured content analysis
- **Analysis Engine:** AI-powered multi-modal assessment
- **Quality Assurance:** Cross-validated insights with fallback mechanisms
- **Generated By:** NC_IM Final Channel Stat Analyzer v2.0

---

*This comprehensive analysis combines insights from {len(video_analyses)} individual video analyses to provide strategic content and creator development insights. All metrics and recommendations are based on quantitative analysis of actual content.*
"""
                    
                    collective_file.write_text(collective_content, encoding='utf-8')
                    
                    return {
                        "success": True,
                        "analysis": collective_analysis,
                        "videos_count": len(video_analyses),
                        "file_path": collective_file,
                        "metrics": {
                            "total_duration": total_duration,
                            "avg_authenticity": avg_authenticity,
                            "products_count": products_count,
                            "content_types": dict(content_type_distribution),
                            "is_brand_analysis": is_brand_analysis,
                            **({
                                "avg_brand_safety": avg_brand_safety,
                                "avg_influence": avg_influence,
                                "community_insights_count": len(community_insights)
                            } if is_brand_analysis and brand_safety_scores else {})
                        }
                    }
                    
                except Exception as e:
                    logger.error(f"LLM collective analysis failed: {e}")
                    return {"error": f"Analysis generation failed: {e}"}
                    
        except Exception as e:
            logger.error(f"Collective analysis failed: {e}")
            return {"error": f"Failed to generate collective analysis: {e}"}
    
    def process_channel_videos(self, channel_id: str, channel_title: str, max_videos: int, include_comments_analysis: bool = False) -> Dict:
        """Process multiple videos from a channel and return results."""
        
        # Use brand-focused analysis if enabled
        if self.enable_brand_analysis and self.brand_service:
            logger.info("Using brand-focused channel analysis")
            return self.brand_service.process_channel_for_brands(channel_id, channel_title, max_videos)
        
        # Original processing logic
        # Create output directory
        output_dir = REPORTS_DIR / "channel_analysis" / channel_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get service
        service, access_type = self.get_service_for_channel(channel_id)
        
        # Get videos list
        videos = self.get_channel_videos(service, channel_id, max_videos)
        
        if not videos:
            return {"error": "No videos found or failed to fetch videos"}
        
        # Process each video
        results = []
        successful_analyses = 0
        failed_analyses = 0
        skipped_analyses = 0
        
        for video_info in videos:
            video_id = video_info["video_id"]
            video_title = video_info["title"]
            
            # Process the video
            result = self.process_single_video(video_id, video_title, output_dir)
            results.append(result)
            
            if result["success"]:
                if result.get("skipped"):
                    skipped_analyses += 1
                else:
                    successful_analyses += 1
            else:
                failed_analyses += 1
        
        return {
            "success": True,
            "videos_processed": len(videos),
            "successful_analyses": successful_analyses,
            "failed_analyses": failed_analyses,
            "skipped_analyses": skipped_analyses,
            "results": results,
            "output_dir": output_dir,
            "access_type": access_type,
            "brand_analysis_enabled": self.enable_brand_analysis
        } 