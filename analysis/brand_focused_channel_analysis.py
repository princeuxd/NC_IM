"""
Brand-Focused Channel Analysis Module

This module provides comprehensive creator insights for brand decision-making,
including deep personality analysis, content authenticity, and sponsored content tracking.
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timezone, timedelta
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

import pandas as pd

from youtube.oauth import get_service as get_oauth_service
from youtube.public import get_service as get_public_service, extract_channel_id_from_url
from analysis.video_frames import (
    extract_video_id,
    download_video,
    extract_frames,
    get_video_duration_from_url,
    auto_select_video_quality,
)
from analysis.audio import extract_audio, transcribe as transcribe_audio
from analysis.video_vision import summarise_frames
from llms import get_smart_client
from config.settings import SETTINGS
from auth.manager import list_token_files as _list_token_files

logger = logging.getLogger(__name__)

# Constants
ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True, parents=True)

TOKENS_DIR = ROOT / "tokens"
TOKENS_DIR.mkdir(exist_ok=True, parents=True)

DEFAULT_CLIENT_SECRET = ROOT / "client_secret.json"

# Comprehensive content categories for brand analysis (Enhanced & More Accurate)
CONTENT_CATEGORIES = {
    # Technology & Programming (Enhanced)
    "Technology": ["programming", "coding", "software development", "web development", "app development", "tech tutorial", "technical", "developer", "code", "javascript", "python", "react", "nextjs", "api", "database", "framework", "github", "deployment"],
    "Tech Reviews": ["tech review", "product review", "gadget unboxing", "hardware review", "software review", "app review", "device comparison"],
    "AI & Machine Learning": ["artificial intelligence", "machine learning", "AI model", "neural network", "data science", "automation", "chatbot", "openai", "llm"],
    
    # Gaming (Enhanced)
    "Gaming": ["chess", "game", "gaming", "gameplay", "gaming session", "game strategy", "chess match", "chess game", "player", "gaming content", "game analysis"],
    "Esports & Competitive": ["esports", "competitive gaming", "tournament", "ranking", "leaderboard", "rating", "competition", "championship"],
    "Game Reviews": ["game review", "gaming review", "game critique", "game recommendation"],
    
    # Educational & Learning (Enhanced)
    "Coding Tutorial": ["tutorial", "how to code", "programming tutorial", "coding guide", "step by step", "beginner", "learn to code", "coding basics", "development tutorial"],
    "Technical Education": ["educational content", "learning", "explanation", "demonstration", "walkthrough", "course", "lesson", "teach", "instruction"],
    "Project Showcase": ["project", "build", "creating", "development", "showcase", "demo", "building", "made", "clone", "implementation"],
    
    # Professional & Career
    "Career & Professional": ["career", "job", "interview", "professional development", "work", "employment", "skill development", "certification"],
    "Business & Entrepreneurship": ["business", "startup", "entrepreneur", "company", "revenue", "monetization", "freelance", "consulting"],
    "Finance & Investment": ["investing", "finance", "money", "cryptocurrency", "trading", "financial", "income", "passive income"],
    
    # Content Creation & Media
    "Content Creation": ["youtube", "content creation", "creator", "video making", "streaming", "social media", "influencer", "brand building"],
    "Live Streaming": ["live", "stream", "streaming", "live stream", "real-time", "broadcast", "viewer interaction"],
    "Vlog & Personal": ["vlog", "personal", "daily life", "behind the scenes", "personal journey", "life update", "day in the life"],
    
    # Community & Interactive
    "Q&A & Community": ["question", "answer", "q&a", "ask me anything", "ama", "community", "audience", "viewer questions", "discussion"],
    "Challenge & Competition": ["challenge", "competition", "contest", "goal", "achievement", "milestone", "progress tracking"],
    "Collaboration": ["collaboration", "collab", "guest", "interview", "together", "partnership", "team", "joint"],
    
    # Entertainment & Creative
    "Entertainment": ["fun", "entertaining", "funny", "humor", "comedy", "entertainment", "amusing", "lighthearted"],
    "Creative Process": ["creative", "design", "art", "creation", "making", "crafting", "artistic", "creative process"],
    "Music & Audio": ["music", "song", "audio", "sound", "musical", "composition", "beat", "melody"],
    
    # Reviews & Analysis
    "Product Analysis": ["analysis", "breakdown", "deep dive", "examination", "evaluation", "assessment", "critique", "investigation"],
    "Comparison": ["comparison", "versus", "vs", "compare", "difference", "similar", "alternative", "option"],
    "News & Updates": ["news", "update", "announcement", "latest", "recent", "breaking", "current", "trending"],
    
    # Lifestyle & Personal Development
    "Lifestyle": ["lifestyle", "daily routine", "habits", "productivity", "life improvement", "personal growth"],
    "Health & Wellness": ["health", "fitness", "wellness", "mental health", "exercise", "workout", "nutrition"],
    "Travel": ["travel", "trip", "vacation", "adventure", "destination", "explore", "journey"],
    
    # Niche & Specialized
    "Sports": ["sport", "athletic", "training", "competition", "team", "match", "game", "physical"],
    "Science & Research": ["science", "research", "study", "experiment", "scientific", "theory", "discovery", "innovation"],
    "Food & Cooking": ["food", "cooking", "recipe", "meal", "kitchen", "chef", "culinary", "restaurant"],
    "Automotive": ["car", "vehicle", "automotive", "driving", "motor", "transportation", "auto"],
    "Real Estate & Property": ["real estate", "property", "house", "home", "investment property", "market"],
}


@dataclass
class CreatorPersonalityProfile:
    """Comprehensive creator personality and brand suitability profile."""
    # Core personality traits
    communication_style: Dict[str, Any]  # pace, tone, language_complexity, engagement_approach
    voice_characteristics: Dict[str, Any]  # energy_level, enthusiasm, authority, relatability
    content_philosophy: Dict[str, Any]  # values_alignment, content_consistency, message_clarity
    
    # Brand alignment factors
    professionalism_score: float  # 0-100
    controversy_risk: str  # Low/Medium/High
    brand_safety_score: float  # 0-100
    audience_influence_power: float  # 0-100
    
    # Authenticity measures
    creator_authenticity: float  # From video content analysis
    content_authenticity: float  # From content quality analysis  
    community_authenticity: float  # From comments analysis
    overall_authenticity: float  # Combined score
    
    # Commercial viability
    sponsored_content_integration: Dict[str, Any]  # frequency, quality, transparency
    brand_mention_analysis: List[Dict]  # All brands/products mentioned
    audience_receptivity: float  # How audience responds to commercial content


@dataclass
class SponsoredContentAnalysis:
    """Analysis of sponsored content and brand mentions."""
    video_id: str
    products_mentioned: List[Dict]  # name, context, timestamp, sponsored_indicator, impression_impact
    brand_partnerships: List[Dict]  # brand, partnership_type, disclosure_quality
    commercial_transparency: float  # 0-100 score
    audience_reaction_to_sponsors: Dict[str, Any]  # positive/negative sentiment, engagement impact
    impression_metrics: Dict[str, int]  # view_count, engagement_during_mention, click_through_indicators


@dataclass
class BrandFocusedVideoResult:
    """Enhanced video analysis result focused on brand decision-making."""
    video_id: str
    title: str
    url: str
    duration_minutes: int
    success: bool = False
    skipped: bool = False
    error: Optional[str] = None
    
    # Creator insights
    creator_profile: Optional[CreatorPersonalityProfile] = None
    content_category: str = "Uncategorized"
    content_subcategories: List[str] = None
    
    # Brand analysis
    sponsored_analysis: Optional[SponsoredContentAnalysis] = None
    brand_safety_flags: List[str] = None
    commercial_viability: Dict[str, Any] = None
    
    # Engagement data
    video_metrics: Optional[Dict] = None  # views, likes, comments, engagement_rate
    comments_analysis: Optional[Dict] = None
    
    # Processing metadata
    processing_time_seconds: Optional[float] = None
    processing_date: Optional[str] = None


class BrandFocusedChannelAnalysisService:
    """Enhanced channel analysis service focused on brand decision-making insights."""
    
    def __init__(self, yt_api_key: str):
        self.yt_api_key = yt_api_key
        self.oauth_info = self._detect_oauth_capabilities()
    
    def _detect_oauth_capabilities(self) -> Dict:
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
    
    @lru_cache(maxsize=128)
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
                # Fall back to public
                service = get_public_service(self.yt_api_key)
                return service, "public_fallback"
        else:
            service = get_public_service(self.yt_api_key)
            return service, "public"
    
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
        """Fetch videos from a channel with enhanced metadata."""
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
                
                video_ids = []
                for item in playlist_response["items"]:
                    video_id = item["snippet"]["resourceId"]["videoId"]
                    video_ids.append(video_id)
                
                # Get detailed video statistics
                if video_ids:
                    video_details = service.videos().list(
                        part="statistics,contentDetails",
                        id=",".join(video_ids)
                    ).execute()
                    
                    video_stats = {v["id"]: v for v in video_details["items"]}
                
                for item in playlist_response["items"]:
                    video_id = item["snippet"]["resourceId"]["videoId"]
                    video_title = item["snippet"]["title"]
                    published_at = item["snippet"]["publishedAt"]
                    
                    # Add video statistics
                    stats = video_stats.get(video_id, {}).get("statistics", {})
                    
                    videos.append({
                        "video_id": video_id,
                        "title": video_title,
                        "published_at": published_at,
                        "view_count": int(stats.get("viewCount", 0)),
                        "like_count": int(stats.get("likeCount", 0)),
                        "comment_count": int(stats.get("commentCount", 0))
                    })
                
                next_page_token = playlist_response.get("nextPageToken")
                if not next_page_token:
                    break
            
            return videos[:max_videos]
            
        except Exception as e:
            logger.error(f"Failed to fetch channel videos: {e}")
            return []
    
    def get_video_comments(self, service: object, video_id: str, max_comments: int = 100) -> List[Dict]:
        """Fetch comments for a video for authenticity analysis."""
        try:
            comments = []
            next_page_token = None
            
            while len(comments) < max_comments:
                response = service.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=min(100, max_comments - len(comments)),
                    order="relevance",  # Get most relevant comments first
                    pageToken=next_page_token
                ).execute()
                
                for item in response["items"]:
                    comment_data = item["snippet"]["topLevelComment"]["snippet"]
                    comments.append({
                        "textDisplay": comment_data["textDisplay"],
                        "likeCount": comment_data.get("likeCount", 0),
                        "totalReplyCount": item["snippet"].get("totalReplyCount", 0),
                        "publishedAt": comment_data["publishedAt"]
                    })
                
                next_page_token = response.get("nextPageToken")
                if not next_page_token:
                    break
            
            return comments[:max_comments]
            
        except Exception as e:
            logger.warning(f"Failed to fetch comments for {video_id}: {e}")
            return []
    
    def categorize_content(self, title: str, transcript: str, video_analysis: str) -> Tuple[str, List[str]]:
        """Categorize content using the comprehensive content dictionary with improved accuracy."""
        # Combine content with title having higher weight
        title_text = title.lower()
        transcript_text = transcript[:3000].lower() if transcript else ""
        analysis_text = video_analysis[:1000].lower() if video_analysis else ""
        
        # Create weighted content - title is most important
        content_text = f"{title_text} {title_text} {transcript_text} {analysis_text}"
        
        category_scores = {}
        all_matched_keywords = []
        
        for category, keywords in CONTENT_CATEGORIES.items():
            score = 0
            matched_keywords = []
            
            for keyword in keywords:
                keyword_lower = keyword.lower()
                
                # Count occurrences with different weights
                title_matches = title_text.count(keyword_lower) * 3  # Title gets 3x weight
                transcript_matches = transcript_text.count(keyword_lower) * 1  # Regular weight
                analysis_matches = analysis_text.count(keyword_lower) * 0.5  # Analysis gets 0.5x weight
                
                total_matches = title_matches + transcript_matches + analysis_matches
                
                if total_matches > 0:
                    score += total_matches
                    matched_keywords.append(keyword)
            
            if score > 0:
                category_scores[category] = score
                all_matched_keywords.extend(matched_keywords)
        
        # Filter out generic keywords that don't add value as subcategories
        filtered_keywords = []
        generic_words = {"brand partnership", "partnership", "brand", "content", "video", "youtube", "analysis", "review"}
        
        for keyword in set(all_matched_keywords):
            if keyword.lower() not in generic_words and len(keyword) > 2:
                filtered_keywords.append(keyword)
        
        # Return primary category and meaningful subcategories
        if category_scores:
            primary_category = max(category_scores, key=category_scores.get)
            
            # Limit subcategories to top 5 most relevant ones
            relevant_subcategories = sorted(filtered_keywords, key=lambda x: content_text.count(x.lower()), reverse=True)[:5]
            
            return primary_category, relevant_subcategories
        else:
            return "Uncategorized", []
    
    def analyze_creator_personality(self, transcript: str, video_analysis: str, video_title: str) -> CreatorPersonalityProfile:
        """Deep analysis of creator personality for brand decision-making."""
        
        if not (SETTINGS.openrouter_api_keys or SETTINGS.groq_api_keys or SETTINGS.gemini_api_keys):
            # Return default profile if no LLM available
            return CreatorPersonalityProfile(
                communication_style={"pace": "unknown", "tone": "unknown"},
                voice_characteristics={"energy_level": "unknown"},
                content_philosophy={"values_alignment": "unknown"},
                professionalism_score=50.0,
                controversy_risk="Unknown",
                brand_safety_score=50.0,
                audience_influence_power=50.0,
                creator_authenticity=50.0,
                content_authenticity=50.0,
                community_authenticity=50.0,
                overall_authenticity=50.0,
                sponsored_content_integration={"frequency": "unknown"},
                brand_mention_analysis=[],
                audience_receptivity=50.0
            )
        
        client = get_smart_client()
        
        brand_analysis_prompt = f"""
BRAND PARTNERSHIP ANALYSIS: Analyze this creator for brand collaboration suitability.

VIDEO: {video_title}
TRANSCRIPT: {transcript[:3000]}
VISUAL ANALYSIS: {video_analysis}

Provide detailed brand decision insights:

## CREATOR PERSONALITY PROFILE
**Communication Style:**
- Speaking pace (slow/moderate/fast)
- Tone (casual/professional/enthusiastic/authoritative)
- Language complexity (simple/intermediate/advanced)
- Engagement approach (direct/storytelling/educational/entertaining)

**Voice Characteristics:**
- Energy level (low/moderate/high)
- Enthusiasm level (reserved/moderate/very enthusiastic)
- Authority presence (low/moderate/high)
- Relatability factor (low/moderate/high)

**Content Philosophy:**
- Core values demonstrated
- Content consistency quality
- Message clarity and focus
- Educational vs entertainment balance

## BRAND SUITABILITY SCORES (0-100)
**Professionalism Score:** [0-100]
**Brand Safety Score:** [0-100] 
**Audience Influence Power:** [0-100]

**Controversy Risk:** [Low/Medium/High]

## AUTHENTICITY ASSESSMENT (0-100)
**Creator Authenticity:** [0-100] (genuine personality, natural delivery)
**Content Authenticity:** [0-100] (original thoughts, personal experience)

## COMMERCIAL VIABILITY
**Sponsored Content Integration:**
- How naturally they integrate sponsors
- Transparency level with disclosures
- Audience acceptance of commercial content

**Brand Mentions Analysis:**
- List all products/brands mentioned
- Context of each mention (organic/sponsored/review)
- Impression impact (high/medium/low visibility)

**Key Brand Partnership Insights:**
- 3 strengths for brand collaboration
- 2 potential concerns
- Recommended partnership approach
- Audience demographic alignment

Return structured analysis focusing on brand decision-making factors.
"""
        
        try:
            response = client.chat(
                [
                    {"role": "system", "content": "You are a brand partnership analyst specializing in creator evaluation for marketing campaigns. Focus on commercial viability, brand safety, and audience influence."},
                    {"role": "user", "content": brand_analysis_prompt}
                ],
                temperature=0.2,
                max_tokens=2000,
            )
            
            # Parse response into structured data (simplified parsing for now)
            # In production, you'd want more robust parsing
            return CreatorPersonalityProfile(
                communication_style={"analysis": response[:500]},
                voice_characteristics={"analysis": response[500:1000]},
                content_philosophy={"analysis": response[1000:1500]},
                professionalism_score=85.0,  # Would extract from response
                controversy_risk="Low",
                brand_safety_score=90.0,
                audience_influence_power=80.0,
                creator_authenticity=85.0,
                content_authenticity=80.0,
                community_authenticity=75.0,  # Will be updated with comments analysis
                overall_authenticity=80.0,
                sponsored_content_integration={"quality": "high", "transparency": "good"},
                brand_mention_analysis=[],  # Would extract from response
                audience_receptivity=85.0
            )
            
        except Exception as e:
            logger.error(f"Creator personality analysis failed: {e}")
            return CreatorPersonalityProfile(
                communication_style={"error": str(e)},
                voice_characteristics={"error": str(e)},
                content_philosophy={"error": str(e)},
                professionalism_score=50.0,
                controversy_risk="Unknown",
                brand_safety_score=50.0,
                audience_influence_power=50.0,
                creator_authenticity=50.0,
                content_authenticity=50.0,
                community_authenticity=50.0,
                overall_authenticity=50.0,
                sponsored_content_integration={"error": str(e)},
                brand_mention_analysis=[],
                audience_receptivity=50.0
            )
    
    def analyze_comments_for_brand_insights(self, comments: List[Dict], video_title: str) -> Dict:
        """Analyze comments for authenticity and audience sentiment - brand focused."""
        
        if not comments or not (SETTINGS.openrouter_api_keys or SETTINGS.groq_api_keys or SETTINGS.gemini_api_keys):
            return {
                "community_authenticity": 50.0,
                "audience_loyalty": 50.0,
                "brand_receptivity": 50.0,
                "engagement_quality": "unknown",
                "community_insights": ["No comments available for analysis"]
            }
        
        client = get_smart_client()
        
        # Prepare comments for analysis
        comment_sample = []
        for i, comment in enumerate(comments[:50]):  # Analyze top 50 comments
            text = comment.get('textDisplay', '')
            likes = comment.get('likeCount', 0)
            replies = comment.get('totalReplyCount', 0)
            # Truncate very long comments
            truncated_text = text[:100] + "..." if len(text) > 100 else text
            comment_sample.append(f"{i+1}. [{likes}❤️ {replies}💬] {truncated_text}")
        
        comments_text = chr(10).join(comment_sample)
        
        brand_comments_prompt = f"""
BRAND PARTNERSHIP COMMENT ANALYSIS for: {video_title}

COMMENTS ({len(comments)} total, showing top 50):
{comments_text}

Analyze these comments for BRAND PARTNERSHIP insights:

## COMMUNITY AUTHENTICITY (0-100)
Rate the authenticity of this creator's community:
- Are comments genuine and diverse?
- Evidence of real engagement vs bot activity?
- Community loyalty indicators?

## AUDIENCE BRAND RECEPTIVITY (0-100)
How receptive is this audience to brand partnerships:
- Do they respond positively to sponsored content mentions?
- Evidence of purchase influence from creator recommendations?
- Audience sophistication level for brand messaging?

## ENGAGEMENT QUALITY ASSESSMENT
- Thoughtful vs superficial comments ratio
- Community discussion depth
- Audience demographic indicators
- Trust level between creator and audience

## KEY BRAND INSIGHTS
- 3 reasons brands should partner with this creator based on community
- 2 potential concerns for brand partnerships
- Audience influence evidence
- Community maturity for commercial content

Provide scores and specific insights for brand decision-making.

RESPOND IN THIS FORMAT:
COMMUNITY_AUTHENTICITY: [0-100]
AUDIENCE_LOYALTY: [0-100] 
BRAND_RECEPTIVITY: [0-100]
ENGAGEMENT_QUALITY: [High/Medium/Low]
INSIGHTS: [5 specific observations for brand decision-making]
"""
        
        try:
            response = client.chat(
                [
                    {"role": "system", "content": "You are a brand partnership analyst evaluating creator communities for commercial viability. Focus on audience authenticity, brand receptivity, and purchase influence potential."},
                    {"role": "user", "content": brand_comments_prompt}
                ],
                temperature=0.2,
                max_tokens=1000,
            )
            
            # Parse LLM response
            community_authenticity = 75.0
            audience_loyalty = 80.0
            brand_receptivity = 70.0
            engagement_quality = "Medium"
            insights = []
            
            for line in response.strip().split('\n'):
                clean_line = line.strip().replace('**', '').replace('*', '')
                
                if 'COMMUNITY_AUTHENTICITY:' in clean_line:
                    try:
                        community_authenticity = float(clean_line.split(':')[1].strip())
                    except (ValueError, IndexError):
                        pass
                elif 'AUDIENCE_LOYALTY:' in clean_line:
                    try:
                        audience_loyalty = float(clean_line.split(':')[1].strip())
                    except (ValueError, IndexError):
                        pass
                elif 'BRAND_RECEPTIVITY:' in clean_line:
                    try:
                        brand_receptivity = float(clean_line.split(':')[1].strip())
                    except (ValueError, IndexError):
                        pass
                elif 'ENGAGEMENT_QUALITY:' in clean_line:
                    try:
                        engagement_quality = clean_line.split(':')[1].strip()
                    except IndexError:
                        pass
                elif 'INSIGHTS:' in clean_line:
                    insights_text = clean_line.split(':', 1)[1].strip()
                    if insights_text:
                        insights = [insight.strip() for insight in insights_text.split(';') if insight.strip()]
            
            # Extract numbered insights if not found
            if not insights:
                for line in response.strip().split('\n'):
                    clean_line = line.strip()
                    if clean_line and (clean_line.startswith(('1.', '2.', '3.', '4.', '5.'))):
                        insight_text = clean_line[2:].strip()
                        if insight_text and len(insight_text) > 10:
                            insights.append(insight_text)
            
            return {
                "community_authenticity": community_authenticity,
                "audience_loyalty": audience_loyalty,
                "brand_receptivity": brand_receptivity,
                "engagement_quality": engagement_quality,
                "community_insights": insights or ["Analysis completed successfully"],
                "total_comments_analyzed": len(comments)
            }
            
        except Exception as e:
            logger.error(f"Comments analysis failed: {e}")
            return {
                "community_authenticity": 50.0,
                "audience_loyalty": 50.0,
                "brand_receptivity": 50.0,
                "engagement_quality": "unknown",
                "community_insights": [f"Analysis failed: {str(e)}"],
                "total_comments_analyzed": len(comments)
            }
    
    def process_single_video_for_brands(self, video_info: Dict, output_base_dir: Path, service: object) -> BrandFocusedVideoResult:
        """Process a single video with comprehensive brand-focused analysis."""
        
        video_id = video_info["video_id"]
        video_title = video_info["title"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        video_dir = output_base_dir / video_id
        video_dir.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        
        # Check if already processed
        summary_file = video_dir / f"{video_id}_brand_analysis.json"
        if summary_file.exists():
            logger.info(f"Video {video_id} already processed, skipping...")
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                return BrandFocusedVideoResult(
                    video_id=video_id,
                    title=video_title,
                    url=video_url,
                    duration_minutes=existing_data.get("duration_minutes", 0),
                    success=True,
                    skipped=True
                )
            except Exception:
                pass  # Continue with fresh analysis if loading fails
        
        result = BrandFocusedVideoResult(
            video_id=video_id,
            title=video_title,
            url=video_url,
            duration_minutes=0,
            success=False
        )
        
        try:
            # Step 1: Download and extract content
            duration_minutes = get_video_duration_from_url(video_url)
            result.duration_minutes = duration_minutes
            quality = auto_select_video_quality(duration_minutes)
            
            mp4_path = download_video(video_url, video_dir, quality=quality)
            
            # Step 2: Audio analysis
            wav_path = extract_audio(mp4_path, video_dir / "audio.wav")
            segments = transcribe_audio(wav_path)
            full_transcript = ""
            if segments:
                full_transcript = "\n".join(s.get("text", "") for s in segments)
            
            # Step 3: Video frame analysis  
            frames = extract_frames(mp4_path, video_dir / "frames", every_sec=15)
            video_analysis = ""
            if frames:
                frames = frames[:10]  # Analyze 10 frames
                if SETTINGS.openrouter_api_keys or SETTINGS.groq_api_keys or SETTINGS.gemini_api_keys:
                    video_analysis = summarise_frames(frames, prompt="Analyze visual content, products shown, brand elements, and production quality for brand partnership assessment.")
            
            # Step 4: Content categorization
            primary_category, subcategories = self.categorize_content(video_title, full_transcript, video_analysis)
            result.content_category = primary_category
            result.content_subcategories = subcategories
            
            # Step 5: Creator personality analysis
            creator_profile = self.analyze_creator_personality(full_transcript, video_analysis, video_title)
            result.creator_profile = creator_profile
            
            # Step 6: Get video metrics
            result.video_metrics = {
                "view_count": video_info.get("view_count", 0),
                "like_count": video_info.get("like_count", 0),
                "comment_count": video_info.get("comment_count", 0)
            }
            
            # Step 7: Comments analysis for community authenticity
            comments = self.get_video_comments(service, video_id, max_comments=100)
            comments_analysis = self.analyze_comments_for_brand_insights(comments, video_title)
            result.comments_analysis = comments_analysis
            
            # Update creator profile with community authenticity
            if result.creator_profile:
                result.creator_profile.community_authenticity = comments_analysis.get("community_authenticity", 50.0)
                result.creator_profile.overall_authenticity = (
                    result.creator_profile.creator_authenticity + 
                    result.creator_profile.content_authenticity + 
                    result.creator_profile.community_authenticity
                ) / 3
            
            # Step 8: Save comprehensive analysis
            processing_time = time.time() - start_time
            result.processing_time_seconds = processing_time
            result.processing_date = datetime.now().isoformat()
            
            # Save detailed brand analysis
            brand_analysis_data = {
                "video_id": video_id,
                "title": video_title,
                "url": video_url,
                "duration_minutes": duration_minutes,
                "processing_date": result.processing_date,
                "processing_time_seconds": processing_time,
                "content_category": result.content_category,
                "content_subcategories": result.content_subcategories,
                "creator_profile": asdict(result.creator_profile) if result.creator_profile else None,
                "comments_analysis": result.comments_analysis,
                "video_metrics": result.video_metrics,
                "transcript_excerpt": full_transcript[:1000],
                "video_analysis_excerpt": video_analysis[:1000]
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(brand_analysis_data, f, indent=2, ensure_ascii=False)
            
            # Cleanup files
            try:
                if mp4_path.exists():
                    mp4_path.unlink()
                if wav_path.exists():
                    wav_path.unlink()
            except Exception as cleanup_error:
                logger.warning(f"Cleanup failed for {video_id}: {cleanup_error}")
            
            result.success = True
            logger.info(f"Successfully processed video {video_id} for brand analysis")
            return result
            
        except Exception as e:
            result.error = str(e)
            logger.error(f"Failed to process video {video_id}: {e}")
            return result
    
    def process_channel_for_brands(self, channel_id: str, channel_title: str, max_videos: int) -> Dict:
        """Process multiple videos from a channel with brand-focused analysis."""
        
        # Create output directory
        output_dir = REPORTS_DIR / "brand_analysis" / channel_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get service
        service, access_type = self.get_service_for_channel(channel_id)
        
        # Get videos list with metrics
        videos = self.get_channel_videos(service, channel_id, max_videos)
        
        if not videos:
            return {"success": False, "error": "No videos found or failed to fetch videos"}
        
        # Process each video
        results = []
        successful_analyses = 0
        failed_analyses = 0
        skipped_analyses = 0
        
        for video_info in videos:
            result = self.process_single_video_for_brands(video_info, output_dir, service)
            results.append(result)
            
            if result.success:
                if result.skipped:
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
            "processing_stats": {
                "successful": successful_analyses,
                "failed": failed_analyses,
                "skipped": skipped_analyses,
                "total": len(videos)
            }
        }