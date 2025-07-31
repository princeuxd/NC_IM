"""Public YouTube Data API v3 helpers.

No authentication required – works with API key only.
"""

from __future__ import annotations

import re
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import isodate

from googleapiclient.discovery import build  # type: ignore


def get_service(api_key: str):
    """Build YouTube Data API v3 service with API key."""
    return build("youtube", "v3", developerKey=api_key)


def extract_channel_id_from_url(url: str) -> Optional[str]:
    """Extract channel ID from various YouTube channel URL formats."""
    
    # Clean the URL
    url = url.strip()
    
    # Pattern 1: /channel/UC...
    channel_match = re.search(r'/channel/([a-zA-Z0-9_-]+)', url)
    if channel_match:
        return channel_match.group(1)
    
    # Pattern 2: /@username (handle format)
    handle_match = re.search(r'/@([a-zA-Z0-9_-]+)', url)
    if handle_match:
        return handle_match.group(1)
    
    # Pattern 3: /c/username 
    c_match = re.search(r'/c/([a-zA-Z0-9_-]+)', url)
    if c_match:
        return c_match.group(1)
    
    # Pattern 4: /user/username
    user_match = re.search(r'/user/([a-zA-Z0-9_-]+)', url)
    if user_match:
        return user_match.group(1)
    
    # Pattern 5: If it's just a plain channel ID
    if url.startswith('UC') and len(url) == 24:
        return url
    
    # Pattern 6: If it's just @username without URL
    if url.startswith('@'):
        return url[1:]  # Remove the @ symbol
    
    # Fallback: assume it's a username/handle
    return url


def get_channel_by_url(service, channel_url: str) -> Optional[Dict[str, Any]]:
    """Get channel information from various URL formats."""
    
    identifier = extract_channel_id_from_url(channel_url)
    if not identifier:
        return None
    
    try:
        # Try as channel ID first
        if identifier.startswith('UC'):
            response = service.channels().list(
                part="id,snippet,statistics,contentDetails,brandingSettings",
                id=identifier
            ).execute()
        else:
            # Try as legacy username first
            response = service.channels().list(
                part="id,snippet,statistics,contentDetails,brandingSettings",
                forUsername=identifier
            ).execute()
            
            if not response.get('items'):
                # Search for channels with this custom URL or handle
                search_response = service.search().list(
                    part="snippet",
                    q=identifier,
                    type="channel",
                    maxResults=15  # search more results to increase accuracy
                ).execute()
                
                if search_response.get('items'):
                    # Build helper function to normalise strings
                    def _norm(s: str) -> str:
                        return re.sub(r"[^a-z0-9]", "", s.lower())
                
                    id_norm = _norm(identifier)
                
                    # Rank candidates
                    candidates = []
                
                    for item in search_response['items']:
                        channel_id = item['snippet']['channelId']
                        channel_title = item['snippet']['title']
                        title_norm = _norm(channel_title)
                        
                        # Score match quality (higher is better)
                        score = 0
                        if id_norm == title_norm:
                            score += 4  # exact title match
                        if id_norm in title_norm or title_norm in id_norm:
                            score += 2  # partial title match
                        
                        # Get custom URL to increase score
                        ch_resp = service.channels().list(
                            part="snippet",
                            id=channel_id
                        ).execute()
                        if ch_resp.get('items'):
                            ch_item = ch_resp['items'][0]
                            custom_url = ch_item['snippet'].get('customUrl', '')
                            custom_norm = _norm(custom_url)
                            if custom_norm == id_norm:
                                score += 5  # exact custom URL match
                            elif id_norm in custom_norm or custom_norm in id_norm:
                                score += 3
                            
                            candidates.append((score, ch_item))
                
                    # Select best candidate by highest score
                    if candidates:
                        best = sorted(candidates, key=lambda x: x[0], reverse=True)[0]
                        if best[0] > 0:
                            return best[1]
        
        return response.get('items', [None])[0]
    
    except Exception as e:
        print(f"Error fetching channel: {e}")
        return None


def get_comprehensive_channel_data(service, channel_url: str) -> Dict[str, Any]:
    """Get comprehensive public data for any channel."""
    
    result = {
        "channel_info": None,
        "recent_videos": [],
        "popular_videos": [],
        "playlists": [],
        "upload_patterns": {},
        "engagement_analysis": {},
        "content_analysis": {},
        "error": None
    }
    
    # Get basic channel info
    channel = get_channel_by_url(service, channel_url)
    if not channel:
        result["error"] = "Channel not found or invalid URL"
        return result
    
    channel_id = channel["id"]
    result["channel_info"] = channel
    
    try:
        # Get uploads playlist
        uploads_playlist_id = channel["contentDetails"]["relatedPlaylists"]["uploads"]
        
        # Get recent videos (last 50)
        # Fetch *all* videos in uploads playlist (may be paginated)
        video_ids = []
        next_page = None
        while True:
            pl_resp = service.playlistItems().list(
                part="contentDetails",
                playlistId=uploads_playlist_id,
                maxResults=50,
                pageToken=next_page
            ).execute()
            video_ids.extend([it["contentDetails"]["videoId"] for it in pl_resp["items"]])
            next_page = pl_resp.get("nextPageToken")
            if not next_page:
                break
        
        if video_ids:
            # Get detailed video statistics
            all_video_items = []
            # API allows 50 ids per call
            for i in range(0, len(video_ids), 50):
                chunk_ids = video_ids[i:i+50]
                videos_response = service.videos().list(
                    part="snippet,statistics,contentDetails",
                    id=",".join(chunk_ids)
                ).execute()
                all_video_items.extend(videos_response["items"])
            
            result["recent_videos"] = all_video_items
            
            # Analyze upload patterns
            result["upload_patterns"] = analyze_upload_patterns(all_video_items)
            
            # Get engagement analysis
            result["engagement_analysis"] = analyze_engagement_metrics(all_video_items)
            
            # Get content analysis
            result["content_analysis"] = analyze_content_patterns(all_video_items)
            
            # Get popular videos (sorted by views)
            popular_videos = sorted(
                all_video_items, 
                key=lambda x: int(x["statistics"].get("viewCount", 0)), 
                reverse=True
            )
            result["popular_videos"] = popular_videos[:20]
        
        # Get channel playlists
        playlists_response = service.playlists().list(
            part="snippet,contentDetails",
            channelId=channel_id,
            maxResults=20
        ).execute()
        result["playlists"] = playlists_response["items"]
        
    except Exception as e:
        result["error"] = f"Error fetching channel data: {str(e)}"
    
    return result


def analyze_upload_patterns(videos: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze upload frequency and timing patterns."""
    
    if not videos:
        return {}
    
    upload_dates = []
    durations = []
    
    for video in videos:
        try:
            # Parse upload date
            published_at = datetime.fromisoformat(
                video["snippet"]["publishedAt"].replace('Z', '+00:00')
            )
            upload_dates.append(published_at)
            
            # Parse duration
            duration_str = video["contentDetails"]["duration"]
            duration = isodate.parse_duration(duration_str)
            durations.append(duration.total_seconds())
            
        except Exception:
            continue
    
    if not upload_dates:
        return {}
    
    # Calculate patterns
    upload_dates.sort(reverse=True)
    
    # Upload frequency
    if len(upload_dates) > 1:
        time_diffs = [(upload_dates[i-1] - upload_dates[i]).days for i in range(1, len(upload_dates))]
        avg_days_between = sum(time_diffs) / len(time_diffs) if time_diffs else 0
    else:
        avg_days_between = 0
    
    # Day of week analysis
    day_counts = {}
    hour_counts = {}
    
    for date in upload_dates:
        day_name = date.strftime('%A')
        hour = date.hour
        
        day_counts[day_name] = day_counts.get(day_name, 0) + 1
        hour_counts[hour] = hour_counts.get(hour, 0) + 1
    
    # Duration analysis
    avg_duration = sum(durations) / len(durations) if durations else 0
    
    return {
        "total_videos": len(videos),
        "avg_days_between_uploads": round(avg_days_between, 1),
        "most_common_upload_day": max(day_counts.items(), key=lambda x: x[1])[0] if day_counts else None,
        "most_common_upload_hour": max(hour_counts.items(), key=lambda x: x[1])[0] if hour_counts else None,
        "avg_video_duration_seconds": round(avg_duration, 0),
        "day_distribution": day_counts,
        "hour_distribution": hour_counts,
        "latest_upload": upload_dates[0].isoformat() if upload_dates else None,
        "upload_consistency": "High" if avg_days_between < 7 else "Medium" if avg_days_between < 30 else "Low"
    }


def analyze_engagement_metrics(videos: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze engagement patterns across videos."""
    
    if not videos:
        return {}
    
    views = []
    likes = []
    comments = []
    engagement_rates = []
    
    for video in videos:
        stats = video["statistics"]
        
        view_count = int(stats.get("viewCount", 0))
        like_count = int(stats.get("likeCount", 0))
        comment_count = int(stats.get("commentCount", 0))
        
        if view_count > 0:
            views.append(view_count)
            likes.append(like_count)
            comments.append(comment_count)
            
            # Calculate engagement rate
            engagement_rate = ((like_count + comment_count) / view_count) * 100
            engagement_rates.append(engagement_rate)
    
    if not views:
        return {}
    
    # Calculate averages and totals
    total_views = sum(views)
    total_likes = sum(likes)
    total_comments = sum(comments)
    avg_views = total_views / len(views)
    avg_likes = total_likes / len(likes)
    avg_comments = total_comments / len(comments)
    avg_engagement_rate = sum(engagement_rates) / len(engagement_rates)
    
    # Find best performing video
    best_video_idx = views.index(max(views))
    best_video = videos[best_video_idx]
    
    # Calculate consistency (coefficient of variation)
    if avg_views > 0:
        view_variance = sum([(v - avg_views) ** 2 for v in views]) / len(views)
        view_std = view_variance ** 0.5
        consistency_score = 1 - (view_std / avg_views)  # Higher is more consistent
    else:
        consistency_score = 0
    
    return {
        "total_videos_analyzed": len(videos),
        "total_views": total_views,
        "total_likes": total_likes,
        "total_comments": total_comments,
        "avg_views_per_video": round(avg_views, 0),
        "avg_likes_per_video": round(avg_likes, 1),
        "avg_comments_per_video": round(avg_comments, 1),
        "avg_engagement_rate": round(avg_engagement_rate, 2),
        "like_to_view_ratio": round((total_likes / total_views) * 100, 2) if total_views > 0 else 0,
        "comment_to_view_ratio": round((total_comments / total_views) * 100, 2) if total_views > 0 else 0,
        "best_performing_video": {
            "title": best_video["snippet"]["title"],
            "views": int(best_video["statistics"].get("viewCount", 0)),
            "likes": int(best_video["statistics"].get("likeCount", 0)),
            "video_id": best_video["id"]
        },
        "consistency_score": round(consistency_score, 2),
        "performance_tier": "High" if avg_views > 10000 else "Medium" if avg_views > 1000 else "Growing"
    }


def analyze_content_patterns(videos: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze content themes, title patterns, and tags."""
    
    if not videos:
        return {}
    
    titles = [video["snippet"]["title"] for video in videos]
    descriptions = [video["snippet"].get("description", "") for video in videos]
    
    # Common words in titles
    title_words = []
    for title in titles:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', title.lower())
        title_words.extend(words)
    
    word_counts = {}
    for word in title_words:
        if word not in ['the', 'and', 'for', 'with', 'how', 'you', 'can', 'are', 'this', 'that']:
            word_counts[word] = word_counts.get(word, 0) + 1
    
    # Most common words
    common_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Title length analysis
    title_lengths = [len(title) for title in titles]
    avg_title_length = sum(title_lengths) / len(title_lengths) if title_lengths else 0
    
    # Content categories (basic keyword detection)
    categories = {
        "Tutorial/How-to": 0,
        "Review": 0,
        "Gaming": 0,
        "Music": 0,
        "Tech": 0,
        "Vlog": 0,
        "Educational": 0
    }
    
    for title in titles:
        title_lower = title.lower()
        if any(word in title_lower for word in ['how', 'tutorial', 'guide', 'learn']):
            categories["Tutorial/How-to"] += 1
        if any(word in title_lower for word in ['review', 'unboxing', 'reaction']):
            categories["Review"] += 1
        if any(word in title_lower for word in ['game', 'gaming', 'play', 'minecraft', 'fortnite']):
            categories["Gaming"] += 1
        if any(word in title_lower for word in ['music', 'song', 'cover', 'remix']):
            categories["Music"] += 1
        if any(word in title_lower for word in ['tech', 'technology', 'coding', 'programming', 'app']):
            categories["Tech"] += 1
        if any(word in title_lower for word in ['vlog', 'daily', 'life', 'day']):
            categories["Vlog"] += 1
        if any(word in title_lower for word in ['explain', 'science', 'history', 'math']):
            categories["Educational"] += 1
    
    # Most common category
    top_category = max(categories.items(), key=lambda x: x[1])[0] if any(categories.values()) else "General"
    
    return {
        "avg_title_length": round(avg_title_length, 1),
        "common_title_words": common_words,
        "content_categories": categories,
        "primary_content_type": top_category,
        "title_optimization": "Good" if 40 <= avg_title_length <= 70 else "Could improve",
        "content_diversity": len([cat for cat, count in categories.items() if count > 0])
    }


def get_channel_recent_performance(service, channel_id: str, days: int = 30) -> Dict[str, Any]:
    """Get recent performance metrics for public analysis."""
    
    try:
        # Get uploads playlist
        channel_response = service.channels().list(
            part="contentDetails",
            id=channel_id
        ).execute()
        
        if not channel_response["items"]:
            return {"error": "Channel not found"}
        
        uploads_playlist_id = channel_response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
        
        # Calculate date threshold
        since_date = datetime.now() - timedelta(days=days)
        
        # Get recent videos
        recent_videos = []
        next_page_token = None
        
        while len(recent_videos) < 50:  # Limit to prevent excessive API calls
            playlist_response = service.playlistItems().list(
                part="snippet,contentDetails",
                playlistId=uploads_playlist_id,
                maxResults=50,
                pageToken=next_page_token
            ).execute()
            
            for item in playlist_response["items"]:
                published_at = datetime.fromisoformat(
                    item["snippet"]["publishedAt"].replace('Z', '+00:00')
                )
                if published_at >= since_date:
                    recent_videos.append(item["contentDetails"]["videoId"])
                else:
                    break
            
            next_page_token = playlist_response.get("nextPageToken")
            if not next_page_token:
                break
        
        if not recent_videos:
            return {"videos_count": 0, "message": f"No videos uploaded in the last {days} days"}
        
        # Get video statistics
        videos_response = service.videos().list(
            part="snippet,statistics,contentDetails",
            id=",".join(recent_videos[:50])  # API limit
        ).execute()
        
        # Calculate metrics
        total_views = sum(int(video["statistics"].get("viewCount", 0)) for video in videos_response["items"])
        total_likes = sum(int(video["statistics"].get("likeCount", 0)) for video in videos_response["items"])
        total_comments = sum(int(video["statistics"].get("commentCount", 0)) for video in videos_response["items"])
        
        return {
            "period_days": days,
            "videos_count": len(videos_response["items"]),
            "total_views": total_views,
            "total_likes": total_likes,
            "total_comments": total_comments,
            "avg_views_per_video": round(total_views / len(videos_response["items"]), 0) if videos_response["items"] else 0,
            "videos": videos_response["items"]
        }
        
    except Exception as e:
        return {"error": str(e)}


__all__ = [
    "get_service",
    "extract_channel_id_from_url",
    "get_channel_by_url", 
    "get_comprehensive_channel_data",
    "analyze_upload_patterns",
    "analyze_engagement_metrics",
    "analyze_content_patterns",
    "get_channel_recent_performance"
] 