"""Analytics helper functions for unified public + OAuth channel analysis."""

from typing import Dict, Any, Optional
from googleapiclient.discovery import Resource  # type: ignore


def get_full_channel_analytics(
    oauth_service: Resource,
    public_service: Resource,
    channel_id: str,
    days_back: int = 30
) -> Dict[str, Any]:
    """
    Get comprehensive channel analytics combining public data + OAuth extras.
    
    Returns:
        {
            "channel_info": {...},
            "recent_videos": [...],
            "popular_videos": [...],
            "playlists": [...],
            "upload_patterns": {...},
            "engagement_analysis": {...},
            "content_analysis": {...},
            "oauth": {
                "growth_metrics": {...},
                "performance_summary": {...},
                "demographics": {...},
                "geography": {...},
                "traffic_sources": {...},
                "monetization": {...},
                "impressions": {...},
                "engagement_breakdown": {...},
                "video_stats": {...}
            }
        }
    """
    from youtube.public import get_comprehensive_channel_data
    from youtube.analytics import get_comprehensive_channel_analytics
    
    # Get public data (all uploads, basic stats, etc.)
    public_data = get_comprehensive_channel_data(public_service, channel_id)
    
    # Get OAuth-only extras if available
    oauth_data = None
    if oauth_service:
        try:
            oauth_data = get_comprehensive_channel_analytics(
                oauth_service, channel_id, days_back=days_back
            )
        except Exception as e:
            print(f"OAuth analytics failed: {e}")
            oauth_data = {"error": str(e)}
    
    # Merge data
    result = {**public_data}
    if oauth_data:
        result["oauth"] = oauth_data
    
    return result 