"""Convenience wrappers for the YouTube Analytics API.

All helpers expect a *youtube* Data API v3 service returned by
``youtube.oauth.get_service``.  We re-use its underlying OAuth credentials to
build a youtubeAnalytics v2 client.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import List, Dict, Any

from googleapiclient.discovery import build  # type: ignore
from googleapiclient.discovery import Resource  # type: ignore


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _get_analytics_service(data_service: Resource):
    """Build and cache a youtubeAnalytics service from the same credentials."""

    creds = data_service._http.credentials  # type: ignore[attr-defined]
    return build("youtubeAnalytics", "v2", credentials=creds)


# ---------------------------------------------------------------------------
# Video-level analytics (existing functions)
# ---------------------------------------------------------------------------

def video_summary_metrics(
    data_service: Resource,
    video_id: str,
    channel_id: str,
    *,
    days_back: int = 28,
) -> Dict[str, Any]:
    """Views, watch-time, avg duration, likes & comments for a single video."""

    end = date.today()
    start = end - timedelta(days=days_back)

    analytics = _get_analytics_service(data_service)
    response = (
        analytics.reports()
        .query(
            ids=f"channel=={channel_id}",
            startDate=start.isoformat(),
            endDate=end.isoformat(),
            metrics="views,estimatedMinutesWatched,averageViewDuration,likes,comments",
            filters=f"video=={video_id}",
            dimensions="video",
            maxResults=1,
        )
        .execute()
    )
    return response


def video_monetization_metrics(
    data_service: Resource,
    video_id: str,
    channel_id: str,
    *,
    days_back: int = 28,
) -> Dict[str, Any]:
    """Get monetization metrics for a specific video."""
    
    end = date.today()
    start = end - timedelta(days=days_back)

    analytics = _get_analytics_service(data_service)
    
    try:
        response = (
            analytics.reports()
            .query(
                ids=f"channel=={channel_id}",
                startDate=start.isoformat(),
                endDate=end.isoformat(),
                metrics="estimatedRevenue,estimatedAdRevenue,estimatedRedPartnerRevenue,grossRevenue,cpm,playbackBasedCpm,impressionBasedCpm",
                filters=f"video=={video_id}",
                maxResults=1,
            )
            .execute()
        )
        return response
    except Exception:
        # Monetization data might not be available
        return {"rows": [], "error": "Monetization data not available"}


def video_time_series_metrics(
    data_service: Resource,
    video_id: str,
    channel_id: str,
    *,
    days_back: int = 28,
) -> Dict[str, Any]:
    """Get daily time series data for views, likes, and subscribers gained."""
    
    end = date.today()
    start = end - timedelta(days=days_back)

    analytics = _get_analytics_service(data_service)
    
    response = (
        analytics.reports()
        .query(
            ids=f"channel=={channel_id}",
            startDate=start.isoformat(),
            endDate=end.isoformat(),
            metrics="views,likes,subscribersGained,estimatedMinutesWatched,shares,comments",
            dimensions="day",
            filters=f"video=={video_id}",
            maxResults=days_back,
        )
        .execute()
    )
    return response


def video_engagement_metrics(
    data_service: Resource,
    video_id: str,
    channel_id: str,
    *,
    days_back: int = 28,
) -> Dict[str, Any]:
    """Get comprehensive engagement metrics for a video."""
    
    end = date.today()
    start = end - timedelta(days=days_back)

    analytics = _get_analytics_service(data_service)
    
    response = (
        analytics.reports()
        .query(
            ids=f"channel=={channel_id}",
            startDate=start.isoformat(),
            endDate=end.isoformat(),
            metrics="views,likes,dislikes,comments,shares,subscribersGained,subscribersLost,videosAddedToPlaylists,savesAdded,savesRemoved",
            filters=f"video=={video_id}",
            maxResults=1,
        )
        .execute()
    )
    return response


def video_impressions_metrics(
    data_service: Resource,
    video_id: str,
    channel_id: str,
    *,
    days_back: int = 28,
) -> Dict[str, Any]:
    """Get impressions and click-through rate data for a specific video."""
    
    end = date.today()
    start = end - timedelta(days=days_back)

    analytics = _get_analytics_service(data_service)
    
    try:
        response = (
            analytics.reports()
            .query(
                ids=f"channel=={channel_id}",
                startDate=start.isoformat(),
                endDate=end.isoformat(),
                metrics="impressions,impressionClickThroughRate,uniqueViewers",
                filters=f"video=={video_id}",
                maxResults=1,
            )
            .execute()
        )
        return response
    except Exception:
        # Impressions data might not be available
        return {"rows": [], "error": "Impressions data not available"}


def video_subscriber_status_breakdown(
    data_service: Resource,
    video_id: str,
    channel_id: str,
    *,
    days_back: int = 28,
) -> Dict[str, Any]:
    """Get views, watch time, and avg view duration broken down by subscriber status for a video (OAuth only)."""
    end = date.today()
    start = end - timedelta(days=days_back)
    analytics = _get_analytics_service(data_service)
    response = (
        analytics.reports()
        .query(
            ids=f"channel=={channel_id}",
            startDate=start.isoformat(),
            endDate=end.isoformat(),
            metrics="views,estimatedMinutesWatched,averageViewDuration",
            dimensions="subscribedStatus",
            filters=f"video=={video_id}",
            maxResults=2,
        )
        .execute()
    )
    return response


def audience_retention(
    data_service: Resource,
    video_id: str,
    channel_id: str,
    *,
    days_back: int = 28,
) -> List[List[Any]]:
    """Audience-watch-ratio curve (elapsedVideoTimeRatio vs audienceWatchRatio)."""

    end = date.today()
    start = end - timedelta(days=days_back)

    analytics = _get_analytics_service(data_service)
    resp = (
        analytics.reports()
        .query(
            ids=f"channel=={channel_id}",
            startDate=start.isoformat(),
            endDate=end.isoformat(),
            metrics="audienceWatchRatio",
            dimensions="elapsedVideoTimeRatio",
            filters=f"video=={video_id}",
            maxResults=100,
        )
        .execute()
    )
    return resp.get("rows", [])


def traffic_sources(
    data_service: Resource,
    video_id: str,
    channel_id: str,
    *,
    days_back: int = 28,
) -> List[List[Any]]:
    """Break-down of views by traffic source type."""

    end = date.today()
    start = end - timedelta(days=days_back)

    analytics = _get_analytics_service(data_service)
    resp = (
        analytics.reports()
        .query(
            ids=f"channel=={channel_id}",
            startDate=start.isoformat(),
            endDate=end.isoformat(),
            metrics="views",
            dimensions="insightTrafficSourceType",
            filters=f"video=={video_id}",
            maxResults=25,
        )
        .execute()
    )
    return resp.get("rows", [])


def geography_breakdown(
    data_service: Resource,
    video_id: str,
    channel_id: str,
    *,
    days_back: int = 28,
) -> List[List[Any]]:
    """Views by country code (ISO-3166)."""

    end = date.today()
    start = end - timedelta(days=days_back)

    analytics = _get_analytics_service(data_service)
    resp = (
        analytics.reports()
        .query(
            ids=f"channel=={channel_id}",
            startDate=start.isoformat(),
            endDate=end.isoformat(),
            metrics="views",
            dimensions="country",
            filters=f"video=={video_id}",
            maxResults=250,
        )
        .execute()
    )
    return resp.get("rows", [])


def demographics_breakdown(
    data_service: Resource,
    video_id: str,
    channel_id: str,
    *,
    days_back: int = 28,
) -> List[List[Any]]:
    """Viewer percentage by age group and gender.

    Returns list rows of [ageGroup, gender, viewerPercentage].
    """

    end = date.today()
    start = end - timedelta(days=days_back)

    analytics = _get_analytics_service(data_service)
    resp = (
        analytics.reports()
        .query(
            ids=f"channel=={channel_id}",
            startDate=start.isoformat(),
            endDate=end.isoformat(),
            metrics="viewerPercentage",
            dimensions="ageGroup,gender",
            filters=f"video=={video_id}",
            maxResults=200,
        )
        .execute()
    )
    return resp.get("rows", [])


# ---------------------------------------------------------------------------
# Channel-level analytics (new functions)
# ---------------------------------------------------------------------------

def channel_growth_metrics(
    data_service: Resource,
    channel_id: str,
    *,
    days_back: int = 90,
) -> Dict[str, Any]:
    """Get channel growth metrics over time including subscribers and views."""
    
    end = date.today()
    start = end - timedelta(days=days_back)

    analytics = _get_analytics_service(data_service)
    
    # Get daily growth data
    response = (
        analytics.reports()
        .query(
            ids=f"channel=={channel_id}",
            startDate=start.isoformat(),
            endDate=end.isoformat(),
            metrics="views,subscribersGained,subscribersLost,estimatedMinutesWatched,videosAddedToPlaylists,videosRemovedFromPlaylists",
            dimensions="day",
            maxResults=days_back,
        )
        .execute()
    )
    return response


def channel_performance_summary(
    data_service: Resource,
    channel_id: str,
    *,
    days_back: int = 28,
) -> Dict[str, Any]:
    """Get overall channel performance summary for a time period."""
    
    end = date.today()
    start = end - timedelta(days=days_back)

    analytics = _get_analytics_service(data_service)
    
    response = (
        analytics.reports()
        .query(
            ids=f"channel=={channel_id}",
            startDate=start.isoformat(),
            endDate=end.isoformat(),
            metrics="views,estimatedMinutesWatched,averageViewDuration,subscribersGained,subscribersLost,likes,comments,shares,videosAddedToPlaylists,savesAdded,savesRemoved",
            maxResults=1,
        )
        .execute()
    )
    return response


def channel_top_videos(
    data_service: Resource,
    channel_id: str,
    *,
    days_back: int = 90,
    max_results: int = 20,
) -> Dict[str, Any]:
    """Get top performing videos for the channel in the specified period."""
    
    end = date.today()
    start = end - timedelta(days=days_back)

    analytics = _get_analytics_service(data_service)
    
    response = (
        analytics.reports()
        .query(
            ids=f"channel=={channel_id}",
            startDate=start.isoformat(),
            endDate=end.isoformat(),
            metrics="views,estimatedMinutesWatched,averageViewDuration,likes,comments,shares",
            dimensions="video",
            sort="-views",
            maxResults=max_results,
        )
        .execute()
    )
    return response


def channel_audience_demographics(
    data_service: Resource,
    channel_id: str,
    *,
    days_back: int = 90,
) -> Dict[str, Any]:
    """Get comprehensive audience demographics for the channel."""
    
    end = date.today()
    start = end - timedelta(days=days_back)

    analytics = _get_analytics_service(data_service)
    
    # Age and gender breakdown
    demographics_response = (
        analytics.reports()
        .query(
            ids=f"channel=={channel_id}",
            startDate=start.isoformat(),
            endDate=end.isoformat(),
            metrics="viewerPercentage",
            dimensions="ageGroup,gender",
            maxResults=200,
        )
        .execute()
    )
    
    return demographics_response


def channel_geography_stats(
    data_service: Resource,
    channel_id: str,
    *,
    days_back: int = 90,
) -> Dict[str, Any]:
    """Get geographic distribution of channel audience."""
    
    end = date.today()
    start = end - timedelta(days=days_back)

    analytics = _get_analytics_service(data_service)
    
    response = (
        analytics.reports()
        .query(
            ids=f"channel=={channel_id}",
            startDate=start.isoformat(),
            endDate=end.isoformat(),
            metrics="views,estimatedMinutesWatched,averageViewDuration",
            dimensions="country",
            sort="-views",
            maxResults=50,
        )
        .execute()
    )
    return response


def channel_traffic_sources(
    data_service: Resource,
    channel_id: str,
    *,
    days_back: int = 90,
) -> Dict[str, Any]:
    """Get traffic sources breakdown for the entire channel."""
    
    end = date.today()
    start = end - timedelta(days=days_back)

    analytics = _get_analytics_service(data_service)
    
    response = (
        analytics.reports()
        .query(
            ids=f"channel=={channel_id}",
            startDate=start.isoformat(),
            endDate=end.isoformat(),
            metrics="views,estimatedMinutesWatched",
            dimensions="insightTrafficSourceType",
            sort="-views",
            maxResults=20,
        )
        .execute()
    )
    return response


def channel_content_type_performance(
    data_service: Resource,
    channel_id: str,
    *,
    days_back: int = 90,
) -> Dict[str, Any]:
    """Get performance breakdown by content type/category."""
    
    end = date.today()
    start = end - timedelta(days=days_back)

    analytics = _get_analytics_service(data_service)
    
    # Get upload activity by time
    upload_timing = (
        analytics.reports()
        .query(
            ids=f"channel=={channel_id}",
            startDate=start.isoformat(),
            endDate=end.isoformat(),
            metrics="views",
            dimensions="uploaderType",
            maxResults=10,
        )
        .execute()
    )
    
    return upload_timing


def channel_monetization_metrics(
    data_service: Resource,
    channel_id: str,
    *,
    days_back: int = 90,
) -> Dict[str, Any]:
    """Get monetization and revenue metrics (if available)."""
    
    end = date.today()
    start = end - timedelta(days=days_back)

    analytics = _get_analytics_service(data_service)
    
    try:
        response = (
            analytics.reports()
            .query(
                ids=f"channel=={channel_id}",
                startDate=start.isoformat(),
                endDate=end.isoformat(),
                metrics="estimatedRevenue,estimatedAdRevenue,estimatedRedPartnerRevenue,grossRevenue,cpm,playbackBasedCpm,impressionBasedCpm",
                maxResults=1,
            )
            .execute()
        )
        return response
    except Exception:
        # Monetization data might not be available for all channels
        return {"rows": [], "error": "Monetization data not available"}


def channel_impressions_metrics(
    data_service: Resource,
    channel_id: str,
    *,
    days_back: int = 28,
) -> Dict[str, Any]:
    """Get impressions and click-through rate data for the channel."""
    
    end = date.today()
    start = end - timedelta(days=days_back)

    analytics = _get_analytics_service(data_service)
    
    try:
        response = (
            analytics.reports()
            .query(
                ids=f"channel=={channel_id}",
                startDate=start.isoformat(),
                endDate=end.isoformat(),
                metrics="impressions,impressionClickThroughRate,uniqueViewers",
                maxResults=1,
            )
            .execute()
        )
        return response
    except Exception:
        # Impressions data might not be available for all channels
        return {"rows": [], "error": "Impressions data not available"}


def channel_engagement_breakdown(
    data_service: Resource,
    channel_id: str,
    *,
    days_back: int = 28,
) -> Dict[str, Any]:
    """Get detailed engagement breakdown including saves and playlist adds."""
    
    end = date.today()
    start = end - timedelta(days=days_back)

    analytics = _get_analytics_service(data_service)
    
    response = (
        analytics.reports()
        .query(
            ids=f"channel=={channel_id}",
            startDate=start.isoformat(),
            endDate=end.isoformat(),
            metrics="views,likes,dislikes,comments,shares,savesAdded,savesRemoved,videosAddedToPlaylists,videosRemovedFromPlaylists",
            maxResults=1,
        )
        .execute()
    )
    return response


def channel_video_performance_stats(
    data_service: Resource,
    channel_id: str,
    *,
    days_back: int = 90,
) -> Dict[str, Any]:
    """Get aggregated video performance statistics for calculating averages."""
    
    end = date.today()
    start = end - timedelta(days=days_back)

    analytics = _get_analytics_service(data_service)
    
    # Get all videos' performance to calculate averages
    response = (
        analytics.reports()
        .query(
            ids=f"channel=={channel_id}",
            startDate=start.isoformat(),
            endDate=end.isoformat(),
            metrics="views,likes,comments,shares,averageViewDuration,estimatedMinutesWatched",
            dimensions="video",
            maxResults=200,  # Get more videos for better averages
        )
        .execute()
    )
    return response


def channel_subscriber_status_breakdown(
    data_service: Resource,
    channel_id: str,
    *,
    days_back: int = 28,
) -> Dict[str, Any]:
    """Get views, watch time, and avg view duration broken down by subscriber status for the channel (OAuth only)."""
    end = date.today()
    start = end - timedelta(days=days_back)
    analytics = _get_analytics_service(data_service)
    response = (
        analytics.reports()
        .query(
            ids=f"channel=={channel_id}",
            startDate=start.isoformat(),
            endDate=end.isoformat(),
            metrics="views,estimatedMinutesWatched,averageViewDuration",
            dimensions="subscribedStatus",
            maxResults=2,
        )
        .execute()
    )
    return response


def get_comprehensive_channel_analytics(
    data_service: Resource,
    channel_id: str,
    *,
    days_back: int = 90,
) -> Dict[str, Any]:
    """Get all available channel analytics in one convenient function."""
    
    analytics_data = {}
    
    try:
        analytics_data["growth_metrics"] = channel_growth_metrics(data_service, channel_id, days_back=days_back)
    except Exception as e:
        analytics_data["growth_metrics"] = {"error": str(e)}
    
    try:
        analytics_data["performance_summary"] = channel_performance_summary(data_service, channel_id, days_back=days_back)
    except Exception as e:
        analytics_data["performance_summary"] = {"error": str(e)}
    
    try:
        analytics_data["top_videos"] = channel_top_videos(data_service, channel_id, days_back=days_back)
    except Exception as e:
        analytics_data["top_videos"] = {"error": str(e)}
    
    try:
        analytics_data["demographics"] = channel_audience_demographics(data_service, channel_id, days_back=days_back)
    except Exception as e:
        analytics_data["demographics"] = {"error": str(e)}
    
    try:
        analytics_data["geography"] = channel_geography_stats(data_service, channel_id, days_back=days_back)
    except Exception as e:
        analytics_data["geography"] = {"error": str(e)}
    
    try:
        analytics_data["traffic_sources"] = channel_traffic_sources(data_service, channel_id, days_back=days_back)
    except Exception as e:
        analytics_data["traffic_sources"] = {"error": str(e)}
    
    try:
        analytics_data["monetization"] = channel_monetization_metrics(data_service, channel_id, days_back=days_back)
    except Exception as e:
        analytics_data["monetization"] = {"error": str(e)}
    
    try:
        analytics_data["impressions"] = channel_impressions_metrics(data_service, channel_id, days_back=days_back)
    except Exception as e:
        analytics_data["impressions"] = {"error": str(e)}
    
    try:
        analytics_data["engagement_breakdown"] = channel_engagement_breakdown(data_service, channel_id, days_back=days_back)
    except Exception as e:
        analytics_data["engagement_breakdown"] = {"error": str(e)}
    
    try:
        analytics_data["video_stats"] = channel_video_performance_stats(data_service, channel_id, days_back=days_back)
    except Exception as e:
        analytics_data["video_stats"] = {"error": str(e)}
    
    analytics_data["period"] = f"{date.today() - timedelta(days=days_back)} to {date.today()}"
    
    return analytics_data


def get_comprehensive_video_analytics(
    data_service: Resource,
    video_id: str,
    channel_id: str,
    *,
    days_back: int = 28,
) -> Dict[str, Any]:
    """Get all available video analytics in one convenient function."""
    
    analytics_data = {}
    
    try:
        analytics_data["summary_metrics"] = video_summary_metrics(data_service, video_id, channel_id, days_back=days_back)
    except Exception as e:
        analytics_data["summary_metrics"] = {"error": str(e)}
    
    try:
        analytics_data["audience_retention"] = audience_retention(data_service, video_id, channel_id, days_back=days_back)
    except Exception as e:
        analytics_data["audience_retention"] = {"error": str(e)}
    
    try:
        analytics_data["demographics"] = demographics_breakdown(data_service, video_id, channel_id, days_back=days_back)
    except Exception as e:
        analytics_data["demographics"] = {"error": str(e)}
    
    try:
        analytics_data["geography"] = geography_breakdown(data_service, video_id, channel_id, days_back=days_back)
    except Exception as e:
        analytics_data["geography"] = {"error": str(e)}
    
    try:
        analytics_data["traffic_sources"] = traffic_sources(data_service, video_id, channel_id, days_back=days_back)
    except Exception as e:
        analytics_data["traffic_sources"] = {"error": str(e)}
    
    try:
        analytics_data["monetization"] = video_monetization_metrics(data_service, video_id, channel_id, days_back=days_back)
    except Exception as e:
        analytics_data["monetization"] = {"error": str(e)}
    
    try:
        analytics_data["time_series"] = video_time_series_metrics(data_service, video_id, channel_id, days_back=days_back)
    except Exception as e:
        analytics_data["time_series"] = {"error": str(e)}
    
    try:
        analytics_data["engagement_metrics"] = video_engagement_metrics(data_service, video_id, channel_id, days_back=days_back)
    except Exception as e:
        analytics_data["engagement_metrics"] = {"error": str(e)}
    
    try:
        analytics_data["impressions"] = video_impressions_metrics(data_service, video_id, channel_id, days_back=days_back)
    except Exception as e:
        analytics_data["impressions"] = {"error": str(e)}
    
    analytics_data["period"] = f"{date.today() - timedelta(days=days_back)} to {date.today()}"
    
    return analytics_data


__all__ = [
    # Video-level analytics
    "video_summary_metrics",
    "audience_retention",
    "traffic_sources",
    "geography_breakdown",
    "demographics_breakdown",
    "video_monetization_metrics",
    "video_time_series_metrics",
    "video_engagement_metrics",
    "video_impressions_metrics",
    "video_subscriber_status_breakdown",
    # Channel-level analytics
    "channel_growth_metrics",
    "channel_performance_summary",
    "channel_top_videos",
    "channel_audience_demographics",
    "channel_geography_stats",
    "channel_traffic_sources",
    "channel_content_type_performance",
    "channel_monetization_metrics",
    "channel_impressions_metrics",
    "channel_engagement_breakdown",
    "channel_video_performance_stats",
    "channel_subscriber_status_breakdown",
    "get_comprehensive_channel_analytics",
    "get_comprehensive_video_analytics",
] 