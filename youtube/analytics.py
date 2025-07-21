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
# Public report helpers
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


__all__ = [
    "video_summary_metrics",
    "audience_retention",
    "traffic_sources",
    "geography_breakdown",
] 