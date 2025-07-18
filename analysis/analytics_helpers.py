"""YouTube Analytics helpers for engagement time-series.

These functions require OAuth credentials with the scope
``https://www.googleapis.com/auth/yt-analytics.readonly``. They gracefully
return an empty list if credentials or requested dimensions are not available
so that the rest of the pipeline can continue in public-only mode.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, List, Sequence

from googleapiclient.discovery import Resource  # type: ignore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------


SUPPORTED_FREQUENCIES = {"day": "day", "week": "week", "month": "month"}
DEFAULT_METRICS = "views,likes,comments,shares"


def fetch_engagement_timeseries(
    analytics: Resource | None,
    video_id: str,
    channel_id: str,
    start_date: date,
    end_date: date,
    *,
    frequency: str = "day",
    metrics: str = DEFAULT_METRICS,
) -> List[list[Any]]:
    """Return list of rows for the requested frequency.

    Row format is identical to yt-analytics API response: the first column is
    the *period* (YYYY-MM-DD for day, etc.) followed by metric columns in the
    same order as ``metrics``.
    """

    if analytics is None:
        logger.info("No analytics client – skipping engagement timeseries fetch.")
        return []

    dim = SUPPORTED_FREQUENCIES.get(frequency)
    if dim is None:
        raise ValueError(f"Unsupported frequency '{frequency}'.")

    try:
        response = (
            analytics.reports()  # type: ignore[attr-defined]
            .query(
                ids=f"channel=={channel_id}",
                metrics=metrics,
                dimensions=dim,
                filters=f"video=={video_id}",
                startDate=start_date.isoformat(),
                endDate=end_date.isoformat(),
            )
            .execute()
        )
        return response.get("rows", [])  # type: ignore[return-value]
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Engagement timeseries fetch failed (%s)", exc)
        return []


# ---------------------------------------------------------------------------
# Simple correlation utility
# ---------------------------------------------------------------------------


def correlate_products_with_engagement(
    detections: Sequence[dict[str, Any]],
    timeseries: Sequence[list[Any]],
    *,
    window_days: int = 1,
) -> List[dict[str, Any]]:
    """Attach engagement delta before/after product appearance (daily).

    For each detection timestamp, we take the publication date (first column in
    ``timeseries``) that is closest *after* the appearance (floor to day). We
    compare metrics at \-window_days and +window_days to compute deltas.
    """

    if not timeseries:
        # Return detections unchanged if no data.
        return list(detections)

    # Build date->metrics mapping
    ts_map = {row[0]: row[1:] for row in timeseries}  # type: ignore[index]
    dates_sorted = sorted(ts_map.keys())

    enriched: list[dict[str, Any]] = []
    for det in detections:
        det_date = _timestamp_to_date(det["timestamp"])
        # Find nearest date on/after
        target_date = next((d for d in dates_sorted if d >= det_date), None)
        if target_date is None:
            enriched.append(det)
            continue
        before_date = _shift_date(target_date, -window_days)
        after_date = _shift_date(target_date, window_days)
        before_metrics = ts_map.get(before_date)
        after_metrics = ts_map.get(after_date)
        if before_metrics and after_metrics:
            delta = [a - b for a, b in zip(after_metrics, before_metrics)]
            det = {**det, "engagement_delta": delta}
        enriched.append(det)
    return enriched


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------


from datetime import datetime, timedelta


def _timestamp_to_date(seconds: float) -> str:
    # Placeholder: we don't have actual publish date/time here, so we just use
    # today's date. In a full implementation you would map video timestamp to
    # real calendar date relative to publishAt.
    return date.today().isoformat()


def _shift_date(d: str, days: int) -> str:
    dt = datetime.fromisoformat(d)
    dt2 = dt + timedelta(days=days)
    return dt2.date().isoformat()


__all__ = [
    "fetch_engagement_timeseries",
    "correlate_products_with_engagement",
] 