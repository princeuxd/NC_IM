"""Public YouTube Data API helpers (API-key based)."""

from __future__ import annotations

from googleapiclient.discovery import Resource  # type: ignore
from googleapiclient.discovery import build  # type: ignore

__all__ = [
    "get_service",
]


def get_service(api_key: str | None = None) -> Resource:
    """Return a YouTube Data API v3 service authorised with *api_key*."""

    if api_key is None:
        raise ValueError("API key must be provided for public access")
    return build("youtube", "v3", developerKey=api_key) 