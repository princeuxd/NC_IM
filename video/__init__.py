"""Video processing helpers (download, analytics, etc.)."""

from importlib import import_module
from typing import Any


def __getattr__(name: str) -> Any:  # noqa: D401
    mod = import_module("video.core")
    value = getattr(mod, name)
    globals()[name] = value
    return value

__all__ = [
    "extract_video_id",
    "fetch_and_save_video_metadata",
    "fetch_video_metadata",
    "fetch_video_analytics",
    "fetch_video_metrics",
    "download_video",
    "extract_audio",
    "extract_frames",
] 