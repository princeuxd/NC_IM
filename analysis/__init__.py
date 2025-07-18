"""Analysis utilities namespace (sentiment, comments, etc.)."""

from importlib import import_module
from typing import Any


def __getattr__(name: str) -> Any:  # noqa: D401
    mod = import_module("analysis.core")
    value = getattr(mod, name)
    globals()[name] = value
    return value


__all__ = [
    "transcribe_audio",
    "analyze_transcript_sentiment",
    "fetch_comments",
    "analyze_comment_sentiment",
    "detect_logos",
    "save",
] 