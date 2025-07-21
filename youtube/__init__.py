"""Unified YouTube Data API helpers.

This package separates concerns between *public* API access (API key) and
OAuth-authenticated access while providing a stable surface to the rest of
NC_IM.  Import from:

    from youtube.public import get_service, fetch_video_metadata, fetch_comments
    from youtube.oauth import get_service, fetch_video_analytics

Existing higher-level modules may still import legacy helpers; those now call
into these wrappers so behaviour is unchanged while the structure is clearer.
"""

__all__ = [] 