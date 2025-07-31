"""Unified YouTube Data API helpers.

This package separates concerns between *public* API access (API key) and
OAuth-authenticated access while providing a stable surface to the rest of
NC_IM.  Import from:

    from src.youtube.public import get_service, fetch_video_metadata, fetch_comments
    from src.youtube.oauth import get_service, fetch_video_analytics

Existing higher-level modules may still import legacy helpers; those now call
into these wrappers so behaviour is unchanged while the structure is clearer.
"""

# Import key functions for easier access
from src.youtube.public import get_service as get_public_service
from src.youtube.oauth import get_service as get_oauth_service

__all__ = [
    'get_public_service',
    'get_oauth_service'
] 