"""YouTube API helper utilities.

Provides functions to build YouTube Data API v3 service objects using either:
1. A simple public API key.
2. OAuth 2.0 user authorization to access a creator's private data.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from google.auth.transport.requests import Request  # type: ignore
from google.oauth2.credentials import Credentials  # type: ignore
from google_auth_oauthlib.flow import InstalledAppFlow  # type: ignore
from googleapiclient.discovery import build  # type: ignore

# Minimum scopes required for full pipeline features.
# - youtube.readonly → public & private YouTube Data API v3 access
# - yt-analytics.readonly → channel-level analytics (audience retention, impressions, etc.)
# Feel free to append additional scopes if you extend the pipeline.
SCOPES = [
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/yt-analytics.readonly",
]

logger = logging.getLogger(__name__)


def get_public_service(api_key: str):
    """Return a YouTube API service using an API key for public data."""
    if not api_key:
        raise ValueError("API key must be provided for public access.")
    logger.debug("Creating YouTube service with API key.")
    return build("youtube", "v3", developerKey=api_key)


def _load_credentials(token_file: Path) -> Optional[Credentials]:
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)
        if creds and creds.expired and creds.refresh_token:
            logger.debug("Refreshing expired credentials.")
            creds.refresh(Request())
            # Save the refreshed token
            token_file.write_text(creds.to_json())
        return creds
    return None


def get_oauth_service(client_secrets_file: Path, token_file: Path, scopes: list[str] | None = None):
    """Return a YouTube API service authorized via OAuth 2.0.

    Args:
        client_secrets_file: Path to GCP OAuth client secret JSON.
        token_file: Where to store the credentials for future runs.
        scopes: Optional custom scopes list; defaults to SCOPES.
    """
    scopes = scopes or SCOPES

    creds = _load_credentials(token_file)

    if not creds:
        logger.info("No valid credentials found; starting OAuth flow.")
        flow = InstalledAppFlow.from_client_secrets_file(str(client_secrets_file), scopes)

        # Use run_console when available (Google's newer versions). If not, fall back
        # to run_local_server which opens a browser for the OAuth flow.
        if hasattr(flow, "run_console"):
            creds = flow.run_console()  # type: ignore[attr-defined]
        else:
            # 0 selects an arbitrary free port.
            creds = flow.run_local_server(port=0)
        # Save credentials
        logger.info("Saving new credentials to %s", token_file)
        token_file.write_text(creds.to_json())

    service = build("youtube", "v3", credentials=creds)
    return service 