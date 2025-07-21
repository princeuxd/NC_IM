"""OAuth-enhanced YouTube helpers.

Provides wrappers that extend the public helpers with private analytics
capabilities when OAuth credentials are available.
"""

from __future__ import annotations

from typing import Any, Dict, List

from pathlib import Path
from google_auth_oauthlib.flow import InstalledAppFlow  # type: ignore
from googleapiclient.discovery import build  # type: ignore
from google.oauth2.credentials import Credentials  # type: ignore
from google.auth.transport.requests import Request  # type: ignore

# Minimal scopes required for read operations and comments
SCOPES = [
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/youtube.force-ssl",
    "https://www.googleapis.com/auth/yt-analytics.readonly",
]

from youtube.public import get_service as get_public_service  # re-export

__all__ = [
    "get_service",
    "get_public_service",
]


def _load_credentials(token_file: Path):
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())  # type: ignore
            token_file.write_text(creds.to_json())
        return creds
    return None


def get_service(client_secrets_file: Path | str, token_file: Path | str, *, scopes: list[str] | None = None, **_ignored):  # type: ignore[override]
    """Create or refresh an OAuth credential and return a YouTube service."""

    client_secrets_file = Path(client_secrets_file)
    token_file = Path(token_file)

    creds = _load_credentials(token_file)

    if not creds:
        use_scopes = scopes or SCOPES
        flow = InstalledAppFlow.from_client_secrets_file(str(client_secrets_file), use_scopes)

        if hasattr(flow, "run_console"):
            creds = flow.run_console()  # type: ignore[attr-defined]
        else:
            # Older google-auth-oauthlib versions only support run_local_server
            creds = flow.run_local_server(port=0)
        token_file.write_text(creds.to_json())

    return build("youtube", "v3", credentials=creds) 