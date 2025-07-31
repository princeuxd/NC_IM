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
import sys

# Minimal scopes required for read operations and comments
SCOPES = [
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/youtube.force-ssl",
    "https://www.googleapis.com/auth/yt-analytics.readonly",
]

from src.youtube.public import get_service as get_public_service  # re-export

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

    # ------------------------------------------------------------------
    # Perform OAuth flow if no valid credentials are cached
    # ------------------------------------------------------------------
    if not creds:
        use_scopes = scopes or SCOPES
        flow = InstalledAppFlow.from_client_secrets_file(
            str(client_secrets_file), use_scopes
        )

        is_streamlit = "streamlit" in sys.modules

        if is_streamlit:
            # ---------------- Streamlit copy-and-paste flow -------------
            # Import Streamlit lazily to avoid hard dependency when running
            # this function in non-Streamlit contexts (e.g. CLI scripts).
            import streamlit as st  # type: ignore

            # Use the legacy out-of-band flow by explicitly setting the
            # redirect URI on the flow object (mirrors what run_console()
            # does internally).
            flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"

            auth_url, _ = flow.authorization_url(
                prompt="consent",
                access_type="offline",
                include_granted_scopes="true",
            )

            st.markdown(
                "### 🔐 Google authorisation required\n"
                "1. Click the link below and complete the consent screen.\n"
                "2. Google will show you a **verification code** – copy it.\n"
                "3. Paste the code in the box and press Enter."
            )
            st.markdown(
                f"[Open consent page →]({auth_url})",
                unsafe_allow_html=True,
            )

            code = st.text_input("Paste verification code:")
            if not code:
                # Stop execution until the user supplies the code; Streamlit
                # reruns the script automatically on each input change.
                st.stop()

            # Exchange code for tokens
            flow.fetch_token(code=code.strip())
            creds = flow.credentials
        else:
            # ---------------- Console / local-server fallback ------------
            if hasattr(flow, "run_console"):
                creds = flow.run_console()  # type: ignore[attr-defined]
            else:
                # Older google-auth-oauthlib versions only support local server
                creds = flow.run_local_server(port=0)

        # Persist newly obtained credentials for future sessions
        token_file.write_text(creds.to_json())

    return build("youtube", "v3", credentials=creds) 