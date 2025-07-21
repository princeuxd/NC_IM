"""Utilities for managing YouTube creator OAuth credentials.

This module centralizes helper functions related to onboarding creators (obtaining
and storing OAuth 2.0 credentials), listing existing creators, and removing
creators from the local credential store. The functions here are reusable from
both the Streamlit UI and command-line scripts.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

from google.oauth2.credentials import Credentials  # type: ignore
from youtube.oauth import get_service as get_oauth_service

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

# Project root two levels up from this file: <root>/auth/manager.py → <root>
ROOT_DIR = Path(__file__).resolve().parent.parent

# Directory where OAuth credential JSON files are stored (one per creator)
TOKENS_DIR = ROOT_DIR / "tokens"
TOKENS_DIR.mkdir(parents=True, exist_ok=True)

# Default client secret JSON (generated in Google Cloud Console and placed in
# project root). Callers can override this path if they wish.
DEFAULT_CLIENT_SECRET = ROOT_DIR / "client_secret.json"


# ---------------------------------------------------------------------------
# Public helper functions
# ---------------------------------------------------------------------------

def list_token_files() -> List[Path]:
    """Return a sorted list of credential JSON files for onboarded creators.

    We exclude files that are not authorised-user credential JSONs (e.g. raw
    `client_secret.json` files uploaded by mistake) by ensuring the JSON
    payload includes a ``refresh_token`` key.
    """

    candidate_files = sorted(TOKENS_DIR.glob("*.json"))
    valid_files: List[Path] = []

    for path in candidate_files:
        try:
            data = json.loads(path.read_text())
            # An authorised-user credential always contains these fields
            if "refresh_token" in data and "client_id" in data:
                valid_files.append(path)
        except Exception:
            # Skip unreadable or malformed JSON files silently – they will be
            # surfaced later if needed during individual credential loading.
            continue

    return valid_files


def validate_client_secret(client_secret_path: Path) -> Dict[str, Any]:
    """Validate a client secret JSON file and return metadata.
    
    Returns:
        Dict with 'valid', 'error', 'project_id', 'client_id' keys
    """
    try:
        if not client_secret_path.exists():
            return {"valid": False, "error": "File does not exist"}
        
        with open(client_secret_path) as f:
            data = json.load(f)
        
        # Check for required OAuth structure
        if "web" not in data and "installed" not in data:
            return {"valid": False, "error": "Invalid OAuth client format"}
        
        oauth_data = data.get("web") or data.get("installed", {})
        client_id = oauth_data.get("client_id", "")
        project_id = oauth_data.get("project_id", "")
        
        if not client_id:
            return {"valid": False, "error": "Missing client_id"}
        
        return {
            "valid": True,
            "error": None,
            "project_id": project_id,
            "client_id": client_id,
            "type": "web" if "web" in data else "installed"
        }
    except json.JSONDecodeError:
        return {"valid": False, "error": "Invalid JSON format"}
    except Exception as e:
        return {"valid": False, "error": f"Validation error: {str(e)}"}


def get_creator_details(token_file: Path) -> Dict[str, Any]:
    """Get detailed information about a creator including credential status.
    
    Returns comprehensive creator info including token validity and channel stats.
    """
    try:
        # Load credentials to check validity
        creds = Credentials.from_authorized_user_file(str(token_file))
        is_valid = creds and not creds.expired
        
        # Get channel info
        if is_valid:
            svc = get_oauth_service(DEFAULT_CLIENT_SECRET, token_file)
            resp = svc.channels().list(part="snippet,statistics", mine=True).execute()  # type: ignore[attr-defined]
            
            if resp["items"]:
                channel = resp["items"][0]
                snippet = channel["snippet"]
                stats = channel["statistics"]
                
                return {
                    "channel_id": channel["id"],
                    "title": snippet["title"],
                    "description": snippet.get("description", ""),
                    "thumbnail_url": snippet.get("thumbnails", {}).get("default", {}).get("url"),
                    "subscriber_count": int(stats.get("subscriberCount", 0)),
                    "video_count": int(stats.get("videoCount", 0)),
                    "view_count": int(stats.get("viewCount", 0)),
                    "created_at": snippet.get("publishedAt"),
                    "is_valid": True,
                    "token_file": token_file.name,
                    "last_checked": datetime.now().isoformat()
                }
        
        # Fallback for invalid/expired tokens
        return {
            "channel_id": token_file.stem,
            "title": "<Unknown - Token Invalid>",
            "description": "",
            "thumbnail_url": None,
            "subscriber_count": 0,
            "video_count": 0,
            "view_count": 0,
            "created_at": None,
            "is_valid": False,
            "token_file": token_file.name,
            "last_checked": datetime.now().isoformat()
        }
        
    except Exception as exc:
        logger.warning("Failed to load creator details from %s: %s", token_file, exc)
        return {
            "channel_id": token_file.stem,
            "title": "<Error Loading>",
            "description": "",
            "thumbnail_url": None,
            "subscriber_count": 0,
            "video_count": 0,
            "view_count": 0,
            "created_at": None,
            "is_valid": False,
            "token_file": token_file.name,
            "last_checked": datetime.now().isoformat(),
            "error": str(exc)
        }


def channel_info_from_token(
    token_file: Path, *, client_secret_file: Optional[Path] = None
) -> Tuple[str, str]:
    """Return ``(channel_id, channel_title)`` for a stored credential file.

    If the credentials are invalid / expired we fall back to returning the file
    stem as the channel ID and "<unknown>" as the title to avoid hard failures.
    """
    client_secret_file = client_secret_file or DEFAULT_CLIENT_SECRET

    try:
        svc = get_oauth_service(client_secret_file, token_file)
        resp = svc.channels().list(part="snippet", mine=True).execute()  # type: ignore[attr-defined]
        item = resp["items"][0]
        return item["id"], item["snippet"]["title"]
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Failed to load channel info from %s: %s", token_file, exc)
        return token_file.stem, "<unknown>"


def onboard_creator(
    client_secret_file: Optional[Path] = None,
    *,
    tokens_dir: Optional[Path] = None,
    scopes: List[str] | None = None,
) -> Tuple[Path, str, str]:
    """Run the OAuth flow for a new creator and persist their credentials.

    Args:
        client_secret_file: Path to the OAuth client_secret.json. If omitted we
            look for ``client_secret.json`` in the project root.
        tokens_dir: Override where the credentials are written (defaults to
            ``<project_root>/tokens``).
        scopes: Optional list of scopes to request instead of the defaults.

    Returns:
        token_path: Final JSON file containing the credentials (named
            ``<channel_id>.json``).
        channel_id: YouTube channel ID of the onboarded creator.
        channel_title: Human-readable channel title.

    Raises:
        RuntimeError: If the OAuth flow fails for any reason.
    """
    client_secret_file = client_secret_file or DEFAULT_CLIENT_SECRET
    tokens_dir = tokens_dir or TOKENS_DIR

    if not client_secret_file.exists():
        raise FileNotFoundError(
            f"Client secret JSON not found at {client_secret_file}. "
            "Provide it explicitly or place it in the project root."
        )

    temp_token = tokens_dir / "_temp_creds.json"
    try:
        svc = get_oauth_service(client_secret_file, temp_token, scopes=scopes)
        me = svc.channels().list(part="id,snippet", mine=True).execute()  # type: ignore[attr-defined]
        item = me["items"][0]
        channel_id: str = item["id"]
        channel_title: str = item["snippet"]["title"]

        final_token_path = tokens_dir / f"{channel_id}.json"
        # Overwrite existing credentials atomically (replace is atomic on POSIX)
        temp_token.replace(final_token_path)
        logger.info("Onboarded channel '%s' (%s)", channel_title, channel_id)
        return final_token_path, channel_id, channel_title
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("OAuth onboarding failed: %s", exc)
        temp_token.unlink(missing_ok=True)
        raise RuntimeError("OAuth onboarding failed") from exc


def remove_creator(channel_id: str, *, tokens_dir: Optional[Path] = None) -> bool:
    """Delete stored credentials for a creator.

    Returns ``True`` if the credentials were removed, ``False`` if no matching
    credential file was found.
    """
    tokens_dir = tokens_dir or TOKENS_DIR
    token_file = tokens_dir / f"{channel_id}.json"
    if token_file.exists():
        token_file.unlink()
        logger.info("Removed credentials for channel %s", channel_id)
        return True

    logger.warning("No credentials found for channel %s", channel_id)
    return False


def refresh_creator_token(channel_id: str, *, tokens_dir: Optional[Path] = None) -> bool:
    """Attempt to refresh an expired token for a creator.
    
    Returns True if refresh was successful, False otherwise.
    """
    tokens_dir = tokens_dir or TOKENS_DIR
    token_file = tokens_dir / f"{channel_id}.json"
    
    if not token_file.exists():
        return False
    
    try:
        creds = Credentials.from_authorized_user_file(str(token_file))
        if creds and creds.expired and creds.refresh_token:
            from google.auth.transport.requests import Request
            creds.refresh(Request())
            # Save refreshed credentials
            token_file.write_text(creds.to_json())
            logger.info("Refreshed credentials for channel %s", channel_id)
            return True
    except Exception as exc:
        logger.warning("Failed to refresh token for %s: %s", channel_id, exc)
    
    return False 