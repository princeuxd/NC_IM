"""Business-logic helpers for managing creator OAuth credentials.

This module simply re-exports the rich functionality already implemented in
`auth.manager`, providing a clean, UI-friendly facade (and maintaining backward
compatibility for earlier underscore-prefixed helper names).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Any, Dict, Optional

from src.auth.manager import (
    list_token_files as _list_token_files_original,
    get_creator_details as _get_creator_details_original,
    refresh_creator_token as _refresh_creator_token_original,
    remove_creator as _remove_creator_original,
    onboard_creator as _onboard_creator_original,
    validate_env_oauth_config,
    create_temp_client_secret_file,
    TOKENS_DIR,
)

# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------

def list_token_files() -> List[Path]:
    """Return credential files for all onboarded creators."""
    return _list_token_files_original()


def get_creator_details(token_file: Path) -> Dict[str, Any]:
    """Return status and channel stats for a given credential file."""
    return _get_creator_details_original(token_file)


def refresh_creator_token(channel_id: str) -> bool:
    """Attempt to refresh an expired token for the given channel ID."""
    return _refresh_creator_token_original(channel_id)


def remove_creator(channel_id: str) -> bool:
    """Delete stored credentials for the given channel ID."""
    return _remove_creator_original(channel_id)


def onboard_creator(
    client_secret_file: Optional[Path] = None,
    *,
    tokens_dir: Optional[Path] = None,
    scopes: List[str] | None = None,
) -> Tuple[Path, str, str]:
    """Run OAuth flow and persist credentials for a new creator."""
    return _onboard_creator_original(
        client_secret_file, tokens_dir=tokens_dir, scopes=scopes
    )

# Environment helpers (re-exported directly)
validate_env_oauth_config = validate_env_oauth_config  # noqa: E305
create_temp_client_secret_file = create_temp_client_secret_file  # noqa: E305

# Export path constant for convenience
TOKENS_DIR = TOKENS_DIR  # type: ignore[misc]

# ---------------------------------------------------------------------------
# Backward-compatibility aliases (keeps legacy underscore-prefixed names alive)
# ---------------------------------------------------------------------------

# Older code referenced underscore-prefixed helpers imported from the
# monolithic Streamlit file. We keep those names as thin aliases so that legacy
# imports continue to work while new code can switch to the clearer names.

_list_token_files = list_token_files  # type: ignore[invalid-name]
_get_creator_details = get_creator_details  # type: ignore[invalid-name]
_refresh_creator_token = refresh_creator_token  # type: ignore[invalid-name]
_remove_creator = remove_creator  # type: ignore[invalid-name]
