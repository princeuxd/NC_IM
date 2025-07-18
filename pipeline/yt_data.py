#!/usr/bin/env python3
"""Example script to fetch YouTube data.

Usage examples:
    # Using API key only (public data)
    python fetch_youtube_data.py --api-key YOUR_KEY --channel-id UC_x5XG1OV2P6uZZ5FSM9Ttw

    # Using OAuth for private channel data (asks to authenticate on first run)
    python fetch_youtube_data.py \
        --api-key YOUR_KEY \
        --client-secrets-file client_secret.json \
        --token-file oauth_token.json \
        --use-oauth --channel-id mine
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path so that ``import config`` etc. work when this
# file is executed directly (``python pipeline/yt_data.py``).
# ---------------------------------------------------------------------------

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------

import argparse
import json
from typing import Any

from config import parse_args  # type: ignore[attr-defined]
from auth import get_oauth_service, get_public_service  # type: ignore[attr-defined]


def parse_extra() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--channel-id",
        required=True,
        help="Target channel ID. Use 'mine' to fetch authenticated user's channel when --use-oauth is set.",
    )
    parser.add_argument(
        "--use-oauth",
        action="store_true",
        help="Use OAuth service instead of public API-only requests.",
    )
    return parser.parse_known_args()[0]


def dump_json(data: Any):
    print(json.dumps(data, indent=2, ensure_ascii=False))


def main():
    settings = parse_args()  # type: ignore[misc]
    extra = parse_extra()

    if extra.use_oauth:
        service = get_oauth_service(settings.client_secrets_file, settings.token_file)  # type: ignore[misc]
    else:
        service = get_public_service(settings.yt_api_key)  # type: ignore[misc]

    request = service.channels().list(part="snippet,statistics", id=extra.channel_id)
    if extra.channel_id == "mine":
        # When using OAuth and channel_id 'mine', set mine=True instead of id param.
        request = service.channels().list(part="snippet,statistics", mine=True)

    response = request.execute()
    dump_json(response)


if __name__ == "__main__":
    main() 