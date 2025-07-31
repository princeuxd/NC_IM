import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv  # type: ignore

# Load .env if present
load_dotenv()


@dataclass
class Settings:
    yt_api_key: str | None
    client_secrets_file: Path
    token_file: Path


def parse_args() -> Settings:
    parser = argparse.ArgumentParser(description="YouTube API configuration")
    parser.add_argument("--api-key", dest="api_key", help="YouTube Data API v3 key", default=os.getenv("YT_API_KEY"))
    parser.add_argument(
        "--client-secrets-file",
        dest="client_secrets_file",
        type=Path,
        help="Path to OAuth client secret JSON file",
        default=os.getenv("OAUTH_CLIENT_SECRETS_FILE", "client_secret.json"),
    )
    parser.add_argument(
        "--token-file",
        dest="token_file",
        type=Path,
        help="Path to store OAuth token JSON file",
        default=os.getenv("OAUTH_TOKEN_FILE", "oauth_token.json"),
    )

    # Use parse_known_args so that scripts can define additional CLI flags
    # (e.g., --channel-id) without config.py failing due to unknown args.
    args = parser.parse_known_args()[0]

    # API key can be omitted if OAuth client secrets exist – scripts can then
    # build an authenticated service instead of the public key-based one.
    if not args.api_key and not args.client_secrets_file.exists():
        parser.error(
            "Provide --api-key / YT_API_KEY or have a valid client_secret.json for OAuth."
        )

    return Settings(
        yt_api_key=args.api_key,
        client_secrets_file=args.client_secrets_file,
        token_file=args.token_file,
    ) 