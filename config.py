import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv  # type: ignore

# Load .env if present
load_dotenv()


@dataclass
class Settings:
    yt_api_key: str
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

    if not args.api_key:
        parser.error("You must provide --api-key or set YT_API_KEY in environment.")

    return Settings(
        yt_api_key=args.api_key,
        client_secrets_file=args.client_secrets_file,
        token_file=args.token_file,
    ) 