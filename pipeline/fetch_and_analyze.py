#!/usr/bin/env python3
"""End-to-end pipeline: download video, fetch metadata, owner analytics (if possible),
transcribe, sentiment, logo detection, correlation.

Usage examples:
    python fetch_and_analyze.py --url https://youtu.be/7lCDEYXw3mM \
        --api-key $YT_API_KEY \
        --client-secrets-file client_secret.json \
        --token-file oauth_token.json

If OAuth flags are omitted the script runs in public-only mode (skips analytics).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Make sibling top-level packages importable when this file is executed as a
# script (``python pipeline/fetch_and_analyze.py``). Python normally puts the
# *containing* directory (``pipeline/``) on ``sys.path`` which means sibling
# packages like ``analysis`` are not visible.  We prepend the parent directory
# to ``sys.path`` at runtime so that ``import analysis`` etc. work without the
# user needing to set PYTHONPATH or use ``python -m``.
# ---------------------------------------------------------------------------

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------

import argparse
import logging
from config import parse_args as parse_base
from pipeline.core import run_pipeline

logging.basicConfig(level=logging.INFO)


def parse_cli():
    """Parse command-line arguments."""
    base = parse_base()
    parser = argparse.ArgumentParser(
        add_help=True, description="Fetch and analyze a YouTube video."
    )
    parser.add_argument("--url", required=True, help="YouTube video URL")
    parser.add_argument("--output", default="reports", help="Output directory root")
    args = parser.parse_args(namespace=base)
    return args


def main():
    """Run the main analysis pipeline from the command line."""
    args = parse_cli()
    run_pipeline(
        url=args.url,
        output_dir=Path(args.output),
        public_api_key=args.yt_api_key,
        client_secrets_file=args.client_secrets_file,
        token_file=args.token_file,
    )


if __name__ == "__main__":
    main() 