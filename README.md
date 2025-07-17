# YouTube Data Fetcher

A small utility to access YouTube Data API v3 using either:

1. **Public API key** – fetch publicly available metadata without user authorization.
2. **OAuth 2.0** – obtain an influencer’s authorization to access additional (private) channel data.

## Features

- Load configuration from `.env` or CLI arguments.
- Simple helpers to build YouTube service objects.
- Example script to fetch channel details.

## Quick Start

### 1. Clone & set up a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Google Cloud Console configuration

1. Create / select a project.
2. Enable **YouTube Data API v3**.
3. **Credentials → Create credentials → API key** (public access).
4. **Credentials → Create credentials → OAuth client ID**.
   - Application type: **Desktop** or **Other**.
   - Download the JSON file, rename/move to `client_secret.json` in project root (or point to it via `--client-secrets-file`).

### 3. Environment variables

```ini
YT_API_KEY=YOUR_PUBLIC_API_KEY
OAUTH_CLIENT_SECRETS_FILE=client_secret.json
OAUTH_TOKEN_FILE=oauth_token.json
```

### 4. Fetch data

Public metadata (no OAuth required):

Ncompas Channel

```bash
python yt_data.py --channel-id UCBwUDcDZaYsfbscchwL_V0Q
```

Access private data with OAuth (first run opens browser / console prompt):

```bash
python yt_data.py --use-oauth --channel-id mine
```

Subsequent runs reuse `oauth_token.json`.

## Project Structure

```
├── config.py              # Config loader
├── youtube_client.py      # Authentication helpers
├── yt_data.py  # Example script
├── requirements.txt
├── env
└── README.md
```
