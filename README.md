# Ncompas Influencer Marketing MVP

End-to-end toolkit that downloads a video, pulls analytics/metrics, transcribes the audio,
does basic sentiment + logo analysis and writes everything to `reports/<video-id>/` – all
from a single URL.

The script automatically chooses the best authentication method that is available:

1. **OAuth 2.0** – if `client_secret.json` is present (and the token can be created) the
   pipeline fetches owner-only analytics such as audience-retention, watch-time, etc.
2. **Public API key** – otherwise it falls back to a simple API-key and fetches everything
   that is public.

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

### 4. Run the pipeline (video URL → full report)

```bash
# Public-only mode
python pipeline/fetch_and_analyze.py \
       --url "https://youtu.be/VLwhqqEm2L8" \
       --api-key "$YT_API_KEY"

# Owner analytics mode – same command, OAuth files present in project root
python pipeline/fetch_and_analyze.py \
       --url "https://youtu.be/VLwhqqEm2L8"
```

For simple channel metadata retrieval you can still use the helper:

```bash
python pipeline/yt_data.py --channel-id UCBwUDcDZaYsfbscchwL_V0Q
```

## Project Structure

```
├── config/        ← config.core.py (arg/env loader)
├── auth/          ← auth.youtube.py (API key & OAuth helpers)
├── video/         ← video.core.py (download, analytics, metrics)
├── analysis/      ← analysis.core.py (transcript + comment sentiment, logos)
├── pipeline/
│   ├── fetch_and_analyze.py  ← main pipeline entry-point
│   └── yt_data.py            ← lightweight channel-info helper
├── requirements.txt
└── README.md
```
