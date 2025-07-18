# Ncompas Influencer Marketing MVP

End-to-end toolkit that downloads a video, pulls analytics/metrics, transcribes the audio,
detects products in video frames via a multimodal LLM, analyses audience engagement &
sentiment, and writes everything to `reports/<video-id>/` – all from a single URL or
via a Streamlit UI.

The script automatically chooses the best authentication method that is available:

1. **OAuth 2.0** – if `client_secret.json` is present (and the token can be created) the
   pipeline fetches owner-only analytics such as audience-retention, watch-time, etc.
2. **Public API key** – otherwise it falls back to a simple API-key and fetches everything
   that is public.

## Features

- Load configuration from `.env` or CLI arguments.
- Multimodal LLM object-detection (default: `x-ai/grok-2-vision-1212`) to timestamp when
  products/brands appear on screen.
- Daily engagement-delta correlation: views / likes / comments before & after each
  product appearance (requires OAuth).
- LLM-based sentiment analysis for comments & transcript (falls back to TextBlob when
  no API key).
- One-click executive summary (`summary.md`) generated with an LLM.
- Streamlit dashboard with creator OAuth onboarding and dynamic settings (frame interval,
  vision model route).
- Simple helpers to build YouTube service objects and fetch channel details.

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
# Optional – unlock LLM enhancements (vision detection, sentiment, summary)
OPENROUTER_API_KEY=...
# or
GROQ_API_KEY=...

# Frame extraction / model route defaults (can also be changed in Streamlit UI)
FRAME_INTERVAL_SEC=5
OBJECT_DETECTION_MODEL=x-ai/grok-2-vision-1212
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

### 5. Run the Streamlit dashboard

```bash
streamlit run streamlit_app.py
# Tab 1: Creator onboarding (OAuth flow)
# Tab 2: Paste YouTube URL → analysis, product timeline & executive summary
```

## Project Structure

```
├── config/        ← config.core.py (arg/env loader)
├── auth/          ← auth.youtube.py (API key & OAuth helpers)
├── video/         ← video.core.py (download, analytics, metrics)
├── analysis/      ← analysis.core.py (transcript + comment sentiment, logos)
│   ├── object_detection.py     ← vision-LLM batch helper
│   ├── analytics_helpers.py    ← engagement time-series & correlation
│   ├── sentiment_llm.py        ← LLM sentiment scorer
│   └── summarizer.py           ← executive summary generator
├── pipeline/
│   ├── fetch_and_analyze.py  ← main pipeline entry-point
│   └── yt_data.py            ← lightweight channel-info helper
├── streamlit_app.py  ← interactive dashboard
├── requirements.txt
└── README.md
```
