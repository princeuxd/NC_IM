# NC_IM - Ncompas Influencer Marketing

## Key Packages & Entry-points

| Area         | Module / Package                         | Highlights                                           |
| ------------ | ---------------------------------------- | ---------------------------------------------------- |
| Sentiment    | `analysis/sentiment.py`                  | Local HF model, single `sentiment_scores()` function |
| Comments     | `analysis/comments.py`                   | Fetch + attach sentiment + JSON helper               |
| Audio        | `analysis/audio.py`                      | Whisper transcription + sentiment                    |
| Vision       | `analysis/video_frames.py`               | Frame extraction + multimodal LLM object detection   |
| YouTube API  | `youtube/public.py` / `youtube/oauth.py` | Clean wrappers for public-key & OAuth access         |
| Streamlit UI | `simple_streamlit_app.py`                | Modern UI built on top of the modular API            |

## Quick Start

```bash
# Install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Launch the Streamlit UI
streamlit run simple_streamlit_app.py
```

## Environment Variables

| Variable              | Purpose                                           |
| --------------------- | ------------------------------------------------- |
| `YT_API_KEY`          | Google Data API key for public calls              |
| `OAUTH_CLIENT_ID`     | OAuth 2.0 Client ID from Google Cloud Console     |
| `OAUTH_CLIENT_SECRET` | OAuth 2.0 Client Secret from Google Cloud Console |
| `OAUTH_PROJECT_ID`    | Google Cloud Project ID (optional)                |
| `OPENROUTER_API_KEY`  | LLM provider key (OpenRouter)                     |
| `GROQ_API_KEY`        | LLM provider key (Groq)                           |

### OAuth Setup

Instead of uploading a `client_secret.json` file, configure OAuth credentials via environment variables:

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create or select a project
3. Enable the YouTube Data API v3 and YouTube Analytics API
4. Go to **Credentials** → **Create Credentials** → **OAuth 2.0 Client IDs**
5. Choose **Desktop application** as the application type
6. Extract the values and add to your `.env` file:

```bash
# OAuth Configuration
OAUTH_CLIENT_ID=your_client_id_here
OAUTH_CLIENT_SECRET=your_client_secret_here
OAUTH_PROJECT_ID=your_project_id_here  # Optional
```

_All settings can also be overridden via `.env` or programmatically through
`config.settings.update_from_kwargs()`._

## Contributing

Pull requests are welcome. Please ensure `black` and `ruff` pass before
submitting. Unit tests live under `tests/` (coming soon).
