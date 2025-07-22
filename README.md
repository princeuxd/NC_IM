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

| Variable             | Purpose                              |
| -------------------- | ------------------------------------ |
| `YT_API_KEY`         | Google Data API key for public calls |
| `OPENROUTER_API_KEY` | LLM provider key (OpenRouter)        |
| `GROQ_API_KEY`       | LLM provider key (Groq)              |

_All settings can also be overridden via `.env` or programmatically through
`config.settings.update_from_kwargs()`._

## Contributing

Pull requests are welcome. Please ensure `black` and `ruff` pass before
submitting. Unit tests live under `tests/` (coming soon).
