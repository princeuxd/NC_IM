"""Generate executive summary of analysis via LLM (OpenRouter / Groq).

Reads core artefacts (metadata, product_impact, comments_sentiment, etc.) and
prompts an LLM to produce a concise markdown summary aimed at brand managers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

import openai  # type: ignore
from openai import OpenAI  # type: ignore

from config.settings import SETTINGS, PipelineSettings

logger = logging.getLogger(__name__)

# Reduce noise from HTTP request logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------


KEY_FILES = [
    "metadata.json",
    "product_impact.json",
    "comments_sentiment.json",
    "transcript_sentiment.json",
]


def _configure_client(settings: PipelineSettings = SETTINGS):
    """Configure OpenAI client with proper API key and base URL."""
    try:
        if settings.openrouter_api_key:
            # Use proper client initialization
            client = OpenAI(
                api_key=settings.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
                default_headers={
                    "HTTP-Referer": "https://github.com/your-repo",  # Optional
                    "X-Title": "YouTube Video Summarizer"  # Optional
                }
            )
            # Store client globally for use in _try_generate
            globals()['_openai_client'] = client
            logger.info("Configured OpenRouter client")
            return True
        if settings.groq_api_key:
            client = OpenAI(
                api_key=settings.groq_api_key,
                base_url="https://api.groq.com/openai/v1"
            )
            globals()['_openai_client'] = client
            logger.info("Configured Groq client")
            return True
        logger.info("No API key found in settings")
        return False
    except Exception as e:
        logger.error("Failed to configure client: %s", e)
        return False


def _load_snippets(folder: Path, limit_chars: int = 4_000) -> str:
    """Concatenate truncated JSON snippets to stay within token limits."""
    parts: List[str] = []
    total = 0
    for fname in KEY_FILES:
        fpath = folder / fname
        if not fpath.exists():
            continue
        data = fpath.read_text()[:limit_chars]
        parts.append(f"### {fname}\n```json\n{data}\n```")
        total += len(data)
        if total > limit_chars * 2:  # rough limit
            break
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _try_generate(folder: Path, model: str, settings: PipelineSettings) -> Optional[str]:
    """Attempt one LLM call and return markdown or None on provider error."""
    context = _load_snippets(folder)
    if not context:
        return None

    system = (
        "You are an analytical assistant. Given structured JSON artefacts from "
        "a YouTube video analysis, produce a concise markdown report (max 400 "
        "words) summarising product appearances, engagement impact, audience "
        "sentiment trends, and actionable insights for a brand manager. Use "
        "bullet points where appropriate and include an overall sentiment score "
        "(-1..1)."
    )

    user_msg = (
        "Here are the artefacts. Summarise insights following the instructions.\n\n"
        + context
    )

    try:
        # Use the properly initialized client
        client = globals().get('_openai_client')
        if client:
            # Set lower timeout to fail fast on rate limits
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.3,
                max_tokens=800,  # Limit response length
                timeout=30  # Fail fast instead of retrying
            )
            # Handle None response content properly
            if resp.choices and len(resp.choices) > 0 and resp.choices[0].message:
                md = resp.choices[0].message.content or ""
            else:
                logger.error("Empty response from model %s", model)
                return None
        else:
            logger.error("No OpenAI client configured")
            return None
            
        if md.lstrip().startswith("<!"):
            logger.warning("Provider returned HTML error page for model %s", model)
            return None  # provider returned HTML error
        
        (folder / "summary.md").write_text(md)
        logger.info("Successfully generated summary with model %s", model)
        return md
    except Exception as e:
        logger.error("Summary generation failed for model %s: %s", model, e)
        return None


def generate_summary(folder: Path, *, settings: PipelineSettings = SETTINGS) -> str | None:
    """Create markdown executive summary and write summary.md.

    Returns the markdown string or None if generation skipped.
    """

    if not _configure_client(settings):
        logger.info("No LLM API key – skipping summary generation.")
        return None

    fallback_models = [
        settings.summary_model, 
        "google/gemma-3-27b-it:free"  # Only use models that work reliably
    ]
    for m in fallback_models:
        md = _try_generate(folder, m, settings)
        if md:
            logger.info("Generated summary with model %s", m)
            return md

    logger.warning("All summary generation attempts failed.")
    return None


__all__ = ["generate_summary"] 