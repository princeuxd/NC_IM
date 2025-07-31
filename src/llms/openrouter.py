from __future__ import annotations

from typing import Any, Dict, List

from openai import OpenAI

from .base import LLMClient
from src.config.settings import SETTINGS


class OpenRouterClient(LLMClient):
    """Wrapper around the OpenRouter API compatible with the OpenAI SDK."""

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str | None):
        if not api_key:
            raise ValueError("OpenRouter API key must be provided.")

        # The OpenAI Python library v1.0+ supports custom hosts via *base_url*.
        # We initialise a dedicated client instance to avoid global side-effects.
        self._client = OpenAI(
            api_key=api_key,
            base_url=self.BASE_URL,
            default_headers={
                "HTTP-Referer": "https://github.com/prince/NC_IM",  # project attribution per provider policy
                "X-Title": "NC_IM Video Analyzer",
            },
        )

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def chat(self, messages: Any, model: str | None = None, **kwargs: Any) -> str:  # type: ignore[override]
        """Return the assistant reply for *messages* using *model*."""
        if model is None:
            model = SETTINGS.openrouter_chat_model

        completion = self._client.chat.completions.create(  # type: ignore[arg-type]
            model=model,
            messages=messages,
            **kwargs,
        )
        return completion.choices[0].message.content or ""

    # OpenRouter does not currently expose embeddings on the free tier.
    # We raise by default so callers can handle gracefully.
    def embed(self, texts: List[str], **kwargs: Any):  # type: ignore[override]
        raise NotImplementedError("Embeddings not supported by OpenRouter client.") 