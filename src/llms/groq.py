from __future__ import annotations

from typing import Any

from openai import OpenAI

from .base import LLMClient
from src.config.settings import SETTINGS


class GroqClient(LLMClient):
    """Wrapper for Groq Cloud LLMs via the OpenAI-compatible endpoint."""

    BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(self, api_key: str | None):
        if not api_key:
            raise ValueError("Groq API key must be provided.")
        self._client = OpenAI(api_key=api_key, base_url=self.BASE_URL)

    def chat(self, messages: Any, model: str | None = None, **kwargs: Any) -> str:  # type: ignore[override]
        # The OpenAI SDK type hints expect a specific message schema; we bypass strict
        # checking here because Groq follows the same runtime contract.
        if model is None:
            model = SETTINGS.groq_chat_model

        completion = self._client.chat.completions.create(  # type: ignore[arg-type]
            model=model, messages=messages, **kwargs
        )  # type: ignore
        return completion.choices[0].message.content or ""

    def embed(self, texts: List[str], **kwargs: Any):  # type: ignore[override]
        raise NotImplementedError("Embeddings not supported by Groq client yet.") 