from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class LLMClient(ABC):
    """Abstract interface for language model providers (chat focus)."""

    @abstractmethod
    def chat(self, messages: List[Dict[str, Any]], **kwargs: Any) -> str:  # noqa: D401
        """Send chat messages and return the assistant reply text."""

    # Optional API: embeddings or other tasks can be implemented by providers.
    def embed(self, texts: List[str], **kwargs: Any) -> List[List[float]]:  # noqa: D401
        """Return vector embeddings for *texts* or raise if unsupported."""
        raise NotImplementedError("Embedding not implemented for this client.")


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def get_client(provider: str, api_key: str | None = None) -> "LLMClient":
    """Return an LLMClient for *provider* ('openrouter', 'groq', or 'gemini')."""

    provider = provider.lower().strip()
    if provider == "openrouter":
        from .openrouter import OpenRouterClient  # local import to avoid heavy deps

        return OpenRouterClient(api_key=api_key)
    if provider == "groq":
        from .groq import GroqClient

        return GroqClient(api_key=api_key)
    if provider == "gemini":
        from .gemini import GeminiClient

        return GeminiClient(api_key=api_key)

    raise ValueError(f"Unknown LLM provider: {provider}")


def get_smart_client() -> "LLMClient":
    """Get an LLM client with automatic key rotation and provider fallback.
    
    This is the recommended way to get an LLM client as it handles:
    - Automatic key rotation when rate limits are hit
    - Fallback between providers: OpenRouter → Groq → Gemini
    - Error handling and retry logic
    
    Returns:
        LLMClient: A client instance that automatically handles failures
    
    Raises:
        RuntimeError: If all keys are exhausted
    """
    from .smart_client import SmartLLMClient
    return SmartLLMClient() 