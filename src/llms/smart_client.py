"""Smart LLM client with automatic key rotation and provider fallback."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from .base import LLMClient
from .key_manager import key_manager

logger = logging.getLogger(__name__)


class SmartLLMClient(LLMClient):
    """LLM client that automatically rotates keys and falls back between providers."""
    
    def __init__(self):
        self._current_client = None
        self._current_provider = None
        self._current_key_status = None

    def chat(self, messages: List[Dict[str, Any]], **kwargs: Any) -> str:
        """Send chat messages with automatic key rotation and provider fallback."""
        # Check if this is a vision request by looking for image content
        require_vision = self._has_image_content(messages)
        
        # Dynamically determine how many times we should retry before giving up.
        # One retry per available key (across all providers) is usually enough
        # to guarantee we will eventually hit a non-rate-limited key or exhaust
        # the pool.  We cap it to 20 just in case someone mis-configures an
        # excessive amount of keys.
        try:
            summary = key_manager.get_status_summary()
            if require_vision:
                # Only count vision-capable providers (OpenRouter + Gemini)
                total_keys = (
                    summary["openrouter"]["total"]
                    + summary["gemini"]["total"]
                )
            else:
                total_keys = (
                    summary["openrouter"]["total"]
                    + summary["groq"]["total"]
                    + summary["gemini"]["total"]
                )
            max_retries = min(max(total_keys, 3), 20)
        except Exception:
            # Fallback to previous default if for some reason the summary call
            # fails (should be rare).
            max_retries = 10
        
        for attempt in range(max_retries):
            try:
                # Get a fresh client if we don't have one or need to retry
                if self._current_client is None or attempt > 0:
                    self._current_client, self._current_provider, self._current_key_status = (
                        key_manager.get_client_with_fallback(require_vision=require_vision)
                    )
                
                # Make the API call
                result = self._current_client.chat(messages, **kwargs)
                
                # Mark success
                key_manager.mark_success(self._current_key_status)
                
                return result
                
            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
                
                # Handle the error through key manager
                if self._current_key_status:
                    key_manager.handle_api_error(self._current_key_status, e)
                
                # Reset client to force getting a new one on retry
                self._current_client = None
                self._current_provider = None
                self._current_key_status = None
                
                # If this was the last attempt, re-raise
                if attempt == max_retries - 1:
                    raise

        # This should never be reached due to the raise in the loop
        raise RuntimeError("Unexpected error in smart client")

    def _has_image_content(self, messages: List[Dict[str, Any]]) -> bool:
        """Check if messages contain image content requiring vision capabilities."""
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        return True
        return False

    def embed(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        """Get embeddings with automatic key rotation and provider fallback."""
        # Embeddings are text-only, so no vision required
        require_vision = False
        
        # Same dynamic retry calculation as in chat()
        try:
            summary = key_manager.get_status_summary()
            total_keys = (
                summary["openrouter"]["total"]
                + summary["groq"]["total"]
                + summary["gemini"]["total"]
            )
            max_retries = min(max(total_keys, 3), 20)
        except Exception:
            max_retries = 10
        
        for attempt in range(max_retries):
            try:
                # Get a fresh client if we don't have one or need to retry
                if self._current_client is None or attempt > 0:
                    self._current_client, self._current_provider, self._current_key_status = (
                        key_manager.get_client_with_fallback(require_vision=require_vision)
                    )
                
                # Make the API call
                result = self._current_client.embed(texts, **kwargs)
                
                # Mark success
                key_manager.mark_success(self._current_key_status)
                
                return result
                
            except Exception as e:
                logger.warning(f"Embedding call failed (attempt {attempt + 1}/{max_retries}): {e}")
                
                # Handle the error through key manager
                if self._current_key_status:
                    key_manager.handle_api_error(self._current_key_status, e)
                
                # Reset client to force getting a new one on retry
                self._current_client = None
                self._current_provider = None
                self._current_key_status = None
                
                # If this was the last attempt, re-raise
                if attempt == max_retries - 1:
                    raise

        # This should never be reached due to the raise in the loop
        raise RuntimeError("Unexpected error in smart client")

    def get_current_provider(self) -> str | None:
        """Get the name of the currently active provider."""
        return self._current_provider

    def get_status_summary(self) -> Dict[str, Any]:
        """Get a summary of all keys and their status."""
        return key_manager.get_status_summary() 