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
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Get a fresh client if we don't have one or need to retry
                if self._current_client is None or attempt > 0:
                    self._current_client, self._current_provider, self._current_key_status = (
                        key_manager.get_client_with_fallback()
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

    def embed(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        """Get embeddings with automatic key rotation and provider fallback."""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Get a fresh client if we don't have one or need to retry
                if self._current_client is None or attempt > 0:
                    self._current_client, self._current_provider, self._current_key_status = (
                        key_manager.get_client_with_fallback()
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