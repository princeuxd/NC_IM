"""Advanced key rotation manager for LLM providers.

Handles automatic key switching when rate limits are encountered, with fallback
between providers: OpenRouter → Groq → Gemini.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from config.settings import SETTINGS

logger = logging.getLogger(__name__)

# Path for persistent state storage
STATE_FILE = Path(".llm_key_state.json")


@dataclass
class KeyStatus:
    """Track the status of an API key."""
    key: str
    provider: str
    last_used: float = field(default_factory=time.time)
    rate_limited_until: float = 0.0
    error_count: int = 0
    success_count: int = 0
    
    @property
    def is_available(self) -> bool:
        """Check if key is currently available (not rate limited)."""
        return time.time() > self.rate_limited_until
    
    def mark_rate_limited(self, duration_minutes: int = 60) -> None:
        """Mark key as rate limited for specified duration."""
        self.rate_limited_until = time.time() + (duration_minutes * 60)
        self.error_count += 1
        logger.warning(f"Key {self.key[:20]}... rate limited for {duration_minutes} minutes")
    
    def mark_success(self) -> None:
        """Mark successful API call."""
        self.success_count += 1
        self.last_used = time.time()
        # Reset rate limit if it was temporary
        if self.rate_limited_until > 0:
            self.rate_limited_until = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "key": self.key,
            "provider": self.provider,
            "last_used": self.last_used,
            "rate_limited_until": self.rate_limited_until,
            "error_count": self.error_count,
            "success_count": self.success_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KeyStatus":
        """Create from dictionary."""
        return cls(
            key=data["key"],
            provider=data["provider"],
            last_used=data.get("last_used", time.time()),
            rate_limited_until=data.get("rate_limited_until", 0.0),
            error_count=data.get("error_count", 0),
            success_count=data.get("success_count", 0)
        )


class KeyRotationManager:
    """Manages API key rotation with automatic fallback between providers."""
    
    def __init__(self):
        self._openrouter_keys: List[KeyStatus] = []
        self._groq_keys: List[KeyStatus] = []
        self._gemini_keys: List[KeyStatus] = []
        
        self._openrouter_index = 0
        self._groq_index = 0
        self._gemini_index = 0
        
        self._load_state()
        self._initialize_keys()
        self._save_state()
    
    def _load_state(self) -> None:
        """Load persistent state from file."""
        if not STATE_FILE.exists():
            return
        
        try:
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
            
            # Load key states
            for key_data in state.get("openrouter_keys", []):
                self._openrouter_keys.append(KeyStatus.from_dict(key_data))
            for key_data in state.get("groq_keys", []):
                self._groq_keys.append(KeyStatus.from_dict(key_data))
            for key_data in state.get("gemini_keys", []):
                self._gemini_keys.append(KeyStatus.from_dict(key_data))
            
            # Load indices
            self._openrouter_index = state.get("openrouter_index", 0)
            self._groq_index = state.get("groq_index", 0)
            self._gemini_index = state.get("gemini_index", 0)
            
            logger.debug("Loaded key manager state from disk")
            
        except Exception as e:
            logger.warning(f"Failed to load key manager state: {e}")
    
    def _save_state(self) -> None:
        """Save persistent state to file."""
        try:
            state = {
                "openrouter_keys": [key.to_dict() for key in self._openrouter_keys],
                "groq_keys": [key.to_dict() for key in self._groq_keys],
                "gemini_keys": [key.to_dict() for key in self._gemini_keys],
                "openrouter_index": self._openrouter_index,
                "groq_index": self._groq_index,
                "gemini_index": self._gemini_index,
                "last_updated": time.time()
            }
            
            with open(STATE_FILE, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save key manager state: {e}")
    
    def _initialize_keys(self) -> None:
        """Initialize key tracking from settings, merging with existing state."""
        # Get current keys from environment
        current_or_keys = set(SETTINGS.openrouter_api_keys)
        current_groq_keys = set(SETTINGS.groq_api_keys)
        current_gemini_keys = set(SETTINGS.gemini_api_keys)
        
        # Update OpenRouter keys
        existing_or_keys = {key.key for key in self._openrouter_keys}
        for key in current_or_keys:
            if key not in existing_or_keys:
                self._openrouter_keys.append(KeyStatus(key, "openrouter"))
        
        # Remove keys that are no longer in config
        self._openrouter_keys = [key for key in self._openrouter_keys if key.key in current_or_keys]
        
        # Update Groq keys
        existing_groq_keys = {key.key for key in self._groq_keys}
        for key in current_groq_keys:
            if key not in existing_groq_keys:
                self._groq_keys.append(KeyStatus(key, "groq"))
        
        self._groq_keys = [key for key in self._groq_keys if key.key in current_groq_keys]
        
        # Update Gemini keys
        existing_gemini_keys = {key.key for key in self._gemini_keys}
        for key in current_gemini_keys:
            if key not in existing_gemini_keys:
                self._gemini_keys.append(KeyStatus(key, "gemini"))
        
        self._gemini_keys = [key for key in self._gemini_keys if key.key in current_gemini_keys]
        
        logger.info(f"Initialized key manager: {len(self._openrouter_keys)} OpenRouter, "
                   f"{len(self._groq_keys)} Groq, {len(self._gemini_keys)} Gemini keys")
    
    def _get_next_key(self, provider: str) -> Optional[KeyStatus]:
        """Get the next available key for a provider."""
        if provider == "openrouter":
            keys = self._openrouter_keys
            current_idx = self._openrouter_index
        elif provider == "groq":
            keys = self._groq_keys
            current_idx = self._groq_index
        elif provider == "gemini":
            keys = self._gemini_keys
            current_idx = self._gemini_index
        else:
            return None
        
        if not keys:
            return None
        
        # Try to find an available key starting from current index
        for i in range(len(keys)):
            idx = (current_idx + i) % len(keys)
            key_status = keys[idx]
            
            if key_status.is_available:
                # Update the index for next call
                if provider == "openrouter":
                    self._openrouter_index = (idx + 1) % len(keys)
                elif provider == "groq":
                    self._groq_index = (idx + 1) % len(keys)
                elif provider == "gemini":
                    self._gemini_index = (idx + 1) % len(keys)
                
                return key_status
        
        return None
    
    def get_client_with_fallback(self) -> Tuple[Any, str, KeyStatus]:
        """Get an LLM client with automatic provider fallback.
        
        Returns:
            Tuple of (client_instance, provider_name, key_status)
        
        Raises:
            RuntimeError: If all keys are exhausted
        """
        from .base import get_client  # Import here to avoid circular imports
        
        # Try providers in order: OpenRouter → Groq → Gemini
        providers = ["openrouter", "groq", "gemini"]
        
        for provider in providers:
            key_status = self._get_next_key(provider)
            if key_status:
                try:
                    client = get_client(provider, key_status.key)
                    logger.debug(f"Using {provider} with key {key_status.key[:20]}...")
                    return client, provider, key_status
                except Exception as e:
                    logger.warning(f"Failed to create {provider} client: {e}")
                    key_status.mark_rate_limited(5)  # Short cooldown for client creation failures
                    self._save_state()  # Save state after marking key as rate limited
        
        raise RuntimeError("All API keys exhausted across all providers")
    
    def handle_api_error(self, key_status: KeyStatus, error: Exception) -> None:
        """Handle API errors and update key status accordingly."""
        error_str = str(error).lower()
        
        # Check for rate limit indicators
        if any(indicator in error_str for indicator in [
            "rate limit", "429", "quota", "too many requests", 
            "rate_limit_exceeded", "insufficient_quota"
        ]):
            # Rate limited - mark key as unavailable
            key_status.mark_rate_limited(60)  # 1 hour cooldown
        elif any(indicator in error_str for indicator in [
            "unauthorized", "401", "invalid", "api_key"
        ]):
            # Invalid key - mark as permanently unavailable
            key_status.mark_rate_limited(24 * 60)  # 24 hour cooldown
        else:
            # Other error - short cooldown
            key_status.error_count += 1
            if key_status.error_count >= 3:
                key_status.mark_rate_limited(10)  # 10 minute cooldown after 3 errors
        
        # Save state after any error handling
        self._save_state()
    
    def mark_success(self, key_status: KeyStatus) -> None:
        """Mark a successful API call."""
        key_status.mark_success()
        self._save_state()  # Save state after successful call
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get a summary of all keys and their status."""
        def _key_summary(keys: List[KeyStatus]) -> Dict[str, Any]:
            available = sum(1 for k in keys if k.is_available)
            total_success = sum(k.success_count for k in keys)
            total_errors = sum(k.error_count for k in keys)
            
            return {
                "total": len(keys),
                "available": available,
                "rate_limited": len(keys) - available,
                "total_success_calls": total_success,
                "total_error_calls": total_errors
            }
        
        return {
            "openrouter": _key_summary(self._openrouter_keys),
            "groq": _key_summary(self._groq_keys),
            "gemini": _key_summary(self._gemini_keys),
            "timestamp": time.time()
        }

    # ------------------------------------------------------------------
    # Maintenance helpers
    # ------------------------------------------------------------------

    def clear_rate_limits(self, providers: Optional[List[str]] | None = None) -> None:
        """Reset *rate_limited_until* for all (or specific) provider keys.

        Example
        -------
        >>> from llms.key_manager import key_manager
        >>> key_manager.clear_rate_limits()  # reset everything
        >>> key_manager.clear_rate_limits(["openrouter"])  # only OR keys
        """

        if providers is None:
            providers = ["openrouter", "groq", "gemini"]

        def _reset(keys: List[KeyStatus]):
            for k in keys:
                if k.rate_limited_until > 0:
                    k.rate_limited_until = 0.0

        if "openrouter" in providers:
            _reset(self._openrouter_keys)
        if "groq" in providers:
            _reset(self._groq_keys)
        if "gemini" in providers:
            _reset(self._gemini_keys)

        self._save_state()

        logger.info(
            "Rate-limit timers cleared for providers: %s", ", ".join(providers)
        )


# Global instance
key_manager = KeyRotationManager() 