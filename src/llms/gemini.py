"""Google Gemini client using the official Google AI Python SDK.

Provides chat completions using Gemini models (gemini-2.0-flash-exp, etc.)
through the Google AI API.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from .base import LLMClient
from src.config.settings import SETTINGS

logger = logging.getLogger(__name__)


class GeminiClient(LLMClient):
    """Wrapper for Google Gemini models via the Google AI Python SDK."""

    def __init__(self, api_key: str | None):
        if not api_key:
            raise ValueError("Gemini API key must be provided.")
        
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "google-generativeai package not installed. "
                "Install with: pip install google-generativeai"
            ) from exc
        
        # Configure the API key
        genai.configure(api_key=api_key)
        self._genai = genai
        self._api_key = api_key

    def chat(self, messages: Any, model: str | None = None, **kwargs: Any) -> str:  # type: ignore[override]
        """Send chat messages and return the assistant reply text."""
        if model is None:
            model = SETTINGS.gemini_chat_model

        try:
            # Create the model instance
            gemini_model = self._genai.GenerativeModel(model)
            
            # Convert OpenAI-style messages to Gemini format
            gemini_messages = self._convert_messages(messages)
            
            # Extract temperature and other generation config
            generation_config = {}
            if "temperature" in kwargs:
                generation_config["temperature"] = kwargs["temperature"]
            if "max_tokens" in kwargs:
                generation_config["max_output_tokens"] = kwargs["max_tokens"]
            
            # Generate response
            response = gemini_model.generate_content(
                gemini_messages,
                generation_config=generation_config if generation_config else None
            )
            
            return response.text or ""
            
        except Exception as e:
            # Re-raise with provider context for better error handling
            error_msg = f"Gemini API error: {e}"
            logger.error(error_msg)
            raise Exception(error_msg) from e

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI-style messages to Gemini format."""
        gemini_messages = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Handle multimodal content (list of content items)
            if isinstance(content, list):
                parts = []
                for item in content:
                    if item.get("type") == "text":
                        parts.append(item["text"])
                    elif item.get("type") == "image_url":
                        # Extract base64 data from data URL
                        image_url = item["image_url"]["url"]
                        if image_url.startswith("data:image/"):
                            # Remove data:image/jpeg;base64, prefix
                            base64_data = image_url.split(",", 1)[1]
                            # Create Gemini-compatible blob
                            import base64
                            image_data = base64.b64decode(base64_data)
                            
                            # Use the genai library's blob creation
                            try:
                                blob = self._genai.types.Blob(
                                    mime_type="image/jpeg",
                                    data=image_data
                                )
                                parts.append(blob)
                            except Exception:
                                # Fallback: try creating blob dictionary directly
                                parts.append({
                                    "mime_type": "image/jpeg",
                                    "data": image_data
                                })
                
                # Map OpenAI roles to Gemini roles for multimodal
                if role == "system":
                    if not gemini_messages:
                        # Add system instruction as text part
                        system_parts = [f"System instructions: {parts[0] if parts else ''}"]
                        gemini_messages.append({
                            "role": "user",
                            "parts": system_parts
                        })
                    else:
                        # Prepend to existing first user message
                        if gemini_messages[0]["role"] == "user":
                            gemini_messages[0]["parts"].insert(0, f"System instructions: {parts[0] if parts else ''}")
                elif role == "user":
                    gemini_messages.append({
                        "role": "user", 
                        "parts": parts
                    })
                elif role == "assistant":
                    gemini_messages.append({
                        "role": "model",  # Gemini uses "model" instead of "assistant"
                        "parts": parts
                    })
            
            # Handle simple text content (string)
            else:
                # Map OpenAI roles to Gemini roles
                if role == "system":
                    # Gemini doesn't have system role, prepend to first user message
                    if not gemini_messages:
                        gemini_messages.append({
                            "role": "user",
                            "parts": [f"System instructions: {content}"]
                        })
                    else:
                        # Add to existing first user message
                        if gemini_messages[0]["role"] == "user":
                            gemini_messages[0]["parts"][0] = f"System instructions: {content}\n\nUser: {gemini_messages[0]['parts'][0]}"
                elif role == "user":
                    gemini_messages.append({
                        "role": "user", 
                        "parts": [content]
                    })
                elif role == "assistant":
                    gemini_messages.append({
                        "role": "model",  # Gemini uses "model" instead of "assistant"
                        "parts": [content]
                    })
        
        return gemini_messages

    def embed(self, texts: List[str], **kwargs: Any):  # type: ignore[override]
        """Return vector embeddings for texts."""
        try:
            # Use Gemini's embedding model
            embeddings = []
            for text in texts:
                result = self._genai.embed_content(
                    model="models/text-embedding-004",  # Gemini's latest embedding model
                    content=text
                )
                embeddings.append(result["embedding"])
            return embeddings
        except Exception as e:
            logger.warning(f"Gemini embedding failed: {e}")
            raise NotImplementedError("Gemini embeddings not available") from e 