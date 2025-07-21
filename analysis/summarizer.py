"""Generate executive summary of analysis via LLM (OpenRouter / Groq).

Reads core artefacts (metadata, product_impact, comments_sentiment, etc.) and
prompts an LLM to produce a concise markdown summary aimed at brand managers.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

import openai
from openai import OpenAI

from config.settings import SETTINGS, PipelineSettings

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)

KEY_FILES = [
    "metadata.json",
    "comments_sentiment.json",
    "transcript_sentiment.json",
]

def _encode_image(image_path: Path) -> str:
    """Encode image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def _load_text_artefacts(folder: Path) -> Dict[str, Any]:
    """Load JSON artefacts from the video analysis."""
    artefacts = {}
    for fname in KEY_FILES:
        fpath = folder / fname
        if fpath.exists():
            try:
                artefacts[fname] = json.loads(fpath.read_text())
            except json.JSONDecodeError:
                logger.warning(f"Could not parse JSON from {fname}")
    return artefacts

def _build_multimodal_prompt(
    text_artefacts: Dict[str, Any], frames: List[Path]
) -> List[Dict[str, Any]]:
    """Construct the full multimodal prompt for the vision model."""
    
    system_prompt = (
        "You are a senior brand strategist at a top marketing agency. Your task is to analyze "
        "a YouTube creator's video to determine their suitability for a brand collaboration. "
        "Evaluate the creator's content quality, audience engagement, and overall brand safety "
        "based *exclusively* on the provided data (JSON artefacts and video frames). "
        "Produce a concise, professional recommendation in markdown format."
    )
    
    user_content = []
    user_content.append({
        "type": "text",
        "text": "Please analyze the following YouTube video data for a potential brand collaboration and provide your recommendation.\n"
    })

    # Add text artefacts
    for fname, data in text_artefacts.items():
        user_content.append({"type": "text", "text": f"--- {fname.upper()} ---\n{json.dumps(data, indent=2)}\n"})

    # Add image frames
    if frames:
        user_content.append({"type": "text", "text": "--- KEY VIDEO FRAMES ---"})
        for frame_path in frames:
            if frame_path.exists():
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{_encode_image(frame_path)}"},
                })

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def _try_generate(
    model: str, messages: List[Dict[str, Any]], settings: PipelineSettings
) -> Optional[str]:
    """Attempt one LLM call and return the markdown summary."""
    client = OpenAI(
        api_key=settings.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={"HTTP-Referer": "http://localhost:8501", "X-Title": "NC_IM Video Analyzer"}
    )
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore
            max_tokens=1024,
            temperature=0.3,
        )
        return completion.choices[0].message.content
    except openai.APIError as e:
        logger.error(f"API Error during summary generation with {model}: {e}")
        return None

def generate_summary(
    folder: Path,
    frames: List[Path],
    *,
    settings: PipelineSettings = SETTINGS,
) -> str:
    """Generate an executive summary from text and image artefacts."""
    if not settings.openrouter_api_key:
        return "❗ OpenRouter API key not configured. Summary generation skipped."

    text_artefacts = _load_text_artefacts(folder)
    if not text_artefacts and not frames:
        return "❗ No data available to generate a summary."

    messages = _build_multimodal_prompt(text_artefacts, frames)
    
    # Use the designated summary model from settings
    summary_model = settings.summary_model
    summary = _try_generate(summary_model, messages, settings)
    
    if not summary:
        return f"❗ Failed to generate summary from the model: {summary_model}."
        
    (folder / "summary.md").write_text(summary)
    logger.info(f"Successfully generated summary with model {summary_model}")
    return summary

__all__ = ["generate_summary"] 