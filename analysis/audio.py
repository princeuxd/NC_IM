"""Audio transcription using Whisper.

Main function:
transcribe()          – Whisper (or openai-whisper) segments

This is a trimmed-down replacement for the original ``analysis.core`` module.
It re-uses the battle-tested Whisper logic from *analysis.core* and the new
LLM summarization logic from ``analysis.sentiment`` for general-purpose use.

For most use cases, you probably want:
    from analysis.audio import transcribe
    segments = transcribe(path_to_audio_file)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, cast

# Local whisper helper (copied from old analysis.core)

def _whisper_transcribe(audio_path: str | os.PathLike) -> List[Dict[str, Any]]:
    """Transcribe *audio_path* with faster-whisper."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        try:
            # Fallback to openai-whisper if faster-whisper not available
            import whisper  # type: ignore
            if not hasattr(whisper, "load_model"):
                raise ImportError("whisper.load_model not found") from exc
        except ImportError:
            import openai_whisper as whisper  # type: ignore
        except ImportError:
            raise RuntimeError("faster-whisper or openai-whisper not installed") from exc
        
        # Use openai-whisper fallback
        model = whisper.load_model("base")  # type: ignore[attr-defined]
        result = model.transcribe(str(audio_path))  # type: ignore[attr-defined]
        return result.get("segments", [])
    
    # Use faster-whisper (preferred)
    try:
        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(str(audio_path), beam_size=5)
        
        # Convert faster-whisper segments to openai-whisper format for compatibility
        result_segments = []
        for segment in segments:
            result_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text
            })
        return result_segments
        
    except Exception as exc:
        logging.getLogger(__name__).warning("Faster-Whisper transcription failed: %s", exc)
        return []

from analysis.sentiment import sentiment_scores

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def transcribe(audio_path: Path | str):
    """Return list of Whisper segments for *audio_path*."""

    return _whisper_transcribe(audio_path)


def attach_sentiment(segments: List[Dict[str, Any]]):
    """Return new list where each segment includes a *sentiment* score."""

    if not segments:
        return []

    texts = [s.get("text", "") for s in segments]
    scores = sentiment_scores(texts)

    enriched: List[Dict[str, Any]] = []
    for seg, score in zip(segments, scores):
        new_seg = dict(seg)
        new_seg["sentiment"] = score
        enriched.append(new_seg)
    return enriched


def analyze_audio(
    audio_path: Path | str,
    *,
    out_path: Path | None = None,
) -> List[Dict[str, Any]]:
    """Transcribe *audio_path*, attach sentiment, optionally save JSON."""

    segments = transcribe(audio_path)
    enriched = attach_sentiment(segments)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(enriched, indent=2, ensure_ascii=False))
        logger.info("Saved transcript sentiment JSON → %s", out_path)

    return enriched


def extract_audio(video_file: Path | str, wav_path: Path | str | None = None) -> Path:
    """Extract mono 16-kHz WAV audio using ffmpeg."""

    video_file = Path(video_file)
    if wav_path is None:
        wav_path = video_file.with_suffix(".wav")
    wav_path = Path(wav_path)

    if shutil.which("ffmpeg") is None:
        logger.warning("ffmpeg not in PATH; skipping audio extraction.")
        logger.info("To fix: Install ffmpeg or add 'ffmpeg' to packages.txt for Streamlit Cloud")
        # Return the original video file if it's already audio format
        if video_file.suffix.lower() in ['.m4a', '.mp3', '.wav', '.aac']:
            logger.info("Video file is already audio format, using directly: %s", video_file)
            return video_file
        raise RuntimeError("ffmpeg is required for audio extraction. Install ffmpeg or use audio-only download.")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_file),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(wav_path),
    ]
    logger.info("Extracting audio track to %s", wav_path)
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        logger.error("ffmpeg failed to extract audio: %s", e)
        # If input is already audio format, try using it directly
        if video_file.suffix.lower() in ['.m4a', '.mp3', '.wav', '.aac']:
            logger.info("Using original audio file directly: %s", video_file)
            return video_file
        raise RuntimeError(f"Audio extraction failed: {e}")
    return wav_path


__all__ = [
    "transcribe",
    "attach_sentiment",
    "analyze_audio",
] 