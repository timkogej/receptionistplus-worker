"""Pre-rendered greeting audio, so the first words play without waiting for TTS.

The first call with a given greeting text speaks it through TTS as usual and
renders a copy in the background; every later call plays the copy straight
away (TTS first byte for the opening line measured ~0.6-0.9s, 2026-10).
The cache is a folder of WAV files keyed by voice and text: a new voice,
company name or time of day is simply a new file. Losing the folder (a new
container) costs one TTS-spoken greeting per text, nothing else.
"""

from __future__ import annotations

import hashlib
import logging
import os
import wave
from pathlib import Path

logger = logging.getLogger("receptionistplus-worker.greeting_audio")

CACHE_DIR = Path(os.environ.get("GREETING_AUDIO_DIR", "/tmp/receptionistplus-greetings"))
# Anything smaller is a failed render, not a greeting.
_MIN_BYTES = 4000


def _path(text: str, voice_key: str) -> Path:
    digest = hashlib.sha256(f"{voice_key}\n{text}".encode()).hexdigest()[:32]
    return CACHE_DIR / f"{digest}.wav"


def cached(text: str, voice_key: str) -> Path | None:
    path = _path(text, voice_key)
    try:
        return path if path.stat().st_size >= _MIN_BYTES else None
    except OSError:
        return None


async def render(tts, text: str, voice_key: str) -> None:
    """Synthesize `text` and store it. Best effort: never raises."""
    path = _path(text, voice_key)
    if cached(text, voice_key):
        return
    try:
        frames = []
        stream = tts.synthesize(text)
        try:
            async for event in stream:
                frames.append(event.frame)
        finally:
            await stream.aclose()
        if not frames:
            return
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        with wave.open(str(tmp), "wb") as out:
            out.setnchannels(frames[0].num_channels)
            out.setsampwidth(2)
            out.setframerate(frames[0].sample_rate)
            for frame in frames:
                out.writeframes(bytes(frame.data))
        os.replace(tmp, path)
        logger.info("greeting audio cached (%d bytes)", path.stat().st_size)
    except Exception:
        logger.exception("greeting audio render failed")
