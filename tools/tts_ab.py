#!/usr/bin/env python3
"""A/B compare Soniox TTS vs. ElevenLabs Flash v2.5 on a fixed set of Slovenian
sentences that stress declension, dual, names, times, prices, and digit read-out.

Writes audio to tts_out/<provider>/<n>.<ext> and prints a summary table.

Usage:
    export SONIOX_API_KEY=...
    export ELEVENLABS_API_KEY=...
    python tools/tts_ab.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "tts_out"

SONIOX_TTS_URL = "https://tts-rt.soniox.com/tts"
SONIOX_MODEL = os.environ.get("SONIOX_TTS_MODEL", "tts-rt-v1")
SONIOX_VOICE = os.environ.get("SONIOX_TTS_VOICE", "Maya")

ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
ELEVENLABS_MODEL = "eleven_flash_v2_5"
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
ELEVENLABS_OUTPUT_FORMAT = "mp3_44100_128"

LANGUAGE = "sl"

SENTENCES = [
    "Pozdravljeni, tukaj digitalni asistent salona Lepote. Kako vam lahko pomagam?",
    "Imamo prosto v torek ob petnajstih ali v sredo ob pol enajstih.",
    "Striženje in barvanje pri Maji stane petinštirideset evrov in traja uro in pol.",
    "Vaš termin sem rezervirala za četrtek, dvajsetega marca, ob devetih zjutraj.",
    "Ali mi lahko poveste vašo telefonsko številko? Ponavljam: nič štiri nič, šest šest tri, štiri deset.",
    "Naročila sem vas k frizerki Špeli Kovačič, se vidiva kmalu.",
]


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip()


@dataclass
class Result:
    provider: str
    index: int
    path: Path | None
    error: str | None = None

    @property
    def status(self) -> str:
        return "ok" if self.path else f"FAILED: {self.error}"


def synthesize_soniox(text: str, api_key: str) -> bytes:
    resp = requests.post(
        SONIOX_TTS_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": SONIOX_MODEL,
            "language": LANGUAGE,
            "voice": SONIOX_VOICE,
            "audio_format": "wav",
            "text": text,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.content


def synthesize_elevenlabs(text: str, api_key: str) -> bytes:
    resp = requests.post(
        ELEVENLABS_TTS_URL.format(voice_id=ELEVENLABS_VOICE_ID),
        params={"output_format": ELEVENLABS_OUTPUT_FORMAT},
        headers={
            "xi-api-key": api_key,
            "Content-Type": "application/json",
        },
        json={
            "text": text,
            "model_id": ELEVENLABS_MODEL,
            "language_code": LANGUAGE,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.content


def main() -> int:
    _load_dotenv(REPO_ROOT / ".env")

    soniox_key = os.environ.get("SONIOX_API_KEY")
    elevenlabs_key = os.environ.get("ELEVENLABS_API_KEY")
    if not soniox_key:
        print("error: SONIOX_API_KEY is not set", file=sys.stderr)
        return 1
    if not elevenlabs_key:
        print("error: ELEVENLABS_API_KEY is not set", file=sys.stderr)
        return 1

    soniox_dir = OUT_DIR / "soniox"
    elevenlabs_dir = OUT_DIR / "elevenlabs"
    soniox_dir.mkdir(parents=True, exist_ok=True)
    elevenlabs_dir.mkdir(parents=True, exist_ok=True)

    results: list[Result] = []

    for i, sentence in enumerate(SENTENCES, start=1):
        try:
            audio = synthesize_soniox(sentence, soniox_key)
            out_path = soniox_dir / f"{i}.wav"
            out_path.write_bytes(audio)
            results.append(Result("soniox", i, out_path))
        except Exception as exc:  # noqa: BLE001
            results.append(Result("soniox", i, None, str(exc)))

        try:
            audio = synthesize_elevenlabs(sentence, elevenlabs_key)
            out_path = elevenlabs_dir / f"{i}.mp3"
            out_path.write_bytes(audio)
            results.append(Result("elevenlabs", i, out_path))
        except Exception as exc:  # noqa: BLE001
            results.append(Result("elevenlabs", i, None, str(exc)))

    header = f"{'#':<3} {'provider':<11} {'file':<28} status"
    print(header)
    print("-" * len(header))
    for r in results:
        file_col = str(r.path.relative_to(REPO_ROOT)) if r.path else "-"
        print(f"{r.index:<3} {r.provider:<11} {file_col:<28} {r.status}")

    failures = [r for r in results if r.error]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
