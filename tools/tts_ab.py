#!/usr/bin/env python3
"""A/B compare Soniox TTS vs. ElevenLabs Flash v2.5 vs. Azure Neural TTS on a
fixed set of Slovenian sentences that stress declension, dual, names, times,
prices, and digit read-out.

Writes audio to tts_out/<provider>/<n>.<ext> and prints a summary table.

Azure candidate: sl-SI has exactly two neural voices as of 2026-08-15 —
sl-SI-PetraNeural (female) and sl-SI-RokNeural (male), confirmed against
Microsoft's live language-support docs (not assumed from a prior report).
Both are synthesized so the two can be compared alongside Soniox/ElevenLabs.
Note: the official livekit-plugins-azure package exists (AZURE_SPEECH_KEY /
AZURE_SPEECH_REGION, same env vars used here) but its TTS is non-streaming —
see the "Azure" note in the project's TTS provider notes before assuming a
win here means a drop-in swap into the live worker.

Usage:
    export SONIOX_API_KEY=...
    export ELEVENLABS_API_KEY=...
    export AZURE_SPEECH_KEY=...
    export AZURE_SPEECH_REGION=...   # e.g. westeurope — Azure resource
                                      # region, not a language code
    pip install azure-cognitiveservices-speech  # only needed for the Azure leg
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

# Confirmed 2026-08-15 against Microsoft's live sl-SI language-support docs —
# these are the only two Slovenian neural voices Azure currently offers.
AZURE_VOICES = ["sl-SI-PetraNeural", "sl-SI-RokNeural"]

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


def synthesize_azure(text: str, voice: str, key: str, region: str) -> bytes:
    try:
        import azure.cognitiveservices.speech as speechsdk
    except ImportError as exc:
        raise RuntimeError(
            "azure-cognitiveservices-speech is not installed — run: "
            "pip install azure-cognitiveservices-speech"
        ) from exc

    speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
    speech_config.speech_synthesis_voice_name = voice
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm
    )
    # audio_config=None: synthesize to memory (result.audio_data) instead of
    # the default speaker output, matching the byte-returning shape of the
    # other synthesize_* functions here.
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
    result = synthesizer.speak_text_async(text).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        return result.audio_data
    if result.reason == speechsdk.ResultReason.Canceled:
        details = result.cancellation_details
        raise RuntimeError(f"Azure TTS canceled: {details.reason} — {details.error_details}")
    raise RuntimeError(f"Azure TTS failed: {result.reason}")


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

    azure_key = os.environ.get("AZURE_SPEECH_KEY")
    azure_region = os.environ.get("AZURE_SPEECH_REGION")
    if not azure_key or not azure_region:
        print(
            "note: AZURE_SPEECH_KEY / AZURE_SPEECH_REGION not set — skipping "
            "the Azure leg",
            file=sys.stderr,
        )

    soniox_dir = OUT_DIR / "soniox"
    elevenlabs_dir = OUT_DIR / "elevenlabs"
    azure_dir = OUT_DIR / "azure"
    soniox_dir.mkdir(parents=True, exist_ok=True)
    elevenlabs_dir.mkdir(parents=True, exist_ok=True)
    if azure_key and azure_region:
        azure_dir.mkdir(parents=True, exist_ok=True)
        for voice in AZURE_VOICES:
            (azure_dir / voice).mkdir(parents=True, exist_ok=True)

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

        if azure_key and azure_region:
            for voice in AZURE_VOICES:
                provider_label = f"azure-{voice}"
                try:
                    audio = synthesize_azure(sentence, voice, azure_key, azure_region)
                    out_path = azure_dir / voice / f"{i}.wav"
                    out_path.write_bytes(audio)
                    results.append(Result(provider_label, i, out_path))
                except Exception as exc:  # noqa: BLE001
                    results.append(Result(provider_label, i, None, str(exc)))

    provider_width = max(11, max((len(r.provider) for r in results), default=11))
    header = f"{'#':<3} {'provider':<{provider_width}} {'file':<34} status"
    print(header)
    print("-" * len(header))
    for r in results:
        file_col = str(r.path.relative_to(REPO_ROOT)) if r.path else "-"
        print(f"{r.index:<3} {r.provider:<{provider_width}} {file_col:<34} {r.status}")

    failures = [r for r in results if r.error]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
