"""Environment-backed settings for the receptionist worker."""

from __future__ import annotations

import os
from dataclasses import dataclass


class ConfigError(RuntimeError):
    """Raised when a required setting is missing or invalid."""


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ConfigError(f"missing required environment variable: {name}")
    return value


@dataclass(frozen=True)
class Settings:
    livekit_url: str
    livekit_api_key: str
    livekit_api_secret: str

    soniox_api_key: str | None
    elevenlabs_api_key: str | None
    anthropic_api_key: str
    openai_api_key: str | None

    tts_provider: str
    tts_voice_id: str | None

    company_slug: str

    supabase_url: str
    supabase_service_role_key: str

    @classmethod
    def from_env(cls) -> "Settings":
        tts_provider = os.environ.get("TTS_PROVIDER", "soniox").strip().lower()
        if tts_provider not in ("soniox", "elevenlabs"):
            raise ConfigError(
                f"invalid TTS_PROVIDER '{tts_provider}', expected 'soniox' or 'elevenlabs'"
            )

        soniox_api_key = os.environ.get("SONIOX_API_KEY")
        elevenlabs_api_key = os.environ.get("ELEVENLABS_API_KEY")

        if tts_provider == "soniox" and not soniox_api_key:
            raise ConfigError("TTS_PROVIDER=soniox requires SONIOX_API_KEY")
        if tts_provider == "elevenlabs" and not elevenlabs_api_key:
            raise ConfigError("TTS_PROVIDER=elevenlabs requires ELEVENLABS_API_KEY")

        return cls(
            livekit_url=_require("LIVEKIT_URL"),
            livekit_api_key=_require("LIVEKIT_API_KEY"),
            livekit_api_secret=_require("LIVEKIT_API_SECRET"),
            soniox_api_key=soniox_api_key,
            elevenlabs_api_key=elevenlabs_api_key,
            anthropic_api_key=_require("ANTHROPIC_API_KEY"),
            openai_api_key=os.environ.get("OPENAI_API_KEY") or None,
            tts_provider=tts_provider,
            tts_voice_id=os.environ.get("TTS_VOICE_ID") or None,
            company_slug=_require("COMPANY_SLUG"),
            supabase_url=_require("SUPABASE_URL"),
            supabase_service_role_key=_require("SUPABASE_SERVICE_ROLE_KEY"),
        )
