"""LiveKit Agents worker: Slovenian phone receptionist (Phase 1, info-only).

Greets the caller via TTS, then holds a spoken Q&A conversation grounded in a
single injected company's services/hours/FAQ (see `test_company.py`). No
booking tools yet — the agent only answers from the data it's given.
"""

from __future__ import annotations

import logging

from livekit import agents
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions
from livekit.agents.voice.turn import InterruptionOptions
from livekit.plugins import anthropic, elevenlabs, soniox

from agent.config import Settings
from agent.test_company import TEST_COMPANY, render_company_prompt

logger = logging.getLogger("receptionistplus-worker")

STATIC_PROMPT = """You are a warm, competent Slovenian phone receptionist for a service business.

Rules:
- Always speak Slovenian, with correct declension and gender agreement.
- Keep answers concise — this is a phone call, not a chat window.
- Never invent prices, hours, or services that are not given to you below.
- If you don't know something, say the owner will call back.
- You have no tools right now — only answer questions using the information given.
- ALL numbers, prices, times, and durations must be written out as Slovenian words,
  never as digits — your output is read aloud by a TTS engine that mispronounces
  digit-formatted numbers and times.
  - Not "45 evrov" → "petinštirideset evrov"
  - Not "9.00 do 14.00" → "od devetih do štirinajstih" (or "od devete do štirinajste ure")
  - Not "90 minut" → "devetdeset minut" or "uro in pol"
  - Not "30 min" → "trideset minut"
  - Not "V soboto smo odprto od 9.00 do 14.00" → "V soboto smo odprti od devetih do
    štirinajstih" (agreement: "odprti", not "odprto", since the subject is "we"/the salon)
  - Not "Dobra dan" → "Dober dan" (masculine noun "dan" takes "dober", not "dobra")"""


def _build_system_prompt() -> str:
    return STATIC_PROMPT + "\n\n" + render_company_prompt(TEST_COMPANY)


def _build_tts(settings: Settings):
    if settings.tts_provider == "elevenlabs":
        return elevenlabs.TTS(
            api_key=settings.elevenlabs_api_key,
            voice_id=settings.tts_voice_id or "21m00Tcm4TlvDq8ikWAM",
        )
    return soniox.TTS(
        api_key=settings.soniox_api_key,
        voice=settings.tts_voice_id or "Maya",
    )


def _build_stt(settings: Settings) -> soniox.STT:
    return soniox.STT(
        api_key=settings.soniox_api_key,
        params=soniox.STTOptions(language_hints=["sl"]),
    )


def _build_llm(settings: Settings) -> anthropic.LLM:
    return anthropic.LLM(
        model="claude-haiku-4-5",
        api_key=settings.anthropic_api_key,
    )


async def entrypoint(ctx: JobContext) -> None:
    settings = Settings.from_env()
    await ctx.connect()

    session = AgentSession(
        stt=_build_stt(settings),
        llm=_build_llm(settings),
        tts=_build_tts(settings),
        turn_handling={
            "turn_detection": "stt",
            "interruption": InterruptionOptions(enabled=True),
        },
    )
    await session.start(
        room=ctx.room,
        agent=Agent(instructions=_build_system_prompt()),
    )

    await session.say(settings.greeting_text)

    logger.info("greeting delivered, conversation active")


if __name__ == "__main__":
    agents.cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="receptionistplus-worker",
        )
    )
