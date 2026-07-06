"""LiveKit Agents worker: greets inbound SIP callers via TTS, then waits silently.

No STT and no LLM are wired up yet — the agent speaks a fixed, env-configurable
greeting on participant join and does nothing else.
"""

from __future__ import annotations

import logging

from livekit import agents
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions
from livekit.plugins import elevenlabs, soniox

from agent.config import Settings

logger = logging.getLogger("receptionistplus-worker")


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


async def entrypoint(ctx: JobContext) -> None:
    settings = Settings.from_env()
    await ctx.connect()

    session = AgentSession(tts=_build_tts(settings))
    await session.start(
        room=ctx.room,
        agent=Agent(instructions="You greet callers. You do not converse."),
    )

    await session.say(settings.greeting_text)

    logger.info("greeting delivered, waiting silently")


if __name__ == "__main__":
    agents.cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="receptionistplus-worker",
        )
    )
