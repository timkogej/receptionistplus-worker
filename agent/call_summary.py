"""A short summary of each call for the business owner, written after
hang-up from the transcript (receptionist_calls.summary).

Runs on Claude Haiku 5.5 through the same wrapper as the conversation
(agent/llm_factory.py). Best effort: any failure or timeout just leaves the
summary empty — it must never hold up billing or call logging.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from livekit.agents import llm

from agent.llm_factory import build_single_llm

logger = logging.getLogger("receptionistplus-worker.summary")

SUMMARY_MODEL = "claude-haiku-5-5"
SUMMARY_TIMEOUT_SEC = 20.0
# Below this many caller turns there is nothing worth summarising (a
# hang-up during the greeting, a single "napačna številka").
MIN_USER_TURNS = 2

_INSTRUCTIONS = {
    "sl": (
        "Spodaj je prepis telefonskega klica med AI receptorko in stranko. Za "
        "lastnika podjetja napiši povzetek v eni do treh kratkih povedih v "
        "slovenščini: kaj je stranka želela, kaj je bilo dogovorjeno ali "
        "rezervirano (storitev, dan, ura) in ali kaj ostaja odprto (npr. "
        "povratni klic). Brez uvoda, brez naštevanja, brez ocen pogovora. Ne "
        "navajaj e-naslovov ali telefonskih številk."
    ),
    "en": (
        "Below is the transcript of a phone call between an AI receptionist "
        "and a caller. For the business owner, write a one-to-three sentence "
        "summary in English: what the caller wanted, what was agreed or "
        "booked (service, day, time), and whether anything is left open "
        "(e.g. a callback). No preamble, no lists, no judging the call. Don't "
        "include email addresses or phone numbers."
    ),
}


_WEEKDAYS = {
    "sl": ("ponedeljek", "torek", "sreda", "četrtek", "petek", "sobota", "nedelja"),
    "en": ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"),
}


def booking_facts(booked: list[dict], company_data: dict | None, language: str) -> str:
    """What was really booked, from the booking tools, so the summary takes
    the day and time from the system and not from the transcript's words
    (a summary once said 14.30 for a 13:30 booking, 2026-10-26)."""
    if not booked:
        return ""
    names = {str(s.get("id")): s.get("naziv") or "" for s in (company_data or {}).get("services", [])}
    days = _WEEKDAYS.get(language, _WEEKDAYS["en"])
    lines = []
    for b in booked:
        try:
            d = datetime.strptime(b["date"], "%Y-%m-%d").date()
            when = f"{days[d.weekday()]} {d.day}. {d.month}. {b['time'][:5]}"
        except (KeyError, ValueError, TypeError):
            continue
        service = ", ".join(n for n in (names.get(str(i), "") for i in b.get("service_ids", [])) if n)
        parts = [service or "storitev", when]
        if b.get("employee"):
            parts.append(b["employee"])
        if b.get("name"):
            parts.append(b["name"])
        lines.append("- " + ", ".join(parts))
    if not lines:
        return ""
    head = (
        "Rezervirano v sistemu (dan in uro vzemi od tu, ne iz prepisa):"
        if language == "sl"
        else "Booked in the system (take the day and time from here, not from the transcript):"
    )
    return head + "\n" + "\n".join(lines)


def _format_transcript(transcript: list[dict]) -> str:
    who = {"user": "Stranka", "assistant": "Receptorka"}
    return "\n".join(
        f"{who.get(t.get('role'), t.get('role'))}: {t.get('text') or ''}"
        for t in transcript
        if t.get("text")
    )


async def summarize_call(
    transcript: list[dict],
    language: str,
    *,
    anthropic_api_key: str | None = None,
    timeout: float = SUMMARY_TIMEOUT_SEC,
    facts: str = "",
) -> str | None:
    if sum(1 for t in transcript if t.get("role") == "user") < MIN_USER_TURNS:
        return None
    model = build_single_llm(SUMMARY_MODEL, effort="low", anthropic_api_key=anthropic_api_key)
    ctx = llm.ChatContext()
    ctx.add_message(role="system", content=_INSTRUCTIONS.get(language, _INSTRUCTIONS["sl"]))
    body = _format_transcript(transcript)
    ctx.add_message(role="user", content=f"{facts}\n\n{body}" if facts else body)

    async def _run() -> str:
        parts = []
        async with model.chat(chat_ctx=ctx) as stream:
            async for chunk in stream:
                if chunk.delta and chunk.delta.content:
                    parts.append(chunk.delta.content)
        return "".join(parts).strip()

    try:
        text = await asyncio.wait_for(_run(), timeout=timeout)
    except Exception:
        logger.exception("call summary failed")
        return None
    finally:
        try:
            await model.aclose()
        except Exception:
            pass
    return text[:1000] or None
