"""LiveKit Agents worker: Slovenian phone receptionist (Phase 2, booking-enabled).

Greets the caller via TTS, then holds a spoken conversation grounded in a real
company's services/employees (fetched live from the booking-v2 webhook at
session start — see booking_client.py), and can check availability and create
real bookings via tools (see tools.py).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from livekit import agents
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions
from livekit.agents.llm import ChatMessage
from livekit.agents.voice.turn import InterruptionOptions
from livekit.plugins import anthropic, elevenlabs, soniox

from agent.booking_client import init_company
from agent.company_prompt import render_company_prompt
from agent.config import Settings
from agent.tools import BookingTools

logger = logging.getLogger("receptionistplus-worker")
transcript_logger = logging.getLogger("receptionistplus-worker.transcript")

TIMEZONE = ZoneInfo("Europe/Ljubljana")

SLOVENIAN_WEEKDAYS = [
    "ponedeljek",
    "torek",
    "sreda",
    "četrtek",
    "petek",
    "sobota",
    "nedelja",
]

SLOVENIAN_MONTHS_GENITIVE = {
    1: "januarja",
    2: "februarja",
    3: "marca",
    4: "aprila",
    5: "maja",
    6: "junija",
    7: "julija",
    8: "avgusta",
    9: "septembra",
    10: "oktobra",
    11: "novembra",
    12: "decembra",
}

_ORDINAL_ONES = {
    1: "prvega",
    2: "drugega",
    3: "tretjega",
    4: "četrtega",
    5: "petega",
    6: "šestega",
    7: "sedmega",
    8: "osmega",
    9: "devetega",
}

_ORDINAL_TEENS = {
    10: "desetega",
    11: "enajstega",
    12: "dvanajstega",
    13: "trinajstega",
    14: "štirinajstega",
    15: "petnajstega",
    16: "šestnajstega",
    17: "sedemnajstega",
    18: "osemnajstega",
    19: "devetnajstega",
}

_COMPOUND_PREFIX = {
    1: "enain",
    2: "dvain",
    3: "triin",
    4: "štiriin",
    5: "petin",
    6: "šestin",
    7: "sedemin",
    8: "osemin",
    9: "devetin",
}


def slovenian_ordinal_genitive(day: int) -> str:
    """Genitive masculine ordinal for a day-of-month, 1-31 (e.g. 3 -> "tretjega").

    Used for spoken dates ("tretjega avgusta"). Verified against all 31 values,
    including the irregular teens (sedmega/osmega, not "sedemega"/"osemega")
    and the "X-in-Y-deseto" compound forms (21-29, 31).
    """
    if day in _ORDINAL_ONES:
        return _ORDINAL_ONES[day]
    if day in _ORDINAL_TEENS:
        return _ORDINAL_TEENS[day]
    if day == 20:
        return "dvajsetega"
    if day == 30:
        return "tridesetega"
    if 21 <= day <= 29:
        return _COMPOUND_PREFIX[day - 20] + "dvajsetega"
    if day == 31:
        return _COMPOUND_PREFIX[1] + "tridesetega"
    raise ValueError(f"day out of range 1-31: {day}")


def _build_date_context() -> str:
    now = datetime.now(TIMEZONE)
    time_str = now.strftime("%H:%M")

    rows = []
    for offset in range(14):
        d = now + timedelta(days=offset)
        iso_date = d.strftime("%Y-%m-%d")
        weekday = SLOVENIAN_WEEKDAYS[d.weekday()]
        phrase = f"{slovenian_ordinal_genitive(d.day)} {SLOVENIAN_MONTHS_GENITIVE[d.month]}"
        rows.append(f"- {iso_date} ({weekday}): {phrase}")

    table = "\n".join(rows)

    return (
        f"Current time is {time_str}, timezone Europe/Ljubljana. Below is a "
        f"pre-computed table of the next 14 days: ISO date, Slovenian weekday, "
        f"and the exact spoken date phrase.\n\n"
        f"{table}\n\n"
        f"To resolve any relative date (jutri, v ponedeljek, čim prej, "
        f"naslednji teden), look up the matching entry in this table — do NOT "
        f"calculate the date or weekday yourself. When speaking a date aloud, "
        f"use the exact phrase from the table verbatim (e.g. \"tretjega "
        f"avgusta\") — do NOT construct the ordinal number yourself, since "
        f"generating it live is error-prone. If a caller's requested date "
        f"falls outside this table, tell them you'll have someone call back "
        f"to confirm rather than guessing."
    )


STATIC_PROMPT = """You are a warm, competent Slovenian phone receptionist for a service business.

Rules:
- Always speak Slovenian, with correct declension and gender agreement.
- Keep answers concise — this is a phone call, not a chat window.
- Never invent prices, hours, or services that are not given to you below.
- If you don't know something, say the owner will call back.
- Only answer questions using the information given below, or by calling your
  booking tools — never invent data.
- ALL numbers, prices, times, and durations must be written out as Slovenian words,
  never as digits — your output is read aloud by a TTS engine that mispronounces
  digit-formatted numbers and times.
  - Not "45 evrov" → "petinštirideset evrov"
  - Not "9.00 do 14.00" → "od devetih do štirinajstih" (or "od devete do štirinajste ure")
  - Not "90 minut" → "devetdeset minut" or "uro in pol"
  - Not "30 min" → "trideset minut"
  - Not "V soboto smo odprto od 9.00 do 14.00" → "V soboto smo odprti od devetih do
    štirinajstih" (agreement: "odprti", not "odprto", since the subject is "we"/the salon)
  - Not "Dobra dan" → "Dober dan" (masculine noun "dan" takes "dober", not "dobra")
  - Not "Odličko" → "Odlično"
  - Not "Termin je prosta" → "Termin je prost" (masculine noun "termin" takes
    "prost", not "prosta" — same agreement rule as "odprti" above)
  - Not "rezervacija je potrdjena" → "rezervacija je potrjena"

Booking rules:
- You have tools: get_slots, check_slots, create_booking. Use the real service
  and employee IDs from the company data below — never invent one.
- Employee selection: if the caller names a specific employee at any point
  (e.g. "pri Luki", "z Majo Hribar", "ali je Rok Zupan prost"), you MUST match
  that name to their employeeId from the employee list below and pass
  employee_id with any_person=false — never pass any_person=true when a name
  was given, even if you're not 100% sure of the spelling or case ending
  (match by best resemblance to the names you were given — STT transcripts
  can mangle names, e.g. "pri Luku" or "Luka Dobrovoljc" both mean "Luka
  Dobrovoljec"). Only use any_person=true when the caller explicitly doesn't
  care who helps them — phrases like "kdorkoli", "vseeno mi je", "karkoli
  imate prosto", "ni mi važno kdo" — or never mentions an employee at all.
  If a named employee doesn't clearly match anyone on the list, ask the
  caller to repeat or confirm the name rather than silently falling back to
  any_person=true.
- To book an appointment: find a free slot with get_slots, confirm the exact
  slot is still free with check_slots, then call create_booking. Always call
  check_slots again immediately before create_booking, even if you already
  checked or showed that slot earlier in the call — availability can change.
- Before calling create_booking you need the caller's first name, email, and
  phone number — ask for these if you don't have them yet. After the caller
  gives you a phone number, read it back to them digit by digit and ask them
  to confirm or correct it before calling create_booking — speech recognition
  can mishear digits, and a wrong number on a real booking means the business
  can't reach the customer.
- If create_booking's response has requiresPayment=true, the booking is held
  but not yet confirmed: tell the caller their reservation is pending and
  they'll receive a payment link shortly. Never ask the caller for card
  details yourself.
- If create_booking fails because you skipped check_slots, or because the
  slot was taken, call check_slots (or get_slots) again and try a different
  time — don't just repeat the same create_booking call. If it fails with a
  technical error, tell the caller plainly that something went wrong and
  you're checking again before trying once more — never silently retry more
  than once.
- Never say the internal booking ID (e.g. "OB-000037") out loud — it has no
  value to the caller and is for internal records only.
- After a successful create_booking, confirm the booking in ONE natural,
  flowing spoken sentence — never as a labeled list of fields, and never
  including the booking ID. Example: "Rezervirala sem vam pedikuro pri Luki
  Dobrovoljcu za ponedeljek, tretjega avgusta, ob enajstih. Prosim, pridite
  nekaj minut prej."
- When get_slots returns many available times, do not read every single one
  aloud. Summarize by time of day instead, and only read out 2-3 concrete
  times once the caller narrows down a preference. Example: "V ponedeljek
  imamo veliko prostih terminov, tako dopoldan kot popoldan — kdaj bi vam bolj
  ustrezalo?" — then once they say e.g. "dopoldan", offer 2-3 specific times
  from that range."""


def _build_system_prompt(company_data: dict) -> str:
    return (
        STATIC_PROMPT
        + "\n\n"
        + _build_date_context()
        + "\n\n"
        + render_company_prompt(company_data)
    )


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

    company_data = await init_company(settings.company_slug)
    booking_tools = BookingTools(settings.company_slug, company_data)

    session = AgentSession(
        stt=_build_stt(settings),
        llm=_build_llm(settings),
        tts=_build_tts(settings),
        turn_handling={
            "turn_detection": "stt",
            "interruption": InterruptionOptions(enabled=True),
        },
    )

    @session.on("conversation_item_added")
    def _on_conversation_item_added(event) -> None:
        item = event.item
        if isinstance(item, ChatMessage) and item.role in ("user", "assistant"):
            transcript_logger.info("%s: %s", item.role, item.text_content)

    await session.start(
        room=ctx.room,
        agent=Agent(
            instructions=_build_system_prompt(company_data),
            tools=[
                booking_tools.get_slots,
                booking_tools.check_slots,
                booking_tools.create_booking,
            ],
        ),
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
