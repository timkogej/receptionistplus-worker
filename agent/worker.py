"""LiveKit Agents worker: Slovenian phone receptionist (Phase 2, booking-enabled).

Greets the caller via TTS, then holds a spoken conversation grounded in a real
company's services/employees (fetched live from the booking-v2 webhook at
session start — see booking_client.py), and can check availability and create
real bookings via tools (see tools.py).
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from livekit import agents
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, llm
from livekit.agents.llm import ChatMessage
from livekit.agents.metrics import EOUMetrics, LLMMetrics, STTMetrics, TTSMetrics
from livekit.agents.utils.audio import audio_frames_from_file
from livekit.agents.voice.turn import InterruptionOptions
from livekit.plugins import anthropic, elevenlabs, openai, soniox

from agent.booking_client import BookingError, init_company
from agent.company_prompt import render_company_prompt
from agent.config import Settings
from agent.supabase_client import SupabaseClient
from agent.tools import (
    CREATE_BOOKING_FILLER_TEXT,
    GET_SLOTS_FILLER_TEXT,
    BookingTools,
)

logger = logging.getLogger("receptionistplus-worker")
transcript_logger = logging.getLogger("receptionistplus-worker.transcript")

NO_CREDITS_MESSAGE = (
    "Trenutno žal ne moremo sprejeti vašega klica. Prosimo, poskusite kasneje."
)

TECHNICAL_DIFFICULTY_MESSAGE = (
    "Oprostite, trenutno imamo tehnične težave, lastnik vas bo poklical nazaj."
)

# EU AI Act Article 50: callers must be told they're talking to an AI system,
# no later than the first interaction. Prepended in code rather than baked
# into GREETING_TEXT so it can't be dropped by a misconfigured or edited
# per-company greeting — this is the one guaranteed source of disclosure,
# not the only one.
AI_DISCLOSURE_SL = (
    "Prosimo, upoštevajte, da govorite z digitalnim glasovnim asistentom, ne "
    "z osebo. "
)

# Pre-rendered once via a working Soniox TTS call (see assets/README or the
# generation snippet in the Phase 5 Part C notes) so this specific message can
# still be played when the configured TTS provider is the thing that's down —
# session.say(..., audio=...) bypasses TTS synthesis entirely for it.
TECHNICAL_DIFFICULTY_AUDIO_PATH = (
    Path(__file__).resolve().parent.parent / "assets" / "technical_difficulty_sl.wav"
)

# Calls shorter than this are treated as accidental hangups/misdials, not
# billable conversations.
ABANDONED_CALL_THRESHOLD_SEC = 10

# Generic filler (any slow turn, not just the three booking tool calls — see
# the entrypoint's _generic_filler_loop). Set above create_booking's own
# filler delay (4.5s) so create_booking's specific tuned wording ("Samo
# trenutek, urejam rezervacijo...") still gets first chance to fire on that
# path — the most consequential moment (a real booking in progress)  keeps
# its specific message; this generic one is a catch-all for everything else,
# including plain conversational turns which previously had no filler.
GENERIC_FILLER_PHRASES = [
    "Samo trenutek...",
    "En moment...",
    "Samo sekundo...",
]
GENERIC_FILLER_DELAY = 5.0

# Promise-follow-up watchdog (2026-08-16 fix): a fallback-model turn that
# says "checking..."/"trenutek, prosim..." and then never calls the tool it
# just promised leaves the caller hanging indefinitely — observed for real
# during a sustained Anthropic outage where GPT-4o-mini spoke this exact
# pattern and then produced no further activity for 4+ minutes until the
# caller gave up. If no new agent/user activity follows such an utterance
# within this window, treat it as a technical failure (see
# _watch_for_promise_followup in entrypoint).
PROMISE_CUE_PATTERN = re.compile(
    r"trenutek|preverim\b|preverjam\b|preveril|urejam\b|uredila\b|rezerviram\b"
    r"|poskušam\b|počakajte",
    re.IGNORECASE,
)
PROMISE_WATCHDOG_TIMEOUT_SEC = 12.0

# Our own filler utterances also match PROMISE_CUE_PATTERN by design, but
# they're paired with an actual in-flight tool call that already has its own
# real timeout (up to 45s for create_booking) — excluded so the watchdog's
# much shorter window doesn't kill a call that's legitimately still running.
_KNOWN_FILLER_TEXTS = frozenset(
    GENERIC_FILLER_PHRASES + [GET_SLOTS_FILLER_TEXT, CREATE_BOOKING_FILLER_TEXT]
)

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


_RELATIVE_DAY_LABELS = {0: "DANES", 1: "JUTRI", 2: "POJUTRIŠNJEM"}


def _build_date_context() -> str:
    """Build the date-lookup table given to the LLM each turn.

    Bug fixed 2026-08-14: the table used to only give each row's absolute
    date/weekday, leaving the model to work out on its own which row is
    "jutri" (offset 0 vs 1) — exactly the kind of manual calculation the
    instructions claimed to forbid. Observed failure: it labeled TODAY's row
    as "jutri" and offered it as a bookable day, and since tomorrow was
    actually a closed Saturday, this made the mislabeled date look like a
    plausible near-term suggestion instead. Fix: stamp DANES/JUTRI/
    POJUTRIŠNJEM directly onto the first three rows so there's no offset
    arithmetic left for the model to get wrong.
    """
    now = datetime.now(TIMEZONE)
    time_str = now.strftime("%H:%M")

    rows = []
    for offset in range(14):
        d = now + timedelta(days=offset)
        iso_date = d.strftime("%Y-%m-%d")
        weekday = SLOVENIAN_WEEKDAYS[d.weekday()]
        phrase = f"{slovenian_ordinal_genitive(d.day)} {SLOVENIAN_MONTHS_GENITIVE[d.month]}"
        label = _RELATIVE_DAY_LABELS.get(offset)
        label_str = f" [{label}]" if label else ""
        rows.append(f"- {iso_date} ({weekday}){label_str}: {phrase}")

    table = "\n".join(rows)

    return (
        f"Current time is {time_str}, timezone Europe/Ljubljana. Below is a "
        f"pre-computed table of the next 14 days: ISO date, Slovenian weekday, "
        f"and the exact spoken date phrase. The first three rows are tagged "
        f"[DANES] (today), [JUTRI] (tomorrow), and [POJUTRIŠNJEM] (the day "
        f"after tomorrow) directly — use these tags as-is to resolve those "
        f"specific words, do NOT count rows or compute the offset yourself.\n\n"
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
- NEVER state a specific date, day, or time as available (or unavailable)
  unless you have a get_slots or check_slots tool RESULT from THIS turn or
  an earlier turn in THIS SAME conversation backing that exact claim. If the
  caller asks about a date range you have not already queried — including
  "proti koncu tedna", "naslednji teden", or any date outside what you've
  already shown them — call get_slots again with the new range. get_slots
  can be called as many times as needed in one call; there is no limit on
  how many times you may query it. Never say you "don't have data" for a
  date range instead of just calling get_slots for it.
  - Bad: telling the caller "imamo proste termine v ponedeljek in torek"
    without ever having called get_slots this conversation.
  - Bad: caller asks "a bi se dalo kaj proti koncu tedna?" → "Nimam podatkov
    za termine proti koncu tedna." (refusing instead of calling get_slots
    again with a later date range)
  - Good: caller asks about a new date range → call get_slots with that
    range, THEN answer from its actual result.
- If you tell the caller you are about to check something (e.g. "Preverim
  razpoložljivost...", "Trenutek, prosim..."), you MUST immediately call the
  corresponding tool (get_slots/check_slots/create_booking) in that same
  turn — never say you're checking and then stop without calling the tool.
  A spoken promise with no tool call leaves the caller waiting with no
  response.
- When summarizing a get_slots response, check every date key individually
  — do not treat a date as unavailable unless its value is literally the
  string "unavailable". Observed 2026-08-16: a get_slots response with 5
  weekdays of full availability followed by 2 "unavailable" weekend days got
  summarized as only the first 4 weekdays being open, silently dropping the
  5th (a real, fully-available day) as if it were part of the closed
  weekend block. Read the whole object before excluding any day.
  - Bad: response has 2026-08-24 through 2026-08-28 all with real time
    lists, and 2026-08-29/2026-08-30 as "unavailable" → saying "od
    ponedeljka do četrtka" (Mon-Thu only), skipping Friday.
  - Good: every date with a real (non-"unavailable") value is available,
    including the last weekday right before the weekend block.
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
  - Not "Termin je prosta" → "Termin je prost" (masculine noun "termin" takes
    "prost", not "prosta" — same agreement rule as "odprti" above)
  - Not "rezervacija je potrdjena" → "rezervacija je potrjena"
- "Odličko" is NOT a Slovenian word and must never be used, under any
  circumstance — the correct word is "Odlično" (great/excellent). This is a
  recurring model mistake, not a one-off — treat it as a hard-banned word.
  - Not "Odličko! Termin ob deseti uri je prost." → "Odlično! Termin ob
    deseti uri je prost."
- Always use formal address (vikanje: "vi"/"vam"/"ste"), never informal
  "ti"/"tebi"/"si" — even if the caller speaks informally first. Do not
  mirror the caller's register.
  - Caller: "Živjo, kako si?" → Bad: "Živjo! Hvala, da vprašaš. Kako ti lahko
    pomagam danes?" → Good: "Pozdravljeni! Kako vam lahko pomagam?"
- Your output is spoken directly by a TTS engine — never use markdown
  (no "**bold**", "*italic*", "-" bullet lists, headings, etc.). Write plain
  natural spoken sentences only.
  - Bad: "**Refleksna masaža stopal** – šestdeset minut za petnajst evrov"
  - Good: "Refleksna masaža stopal traja šestdeset minut in stane petnajst
    evrov."
- "Samo trenutek" / "trenutek prosim" are the correct ways to ask the caller
  to wait — never "prosimo, da mi trenutek", which is not valid Slovenian.
  - Bad: "Sedaj bom preverila prosti termine. Prosimo, da mi trenutek."
  - Good: "Sedaj bom preverila prosti termine. Trenutek prosim." (or "Samo
    trenutek, prosim.")
- Never enumerate more than 2-3 items aloud in one turn — for any list-like
  answer (services, prices, available dates, etc.). If there are more than
  2-3, summarize/group them and ask a clarifying question to narrow down,
  then list the specific 2-3 that match.
  - Caller: "Kaj ponujate?" → Bad: naming every single service with prices
    in one breath. → Good: "Ponujamo več kozmetičnih storitev — na primer
    pedikuro, nego obraza in masažo. Vas kaj od tega zanima, pa vam povem
    več?"
  - EXCEPTION — time slots specifically are NOT covered by this rule: do
    not apply "just pick any 2-3" here. Follow the more specific time-slot
    rule under "Booking rules" below instead (group by dopoldan/popoldan
    and ask a preference first — never lead with 2-3 exact times, even
    though 2-3 would technically satisfy this generic cap).

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
  from that range.
  - THIS RULE TAKES PRECEDENCE over the generic "cap lists to 2-3 items"
    rule above, specifically for time slots. Even if only 2-3 slots would
    satisfy that generic cap, still group by time-of-day and ask the
    caller's preference first — never lead with specific times.
    - Bad: "V ponedeljek imamo proste termine ob osmih, devetih in deseti
      uri. Kateri čas vam najbolj ustreza?" (jumps straight to 3 exact
      times, skipping the time-of-day question)
    - Good: "V ponedeljek imamo veliko prostih terminov, tako dopoldan kot
      popoldan — kdaj bi vam bolj ustrezalo?" """


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
        # Explicit, not plugin defaults: the plugin defaults to model
        # "tts-rt-v1-preview" and language "en" if unset, which is what the
        # live pipeline was silently running on for Slovenian calls (found
        # 2026-08-16 comparing against tools/tts_ab.py, which explicitly
        # requests these two params and sounded noticeably better).
        model="tts-rt-v1",
        language="sl",
    )


def _build_stt(settings: Settings) -> soniox.STT:
    return soniox.STT(
        api_key=settings.soniox_api_key,
        params=soniox.STTOptions(language_hints=["sl"]),
    )


async def _init_company_with_retry(
    company_slug: str, *, attempts: int = 2, retry_delay: float = 1.0
) -> dict:
    """Phase 5 Part C: booking-v2 is unreachable often enough (transient
    network blips, n8n restarts) that a single failed attempt shouldn't sink
    the call outright. Kept separate from call_booking's own retry policy —
    unlike create_booking, init has no side effects, so blindly retrying it
    is safe.
    """
    last_exc: BookingError | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await init_company(company_slug)
        except BookingError as exc:
            last_exc = exc
            logger.warning(
                "booking-v2 init attempt %d/%d failed for company_slug=%s: %s",
                attempt,
                attempts,
                company_slug,
                exc,
            )
            if attempt < attempts:
                await asyncio.sleep(retry_delay)
    assert last_exc is not None
    raise last_exc


def _build_llm(settings: Settings) -> llm.LLM:
    primary = anthropic.LLM(
        model="claude-haiku-4-5",
        api_key=settings.anthropic_api_key,
    )
    if not settings.openai_api_key:
        return primary

    # Phase 5 Part C: falls back to GPT-4o-mini if Anthropic is down. Only
    # kicks in after Anthropic's own attempt fails outright — this is not a
    # cost-saving load-balance, just an outage safety net.
    fallback = openai.LLM(model="gpt-4o-mini", api_key=settings.openai_api_key)
    return llm.FallbackAdapter([primary, fallback])


async def entrypoint(ctx: JobContext) -> None:
    settings = Settings.from_env()
    await ctx.connect()

    supabase = SupabaseClient(settings.supabase_url, settings.supabase_service_role_key)
    call_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc)
    livekit_room = ctx.room.name

    balance = await supabase.get_balance(settings.company_slug)
    if balance <= 0:
        logger.warning(
            "no credits: company_slug=%s balance=%s, rejecting call",
            settings.company_slug,
            balance,
        )
        gate_session = AgentSession(tts=_build_tts(settings))
        await gate_session.start(
            room=ctx.room,
            agent=Agent(
                instructions=(
                    "You only ever speak one fixed message and take no other "
                    "action; you have no tools."
                )
            ),
        )
        await gate_session.say(NO_CREDITS_MESSAGE)
        await gate_session.aclose()
        await supabase.insert_call(
            {
                "id": call_id,
                "company_slug": settings.company_slug,
                "started_at": started_at.isoformat(),
                "ended_at": datetime.now(timezone.utc).isoformat(),
                "duration_sec": 0,
                "billed_credits": 0,
                "outcome": "no_credits",
                "transcript": [],
                "livekit_room": livekit_room,
            }
        )
        return

    try:
        company_data = await _init_company_with_retry(settings.company_slug)
    except BookingError:
        logger.exception(
            "booking-v2 init failed for company_slug=%s, degrading gracefully",
            settings.company_slug,
        )
        gate_session = AgentSession(tts=_build_tts(settings))
        await gate_session.start(
            room=ctx.room,
            agent=Agent(
                instructions=(
                    "You only ever speak one fixed message and take no other "
                    "action; you have no tools."
                )
            ),
        )
        try:
            audio = audio_frames_from_file(str(TECHNICAL_DIFFICULTY_AUDIO_PATH))
            await asyncio.wait_for(
                gate_session.say(TECHNICAL_DIFFICULTY_MESSAGE, audio=audio), timeout=8.0
            )
        except Exception:
            logger.exception("failed to play technical-difficulty message (init failure path)")
        await gate_session.aclose()
        await supabase.insert_call(
            {
                "id": call_id,
                "company_slug": settings.company_slug,
                "started_at": started_at.isoformat(),
                "ended_at": datetime.now(timezone.utc).isoformat(),
                "duration_sec": 0,
                "billed_credits": 0,
                "outcome": "booking_unavailable",
                "transcript": [],
                "livekit_room": livekit_room,
            }
        )
        return

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

    transcript: list[dict] = []
    # Set from the session's "close" event, which fires right when the call
    # actually ends (participant disconnect). The process/job shutdown that
    # triggers _on_shutdown below can lag well behind that — observed ~26s in
    # testing — so ended_at must NOT be datetime.now() taken inside
    # _on_shutdown, or short calls get billed for dead time after hangup.
    call_ended_at: dict[str, datetime] = {}

    last_assistant_text = {"value": ""}

    @session.on("conversation_item_added")
    def _on_conversation_item_added(event) -> None:
        item = event.item
        if isinstance(item, ChatMessage) and item.role in ("user", "assistant"):
            transcript_logger.info("%s: %s", item.role, item.text_content)
            transcript.append(
                {
                    "role": item.role,
                    "text": item.text_content,
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
            )
            if item.role == "assistant":
                last_assistant_text["value"] = item.text_content or ""

    @session.on("close")
    def _on_close(event) -> None:
        call_ended_at["value"] = datetime.now(timezone.utc)
        generic_filler_task.cancel()

    # Generic filler for ANY slow turn, not just the three booking tool calls
    # (get_slots/check_slots/create_booking already have their own
    # context.with_filler() in tools.py, tuned per call — this is a separate,
    # session-wide safety net for plain conversational turns, which had no
    # filler at all before this and left callers in dead air on a slow
    # LLM/TTS turn, e.g. during an Anthropic degradation). Mirrors the SDK's
    # own private _FillerScheduler (voice/filler_scheduler.py) — wait for the
    # session to go idle, then wait up to GENERIC_FILLER_DELAY for the agent
    # or caller to become active again; if neither happens, speak a short
    # filler and repeat. Built on public session.on()/wait_for_idle()/say()
    # rather than the private class so it doesn't depend on SDK internals.
    _agent_active = asyncio.Event()

    def _on_agent_state_changed(event) -> None:
        if event.new_state in ("speaking", "thinking"):
            _agent_active.set()

    def _on_user_state_changed(event) -> None:
        if event.new_state == "speaking":
            _agent_active.set()

    session.on("agent_state_changed", _on_agent_state_changed)
    session.on("user_state_changed", _on_user_state_changed)

    async def _generic_filler_loop() -> None:
        step = 0
        while True:
            await session.wait_for_idle()
            _agent_active.clear()
            try:
                await asyncio.wait_for(_agent_active.wait(), timeout=GENERIC_FILLER_DELAY)
                continue  # agent/caller became active in time — no filler needed
            except asyncio.TimeoutError:
                pass
            phrase = GENERIC_FILLER_PHRASES[step % len(GENERIC_FILLER_PHRASES)]
            step += 1
            session.say(phrase)

    generic_filler_task = asyncio.create_task(
        _generic_filler_loop(), name="generic_filler_loop"
    )

    # Latency investigation (2026-08-14 voice test): aggregate turn-latency
    # numbers alone can't say whether time is going to STT endpointing, LLM
    # TTFT, or TTS startup — logging only, no behavior change. Reuses the
    # SDK's own built-in per-stage metrics (already computed internally)
    # rather than hand-rolling separate timestamps, and speech_id/request_id
    # let a later investigation correlate STT/LLM/TTS events for one turn.
    @session.on("metrics_collected")
    def _on_metrics_collected(event) -> None:
        m = event.metrics
        if isinstance(m, EOUMetrics):
            logger.info(
                "latency stt_eou: end_of_utterance_delay=%.3fs transcription_delay=%.3fs speech_id=%s",
                m.end_of_utterance_delay,
                m.transcription_delay,
                m.speech_id,
            )
        elif isinstance(m, LLMMetrics):
            logger.info(
                "latency llm: ttft=%.3fs duration=%.3fs speech_id=%s request_id=%s",
                m.ttft,
                m.duration,
                m.speech_id,
                m.request_id,
            )
        elif isinstance(m, TTSMetrics):
            logger.info(
                "latency tts: ttfb=%.3fs duration=%.3fs speech_id=%s request_id=%s",
                m.ttfb,
                m.duration,
                m.speech_id,
                m.request_id,
            )
        elif isinstance(m, STTMetrics):
            logger.info(
                "latency stt: duration=%.3fs audio_duration=%.3fs request_id=%s",
                m.duration,
                m.audio_duration,
                m.request_id,
            )

    # Phase 5 Part C: the SDK retries a broken stt/tts/llm provider on its
    # own, but that retry loop either never gives up (observed with a broken
    # STT/TTS key — retries indefinitely) or gives up silently after a single
    # attempt with no further signal (observed with a broken LLM key — one
    # `recoverable=False` error and then just... nothing). Either way the
    # caller is left in dead air. `error.recoverable=True` events are the SDK
    # telling us it's still retrying on its own, so we ignore those; the
    # first `recoverable=False` event for any provider means the SDK has
    # already given up on that attempt, which is our cue to cut the call
    # short ourselves rather than wait for more events that may never come.
    provider_error_state = {"degrading": False}

    async def _degrade_and_close() -> None:
        logger.error("ending call early: unrecoverable stt/tts/llm provider error")
        try:
            # Play a pre-recorded clip rather than calling session.say() with
            # plain text: the configured TTS provider may itself be the thing
            # that's broken (as it was during Phase 5 Part C testing, where
            # STT and TTS shared one invalid key), in which case synthesizing
            # the apology live would fail silently right along with it.
            audio = audio_frames_from_file(str(TECHNICAL_DIFFICULTY_AUDIO_PATH))
            await asyncio.wait_for(
                session.say(TECHNICAL_DIFFICULTY_MESSAGE, audio=audio), timeout=8.0
            )
        except Exception:
            logger.exception("failed to play technical-difficulty message before closing")
        finally:
            await session.aclose()

    @session.on("error")
    def _on_session_error(event) -> None:
        if event.error.type not in ("stt_error", "tts_error", "llm_error"):
            return
        logger.warning(
            "provider error: type=%s recoverable=%s",
            event.error.type,
            event.error.recoverable,
        )
        if event.error.recoverable or provider_error_state["degrading"]:
            return

        provider_error_state["degrading"] = True
        asyncio.create_task(_degrade_and_close())

    # Promise-follow-up watchdog: catches the case where the agent speaks a
    # "checking..." utterance and then produces no tool call and no further
    # speech at all — no exception, no recoverable=False event, nothing for
    # _on_session_error above to catch. Armed only on the transition FROM
    # "speaking" (i.e. once the promise utterance has finished playing), so
    # the utterance's own "speaking" state doesn't immediately satisfy its
    # own watchdog. Cancelled by any subsequent agent/user activity.
    _activity_since_promise = asyncio.Event()

    def _on_agent_state_changed_watchdog(event) -> None:
        if event.new_state in ("thinking", "speaking"):
            _activity_since_promise.set()
        if event.old_state == "speaking" and event.new_state != "speaking":
            text = last_assistant_text["value"]
            if (
                text
                and text not in _KNOWN_FILLER_TEXTS
                and PROMISE_CUE_PATTERN.search(text)
            ):
                _activity_since_promise.clear()
                asyncio.create_task(_watch_for_promise_followup())

    def _on_user_state_changed_watchdog(event) -> None:
        if event.new_state == "speaking":
            _activity_since_promise.set()

    async def _watch_for_promise_followup() -> None:
        try:
            await asyncio.wait_for(
                _activity_since_promise.wait(), timeout=PROMISE_WATCHDOG_TIMEOUT_SEC
            )
        except asyncio.TimeoutError:
            if provider_error_state["degrading"]:
                return
            logger.error(
                "promise watchdog fired: agent said %r and produced no "
                "follow-up within %.0fs",
                last_assistant_text["value"],
                PROMISE_WATCHDOG_TIMEOUT_SEC,
            )
            provider_error_state["degrading"] = True
            await _degrade_and_close()

    session.on("agent_state_changed", _on_agent_state_changed_watchdog)
    session.on("user_state_changed", _on_user_state_changed_watchdog)

    async def _on_shutdown() -> None:
        ended_at = call_ended_at.get("value") or datetime.now(timezone.utc)
        duration_sec = max(0, math.ceil((ended_at - started_at).total_seconds()))

        if duration_sec < ABANDONED_CALL_THRESHOLD_SEC:
            billed_credits = 0.0
            outcome = "abandoned"
        else:
            billed_credits = round(duration_sec / 60.0, 2)
            outcome = "booked" if booking_tools.created_termin_id else "info_only"

        new_balance = balance
        if billed_credits > 0:
            try:
                new_balance = await supabase.deduct_credits(
                    company_slug=settings.company_slug,
                    call_id=call_id,
                    billed_credits=billed_credits,
                )
            except Exception:
                logger.exception(
                    "failed to deduct credits for call_id=%s company_slug=%s",
                    call_id,
                    settings.company_slug,
                )
                new_balance = balance
            else:
                try:
                    settings_row = await supabase.get_settings(settings.company_slug)
                except Exception:
                    logger.exception(
                        "failed to fetch receptionist_settings for company_slug=%s",
                        settings.company_slug,
                    )
                    settings_row = None
                low_balance_threshold = (
                    settings_row["low_balance_threshold"] if settings_row else 60
                )
                if new_balance < low_balance_threshold:
                    logger.warning(
                        "LOW BALANCE: company_slug=%s balance=%s",
                        settings.company_slug,
                        new_balance,
                    )

        try:
            await supabase.insert_call(
                {
                    "id": call_id,
                    "company_slug": settings.company_slug,
                    "started_at": started_at.isoformat(),
                    "ended_at": ended_at.isoformat(),
                    "duration_sec": duration_sec,
                    "billed_credits": billed_credits,
                    "outcome": outcome,
                    "transcript": transcript,
                    "created_termin_id": booking_tools.created_termin_id,
                    "livekit_room": livekit_room,
                }
            )
        except Exception:
            logger.exception("failed to insert receptionist_calls row for call_id=%s", call_id)

    ctx.add_shutdown_callback(_on_shutdown)

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

    try:
        await session.say(AI_DISCLOSURE_SL + settings.greeting_text)
    except Exception:
        # A broken TTS provider surfaces here too; the "error" handler above
        # will already be counting toward _degrade_and_close, so just avoid
        # crashing the entrypoint over it.
        logger.exception("failed to speak greeting")

    logger.info("greeting delivered, conversation active")


if __name__ == "__main__":
    agents.cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="receptionistplus-worker",
        )
    )
