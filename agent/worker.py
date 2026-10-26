"""LiveKit Agents worker: Slovenian phone receptionist (Phase 2, booking-enabled).

Greets the caller via TTS, then holds a spoken conversation grounded in a real
company's services/employees (fetched live from the booking-v2 webhook at
session start — see booking_client.py), and can check availability and create
real bookings via tools (see tools.py).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from livekit import agents, api, rtc
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RunContext,
    StopResponse,
    UserTurnExceededEvent,
    WorkerOptions,
    function_tool,
    llm,
)
from livekit.agents.llm import ChatMessage
from livekit.agents.metrics import EOUMetrics, LLMMetrics, STTMetrics, TTSMetrics
from livekit.agents.utils.audio import audio_frames_from_file
from livekit.agents.voice.turn import InterruptionOptions
from livekit.agents import JobProcess
from livekit.agents.voice import room_io
from livekit.agents.voice.background_audio import (
    AudioConfig,
    BackgroundAudioPlayer,
    BuiltinAudioClip,
)
from livekit.plugins import elevenlabs, noise_cancellation, silero, soniox

from agent.booking_client import BookingError, init_company
from agent.call_policy import (
    MAX_CALL_SEC,
    bill_call,
    call_time_limit_sec,
    company_slug_from_job_metadata,
    duration_seconds,
    normalize_phone,
)
from agent.company_prompt import render_company_prompt
from agent.config import Settings
from agent.dates import ENGLISH_WEEKDAYS, SLOVENIAN_WEEKDAYS, WEEKDAYS, slovenian_number, spoken_date_phrase
from agent.static_prompt import ALL_FEATURES, build_static_prompt
from agent.supabase_client import SupabaseClient
from agent.business_info import (
    booking_window_days,
    is_open,
    parse_urnik,
    render_business_block,
    render_caller_block,
    render_confirmation_line,
)
from agent.call_metrics import CallMetrics
from agent.call_notes import CallNotes
from agent.languages import (
    DEFAULT_LANGUAGE,
    LANGUAGE_NAMES,
    SUPPORTED_LANGUAGES,
    TO_JEDROPLUS_CODE,
    detect_switch,
    normalize_language,
    pack,
)
from agent.call_summary import summarize_call
from agent.appointments import AppointmentTools
from agent.group_booking import GroupBooking
from agent.booking_extras import BookingExtras, render_extras_block
from agent import greeting_audio
from agent.greeting import build_greeting, render_assistant_identity, spoken_business_name
from agent.owner_alert import owner_alert_text, owner_language, unfinished_booking_message
from agent.sms_client import (
    appointment_change_text,
    booking_confirmation_text,
    group_confirmation_text,
    choose_confirmation_template,
    render_confirmation_template,
    send_sms,
)
from agent.spoken_text import PHONE_FIRST_QUESTIONS, SpokenTextFilter, caller_digits
from agent.tools import _sanitize_free_text as _sanitize
from agent.llm_factory import LLMConfig, build_llm
from agent.tools import BookingTools

logger = logging.getLogger("receptionistplus-worker")
transcript_logger = logging.getLogger("receptionistplus-worker.transcript")
# Conversation lines in journald: off by default (2026-10-16). The transcript
# is stored in receptionist_calls (deleted after 365 days); a second copy in
# the system log has no retention at all. LOG_TRANSCRIPTS=on for debugging.
LOG_TRANSCRIPTS = os.environ.get("LOG_TRANSCRIPTS", "").strip().lower() in ("1", "on", "true", "yes")


# NO_CREDITS_MESSAGE: per language in agent/languages.py (LanguagePack.no_credits).

# Spoken before closing a call the agent is ending because abuse continued
# after one warning (see the end_call_abusive tool). Deliberately a normal,
# warm sign-off rather than a rebuke or an accusation: by this point the
# decision is made, and the last thing the caller hears should not escalate.
# It is a graceful close, NOT a hangup — same shape as
# TECHNICAL_DIFFICULTY_MESSAGE, which is spoken and then followed by
# session.aclose(). Slovenian uses the feminine "zaključila" per the FEMALE
# persona rule in STATIC_PROMPT_SL.
# ABUSE_CLOSING_MESSAGE: per language in agent/languages.py (LanguagePack.abuse_closing).

# TECHNICAL_DIFFICULTY_MESSAGE: per language in agent/languages.py (LanguagePack.technical_difficulty).

# EU AI Act Article 50: callers must be told they're talking to an AI system,
# no later than the first interaction. Since 2026-10-23 the notice is part of
# the greeting's non-interruptible opening ("Sem digitalna asistentka.",
# agent/greeting.py), which the owner can't edit; with the owner's own
# greeting text (style "custom") it is LanguagePack.ai_disclosure, spoken
# before that text. The business name comes from greeting_name or the
# company profile, never a hand-maintained env var (one went stale for real
# in 2026-08: callers heard "Salon Lepote", a test company).

# Pre-rendered once via a working Soniox TTS call (see assets/README or the
# generation snippet in the Phase 5 Part C notes) so this specific message can
# still be played when the configured TTS provider is the thing that's down —
# session.say(..., audio=...) bypasses TTS synthesis entirely for it.
# Slovenian-only for now: English calls fall back to a text-only session.say()
# with no pre-rendered audio (see _say_technical_difficulty) — a deliberate
# scope decision, not an oversight, since this path only matters when the
# live TTS provider itself is down.
TECHNICAL_DIFFICULTY_AUDIO_PATH = (
    Path(__file__).resolve().parent.parent / "assets" / "technical_difficulty_sl.wav"
)

# Spoken when the call reaches its time limit (see call_policy.
# call_time_limit_sec). The callback promise is real: reaching the limit sets
# callback_requested on the call row, which notifies the owner.
# TIME_LIMIT_CLOSING_MESSAGE: per language in agent/languages.py (LanguagePack.time_limit_closing).

# How long to wait, once the time limit is reached, for an in-flight turn
# (e.g. create_booking, up to 45s) to finish before closing anyway.
TIME_LIMIT_IDLE_WAIT_SEC = 50.0

# Spoken right before a SIP transfer to a person (transfer_to_person tool).
# TRANSFER_MESSAGE: per language in agent/languages.py (LanguagePack.transfer_message).

# SIP participant attributes set by LiveKit for telephony calls.
SIP_DIALED_NUMBER_ATTR = "sip.trunkPhoneNumber"
SIP_CALLER_NUMBER_ATTR = "sip.phoneNumber"

# How long to wait for the SIP caller to appear in the room. They are
# normally already there when the job starts (the room is created for the
# call), so this only bounds a pathological case.
PARTICIPANT_WAIT_SEC = 5.0

# Slow turns (2026-10-16). The old generic filler waited for the session to
# go idle and spoke "Samo trenutek..." if nothing happened for 5s — but
# "thinking" counted as activity, so it never fired on a slow LLM turn, and
# DID fire whenever the CALLER was silent for 5s (e.g. thinking about which
# day suits them), telling a silent caller "one moment". It is gone. A slow
# model turn is now covered by the typing sound (BACKGROUND AUDIO below),
# which LiveKit plays for exactly as long as the agent is "thinking"; a
# spoken line can't be used there, because session.say() queues behind the
# reply being generated and would be heard after it. Tool calls keep their
# own spoken fillers (tools.py), which are timed correctly by with_filler.

# Caller silence (2026-10-16). LiveKit marks the user "away" after this many
# seconds in which neither side speaks. First time: check the line is still
# there; second time in a row: close politely. Long enough that a caller
# looking something up (a diary, a card) isn't nagged.
CALLER_SILENCE_SEC = 9.0
# STILL_THERE_PHRASES: per language in agent/languages.py (LanguagePack.still_there).
# SILENCE_CLOSING_MESSAGE: per language in agent/languages.py (LanguagePack.silence_closing).

# Background audio (2026-10-16): keyboard typing while the agent is
# "thinking" (LLM or tool call), so a pause sounds like someone typing into
# a calendar rather than a dropped line, and optionally a quiet office
# ambience under the whole call. Volumes are low on purpose — on an 8 kHz
# phone line anything louder competes with speech. Switch off with
# THINKING_SOUND=off / AMBIENT_SOUND=off in the env.
THINKING_SOUND_VOLUME = 0.55
AMBIENT_SOUND_VOLUME = 0.12

# Rambling-caller redirect (2026-09-08). In "stt" turn detection a turn ends
# on SILENCE, so a caller who monologues without pausing never produces an
# endpoint and the agent stays mute indefinitely — there was no upper bound on
# a single user turn anywhere in the config. The SDK has a purpose-built
# feature for this (turn_handling["user_turn_limit"]) which ships disabled
# (both thresholds default to None); enabled below.
#
# Thresholds are deliberately generous: this must never fire on someone simply
# explaining what they want. A caller describing a problem in detail runs well
# under 45s / 150 words; sustained monologue is what we're catching. Whichever
# trips first wins. The framework accumulates across consecutive user turns
# and only resets the counters once the agent actually SPEAKS, so a caller who
# keeps talking through several short non-responses still trips it.
USER_TURN_MAX_DURATION_SEC = 45.0
USER_TURN_MAX_WORDS = 150

# How often to log in-progress user-turn growth. This exists to produce direct
# evidence that interim transcripts actually arrive mid-monologue at the
# cadence the SDK source implies — the assumption the first attempt at this
# feature got wrong. Throttled so a normal turn logs once or twice, not per
# interim (Soniox emits these several times a second).
USER_TURN_CADENCE_LOG_INTERVAL_SEC = 5.0

# Spoken as a canned line rather than via the SDK's default handler, which
# calls generate_reply() with English instructions ("Politely cut in...").
# That would work — the system prompt forces Slovenian output anyway — but a
# fixed line is predictable, costs no LLM round-trip at the exact moment the
# caller is already monopolising the turn, and can't hallucinate an
# availability claim. Rotated like the other repeated phrases in this prompt
# so a caller who trips it twice doesn't hear the identical sentence back.
# Deliberately worded to hand the turn straight back with a question.
# REDIRECT_PHRASES: per language in agent/languages.py (LanguagePack.redirect_phrases).

# Promise-follow-up watchdog (2026-08-16 fix): a fallback-model turn that
# says "checking..."/"trenutek, prosim..." and then never calls the tool it
# just promised leaves the caller hanging indefinitely — observed for real
# during a sustained Anthropic outage where GPT-4o-mini spoke this exact
# pattern and then produced no further activity for 4+ minutes until the
# caller gave up. If no new agent/user activity follows such an utterance
# within this window, treat it as a technical failure (see
# _watch_for_promise_followup in entrypoint).
# Deliberately narrow to present-tense/promise phrasing, same as the
# Slovenian pattern (e.g. "rezerviram" = "I book", not "rezervirala" = past
# tense "booked") — must NOT match a past-tense confirmation sentence like
# "I've booked you in for a pedicure...", or the watchdog would arm itself
# right after a successful, complete turn and fire a false technical-
# failure close once the call goes quiet afterward.
# PROMISE_CUE_PATTERNS: per language in agent/languages.py (LanguagePack.promise_cues).
PROMISE_WATCHDOG_TIMEOUT_SEC = 12.0
# On the first silent promise the model is nudged to follow through (a reply
# is generated with PROMISE_RECOVERY_INSTRUCTIONS); only if that also leads
# nowhere within another window is the call closed as a technical failure.
# Before 2026-10-16 the first silent promise closed the call outright.
PROMISE_RECOVERY_INSTRUCTIONS = (
    "You told the caller you would check or do something, then stopped "
    "without doing it. Do it now by calling the right tool. If you can't, "
    "tell the caller briefly and honestly what you can do instead."
)

# Our own filler utterances also match PROMISE_CUE_PATTERNS by design, but
# they're paired with an actual in-flight tool call that already has its own
# real timeout (up to 45s for create_booking) — excluded so the watchdog's
# much shorter window doesn't kill a call that's legitimately still running.
# Tool fillers are now built per call (they name the day), so those are
# checked against BookingTools.spoken_fillers at runtime instead.
_KNOWN_FILLER_TEXTS = frozenset(
    p for code in SUPPORTED_LANGUAGES for p in pack(code).still_there
)

# A reply whose first words haven't come out of the LLM this long after the
# caller's turn ended starts with a short "Trenutek." (2026-10-26: Tim found
# the silences on a live call too long; LLM ttft p90 ~1.4s, max ~3s). Only
# the slow replies get it: a typical one has its first sentence by ~1.2s.
THINKING_FILLER_AFTER_SEC = 1.8
_THINKING_FILLERS = tuple(
    sorted({p for code in SUPPORTED_LANGUAGES for p in pack(code).thinking_fillers}, key=len, reverse=True)
)


def _without_thinking_filler(text: str) -> str:
    """The reply minus a leading thinking filler, for the promise watchdog:
    "Trenutek. Masaža glave traja ..." is an answer, not a promise."""
    for filler in _THINKING_FILLERS:
        if text.startswith(filler):
            return text[len(filler):].lstrip()
    return text


async def _with_thinking_filler(stream, make_filler, after_sec: float):
    """Pass `stream` through; if it yields nothing for `after_sec`, first
    yield the text make_filler() returns, so the reply starts with it."""
    it = stream.__aiter__()
    first = asyncio.ensure_future(it.__anext__())
    try:
        done, _ = await asyncio.wait({first}, timeout=after_sec)
        if not done:
            filler = make_filler()
            if filler:
                yield llm.ChatChunk(
                    id="thinking-filler", delta=llm.ChoiceDelta(role="assistant", content=filler + " ")
                )
        try:
            item = await first
        except StopAsyncIteration:
            return
        yield item
        async for item in it:
            yield item
    finally:
        if not first.done():
            first.cancel()
        aclose = getattr(it, "aclose", None)
        if aclose is not None:
            await aclose()

# Post-farewell goodbye loop (2026-09-21). Observed: after "Nasvidenje!" the
# caller said "ja, hvala", the agent answered with another farewell, the
# caller acknowledged that, and so on for four rounds — each "hvala" is a
# user turn, and an LLM turn always produces speech. The prompt now says to
# give at most one short closing after a farewell; this is the code backstop
# (see ReceptionistAgent.on_user_turn_completed): once the agent has said
# goodbye twice with nothing but acknowledgements in between, further pure
# acknowledgements get no reply at all. Deliberately narrow in both
# directions — a turn counts as an acknowledgement only if EVERY word is in
# this vocabulary, so any real question or request after a farewell still
# reaches the model.
# FAREWELL_PATTERNS: per language in agent/languages.py (LanguagePack.farewells).
# ACKNOWLEDGEMENT_WORDS: per language in agent/languages.py (LanguagePack.acknowledgements).
# Farewell turns (with only acknowledgements in between) after which a pure
# acknowledgement is no longer answered: 1 = the farewell itself, 2 = the one
# short "Prosim, lep dan!" the prompt allows in reply to the first "hvala".
FAREWELLS_BEFORE_SILENCE = 2


# First sentence of a reply that talks ABOUT the call instead of TO the
# caller (see ReceptionistAgent.llm_node). Mostly English: the model writes its
# reasoning in the prompt's language, and a real reply to an English caller
# never says "per the rules" or "the caller". The one Slovenian pattern is
# "Ta klic je v hrvaščini" (eval 2026-10-17), never said to a real caller.
_META_COMMENTARY = re.compile(
    r"\bthe caller\b|\bper (?:the |my )?rules?\b|\bswitch(?:ing)? (?:the )?(?:call|language)"
    r"|\bswitch_language\b|\btool[- ]?switch\b|\blanguage first\b|\b(?:language|\w+ian|English|German) detected\b|\bso I switch\b"
    r"|\bswitch(?:ed|ing)? (?:back )?to (?:Slovenian|English|German|Croatian|Italian)\b"
    r"|\bta klic (?:je|vodim|bom vodila) v \w+(?:ščini|sko)\b"
    # Reasoning about the prompt's data (eval 2026-10-18: "\"Naslednji torek\"
    # is the row tagged [NASLEDNJI TEDEN] ... Pedikura is service 18, caller
    # said any person.").
    r"|\brow tagged\b|\bcaller (?:said|says|speaks?|spoke|wants|asked|asks|gave|gives|is|named)\b|\bany[_ ]person\b"
    r"|\b(?:today|tomorrow|yesterday) is\b|\b\d{4}-\d{2}-\d{2}\b"
    r"|\bservice (?:id )?\d+\b|\bemployee(?:_id| id)\b",
    re.IGNORECASE,
)
_SENTENCE_END = re.compile(r"[.!?…](?:\s|$)")
# "Switch first." / "Switching first, ..." in front of a reply or a
# switch_language call (eval 2026-10-17, 2 in 10 Croatian switches): too
# short to be a sentence the filter can judge, so it is cut off the front.
_META_PREFIX = re.compile(
    r"^\s*switch(?:ing|ed)?(?: (?:first|now|over|languages?))?\s*[,.:;!\u2014\u2013-]+\s*",
    re.IGNORECASE,
)


def _strip_meta_prefix(text: str) -> str:
    return _META_PREFIX.sub("", text, count=1)


def _is_meta_commentary(text: str) -> bool:
    return bool(_META_COMMENTARY.search(text))


def _is_pure_acknowledgement(text: str, language: str) -> bool:
    words = re.findall(r"[^\W\d_]+", text.lower())
    return bool(words) and all(
        w in pack(language).acknowledgements for w in words
    )


# How long to wait for update_agent() to finish handing the session from the
# bootstrap placeholder to the real ReceptionistAgent. The handoff first waits
# for queued speech (the disclosure) to finish playing, so this has to cover
# the disclosure's remaining playout, not just the swap itself.
AGENT_SWAP_TIMEOUT_SEC = 15.0

TIMEZONE = ZoneInfo("Europe/Ljubljana")

_RELATIVE_DAY_LABELS = {0: "DANES", 1: "JUTRI", 2: "POJUTRIŠNJEM"}
_RELATIVE_DAY_LABELS_EN = {0: "TODAY", 1: "TOMORROW", 2: "DAY AFTER TOMORROW"}

def _build_date_context(language: str = DEFAULT_LANGUAGE, days: int = 14) -> str:
    """Build the date-lookup table given to the LLM each turn.

    Bug fixed 2026-08-14: the table used to only give each row's absolute
    date/weekday, leaving the model to work out on its own which row is
    "jutri" (offset 0 vs 1) — exactly the kind of manual calculation the
    instructions claimed to forbid. Observed failure: it labeled TODAY's row
    as "jutri" and offered it as a bookable day, and since tomorrow was
    actually a closed Saturday, this made the mislabeled date look like a
    plausible near-term suggestion instead. Fix: stamp DANES/JUTRI/
    POJUTRIŠNJEM directly onto the first three rows so there's no offset
    arithmetic left for the model to get wrong. Same fix applies to the
    English TODAY/TOMORROW/DAY AFTER TOMORROW tags below.

    Second bug fixed 2026-08-29: same failure mode, one level up. A COMPOUND
    reference ("naslednji teden v četrtek" / "next week on Thursday") has no
    single-word tag to key off, so the model fell back to computing the date
    itself — and got it wrong (picked 2026-09-07, a Monday, while calling it
    "četrtek"/Thursday). Fix, following the same pattern as DANES/JUTRI: tag
    every row that falls in the next calendar week (Mon-Sun) with
    NASLEDNJI TEDEN / NEXT WEEK, so a compound query becomes "find the row
    with both tags" instead of "compute an offset."
    """
    now = datetime.now(TIMEZONE)
    time_str = now.strftime("%H:%M")

    # ISO week: Monday=0..Sunday=6. Next week starts this many days out —
    # always lands within a 14-row table (max is 7, when today is Monday).
    next_monday_offset = 7 - now.weekday()
    next_week_offsets = range(next_monday_offset, next_monday_offset + 7)

    if language != "sl":
        # English instructions for every non-Slovenian call; weekday names
        # and spoken phrases in the call's language, so the table and the
        # tools' day labels (same dates.py functions) always agree.
        weekday_names = WEEKDAYS.get(language, ENGLISH_WEEKDAYS)
        relative_labels = _RELATIVE_DAY_LABELS_EN
        next_week_tag = "NEXT WEEK"
    else:
        weekday_names = SLOVENIAN_WEEKDAYS
        relative_labels = _RELATIVE_DAY_LABELS
        next_week_tag = "NASLEDNJI TEDEN"

    rows = []
    # One row per bookable day: the company's booking window
    # (max_dnevi_rezervacija, see business_info.booking_window_days), at
    # least 14 so "naslednji teden" always resolves.
    for offset in range(max(14, days)):
        d = now + timedelta(days=offset)
        iso_date = d.strftime("%Y-%m-%d")
        weekday = weekday_names[d.weekday()]
        phrase = spoken_date_phrase(d, language)
        tags = []
        single_day_label = relative_labels.get(offset)
        if single_day_label:
            tags.append(single_day_label)
        if offset in next_week_offsets:
            tags.append(next_week_tag)
        label_str = f" [{'] ['.join(tags)}]" if tags else ""
        rows.append(f"- {iso_date} ({weekday}){label_str}: {phrase}")

    table = "\n".join(rows)

    if language == "sl":
        tags = ("[DANES]", "[JUTRI]", "[POJUTRIŠNJEM]", "[NASLEDNJI TEDEN]")
        ex = {
            "relative": "jutri, v ponedeljek, naslednji teden v četrtek",
            "compound": "\"naslednji teden v četrtek\"",
            "compound_day": "četrtek",
            "aloud": "\"Naslednja sreda je devetnajstega oktobra? Ne, ...\"",
            "phrase": "\"tretjega avgusta\"",
            "connector": "\"V četrtek, torej tretjega septembra.\"",
            "bad": "caller \"torek\" → \"torek, osemindvajsetega septembra\" (the 28th was a Monday)",
            "decline": " You may add \"v\"/\"za\" and decline the weekday (v sredo), never change it.",
            "later": "after \"naslednji teden\", a later \"torek\"",
            "ask": "\"Mislite ta torek ali torek naslednji teden?\"",
            # Kept in Slovenian on purpose: written in English (2026-10-17
            # trim) the model echoed it aloud as English narration.
            "window": (
                # In words too: "14 dni" was copied into speech as digits.
                f"Rezervacije so odprte samo {days} ({slovenian_number(days)}) dni vnaprej (prvih {days} "
                "vrstic). Za kasnejši datum to preprosto povej in ponudi datum "
                "znotraj tega ali da prošnjo posreduješ lastniku "
                "(take_message) — nikoli ne ugibaj."
            ),
        }
    else:
        tags = ("[TODAY]", "[TOMORROW]", "[DAY AFTER TOMORROW]", "[NEXT WEEK]")
        ex = {
            "relative": "tomorrow, on Monday, next week on Thursday",
            "compound": "\"next week on Thursday\"",
            "compound_day": "Thursday (in the table's language)",
            "aloud": "\"Next Wednesday is the 19th? No, ...\"",
            "phrase": "the table's phrase",
            "connector": "\"On Thursday, so September third.\"",
            "bad": "caller \"Tuesday\" → \"Tuesday, September twenty-eighth\" (the 28th was a Monday)",
            "decline": "",
            "later": "after \"next week\", a later \"Tuesday\"",
            "ask": "\"Do you mean this Tuesday or next Tuesday?\"",
            "window": (
                f"Bookings are open {days} days ahead (the first {days} rows). "
                "For a later date, say so and offer a date within that or "
                "take_message — never guess."
            ),
        }

    # Condensed 2026-10-17 from three paragraphs with two examples each; every
    # rule is still here, each from a real incident (2026-08-14 "jutri" on
    # today's row, 2026-08-29 compound "next week Thursday", 2026-09-21
    # caller's weekday glued to another date).
    return (
        f"Current time is {time_str}, timezone Europe/Ljubljana. Table of the "
        f"next {max(14, days)} days — ISO date, weekday, spoken date phrase. "
        f"Rows are tagged {tags[0]}, {tags[1]}, {tags[2]}, and {tags[3]} for "
        f"every day of next calendar week.\n\n"
        f"{table}\n\n"
        "Date rules:\n"
        f"- Resolve every relative date ({ex['relative']}) by finding its row "
        "— never count days or compute a date or weekday yourself. "
        f"{ex['compound']} = the row tagged {tags[3]} whose weekday is "
        f"{ex['compound_day']}. Look it up silently — never work a date out "
        f"aloud (Bad: {ex['aloud']}).\n"
        f"- Say a date with its row's phrase verbatim ({ex['phrase']}); join "
        f"weekday and date naturally: {ex['connector']}\n"
        "- A weekday and its date always come from ONE row or ONE tool label "
        "(get_slots' \"day_labels\", check_slots/create_booking's "
        "\"day_label\") — never a weekday the caller said next to a date "
        f"from somewhere else. Bad (observed): {ex['bad']}.{ex['decline']}\n"
        f"- A week qualifier said earlier still applies: {ex['later']} means "
        f"the row tagged with both. Only if two rows still fit, ask: {ex['ask']}\n"
        f"- {ex['window']}"
    )


def _build_system_prompt(
    company_data: dict,
    language: str = DEFAULT_LANGUAGE,
    *,
    profile: dict | None = None,
    instructions: str | None = None,
    caller_phone: str | None = None,
    customer: dict | None = None,
    sms_confirmation: bool = False,
    switchable_to: list[str] | tuple = (),
    features=ALL_FEATURES,
    customer_lookup: bool = False,
    booking_extras: BookingExtras | None = None,
    assistant_name: str | None = None,
) -> str:
    static_prompt = build_static_prompt(language, switchable_to, features)
    logger.info(
        "prompt built: language=%s switchable_to=%s",
        language,
        ",".join(switchable_to) or "-",
    )
    business_block, hours_known = render_business_block(profile, instructions, language)
    window = booking_window_days(company_data.get("company"))
    parts = [
        static_prompt,
        _build_date_context(language, window),
        render_company_prompt(company_data, language, hours_known=hours_known),
    ]
    if business_block:
        parts.append(business_block)
    extras_block = render_extras_block(booking_extras, company_data)
    if extras_block:
        parts.append(extras_block)
    parts.append(render_assistant_identity(assistant_name))
    parts.append(render_caller_block(caller_phone, customer, language, can_lookup=customer_lookup))
    parts.append(render_confirmation_line(sms_confirmation, language))
    return "\n\n".join(parts)


class BootstrapAgent(Agent):
    """Placeholder that holds the session while company data is still loading.

    Disclosure-first bootstrap (2026-09-22): the session starts, and the AI
    disclosure starts playing, BEFORE the booking-v2 init webhook returns —
    that webhook (~1.4s measured) used to be dead air at the start of every
    call. The real ReceptionistAgent can't exist yet (its prompt and tools
    are built from the init response), so this stands in until entrypoint()
    swaps it out with session.update_agent().

    It never speaks on its own: anything the caller says in this window is
    recorded to the transcript and gets no reply (StopResponse). Before this
    change, speech in the same window was lost entirely because no session
    was listening yet.
    """

    def __init__(self, *, on_user_turn=None) -> None:
        super().__init__(
            instructions=(
                "You are a phone receptionist whose call is still connecting. "
                "Say nothing."
            )
        )
        self._on_user_turn = on_user_turn

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: ChatMessage
    ) -> None:
        text = new_message.text_content or ""
        logger.info(
            "bootstrap: caller spoke before the agent was ready (%d words, no reply)",
            len(text.split()),
        )
        if self._on_user_turn is not None and text:
            self._on_user_turn(text)
        raise StopResponse


class ReceptionistAgent(Agent):
    """Agent with a canned, language-correct redirect for rambling callers.

    Overrides the SDK default `on_user_turn_exceeded`, which calls
    generate_reply() with English instructions. That would produce Slovenian
    output (the system prompt forces it) but costs an LLM round-trip at the
    worst possible moment and could say anything — including an availability
    claim with no get_slots behind it, which the prompt spends considerable
    effort forbidding. A fixed line avoids both.

    Speaks with allow_interruptions=False, matching the SDK default handler:
    the whole point is that the caller is currently talking over everything,
    so an interruptible cut-in would be talked over too.

    The framework only fires this after checking that a normal end-of-turn
    reply isn't already on its way (see AgentActivity._user_turn_exceeded_task
    — it waits on agent_state "speaking" and bails if the agent got there on
    its own), so this does not double-speak over an ordinary answer.
    """

    def __init__(
        self,
        *,
        language: str,
        on_suppressed_user_turn=None,
        on_turn_language=None,
        private_emails: set[str] | None = None,
        on_thinking_filler=None,
        thinking_filler: bool = True,
        phone_first: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._language = language
        # () -> None, counted in the call metrics when a slow reply got "Trenutek."
        self._on_thinking_filler = on_thinking_filler
        self._thinking_filler = thinking_filler
        # Withheld number with find_customer: the phone number before the name.
        self._phone_first = phone_first
        self._thinking_filler_step = 0
        # Stored customer emails (caller ID match, find_customer, the
        # existing-customer guard). The prompt says never to read them out;
        # llm_node also replaces them, since the model still did now and then
        # (eval sl-znana-stranka-rezervacija, ~1 in 5).
        self._private_emails = private_emails if private_emails is not None else set()
        # async callback(text) run before each reply, while the caller's turn
        # is complete but the LLM hasn't started — the one point where a
        # language switch takes effect for THIS reply (worker.entrypoint).
        self._on_turn_language = on_turn_language
        self._redirect_step = 0
        # Farewells spoken since the caller last said anything other than a
        # pure acknowledgement — see FAREWELL_PATTERNS.
        self._farewell_count = 0
        # A turn swallowed via StopResponse never reaches the chat context or
        # the conversation_item_added event, so the entrypoint passes this in
        # to keep the stored transcript complete.
        self._on_suppressed_user_turn = on_suppressed_user_turn
        # Set once this agent's activity is running, i.e. update_agent() has
        # finished handing the session over from BootstrapAgent. The greeting
        # waits on it: a say() issued mid-handoff can land on either agent
        # depending on timing, and the greeting must belong to this one.
        self.entered = asyncio.Event()

    async def on_enter(self) -> None:
        self.entered.set()

    def set_language(self, language: str) -> None:
        self._language = language

    def _next_thinking_filler(self) -> str:
        fillers = pack(self._language).thinking_fillers
        text = fillers[self._thinking_filler_step % len(fillers)]
        self._thinking_filler_step += 1
        logger.info("slow reply: thinking filler %r", text)
        if self._on_thinking_filler is not None:
            self._on_thinking_filler()
        return text

    async def llm_node(self, chat_ctx, tools, model_settings):
        """_masked_llm_node, opened with "Trenutek." when it is slow to start."""
        if not self._thinking_filler:
            async for chunk in self._masked_llm_node(chat_ctx, tools, model_settings):
                yield chunk
            return
        async for chunk in _with_thinking_filler(
            self._masked_llm_node(chat_ctx, tools, model_settings),
            self._next_thinking_filler,
            THINKING_FILLER_AFTER_SEC,
        ):
            yield chunk

    async def _masked_llm_node(self, chat_ctx, tools, model_settings):
        """_llm_node_filtered, then SpokenTextFilter over the text."""
        stream = self._llm_node_filtered(chat_ctx, tools, model_settings)
        heard = None
        for item in reversed(chat_ctx.items):
            if getattr(item, "role", None) == "user":
                heard = caller_digits(getattr(item, "text_content", "") or "")
                break
        # Only until the caller says a number, and only once: a caller who
        # won't give one isn't asked the same thing forever.
        said = [getattr(i, "text_content", "") or "" for i in chat_ctx.items]
        roles = [getattr(i, "role", None) for i in chat_ctx.items]
        phone_first = (
            self._phone_first
            and not any(caller_digits(t) for t, r in zip(said, roles) if r == "user")
            and not any(PHONE_FIRST_QUESTIONS.get(self._language, "\0") in t for t, r in zip(said, roles) if r == "assistant")
        )
        mask = SpokenTextFilter(
            self._private_emails, self._language, heard_digits=heard, phone_first=phone_first
        )
        chunk_id = ""
        async for chunk in stream:
            if isinstance(chunk, str):
                out = mask.feed(chunk)
                if out:
                    yield out
                continue
            if isinstance(chunk, llm.ChatChunk):
                chunk_id = chunk.id or chunk_id
                delta = chunk.delta
                if delta is not None and delta.content and not delta.tool_calls:
                    out = mask.feed(delta.content)
                    if out:
                        yield llm.ChatChunk(id=chunk_id, delta=llm.ChoiceDelta(role="assistant", content=out))
                    continue
            rest = mask.flush()
            if rest:
                yield llm.ChatChunk(id=chunk_id, delta=llm.ChoiceDelta(role="assistant", content=rest))
            yield chunk
        rest = mask.flush()
        if rest:
            yield llm.ChatChunk(id=chunk_id, delta=llm.ChoiceDelta(role="assistant", content=rest))

    async def _llm_node_filtered(self, chat_ctx, tools, model_settings):
        """The default LLM node, minus a first sentence of meta-commentary.

        With thinking off, Claude sometimes writes its reasoning as text
        before a tool call ("The caller speaks Croatian, so I switch
        first.") — and every word of text is spoken (eval, 2026-10-16: 1 in
        3 Croatian switches). The first sentence of each reply is held until
        it ends (TTS waits for a sentence anyway) and dropped if it is that
        kind of narration; everything else passes through untouched.
        """
        buffer = ""
        chunk_id = ""
        filtering = True
        dropped = 0
        async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
            if not filtering:
                yield chunk
                continue
            content = None
            if isinstance(chunk, str):
                content = chunk
            elif isinstance(chunk, llm.ChatChunk):
                chunk_id = chunk.id or chunk_id
                if chunk.delta is not None and chunk.delta.content and not chunk.delta.tool_calls:
                    content = chunk.delta.content
            if content is None:
                # A tool call (or usage): whatever text is held is narration
                # of that call if it looks like it, a real line otherwise.
                buffer = _strip_meta_prefix(buffer)
                if buffer and not _is_meta_commentary(buffer):
                    yield llm.ChatChunk(
                        id=chunk_id, delta=llm.ChoiceDelta(role="assistant", content=buffer)
                    )
                elif buffer:
                    logger.warning("dropped meta-commentary before reply: %r", buffer[:200])
                buffer = ""
                yield chunk
                continue
            buffer += content
            m = _SENTENCE_END.search(buffer)
            if not m:
                continue
            sentence, buffer = buffer[: m.end()], buffer[m.end():]
            sentence = _strip_meta_prefix(sentence)
            if not sentence.strip():
                continue
            if _is_meta_commentary(sentence) and dropped < 3:
                dropped += 1
                logger.warning("dropped meta-commentary before reply: %r", sentence[:200])
                continue
            filtering = False
            yield llm.ChatChunk(
                id=chunk_id,
                delta=llm.ChoiceDelta(role="assistant", content=sentence + buffer),
            )
            buffer = ""
        buffer = _strip_meta_prefix(buffer) if filtering else buffer
        if filtering and buffer and not _is_meta_commentary(buffer):
            yield llm.ChatChunk(id=chunk_id, delta=llm.ChoiceDelta(role="assistant", content=buffer))

    @property
    def has_said_goodbye(self) -> bool:
        return self._farewell_count > 0

    def note_assistant_message(self, text: str) -> None:
        if pack(self._language).farewells.search(text):
            self._farewell_count += 1

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: ChatMessage
    ) -> None:
        text = new_message.text_content or ""
        if self._on_turn_language is not None:
            await self._on_turn_language(text)
        if not _is_pure_acknowledgement(text, self._language):
            self._farewell_count = 0
            return
        if self._farewell_count >= FAREWELLS_BEFORE_SILENCE:
            logger.info(
                "post-farewell acknowledgement %r after %d farewells — not "
                "replying",
                text,
                self._farewell_count,
            )
            if self._on_suppressed_user_turn is not None:
                self._on_suppressed_user_turn(text)
            raise StopResponse

    async def speak_redirect(self, *, source: str, words: int, duration: float) -> None:
        """Speak the next rotating redirect line. Shared by both trigger paths.

        `source` says which watchdog fired — the entrypoint's interim-based
        one (which is what actually works with our STT config) or the SDK's
        user_turn_limit backstop. Rotation state is shared deliberately, so a
        caller who trips both never hears the same sentence twice.
        """
        phrases = pack(self._language).redirect_phrases
        phrase = phrases[self._redirect_step % len(phrases)]
        self._redirect_step += 1
        logger.info(
            "REDIRECT source=%s words=%s duration=%.1fs — speaking %r",
            source,
            words,
            duration,
            phrase,
        )
        # Awaited, not fire-and-forget: on the SDK path, AgentActivity holds
        # _user_turn_exceeded_locked only for the duration of that callback,
        # so returning before the line finishes would let a still-rambling
        # caller trigger a second redirect on top of the first one.
        await self.session.say(phrase, allow_interruptions=False)

    async def on_user_turn_exceeded(self, ev: UserTurnExceededEvent) -> None:
        # Backstop only. Verified 2026-09-08 not to fire in our configuration:
        # the SDK advances its counters solely on FINAL transcripts
        # (audio_recognition.py, inside the FINAL_TRANSCRIPT branch), and the
        # Soniox plugin only emits a FINAL when it detects an ENDPOINT
        # ("final tokens are accumulated across messages until an endpoint is
        # detected") — which a continuous monologue never produces. When the
        # caller finally pauses, the one final that arrives both trips the
        # limit AND ends the turn, and AgentActivity._user_turn_exceeded_task
        # deliberately bails once the agent starts its normal reply. Kept
        # enabled because it costs nothing and would start working if the STT
        # or turn-detection config ever changes; the real trigger is the
        # interim-based watchdog in entrypoint().
        await self.speak_redirect(
            source="sdk-user-turn-limit",
            words=ev.accumulated_word_count,
            duration=ev.duration,
        )


def build_find_customer_tool(
    lookup, caller_phone: str | None, private_emails: set | None = None, on_found=None
):
    """find_customer: an existing customer by a phone number the caller SAID
    (withheld caller ID, or a different number than the one they call from).
    With caller ID the lookup already happened at call start; a spoken
    number needs the same, or booking-v2 matches only on phone + exact name
    and a misheard surname becomes a duplicate customer.

    lookup: async (phone_e164) -> "Stranke" row (Ime, Priimek, "Email
    stranke") or None. Shared with the eval harness."""

    @function_tool
    async def find_customer(context: RunContext, phone: str) -> dict:
        """Look up an existing customer by the phone number the caller said,
        right after they confirmed your read-back of it — before asking for
        their name.

        Args:
            phone: The number as the caller gave it, e.g. "041 555 123".
        """
        e164 = normalize_phone(phone)
        if not e164:
            return {"found": False, "message": "Not a phone number; ask for it again."}
        if caller_phone and e164 == caller_phone:
            return {"found": False, "message": "That's the number they're calling from — follow the caller line."}
        try:
            row = await lookup(e164)
        except Exception:
            logger.exception("find_customer lookup failed")
            row = None
        first = ((row or {}).get("Ime") or "").strip()
        last = ((row or {}).get("Priimek") or "").strip()
        if not first and not last:
            return {
                "found": False,
                "message": "No customer with this number. Collect the details as usual.",
            }
        email = ((row or {}).get("Email stranke") or "").strip()
        if email and private_emails is not None:
            private_emails.add(email)
        if on_found is not None:
            on_found(e164, row)
        logger.info("find_customer: existing customer for spoken number")
        return {
            "found": True,
            "first_name": first,
            "last_name": last,
            "email": email,
            "message": (
                "This number belongs to an existing customer. Without saying you "
                "looked anything up, ask once whether "
                "you're speaking with them, by first name (in Slovenian in the "
                "instrumental case: \"Ali govorim z Ano?\"). If the caller already "
                "gave you a name, ask with the full stored name instead (\"Ali ste "
                "Ana Novak?\") — what you heard may be misheard. If yes: pass exactly "
                "these first_name, last_name"
                + (" and email" if email else "")
                + " and this phone to create_booking (or, for an existing appointment, "
                "follow the cancel/move rule), don't ask for them again"
                + (", and never read the email aloud" if email else "")
                + ". If no: collect the details as usual."
            ),
        }

    return find_customer


def build_receptionist_agent(
    company_data: dict,
    language: str,
    *,
    booking_tools: BookingTools,
    notes: CallNotes,
    extra_tools: list | tuple = (),
    on_suppressed_user_turn=None,
    profile: dict | None = None,
    instructions: str | None = None,
    caller_phone: str | None = None,
    customer: dict | None = None,
    sms_confirmation: bool = False,
    switchable_to: list[str] | tuple = (),
    on_turn_language=None,
    features=ALL_FEATURES,
    customer_lookup: bool = False,
    booking_extras: BookingExtras | None = None,
    assistant_name: str | None = None,
    on_thinking_filler=None,
    thinking_filler: bool = True,
) -> "ReceptionistAgent":
    """The receptionist for one call. Shared by entrypoint() and the eval
    harness (evals/run_evals.py), so both run the same prompt and tools.
    extra_tools carries the ones bound to the live session (end_call_abusive).
    """
    stored_email = ((customer or {}).get("Email stranke") or "").strip()
    if stored_email:
        booking_tools.private_emails.add(stored_email)
    return ReceptionistAgent(
        language=language,
        private_emails=booking_tools.private_emails,
        on_thinking_filler=on_thinking_filler,
        thinking_filler=thinking_filler,
        phone_first=not caller_phone and customer_lookup,
        on_suppressed_user_turn=on_suppressed_user_turn,
        on_turn_language=on_turn_language,
        instructions=_build_system_prompt(
            company_data,
            language,
            profile=profile,
            instructions=instructions,
            caller_phone=caller_phone,
            customer=customer,
            sms_confirmation=sms_confirmation,
            switchable_to=switchable_to,
            features=features,
            customer_lookup=customer_lookup,
            booking_extras=booking_extras,
            assistant_name=assistant_name,
        ),
        tools=[
            booking_tools.get_slots,
            booking_tools.check_slots,
            booking_tools.create_booking,
            notes.take_message,
            *extra_tools,
        ],
    )


class LanguageControl:
    """Switching the call's language mid-call (sklop 5). Shared by
    entrypoint() and the eval harness, so both switch the same way.

    A switch replaces the agent's instructions with the prompt built for the
    new language, sets the TTS language (Soniox speaks all five with the same
    voice), moves the tools' fillers/day labels over, and updates call_lang,
    which every line spoken by code reads. The conversation so far is kept.
    """

    def __init__(
        self,
        *,
        call_lang: dict,
        allowed: list[str],
        company_data: dict,
        prompt_args: dict,
        booking_tools: BookingTools,
        session=None,
        turn_language: dict | None = None,
        enabled: bool = True,
    ) -> None:
        self.call_lang = call_lang
        self.allowed = allowed
        self.company_data = company_data
        self.prompt_args = prompt_args
        self.booking_tools = booking_tools
        self.session = session
        self.turn_language = turn_language if turn_language is not None else {
            "detected": None,
            "previous": None,
        }
        self.enabled = enabled
        self.agent: "ReceptionistAgent | None" = None
        # One switch per caller turn: the model once called switch_language
        # three times in a row (hr, back to sl, hr again) and narrated it.
        self._switched_this_turn = False

    async def switch(self, new_code: str, why: str) -> bool:
        new = normalize_language(new_code)
        if self.agent is None or not new or new == self.call_lang["code"] or new not in self.allowed:
            return False
        old = self.call_lang["code"]
        self.call_lang["code"] = new
        self.call_lang.setdefault("history", []).append(new)
        self.agent.set_language(new)
        self.booking_tools.set_language(new)
        await self.agent.update_instructions(
            _build_system_prompt(
                self.company_data,
                new,
                switchable_to=[c for c in self.allowed if c != new],
                **self.prompt_args,
            )
        )
        if self.session is not None:
            try:
                self.session.tts.update_options(language=new)
            except Exception:
                logger.exception("TTS language update failed (%s -> %s)", old, new)
        logger.info("language switch %s -> %s (%s)", old, new, why)
        return True

    @function_tool
    async def switch_language(self, context: RunContext, language: str) -> dict:
        """Switch the call to another language.

        Call this BEFORE replying when the caller speaks another supported
        language, or asks to speak one ("can we speak English?", "parlate
        italiano?"). Then answer in that language.

        Args:
            language: ISO code — "sl" Slovenian, "en" English, "de" German,
                "hr" Croatian, "it" Italian.
        """
        target = normalize_language(language)
        if target is None or target not in self.allowed:
            return {
                "success": False,
                "message": (
                    "That language isn't available for this business. "
                    "Continue in the current language and say so politely."
                ),
            }
        if target == self.call_lang["code"]:
            return {"success": True, "message": "Already speaking it; just continue."}
        if self._switched_this_turn:
            current = LANGUAGE_NAMES[self.call_lang["code"]]
            return {
                "success": False,
                "message": (
                    f"Already switched to {current} this turn. Don't switch again "
                    f"— reply to the caller in {current} now."
                ),
            }
        self._switched_this_turn = True
        await self.switch(target, "tool")
        return {
            "success": True,
            "message": f"Switched. From now on reply only in {LANGUAGE_NAMES[target]}.",
        }

    async def on_turn(self, text: str) -> None:
        """Before each reply: switch if Soniox heard this turn in another
        allowed language (languages.detect_switch decides how sure is sure
        enough)."""
        self._switched_this_turn = False
        if not self.enabled:
            return
        detected = self.turn_language["detected"]
        target = detect_switch(
            current=self.call_lang["code"],
            detected=detected,
            words=len(text.split()),
            previous_detected=self.turn_language["previous"],
            allowed=self.allowed,
        )
        self.turn_language["previous"] = detected
        if target:
            await self.switch(target, f"detected {detected}")
            self._switched_this_turn = True


def _build_tts(settings: Settings, language: str = DEFAULT_LANGUAGE):
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
        model=settings.tts_model,
        language=language,
        # 1.0 (default) read as noticeably slow to callers; 1.1 read as too
        # fast in testing (2026-08-30). 1.04 is a subtler nudge (plugin
        # range [0.7, 1.3]) chosen to not reopen the pronunciation-quality
        # tuning already settled on voice/model/language.
        speed=1.04,
    )


def _build_stt(
    settings: Settings, language: str = DEFAULT_LANGUAGE, allowed: list[str] | None = None
) -> soniox.STT:
    # Hints, not a restriction: the starting language first, then any other
    # language the call may switch to. Soniox identifies the language of
    # every token (enable_language_identification, on by default), which is
    # what the switch detection reads.
    hints = [language] + [c for c in (allowed or []) if c != language]
    return soniox.STT(
        api_key=settings.soniox_api_key,
        params=soniox.STTOptions(language_hints=hints),
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
    # Model, effort and fallback come from LLM_MODEL / LLM_EFFORT /
    # LLM_FALLBACK_MODEL (agent/llm_factory.py). Default since 2026-10-16:
    # Claude Haiku 5.5 at effort "low" with thinking off, falling back to
    # gpt-5.4-mini — chosen on evals/run_evals.py (5.5: 21-23/24 per run vs
    # 18/24 for Haiku 4.5 with the same judge; gpt-4o-mini, the previous
    # fallback, 16/24 and once quoted a wrong price).
    config = LLMConfig.from_env()
    logger.info(
        "llm: primary=%s effort=%s fallback=%s",
        config.primary_model,
        config.effort,
        config.fallback_model,
    )
    return build_llm(
        config,
        anthropic_api_key=settings.anthropic_api_key,
        openai_api_key=settings.openai_api_key,
    )


async def _insert_call_logged(supabase: SupabaseClient, row: dict) -> None:
    try:
        await supabase.insert_call(row)
    except Exception:
        logger.exception("failed to insert receptionist_calls row for call_id=%s", row.get("id"))


def _env_flag(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() not in ("0", "off", "false", "no")


def _room_options() -> room_io.RoomOptions:
    """Noise cancellation on the caller's audio: LiveKit Cloud's BVC model
    tuned for telephony (8 kHz, codec artefacts). It runs before VAD and STT,
    so street noise, a TV or a hair dryer neither interrupts the agent nor
    ends up in the transcript. NOISE_CANCELLATION=off disables it — e.g. if
    the SIP trunk already applies its own (applying it twice degrades
    speech)."""
    if not _env_flag("NOISE_CANCELLATION"):
        return room_io.RoomOptions()
    return room_io.RoomOptions(
        audio_input=room_io.AudioInputOptions(
            noise_cancellation=noise_cancellation.BVCTelephony()
        )
    )


async def _start_background_audio(ctx: JobContext, session: AgentSession) -> None:
    """Typing while the agent thinks, optional quiet office ambience. Purely
    cosmetic: any failure is logged and the call goes on without it."""
    thinking = (
        [
            AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=THINKING_SOUND_VOLUME, probability=0.6),
            AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING2, volume=THINKING_SOUND_VOLUME, probability=0.4),
        ]
        if _env_flag("THINKING_SOUND")
        else None
    )
    ambient = (
        AudioConfig(BuiltinAudioClip.OFFICE_AMBIENCE, volume=AMBIENT_SOUND_VOLUME)
        if _env_flag("AMBIENT_SOUND")
        else None
    )
    if thinking is None and ambient is None:
        return
    try:
        player = BackgroundAudioPlayer(ambient_sound=ambient, thinking_sound=thinking)
        await player.start(room=ctx.room, agent_session=session)
    except Exception:
        logger.exception("background audio failed to start — continuing without it")


async def _wait_for_caller(ctx: JobContext) -> rtc.RemoteParticipant | None:
    """The SIP caller, or None if nobody shows up within PARTICIPANT_WAIT_SEC."""
    try:
        return await asyncio.wait_for(
            ctx.wait_for_participant(), timeout=PARTICIPANT_WAIT_SEC
        )
    except asyncio.TimeoutError:
        logger.warning("no caller joined within %.0fs", PARTICIPANT_WAIT_SEC)
        return None


async def _resolve_company_slug(
    ctx: JobContext,
    supabase: SupabaseClient,
    dialed_number: str | None,
    *,
    fallback: str | None,
) -> str | None:
    """Which company this call is for, in order of precedence:

    1. company_slug in the dispatch rule's job metadata (set per number in
       LiveKit, no database needed);
    2. receptionist_phone_numbers row for the dialled number;
    3. COMPANY_SLUG from the worker's env (the pre-routing behaviour, so
       the existing number keeps working before any row exists).

    A failed lookup falls through to the next source rather than dropping
    the call.
    """
    from_metadata = company_slug_from_job_metadata(ctx.job.metadata)
    if from_metadata:
        return from_metadata
    if dialed_number:
        try:
            from_table = await supabase.lookup_company_by_number(dialed_number)
        except Exception:
            logger.exception("company lookup failed for dialed=%s", dialed_number)
            from_table = None
        if from_table:
            return from_table
    return fallback


async def _speak_and_close(
    ctx: JobContext, settings: Settings, language: str, message: str
) -> None:
    """Speak one fixed line on a TTS-only session, then close it. Used for
    calls rejected before the real session exists (no credits, disabled,
    unknown number)."""
    gate_session = AgentSession(tts=_build_tts(settings, language))
    try:
        await gate_session.start(
            room=ctx.room,
            agent=Agent(
                instructions=(
                    "You only ever speak one fixed message and take no other "
                    "action; you have no tools."
                )
            ),
        )
        await gate_session.say(message)
    except Exception:
        logger.exception("failed to speak rejection message")
    finally:
        await gate_session.aclose()


async def entrypoint(ctx: JobContext) -> None:
    # Bootstrap timing (2026-09-22): every "t=+" in the logs below is measured
    # from here, so the pre-greeting gap can be read straight off one call's
    # log lines instead of reconstructed from unrelated timestamps.
    job_t0 = time.monotonic()
    job_wall_t0 = time.time()

    def _t() -> float:
        return time.monotonic() - job_t0

    settings = Settings.from_env()
    await ctx.connect()
    logger.info("bootstrap: room connected t=+%.3fs", _t())

    # Pre-job timing (2026-09-23): the t=+ lines above only start when the
    # job reaches us, but a caller's silence can also come from the time
    # BEFORE that — call answered → room created → job dispatched — which
    # nothing measured. An English test call reported a long pre-disclosure
    # silence while every step our code controls matched the Slovenian
    # calls, and there was no way to tell whether the gap was before or
    # after the job started. These two timestamps come from the LiveKit
    # server, on the same t= scale (negative = before the job started), so
    # the next such call answers that directly. Cross-clock (LiveKit server
    # vs this VPS, both NTP-synced), so trust tens of ms, not single ms.
    room_created = ctx.room.creation_time.timestamp()  # 0.0 if not populated
    logger.info(
        "bootstrap: room created t=%s (LiveKit clock)",
        f"{room_created - job_wall_t0:+.3f}s" if room_created > 0 else "unknown",
    )
    caller_join_logged = False

    # First remote participant of ANY kind, with the kind logged: the first
    # version filtered to SIP and logged nothing on a test call (2026-09-23,
    # AJ_iNF2G7ByKait), so a filter that silently drops the only caller in
    # the room is worse than none. The agent is the only local participant.
    def _log_caller_joined(participant: rtc.RemoteParticipant) -> None:
        nonlocal caller_join_logged
        if caller_join_logged:
            return
        caller_join_logged = True
        joined = participant.joined_at
        logger.info(
            "bootstrap: caller joined room t=%s (LiveKit clock), seen by us "
            "t=+%.3fs kind=%s identity=%s",
            f"{joined.timestamp() - job_wall_t0:+.3f}s" if joined else "unknown",
            _t(),
            rtc.ParticipantKind.Name(participant.kind),
            participant.identity,
        )

    for participant in ctx.room.remote_participants.values():
        _log_caller_joined(participant)
    if not caller_join_logged:
        ctx.room.on("participant_connected", _log_caller_joined)

    supabase = SupabaseClient(settings.supabase_url, settings.supabase_service_role_key)
    call_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc)
    livekit_room = ctx.room.name

    # DTMF: LOGGING ONLY for now (2026-09-08) — deliberately no functional
    # behavior attached yet.
    #
    # Inbound keypresses arrive as a room-level SIP signalling event, NOT
    # through the audio/STT path, so they were previously dropped silently:
    # nothing in this worker registered any room-level handler at all. They
    # are not "misheard as noise" by Soniox — out-of-band DTMF never reaches
    # it.
    #
    # The open question this is here to answer with real data: whether the
    # SIP trunk sends DTMF out-of-band (RFC 2833 / SIP INFO — the normal
    # case, which produces these events) or in-band as audio tones (in which
    # case this handler stays silent and the tones hit STT as garbage). That
    # is trunk configuration and can only be settled by a real call with a
    # keypress. Registered before the credit/booking gates so it captures
    # presses on every call, including ones that end early.
    dtmf_seen: list[str] = []

    def _on_sip_dtmf_received(event: rtc.SipDTMF) -> None:
        dtmf_seen.append(event.digit)
        logger.info(
            "DTMF received: digit=%r code=%s participant=%s total_this_call=%d "
            "sequence=%r (logging only — no action taken)",
            event.digit,
            event.code,
            event.participant.identity if event.participant else None,
            len(dtmf_seen),
            "".join(dtmf_seen),
        )

    ctx.room.on("sip_dtmf_received", _on_sip_dtmf_received)

    # Who is calling, and which of our numbers they dialled (2026-10-16).
    # The dialled number decides the company; before this every call went
    # to the one COMPANY_SLUG in the worker's env, so a company activating
    # Receptionist+ from the dashboard got nothing.
    caller = await _wait_for_caller(ctx)
    caller_attrs = dict(caller.attributes) if caller is not None else {}
    dialed_number = normalize_phone(caller_attrs.get(SIP_DIALED_NUMBER_ATTR))
    caller_phone = normalize_phone(caller_attrs.get(SIP_CALLER_NUMBER_ATTR))
    company_slug = await _resolve_company_slug(
        ctx, supabase, dialed_number, fallback=settings.company_slug
    )
    logger.info(
        "bootstrap: routing dialed=%s caller=%s -> company_slug=%s t=+%.3fs",
        dialed_number,
        "known" if caller_phone else "withheld",
        company_slug,
        _t(),
    )
    if company_slug is None:
        # No mapping and no fallback: there is no company whose data, credits
        # or language we could use. Say a neutral line and hang up; there is
        # no receptionist_calls row to write (company_slug is NOT NULL).
        logger.error("no company for dialed number %s — rejecting call", dialed_number)
        await _speak_and_close(ctx, settings, DEFAULT_LANGUAGE, pack(DEFAULT_LANGUAGE).no_credits)
        return

    # get_settings, get_balance, and the booking-v2 init fetch don't depend
    # on each other's results, so fire all three concurrently rather than
    # sequentially — measured 2026-08-30: sequential awaits added up to a
    # ~3s pre-greeting gap (dominated by the two network round-trips run
    # one after another), during which a caller speaking immediately (e.g.
    # "Živjo") got no response at all. create_task starts each one running
    # right away; each is still awaited individually below so existing
    # per-call error handling (graceful language fallback, the no-credits
    # gate, the booking-v2-failure gate) is unchanged.
    async def _timed(name: str, coro):
        started = time.monotonic()
        try:
            return await coro
        finally:
            logger.info(
                "bootstrap: %s took %.3fs (done at t=+%.3fs)",
                name,
                time.monotonic() - started,
                _t(),
            )

    settings_task = asyncio.create_task(
        _timed("get_settings", supabase.get_settings(company_slug))
    )
    balance_task = asyncio.create_task(
        _timed("get_balance", supabase.get_balance(company_slug))
    )
    init_task = asyncio.create_task(
        _timed("booking-v2 init", _init_company_with_retry(company_slug))
    )

    # Opening hours, address, owner's notes, and whether the caller is an
    # existing customer (by caller ID). Runs alongside init; needed only
    # when the real agent is built, after the disclosure. Best effort: any
    # failure leaves the agent without those extras, never drops the call.
    async def _load_caller_context() -> tuple[dict | None, dict | None]:
        try:
            profile = await supabase.get_company_profile(company_slug)
        except Exception:
            logger.exception("company profile lookup failed for %s", company_slug)
            return None, None
        customer = None
        if caller_phone and profile and profile.get("ID Podjetja"):
            try:
                customer = await supabase.find_customer_by_phone(
                    profile["ID Podjetja"], caller_phone
                )
            except Exception:
                logger.exception("customer lookup by phone failed")
        logger.info("caller context: known_customer=%s", customer is not None)
        return profile, customer

    caller_context_task = asyncio.create_task(
        _timed("caller context", _load_caller_context())
    )

    # Whether an SMS confirmation can actually go out (plan, subscription,
    # quota left) and the company's own confirmation template. Without SMS
    # the call confirms by email instead, so the caller isn't promised a
    # text that never comes.
    async def _load_sms_setup() -> tuple[int | None, dict | None]:
        quota_left, template_row = await asyncio.gather(
            supabase.sms_quota_left(company_slug),
            supabase.get_confirmation_template_row(company_slug),
            return_exceptions=True,
        )
        if isinstance(template_row, BaseException):
            logger.warning("confirmation template lookup failed: %s", template_row)
            template_row = None
        if isinstance(quota_left, BaseException):
            quota_left = None
        return quota_left, template_row

    sms_setup_task = asyncio.create_task(_timed("sms setup", _load_sms_setup()))

    settings_fetch_failed = False
    try:
        settings_row = await settings_task
    except Exception:
        logger.exception(
            "failed to fetch receptionist_settings for company_slug=%s, "
            "defaulting language to %s",
            company_slug,
            DEFAULT_LANGUAGE,
        )
        settings_row = None
        settings_fetch_failed = True
    language = normalize_language((settings_row or {}).get("language")) or DEFAULT_LANGUAGE
    if language not in SUPPORTED_LANGUAGES:
        logger.warning(
            "unsupported language %r for company_slug=%s, defaulting to %s",
            language,
            company_slug,
            DEFAULT_LANGUAGE,
        )
        language = DEFAULT_LANGUAGE

    # The company's own language: the owner reads summaries and messages in
    # it, and the dashboard greeting is written in it. `language` below may
    # become the caller's instead.
    company_language = language
    # Switching languages mid-call is the dashboard's "detect_caller_language"
    # (Receptionist+ settings). Off = the call stays in the company language,
    # exactly as before sklop 5.
    detect_language = bool((settings_row or {}).get("detect_caller_language"))
    allowed_languages = list(SUPPORTED_LANGUAGES) if detect_language else [language]
    # The CURRENT language of the call. Everything spoken by code reads it
    # from here, never from `language`, which is only the starting point.
    call_lang = {"code": language, "history": [language]}
    # Which greeting played and how it went, for receptionist_calls.metrics
    # (filled in at the greeting below; empty if the call never got there).
    greeting_state: dict = {}

    async def _say_technical_difficulty(target_session: AgentSession) -> None:
        """Play the pre-rendered apology clip for Slovenian (bypasses TTS
        synthesis entirely, so it still works if the TTS provider itself is
        the thing that's down); English has no equivalent asset yet (a
        deliberate scope decision), so it falls back to plain session.say()
        text, which won't survive a TTS-provider outage but is otherwise
        fine.
        """
        current = call_lang["code"]
        message = pack(current).technical_difficulty
        try:
            if current == "sl":
                audio = audio_frames_from_file(str(TECHNICAL_DIFFICULTY_AUDIO_PATH))
                await asyncio.wait_for(
                    target_session.say(message, audio=audio), timeout=8.0
                )
            else:
                await asyncio.wait_for(target_session.say(message), timeout=8.0)
        except Exception:
            logger.exception("failed to play technical-difficulty message")

    # A missing balance used to crash the entrypoint here (uncaught
    # SupabaseError → the caller heard nothing at all). A Supabase blip is
    # not the company's fault, so the call goes ahead; it is still billed at
    # the end if Supabase is back by then. With the balance unknown the call
    # gets the full MAX_CALL_SEC limit.
    try:
        balance: float | None = await balance_task
    except Exception:
        logger.exception(
            "failed to fetch balance for company_slug=%s — answering anyway",
            company_slug,
        )
        balance = None

    # enabled=false (switched off in the dashboard) was read but never
    # checked, so a disabled Receptionist+ kept answering. A missing row
    # means the company never activated it. Only a failed FETCH lets the
    # call through, for the same reason as the balance above.
    if settings_row is not None:
        enabled = bool(settings_row.get("enabled"))
    else:
        enabled = settings_fetch_failed

    reject_outcome = None
    if not enabled:
        reject_outcome = "disabled"
    elif balance is not None and balance <= 0:
        reject_outcome = "no_credits"

    if reject_outcome is not None:
        logger.warning(
            "rejecting call: company_slug=%s outcome=%s balance=%s",
            company_slug,
            reject_outcome,
            balance,
        )
        # Booking-v2 init was fired concurrently above but its result is
        # never needed on this path — cancel it rather than let it keep
        # running an unnecessary webhook call in the background.
        init_task.cancel()
        caller_context_task.cancel()
        await _speak_and_close(ctx, settings, language, pack(language).no_credits)
        try:
            await supabase.insert_call(
                {
                    "id": call_id,
                    "company_slug": company_slug,
                    "started_at": started_at.isoformat(),
                    "ended_at": datetime.now(timezone.utc).isoformat(),
                    "duration_sec": 0,
                    "billed_credits": 0,
                    "outcome": reject_outcome,
                    "transcript": [],
                    "livekit_room": livekit_room,
                    "caller_phone": caller_phone,
                    "dialed_number": dialed_number,
                }
            )
        except Exception:
            logger.exception("failed to log rejected call %s", call_id)
        return

    call_limit_sec = MAX_CALL_SEC if balance is None else call_time_limit_sec(balance)

    # A known customer whose stored language ("Stranke".language) differs
    # from the company's starts the call in their language — disclosure
    # included. Waits at most 1s for the lookup (it runs since the call
    # started); otherwise the call starts in the company language and can
    # still switch once the caller speaks.
    if detect_language and caller_phone:
        try:
            _profile, _customer = await asyncio.wait_for(
                asyncio.shield(caller_context_task), timeout=1.0
            )
            remembered = normalize_language((_customer or {}).get("language"))
            if remembered and remembered != language and remembered in allowed_languages:
                logger.info("starting in the known customer's language: %s", remembered)
                language = remembered
                call_lang["code"] = remembered
                call_lang["history"] = [remembered]
        except Exception:
            logger.info("caller context not ready within 1s — starting in %s", language)
    logger.info("call time limit: %ss (balance=%s)", call_limit_sec, balance)

    session = AgentSession(
        stt=_build_stt(settings, language, allowed_languages),
        llm=_build_llm(settings),
        tts=_build_tts(settings, language),
        # Silero VAD (2026-10-16, loaded once per process in prewarm): the
        # caller starting to talk is detected from the audio itself, within
        # a few hundred ms, instead of waiting for the STT's first words — so
        # barge-in stops the agent sooner. Turn END is still decided by
        # Soniox endpointing ("turn_detection": "stt" below); this only adds
        # speech-start detection. Background noise that VAD would mistake for
        # speech is cut first by BVCTelephony (room options at start()).
        vad=ctx.proc.userdata.get("vad"),
        user_away_timeout=CALLER_SILENCE_SEC,
        turn_handling={
            "turn_detection": "stt",
            "interruption": InterruptionOptions(enabled=True),
            # Endpointing is left at the framework defaults on purpose:
            # because turn_detection is the string "stt" (not a
            # _StreamingTurnDetector instance), _resolve_endpointing picks
            # min_delay=0.5 / max_delay=3.0 rather than the tighter streaming
            # defaults (0.3/2.5). That 0.5s grace before taking the turn, and
            # tolerance for a 3s mid-sentence pause, is what keeps us from
            # cutting callers off — don't tighten without a voice test.
            # This limit covers the opposite failure: the caller who never
            # pauses at all, so endpointing never fires. See
            # ReceptionistAgent.on_user_turn_exceeded.
            "user_turn_limit": {
                "max_duration": USER_TURN_MAX_DURATION_SEC,
                "max_words": USER_TURN_MAX_WORDS,
            },
            # The LLM already starts before the turn is confirmed (framework
            # default); since 2026-10-26 TTS does too, so the first words are
            # ready the moment the turn ends (~0.35s less silence per reply).
            # A discarded draft costs a few TTS characters.
            "preemptive_generation": {"preemptive_tts": True},
        },
    )

    # Set by the end_call_abusive tool so _on_shutdown can log the call as
    # "ended_abusive" rather than the duration/booking-derived outcome.
    call_end_state = {"abusive": False, "logged_directly": False, "transferred": False}

    # Message / callback request for the owner: set by the take_message tool
    # and by every code path that tells the caller "we'll call you back", so
    # that promise always reaches the owner (a trigger on receptionist_calls
    # notifies them; see migrations/phase6).
    notes = CallNotes()

    transfer_number = (settings_row or {}).get("transfer_number")
    transfer_enabled = bool((settings_row or {}).get("transfer_enabled")) and bool(
        transfer_number
    )
    call_profile: dict = {"profile": None}
    call_context: dict = {"profile": None, "customer": None}

    @function_tool
    async def transfer_to_person(context: RunContext, reason: str) -> dict:
        """Put the caller through to a person at the business.

        Use when the caller asks for a person, or for something you can't
        handle. A short "putting you through" line is spoken automatically;
        say nothing after calling this.

        Args:
            reason: One short sentence for the person who answers, in the
                caller's language — what the caller needs.
        """
        schedule = parse_urnik((call_profile["profile"] or {}).get("urnik"))
        if schedule and not is_open(schedule, datetime.now(TIMEZONE)):
            return {
                "success": False,
                "error": "closed",
                "message": (
                    "The business is closed right now, so nobody can take the "
                    "call. Tell the caller and offer take_message instead."
                ),
            }
        if caller is None:
            return {"success": False, "error": "no_caller", "message": "Offer take_message."}

        logger.info("transferring call to a person: reason=%r", reason)
        await context.wait_for_playout()
        await session.say(pack(call_lang["code"]).transfer_message, allow_interruptions=False)
        try:
            await ctx.api.sip.transfer_sip_participant(
                api.TransferSIPParticipantRequest(
                    room_name=ctx.room.name,
                    participant_identity=caller.identity,
                    transfer_to=f"tel:{transfer_number}",
                    play_dialtone=False,
                )
            )
        except Exception:
            logger.exception("SIP transfer failed")
            return {
                "success": False,
                "error": "transfer_failed",
                "message": (
                    "The transfer didn't go through. Apologise briefly and "
                    "offer to have someone call them back (take_message)."
                ),
            }
        call_end_state["transferred"] = True
        notes.message = notes.message or f"Klic prenesen na {transfer_number}. {_sanitize(reason)}"
        # The caller has left the room; nothing more to say.
        raise StopResponse

    @function_tool
    async def end_call_abusive(context: RunContext) -> None:
        """End the call because genuine abuse continued after ONE warning.

        Never for frustration or complaints. A closing line is spoken
        automatically; say nothing after calling this.
        """
        logger.warning(
            "ending call for continued abuse after warning: call_id=%s "
            "company_slug=%s",
            call_id,
            company_slug,
        )
        call_end_state["abusive"] = True

        # Let whatever the agent was already saying finish, so the closing
        # line doesn't clip the tail of the warning turn.
        await context.wait_for_playout()
        await session.say(pack(call_lang["code"]).abuse_closing, allow_interruptions=False)

        # NOT awaited: AgentSession.aclose() force-interrupts and then drains
        # in-flight speech and tool tasks — and this tool IS one of those
        # tasks, so awaiting it here would deadlock waiting on ourselves.
        # (_degrade_and_close can await it safely because it runs from an
        # event-handler task, not from inside a tool.) Scheduling it means
        # the close lands on the next loop tick instead.
        asyncio.create_task(session.aclose(), name="close_after_abuse")

        # Suppress the follow-up LLM turn the SDK would otherwise generate
        # from this tool's result — without it the model could start speaking
        # again while the session is closing. StopResponse is handled as a
        # normal "done" outcome by the tool executor, not an error, so it
        # neither logs an exception nor trips the session error handler that
        # drives _degrade_and_close.
        raise StopResponse

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
            if LOG_TRANSCRIPTS:
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
                if agent_ref["receptionist"] is not None:
                    agent_ref["receptionist"].note_assistant_message(
                        item.text_content or ""
                    )

    def _record_suppressed_user_turn(text: str, why: str = "post-farewell") -> None:
        if LOG_TRANSCRIPTS:
            transcript_logger.info("user (no reply, %s): %s", why, text)
        transcript.append(
            {
                "role": "user",
                "text": text,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        )

    @session.on("close")
    def _on_close(event) -> None:
        call_ended_at["value"] = datetime.now(timezone.utc)

    # The real ReceptionistAgent needs company data (prompt + tools), which
    # now arrives AFTER the session has started and the disclosure is already
    # playing — see the bootstrap sequence at the end of entrypoint(). Every
    # handler registered before then reaches it through this holder and must
    # tolerate it being None during that window.
    agent_ref: dict = {"receptionist": None}
    booking_ref: dict = {"tools": None}

    # Soniox's language for the caller's latest final transcript, and the
    # one before it (see languages.detect_switch).
    turn_language: dict = {"detected": None, "previous": None}

    user_turn_watch: dict = {
        "started_at": None,
        "fired": False,
        "last_log_at": 0.0,
        "peak_words": 0,
    }

    def _reset_user_turn_watch(reason: str) -> None:
        if user_turn_watch["started_at"] is not None:
            logger.info(
                "user turn watch reset (%s): peak_words=%d duration=%.1fs "
                "fired=%s",
                reason,
                user_turn_watch["peak_words"],
                time.monotonic() - user_turn_watch["started_at"],
                user_turn_watch["fired"],
            )
        user_turn_watch["started_at"] = None
        user_turn_watch["fired"] = False
        user_turn_watch["last_log_at"] = 0.0
        user_turn_watch["peak_words"] = 0

    def _on_agent_state_changed(event) -> None:
        if event.new_state == "speaking":
            # The caller has been answered, so the current turn is no longer an
            # unanswered monologue — start counting afresh. Mirrors the SDK's
            # own rule that the turn tracker resets when the agent speaks.
            _reset_user_turn_watch("agent spoke")

    # Caller silence: LiveKit sets the user "away" after CALLER_SILENCE_SEC
    # with neither side speaking. Once: "Ste še na liniji?". Twice in a row:
    # a short closing and hang up — before this a silent line (caller walked
    # away, call dropped on their side without a SIP BYE) stayed open, and
    # billed, until something upstream timed out.
    silence_state = {"count": 0}

    def _on_user_state_changed(event) -> None:
        if event.new_state == "speaking":
            silence_state["count"] = 0
            return
        if event.new_state != "away":
            return
        receptionist = agent_ref["receptionist"]
        if receptionist is None or "value" in call_ended_at:
            return
        if promise_state["pending"]:
            # The agent promised to check something and went quiet; the
            # promise watchdog handles that. "Ste še na liniji?" here would
            # blame the caller and also cancel the watchdog.
            return
        silence_state["count"] += 1
        asyncio.create_task(
            _handle_caller_silence(silence_state["count"], receptionist),
            name="caller_silence",
        )

    async def _handle_caller_silence(count: int, receptionist) -> None:
        if receptionist.has_said_goodbye:
            # Goodbye already said and the caller just hasn't hung up: asking
            # "are you still there?" now would reopen a finished call.
            logger.info("caller silent after goodbye — closing")
            await session.aclose()
            return
        if count == 1:
            phrases = pack(call_lang["code"]).still_there
            logger.info("caller silent %.0fs — checking the line", CALLER_SILENCE_SEC)
            session.say(phrases[0])
            return
        logger.info("caller silent again — closing the call")
        try:
            await asyncio.wait_for(
                session.say(pack(call_lang["code"]).silence_closing, allow_interruptions=False),
                timeout=10.0,
            )
        except Exception:
            logger.exception("failed to speak silence closing message")
        await session.aclose()

    session.on("agent_state_changed", _on_agent_state_changed)
    session.on("user_state_changed", _on_user_state_changed)

    # Rambling-caller watchdog, interim-transcript based (2026-09-08, second
    # attempt). The SDK's own turn_handling["user_turn_limit"] is left enabled
    # as a backstop but CANNOT fire in this configuration — see the comment on
    # ReceptionistAgent.on_user_turn_exceeded. Root cause, from a real call
    # (job AJ_k3sjqa3MVBqP): a 365-word, ~2.5-minute monologue produced no
    # event at all, because the SDK only advances its counters on FINAL
    # transcripts and Soniox only emits a FINAL on an endpoint, which
    # continuous speech never produces.
    #
    # Interim transcripts, by contrast, DO flow throughout a monologue:
    # AgentActivity.on_interim_transcript emits user_input_transcribed with
    # is_final=False on every interim, and the Soniox plugin builds that text
    # as _merge_lang_segments(final, non_final) — the full running utterance,
    # growing as the caller talks. So counting from interims measures the turn
    # in real time without ever needing an endpoint.
    #
    # Counters reset when the agent speaks (the caller got a response, so the
    # turn is no longer unanswered) or when a final lands (turn committed).
    @session.on("user_input_transcribed")
    def _on_user_input_transcribed(event) -> None:
        if event.is_final:
            _reset_user_turn_watch("final transcript")
            if event.transcript and event.transcript.strip():
                turn_language["detected"] = event.language
            return

        text = (event.transcript or "").strip()
        if not text:
            # The SDK also emits an empty interim as an end-of-speech marker
            # when VAD is absent; it carries no turn content.
            return

        now = time.monotonic()
        if user_turn_watch["started_at"] is None:
            user_turn_watch["started_at"] = now
            user_turn_watch["last_log_at"] = now
            logger.info("user turn watch armed (first interim of turn)")

        elapsed = now - user_turn_watch["started_at"]
        words = len(text.split())
        user_turn_watch["peak_words"] = max(user_turn_watch["peak_words"], words)

        if now - user_turn_watch["last_log_at"] >= USER_TURN_CADENCE_LOG_INTERVAL_SEC:
            user_turn_watch["last_log_at"] = now
            logger.info(
                "user turn in progress: elapsed=%.1fs words=%d "
                "(thresholds %.0fs / %d) agent_state=%s",
                elapsed,
                words,
                USER_TURN_MAX_DURATION_SEC,
                USER_TURN_MAX_WORDS,
                session.agent_state,
            )

        if user_turn_watch["fired"]:
            return
        if elapsed < USER_TURN_MAX_DURATION_SEC and words < USER_TURN_MAX_WORDS:
            return
        # Don't cut in while the agent is already responding — that turn is
        # answered, and the SDK's own handler bails for the same reason.
        if session.agent_state in ("speaking", "thinking"):
            return
        # Still bootstrapping (disclosure playing, company data not in yet):
        # nothing to redirect to.
        if agent_ref["receptionist"] is None:
            return

        user_turn_watch["fired"] = True
        asyncio.create_task(
            agent_ref["receptionist"].speak_redirect(
                source="interim-watchdog", words=words, duration=elapsed
            ),
            name="rambling_redirect",
        )

    # Latency investigation (2026-08-14 voice test): aggregate turn-latency
    # numbers alone can't say whether time is going to STT endpointing, LLM
    # TTFT, or TTS startup — logging only, no behavior change. Reuses the
    # SDK's own built-in per-stage metrics (already computed internally)
    # rather than hand-rolling separate timestamps, and speech_id/request_id
    # let a later investigation correlate STT/LLM/TTS events for one turn.
    call_metrics = CallMetrics()

    # Caller stops talking -> agent starts talking, per reply.
    reply_wait = {"since": None}

    def _on_user_state_for_gap(event) -> None:
        if event.new_state == "speaking":
            reply_wait["since"] = None
        elif event.old_state == "speaking":
            reply_wait["since"] = time.monotonic()

    def _on_agent_state_for_gap(event) -> None:
        if event.new_state == "speaking" and reply_wait["since"] is not None:
            gap = time.monotonic() - reply_wait["since"]
            reply_wait["since"] = None
            call_metrics.record_response_gap(gap)
            if gap > 2.5:
                logger.info("latency reply_gap: %.2fs (slow)", gap)

    session.on("user_state_changed", _on_user_state_for_gap)
    session.on("agent_state_changed", _on_agent_state_for_gap)

    def _count_thinking_filler() -> None:
        call_metrics.thinking_fillers += 1

    @session.on("metrics_collected")
    def _on_metrics_collected(event) -> None:
        m = event.metrics
        call_metrics.record(m)
        if isinstance(m, EOUMetrics):
            logger.info(
                "latency stt_eou: end_of_utterance_delay=%.3fs transcription_delay=%.3fs speech_id=%s",
                m.end_of_utterance_delay,
                m.transcription_delay,
                m.speech_id,
            )
        elif isinstance(m, LLMMetrics):
            logger.info(
                "latency llm: ttft=%.3fs duration=%.3fs speech_id=%s request_id=%s "
                "prompt_tokens=%d prompt_cached_tokens=%d",
                m.ttft,
                m.duration,
                m.speech_id,
                m.request_id,
                m.prompt_tokens,
                m.prompt_cached_tokens,
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
        # The apology says the owner will call back — make that true.
        notes.request_callback("Klic se je prekinil zaradi tehnične težave asistenta.")
        try:
            await _say_technical_difficulty(session)
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
        call_metrics.record_error(
            f"{event.error.type}{'' if event.error.recoverable else '_fatal'}"
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
    # pending: a promise is being watched right now (the caller-silence
    # handler stays out of it — the silence is ours, not the caller's).
    promise_state = {"pending": False, "recovery_used": False}

    def _on_agent_state_changed_watchdog(event) -> None:
        if event.new_state in ("thinking", "speaking"):
            _activity_since_promise.set()
        if event.old_state == "speaking" and event.new_state != "speaking":
            text = last_assistant_text["value"]
            if (
                text
                and text not in _KNOWN_FILLER_TEXTS
                and not (
                    booking_ref["tools"] is not None
                    and text in booking_ref["tools"].spoken_fillers
                )
                and pack(call_lang["code"]).promise_cues.search(_without_thinking_filler(text))
            ):
                _activity_since_promise.clear()
                asyncio.create_task(_watch_for_promise_followup())

    def _on_user_state_changed_watchdog(event) -> None:
        if event.new_state == "speaking":
            _activity_since_promise.set()

    async def _watch_for_promise_followup() -> None:
        promise_state["pending"] = True
        try:
            await asyncio.wait_for(
                _activity_since_promise.wait(), timeout=PROMISE_WATCHDOG_TIMEOUT_SEC
            )
            return
        except asyncio.TimeoutError:
            pass
        finally:
            promise_state["pending"] = False
        if provider_error_state["degrading"]:
            return
        if not promise_state["recovery_used"]:
            # First stall this call: nudge the model to follow through
            # instead of hanging up on the caller. If the LLM itself is down
            # this reply errors out and _on_session_error closes the call.
            promise_state["recovery_used"] = True
            logger.warning(
                "promise watchdog: agent said %r and stopped for %.0fs — "
                "asking the model to follow through",
                last_assistant_text["value"],
                PROMISE_WATCHDOG_TIMEOUT_SEC,
            )
            _activity_since_promise.clear()
            session.generate_reply(instructions=PROMISE_RECOVERY_INSTRUCTIONS)
            try:
                await asyncio.wait_for(
                    _activity_since_promise.wait(), timeout=PROMISE_WATCHDOG_TIMEOUT_SEC
                )
                return
            except asyncio.TimeoutError:
                pass
            if provider_error_state["degrading"]:
                return
        logger.error(
            "promise watchdog fired: agent said %r and produced no follow-up "
            "(recovery already tried) — closing as a technical failure",
            last_assistant_text["value"],
        )
        provider_error_state["degrading"] = True
        await _degrade_and_close()

    session.on("agent_state_changed", _on_agent_state_changed_watchdog)
    session.on("user_state_changed", _on_user_state_changed_watchdog)

    async def _remember_customer_language() -> None:
        """Language memory without storing anything new: a known customer
        who booked in another language than the one on file gets their
        "Stranke".language updated — the same field Jedro+ already uses for
        their reminders, so those follow too. Only after a booking in this
        call, so a stranger borrowing the phone can't change it."""
        customer = (call_context.get("customer") or {})
        booking_tools = booking_ref["tools"]
        if not customer.get("ID stranke") or not (booking_tools and booking_tools.created_termin_id):
            return
        spoken = call_lang["code"]
        if normalize_language(customer.get("language")) == spoken:
            return
        try:
            await supabase.update_customer_language(
                (call_context.get("profile") or {}).get("ID Podjetja", ""),
                customer["ID stranke"],
                TO_JEDROPLUS_CODE[spoken],
            )
            logger.info("customer language updated to %s", spoken)
        except Exception:
            logger.exception("failed to update customer language")

    async def _send_owner_alert() -> None:
        """SMS to the owner for a call that needs a follow-up (phase10).
        Off until the migration has run (no column -> no alert)."""
        row = settings_row or {}
        if row.get("owner_alert_sms") is not True:
            return
        phone = normalize_phone(row.get("owner_alert_phone")) or normalize_phone(
            (call_profile.get("profile") or {}).get("Kontaktni telefon")
        )
        if not phone:
            logger.info("owner alert skipped: no phone (settings or Kontaktni telefon)")
            return
        lang = owner_language(company_language)
        text = owner_alert_text(
            message=notes.message,
            callback_requested=notes.callback_requested,
            kind=notes.alert_kind,
            caller_phone=caller_phone,
            language=lang,
        )
        try:
            await asyncio.wait_for(
                send_sms(
                    kind="owner_alert",
                    company_slug=company_slug,
                    phone=phone,
                    text=text,
                    call_id=call_id,
                    language=lang,
                    timeout=8.0,
                ),
                timeout=9.0,
            )
        except Exception:
            logger.exception("owner alert SMS failed")

    async def _on_shutdown() -> None:
        if call_end_state["logged_directly"]:
            return
        booking_tools = booking_ref["tools"]
        if booking_tools is not None and booking_tools.background_tasks:
            # A confirmation SMS still in flight: let it finish (it may set
            # a note for the owner) before the call row is written.
            await asyncio.wait(list(booking_tools.background_tasks), timeout=10.0)
        created_termin_id = booking_tools.created_termin_id if booking_tools else None
        if booking_tools is not None and booking_tools.owner_followup_reason:
            notes.request_callback(booking_tools.owner_followup_reason, kind="trouble")
        # Looked for (or picked) a time and hung up before it was booked:
        # the owner gets it as a callback request, like a message taken.
        if (
            booking_tools is not None
            and not created_termin_id
            and booking_tools.booking_interest
            and not notes.has_note
            and not call_end_state["transferred"]
            and not call_end_state["abusive"]
        ):
            notes.request_callback(
                unfinished_booking_message(
                    booking_tools.booking_interest,
                    booking_tools._company_data,
                    owner_language(company_language),
                ),
                kind="unfinished",
            )
        ended_at = call_ended_at.get("value") or datetime.now(timezone.utc)
        duration_sec = duration_seconds(started_at, ended_at)

        billed_credits, outcome = bill_call(
            duration_sec,
            created_termin_id=created_termin_id,
            abusive=call_end_state["abusive"],
            transferred=call_end_state["transferred"],
            message_taken=notes.has_note and not call_end_state["transferred"],
            caller_spoke=any(t.get("role") == "user" and (t.get("text") or "").strip() for t in transcript),
        )

        if billed_credits > 0:
            try:
                new_balance = await supabase.deduct_credits(
                    company_slug=company_slug,
                    call_id=call_id,
                    billed_credits=billed_credits,
                )
            except Exception:
                logger.exception(
                    "failed to deduct credits for call_id=%s company_slug=%s",
                    call_id,
                    company_slug,
                )
            else:
                # The owner is notified by the low-balance trigger on
                # receptionist_credits (Jedro+ repo); this is just for the log.
                threshold = (settings_row or {}).get("low_balance_threshold") or 60
                if new_balance < float(threshold):
                    logger.warning(
                        "LOW BALANCE: company_slug=%s balance=%s",
                        company_slug,
                        new_balance,
                    )

        # The job process gets shutdown_process_timeout (30s, see
        # WorkerOptions) for all of this, and billing + the call row come
        # first. Only when the owner will be NOTIFIED (message/callback) is a
        # quick summary made before the insert, so the notification carries
        # it; otherwise it is written afterwards.
        summary = None
        if notes.has_note:
            summary = await summarize_call(
                transcript,
                company_language,
                anthropic_api_key=settings.anthropic_api_key,
                timeout=6.0,
            )

        try:
            await supabase.insert_call(
                {
                    "id": call_id,
                    "company_slug": company_slug,
                    "started_at": started_at.isoformat(),
                    "ended_at": ended_at.isoformat(),
                    "duration_sec": duration_sec,
                    "billed_credits": billed_credits,
                    "outcome": outcome,
                    "transcript": transcript,
                    "created_termin_id": created_termin_id,
                    "livekit_room": livekit_room,
                    "caller_phone": caller_phone,
                    "dialed_number": dialed_number,
                    "message": notes.full_message,
                    "callback_requested": notes.callback_requested,
                    "summary": summary,
                    "language": call_lang["code"],
                    "metrics": call_metrics.summary(
                        turns=sum(1 for t in transcript if t["role"] == "user"),
                        languages=call_lang.get("history", []),
                        time_limit_sec=call_limit_sec,
                        greeting=greeting_state or None,
                    ),
                }
            )
        except Exception:
            logger.exception("failed to insert receptionist_calls row for call_id=%s", call_id)
            return

        if notes.needs_owner_alert and not call_end_state["transferred"]:
            await _send_owner_alert()

        await _remember_customer_language()

        if summary is None:
            summary = await summarize_call(
                transcript,
                company_language,
                anthropic_api_key=settings.anthropic_api_key,
                timeout=12.0,
            )
            if summary:
                try:
                    await supabase.update_call_summary(call_id, summary)
                except Exception:
                    logger.exception("failed to store call summary for %s", call_id)

    ctx.add_shutdown_callback(_on_shutdown)

    # Time limit (2026-10-16): MAX_CALL_SEC, or less when the balance at the
    # start of the call can't cover that (call_policy.call_time_limit_sec).
    # Measured from the job start, like the rest of the bootstrap timings.
    async def _enforce_call_limit() -> None:
        await asyncio.sleep(max(0.0, call_limit_sec - _t()))
        if "value" in call_ended_at:
            return
        logger.warning("call reached its time limit (%ss) — closing", call_limit_sec)
        # Don't cut into a turn in flight — create_booking can take up to
        # 45s, and closing in the middle of it is the worst moment to hang up.
        deadline = time.monotonic() + TIME_LIMIT_IDLE_WAIT_SEC
        while (
            session.agent_state in ("thinking", "speaking")
            and time.monotonic() < deadline
        ):
            await asyncio.sleep(0.5)
        booking_tools = booking_ref["tools"]
        if not (booking_tools and booking_tools.created_termin_id):
            notes.request_callback("Klic je dosegel časovno omejitev, preden je bilo vse urejeno.")
        try:
            await asyncio.wait_for(
                session.say(pack(call_lang["code"]).time_limit_closing, allow_interruptions=False),
                timeout=15.0,
            )
        except Exception:
            logger.exception("failed to speak time-limit closing message")
        await session.aclose()

    call_limit_task = asyncio.create_task(_enforce_call_limit(), name="call_limit")
    session.on("close", lambda _event: call_limit_task.cancel())

    # Disclosure-first bootstrap (2026-09-22). Measured before this change:
    # every call opened with ~2.5s of silence after the job started — ~0.4s
    # connecting, ~1.4s waiting for the booking-v2 init webhook, ~0.7s for
    # session start + TTS first byte. The disclosure doesn't depend on
    # company data, so it now starts as soon as the session can speak, and
    # init finishes in the background while it plays (~5s of audio).
    #
    # Deliberately NOT moved earlier than the credit gate: settings and
    # balance (both fast Supabase reads, fetched concurrently) are awaited
    # above, and the no-credits path has already run and returned before any
    # of this — so a company with no credits still hears only the
    # no-credits message, from its own gate session, exactly as before.
    #
    # The disclosure is non-interruptible: it is the EU AI Act notice, and
    # the caller has to hear it whole. (Previously it was one interruptible
    # utterance together with the greeting.) The greeting that follows is
    # interruptible, as before, and is spoken by the real agent.
    await session.start(
        room=ctx.room,
        agent=BootstrapAgent(
            on_user_turn=lambda text: _record_suppressed_user_turn(
                text, "before agent ready"
            )
        ),
        room_options=_room_options(),
    )
    logger.info("bootstrap: session started t=+%.3fs", _t())
    await _start_background_audio(ctx, session)

    # The greeting (agent/greeting.py, 2026-10-23): "Salon Maja, dober dan!
    # Sem digitalna asistentka." now, non-interruptible (it carries the AI
    # notice), while init finishes; the question ("Kako vam lahko
    # pomagam?") once the real agent is in, interruptible. The business
    # name is the owner's greeting_name, else the legal name without
    # "d.o.o."/"s.p." from the profile, which has been loading since the
    # call started (waits at most 0.8s; without it the greeting has no name).
    greeting_row = settings_row or {}
    if greeting_row.get("greeting_name"):
        greeting_business = str(greeting_row["greeting_name"]).strip()
    else:
        try:
            _greeting_profile, _ = await asyncio.wait_for(
                asyncio.shield(caller_context_task), timeout=0.8
            )
        except Exception:
            _greeting_profile = None
        greeting_business = spoken_business_name((_greeting_profile or {}).get("Naziv Podjetja"))
    greeting_style = greeting_row.get("greeting_style")
    if greeting_style is None and greeting_row.get("greeting_text"):
        greeting_style = "custom"  # before the phase11 migration
    greeting = build_greeting(
        language=language,
        style=greeting_style,
        business_name=greeting_business,
        assistant_name=greeting_row.get("assistant_name"),
        # The owner's own text is written in the company's language; a call
        # starting in the caller's own language gets the preset instead.
        custom_text=greeting_row.get("greeting_text") if language == company_language else None,
        disclosure=pack(language).ai_disclosure,
    )
    greeting_state.update(style=greeting.style, cached=False, opening_at=None, finished=False)
    voice_key = f"{settings.tts_provider}|{settings.tts_model}|{settings.tts_voice_id or ''}|{language}"
    greeting_render_tasks: set = set()

    def _say_greeting_part(text: str, *, allow_interruptions: bool):
        """Play the pre-rendered copy if there is one, else TTS and render a
        copy for the next call once this one has played."""
        path = greeting_audio.cached(text, voice_key)
        if path is not None:
            return session.say(
                text, audio=audio_frames_from_file(str(path)), allow_interruptions=allow_interruptions
            ), True
        handle = session.say(text, allow_interruptions=allow_interruptions)

        async def _render_after_playout() -> None:
            try:
                await handle.wait_for_playout()
            except Exception:
                pass
            await greeting_audio.render(session.tts, text, voice_key)

        task = asyncio.create_task(_render_after_playout(), name="greeting_render")
        greeting_render_tasks.add(task)
        task.add_done_callback(greeting_render_tasks.discard)
        return handle, False

    greeting_state["opening_at"] = round(_t(), 3)
    disclosure_handle, greeting_state["cached"] = _say_greeting_part(
        greeting.opening, allow_interruptions=False
    )
    logger.info(
        "greeting: style=%s cached=%s t=+%.3fs", greeting.style, greeting_state["cached"], _t()
    )

    try:
        company_data = await init_task
    except BookingError:
        logger.exception(
            "booking-v2 init failed for company_slug=%s, degrading gracefully",
            company_slug,
        )
        # The disclosure is already playing on the main session, so the
        # apology goes out on that same session, after it — no separate gate
        # session (the no-credits path still uses one, but it runs before
        # this session exists). Wait for the disclosure first: the apology's
        # own 8s playout timeout in _say_technical_difficulty would otherwise
        # also be counting the disclosure still ahead of it in the queue.
        # Logged here rather than by _on_shutdown, exactly as before: 0
        # credits, outcome booking_unavailable.
        call_end_state["logged_directly"] = True
        try:
            await disclosure_handle.wait_for_playout()
        except Exception:
            logger.exception("disclosure playout failed before init-failure apology")
        await _say_technical_difficulty(session)
        await session.aclose()
        await _insert_call_logged(
            supabase,
            {
                "id": call_id,
                "company_slug": company_slug,
                "started_at": started_at.isoformat(),
                "ended_at": datetime.now(timezone.utc).isoformat(),
                "duration_sec": 0,
                "billed_credits": 0,
                "outcome": "booking_unavailable",
                "transcript": transcript,
                "livekit_room": livekit_room,
                "caller_phone": caller_phone,
                "dialed_number": dialed_number,
                # The apology promised a call back.
                "message": "Asistent ni mogel naložiti podatkov podjetja (napaka rezervacijskega sistema).",
                "callback_requested": True,
            }
        )
        return

    # Default on when the column isn't there yet is deliberately NOT done:
    # without phase7 the setting is unknown, and asking for an email (the
    # old behaviour) is the safe choice until SMS is confirmed to work.
    sms_confirmation = bool((settings_row or {}).get("sms_confirmation", False))
    confirmation_template_row: dict | None = None
    if sms_confirmation:
        try:
            sms_quota_left, confirmation_template_row = await asyncio.wait_for(
                asyncio.shield(sms_setup_task), timeout=2.0
            )
        except Exception:
            logger.warning("SMS setup not ready — assuming SMS is available")
            sms_quota_left = None
        # Unknown (lookup failed) keeps SMS: the send itself still checks
        # the quota, and the owner is told if it didn't go out.
        if sms_quota_left == 0:
            logger.info("no SMS left for %s — confirming by email instead", company_slug)
            sms_confirmation = False

    async def _send_confirmation(booking: dict) -> None:
        services = {str(s["id"]): s["naziv"] for s in company_data.get("services", [])}
        service_names = [services.get(str(i), "") for i in booking["service_ids"]]
        template = choose_confirmation_template(confirmation_template_row, call_lang["code"])
        if template:
            text = render_confirmation_template(
                template,
                company_row=confirmation_template_row,
                service_names=service_names,
                date=booking["date"],
                time_hhmm=booking["time"],
                first_name=booking.get("first_name") or "",
                last_name=booking.get("last_name") or "",
                employee_name=booking.get("employee_name") or "",
                language=call_lang["code"],
            )
        else:
            text = booking_confirmation_text(
                company_name=company_data["company"]["naziv"],
                service_names=service_names,
                date=booking["date"],
                time_hhmm=booking["time"],
                language=call_lang["code"],
            )
        result = await send_sms(
            kind="confirmation",
            company_slug=company_slug,
            phone=normalize_phone(booking["phone"]) or booking["phone"],
            text=text,
            email=booking.get("email") or "",
            termin_id=booking.get("terminId"),
            call_id=call_id,
            language=call_lang["code"],
        )
        if not result.get("success"):
            # The caller was told a text is coming; make sure the owner
            # knows it didn't go, so they can confirm another way.
            if not notes.message:
                notes.message = (
                    f"SMS potrdilo za rezervacijo {booking.get('terminId')} ni bilo "
                    f"poslano ({result.get('error')}). Potrdite stranki drugače."
                )
                notes.alert_kind = "trouble"

    booking_tools = BookingTools(
        company_slug,
        company_data,
        language=language,
        on_booked=_send_confirmation if sms_confirmation else None,
        email_confirmation=not sms_confirmation,
    )
    booking_extras = BookingExtras.from_settings(settings_row)
    booking_tools.set_booking_extras(booking_extras)
    booking_ref["tools"] = booking_tools

    # Caller hung up while the disclosure was playing / init was running.
    # _on_shutdown has logged (or will log) the call from the "close" event's
    # timestamp; don't go on to hand over and greet into a closed session —
    # that only ends in the swap timeout and a failed say().
    if "value" in call_ended_at:
        logger.info("bootstrap: call ended before the agent was ready t=+%.3fs", _t())
        return

    try:
        profile, customer = await asyncio.wait_for(caller_context_task, timeout=3.0)
    except Exception:
        logger.warning("caller context not ready — continuing without it")
        profile, customer = None, None
    call_profile["profile"] = profile
    call_context["profile"] = profile
    call_context["customer"] = customer
    booking_tools.set_caller_customer(customer, caller_phone)

    customer_email = ((customer or {}).get("Email stranke") or "").strip()

    # The customer find_customer matched to a spoken number: the portal
    # link goes to THAT stored number, never to a number nobody verified.
    spoken_match: dict = {}

    def _remember_spoken_match(phone_e164: str, row: dict) -> None:
        spoken_match.update(
            phone=phone_e164, email=((row or {}).get("Email stranke") or "").strip()
        )

    @function_tool
    async def send_manage_link(context: RunContext) -> dict:
        """Text a link to the customer portal, where the appointment can be
        cancelled or moved, to the number on file of the customer
        find_customer found. Only after the caller said yes. The link is
        valid for one hour."""
        phone = spoken_match.get("phone") or (caller_phone if customer_email else "")
        email = spoken_match.get("email") if spoken_match else customer_email
        if not phone:
            return {"success": False, "error": "no_customer",
                    "message": "No customer found for a number yet: use find_customer first, or offer take_message."}
        if not email:
            # Portal login is by email; without one the link can't work.
            return {"success": False, "error": "no_email_on_file",
                    "message": "This customer has no email, so the portal can't be used. Offer take_message instead."}
        result = await send_sms(
            kind="manage_link",
            company_slug=company_slug,
            phone=phone,
            email=email,
            call_id=call_id,
            language=call_lang["code"],
        )
        if result.get("success"):
            return {
                "success": True,
                "message": (
                    "Sent. Tell the caller the text is on its way and that the "
                    "link works for one hour."
                ),
            }
        return {
            "success": False,
            "error": result.get("error"),
            "message": (
                "The text could not be sent. Apologise briefly and offer "
                "take_message so the owner sorts out the appointment."
            ),
        }

    # Which optional rules this call's prompt carries (static_prompt.
    # _FEATURE_RULES): the transfer and portal-link rules only when their
    # tools are registered below, the prepayment rule only when the company
    # takes payments online.
    features = set()
    if transfer_enabled:
        features.add("transfer")
    company_text_id = (profile or {}).get("ID Podjetja")
    appointment_tools = None
    if caller_phone and customer and customer.get("ID stranke") and company_text_id:
        async def _notify_change(kind, row, new_date, new_time) -> None:
            if not sms_confirmation:
                return
            text = appointment_change_text(
                kind,
                company_name=company_data["company"]["naziv"],
                service=row.get("Storitev") or "",
                date=new_date or row["Datum"][:10],
                time_hhmm=new_time or str(row.get("Čas") or "")[:5],
                language=call_lang["code"],
            )
            result = await send_sms(
                kind="confirmation", company_slug=company_slug, phone=caller_phone,
                text=text, termin_id=row.get("ID termina"), call_id=call_id,
                language=call_lang["code"],
            )
            if not result.get("success"):
                notes.add_change(f"SMS o spremembi termina {row.get('ID termina')} ni bil poslan.")

        appointment_tools = AppointmentTools(
            supabase=supabase,
            company_id=company_text_id,
            company_slug=company_slug,
            customer=customer,
            profile=profile,
            company_data=company_data,
            notes=notes,
            language=call_lang["code"],
            resources_for=booking_tools._resources,
            notify_customer=_notify_change,
        )
    if appointment_tools is not None or company_text_id:
        features.add("manage_link")
    # "payments" (R-PAYMENT-REQUIRED: "you'll get a payment link") stays off
    # until the link exists (sklop 8): booking-v2 books phone bookings as
    # ordinary appointments without prepayment and notes it for the owner
    # (Tim, 2026-10-19). Before that a Stripe company's phone booking sat in
    # pending_payment with a promised link nobody sent.

    prompt_args = dict(
        features=frozenset(features),
        profile=profile,
        instructions=(settings_row or {}).get("instructions"),
        caller_phone=caller_phone,
        customer=customer,
        sms_confirmation=sms_confirmation,
        # find_customer is registered below whenever the company id is known.
        customer_lookup=bool((profile or {}).get("ID Podjetja")),
        booking_extras=booking_extras,
        assistant_name=(settings_row or {}).get("assistant_name"),
    )

    language_control = LanguageControl(
        call_lang=call_lang,
        allowed=allowed_languages,
        company_data=company_data,
        prompt_args=prompt_args,
        booking_tools=booking_tools,
        session=session,
        turn_language=turn_language,
        enabled=detect_language,
    )

    async def _cancel_created(row_id) -> bool:
        if not company_text_id:
            return False
        return bool(await supabase.update_appointment(company_text_id, row_id, {"Status": "cancelled"}))

    async def _send_group_confirmation(bookings: list[dict], phone: str) -> None:
        notes.add_change(
            "Skupinska rezervacija po telefonu: "
            + ", ".join(f"{b['terminId']} ({b['name']}, {b['time']})" for b in bookings) + "."
        )
        if not sms_confirmation:
            return
        text = group_confirmation_text(
            company_name=company_data["company"]["naziv"], bookings=bookings, language=call_lang["code"]
        )
        result = await send_sms(
            kind="confirmation", company_slug=company_slug, phone=normalize_phone(phone) or phone,
            text=text, termin_id=bookings[0]["terminId"], call_id=call_id, language=call_lang["code"],
        )
        if not result.get("success") and not notes.message:
            notes.message = (
                f"SMS potrdilo za skupinsko rezervacijo ({bookings[0]['terminId']} …) ni bilo "
                f"poslano ({result.get('error')}). Potrdite stranki drugače."
            )
            notes.alert_kind = "trouble"

    group_booking = GroupBooking(
        booking_tools, cancel_created=_cancel_created, on_group_booked=_send_group_confirmation
    )
    extra_tools = [group_booking.find_group_times, group_booking.book_group, end_call_abusive]
    if company_text_id:
        async def _lookup_spoken_number(phone_e164: str) -> dict | None:
            return await supabase.find_customer_by_phone(company_text_id, phone_e164)

        extra_tools.insert(
            0,
            build_find_customer_tool(
                _lookup_spoken_number, caller_phone, booking_tools.private_emails,
                on_found=_remember_spoken_match,
            ),
        )
        booking_tools.set_customer_lookup(_lookup_spoken_number, caller_phone)
    if detect_language:
        extra_tools.insert(0, language_control.switch_language)
    if transfer_enabled:
        extra_tools.insert(0, transfer_to_person)
    # Cancel / move in the call only for a caller recognised by caller ID;
    # everyone else can get the portal link texted to the number on file.
    if appointment_tools is not None:
        extra_tools[0:0] = [
            appointment_tools.list_my_appointments,
            appointment_tools.cancel_appointment,
            appointment_tools.reschedule_appointment,
        ]
    elif company_text_id:
        extra_tools.insert(0, send_manage_link)
    receptionist_agent = build_receptionist_agent(
        company_data,
        language,
        booking_tools=booking_tools,
        notes=notes,
        extra_tools=extra_tools,
        on_suppressed_user_turn=_record_suppressed_user_turn,
        switchable_to=[c for c in allowed_languages if c != language],
        on_turn_language=language_control.on_turn,
        on_thinking_filler=_count_thinking_filler,
        **prompt_args,
    )
    agent_ref["receptionist"] = receptionist_agent
    language_control.agent = receptionist_agent

    # The handoff drains the bootstrap agent first, which waits for queued
    # speech to finish rather than interrupting it — the disclosure always
    # plays to the end.
    session.update_agent(receptionist_agent)
    try:
        await asyncio.wait_for(
            receptionist_agent.entered.wait(), timeout=AGENT_SWAP_TIMEOUT_SEC
        )
        logger.info("bootstrap: receptionist agent active t=+%.3fs", _t())
    except asyncio.TimeoutError:
        logger.error(
            "bootstrap: receptionist agent not active after %.0fs — greeting anyway",
            AGENT_SWAP_TIMEOUT_SEC,
        )

    try:
        invitation_handle, _ = _say_greeting_part(greeting.invitation, allow_interruptions=True)
        await invitation_handle
        # False = the caller hung up during the greeting: what the styles
        # are compared on.
        greeting_state["finished"] = "value" not in call_ended_at
    except Exception:
        # A broken TTS provider surfaces here too; the "error" handler above
        # will already be counting toward _degrade_and_close, so just avoid
        # crashing the entrypoint over it. Also reached if the caller hung up
        # during bootstrap (the session is already closed).
        logger.exception("failed to speak greeting")

    logger.info("greeting delivered, conversation active")


def prewarm(proc: JobProcess) -> None:
    """Runs once per worker process, before any call: loading Silero takes
    ~1s and must not land on a caller's first second of silence."""
    try:
        proc.userdata["vad"] = silero.VAD.load()
    except Exception:
        # Without VAD the call still works exactly as before (STT-only
        # speech detection) — never refuse calls over it.
        logger.exception("failed to load Silero VAD — running without it")


if __name__ == "__main__":
    agents.cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            # Room for _on_shutdown: billing, the call row and the summary
            # (default 10s would cut the summary — or worse — short).
            shutdown_process_timeout=30.0,
            agent_name="receptionistplus-worker",
        )
    )
