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

from livekit import agents, rtc
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
from agent.dates import ENGLISH_WEEKDAYS, SLOVENIAN_WEEKDAYS, spoken_date_phrase
from agent.static_prompt import STATIC_PROMPT_EN, STATIC_PROMPT_SL
from agent.supabase_client import SupabaseClient
from agent.call_notes import CallNotes
from agent.llm_factory import LLMConfig, build_llm
from agent.tools import BookingTools

logger = logging.getLogger("receptionistplus-worker")
transcript_logger = logging.getLogger("receptionistplus-worker.transcript")

SUPPORTED_LANGUAGES = ("sl", "en")
DEFAULT_LANGUAGE = "sl"

NO_CREDITS_MESSAGE = {
    "sl": "Trenutno žal ne moremo sprejeti vašega klica. Prosimo, poskusite kasneje.",
    "en": "We're unable to take your call right now. Please try again later.",
}

# Spoken before closing a call the agent is ending because abuse continued
# after one warning (see the end_call_abusive tool). Deliberately a normal,
# warm sign-off rather than a rebuke or an accusation: by this point the
# decision is made, and the last thing the caller hears should not escalate.
# It is a graceful close, NOT a hangup — same shape as
# TECHNICAL_DIFFICULTY_MESSAGE, which is spoken and then followed by
# session.aclose(). Slovenian uses the feminine "zaključila" per the FEMALE
# persona rule in STATIC_PROMPT_SL.
ABUSE_CLOSING_MESSAGE = {
    "sl": "Hvala za klic. Klic bom zdaj zaključila. Lep dan še naprej.",
    "en": "Thank you for calling. I'll end the call here. Have a good day.",
}

TECHNICAL_DIFFICULTY_MESSAGE = {
    "sl": "Oprostite, trenutno imamo tehnične težave, lastnik vas bo poklical nazaj.",
    "en": (
        "Sorry, we're experiencing technical difficulties right now — the "
        "owner will call you back."
    ),
}

# EU AI Act Article 50: callers must be told they're talking to an AI system,
# no later than the first interaction. Prepended in code rather than baked
# into the greeting itself so it can't be dropped by a misconfigured or
# edited per-company greeting — this is the one guaranteed source of
# disclosure, not the only one.
AI_DISCLOSURE = {
    "sl": (
        "Prosimo, upoštevajte, da govorite z digitalnim glasovnim asistentom, "
        "ne z osebo. "
    ),
    "en": (
        "Please note that you are speaking with a digital voice assistant, "
        "not a person. "
    ),
}

# Built from company_data.company.naziv (fetched live via booking-v2 init)
# rather than a hand-maintained env var — a per-company GREETING_TEXT env var
# went stale for real (found 2026-08-28: production was still greeting
# callers with "Salon Lepote", a Phase 1 fake test company name, months
# after switching to real jedroplus-d-o-o data) because nothing forces it to
# stay in sync with the actual company name. This can't go stale the same
# way, at the cost of no longer supporting custom greeting wording beyond
# "Welcome to <company name>" — a deliberate tradeoff, not an oversight.
GREETING_TEMPLATE = {
    "sl": "Pozdravljeni, dobrodošli v {name}. Kako vam lahko pomagam?",
    "en": "Hello, welcome to {name}. How can I help you?",
}

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
TIME_LIMIT_CLOSING_MESSAGE = {
    "sl": (
        "Oprostite, klic moram zdaj zaključiti. Če česa še nismo uredili, vas "
        "bomo poklicali nazaj. Lep dan še naprej."
    ),
    "en": (
        "I'm sorry, I have to end the call now. If anything is still open, "
        "we'll call you back. Have a good day."
    ),
}

# How long to wait, once the time limit is reached, for an in-flight turn
# (e.g. create_booking, up to 45s) to finish before closing anyway.
TIME_LIMIT_IDLE_WAIT_SEC = 50.0

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
STILL_THERE_PHRASES = {
    "sl": ["Ste še na liniji?", "Halo, me slišite?"],
    "en": ["Are you still there?", "Hello, can you hear me?"],
}
SILENCE_CLOSING_MESSAGE = {
    "sl": "Žal vas ne slišim več. Pokličite nas, prosim, še enkrat. Lep dan.",
    "en": "I'm afraid I can't hear you anymore. Please call us again. Have a good day.",
}

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
REDIRECT_PHRASES = {
    "sl": [
        "Oprostite, da vas prekinem — kako vam lahko pomagam?",
        "Oprostite, da vas prekinem — mi lahko na kratko poveste, kaj potrebujete?",
        "Se opravičujem za prekinitev — povejte mi, prosim, s čim vam lahko pomagam.",
    ],
    "en": [
        "Sorry to jump in — how can I help you?",
        "Sorry to interrupt — could you tell me briefly what you need?",
        "Apologies for cutting in — please tell me what I can help you with.",
    ],
}

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
PROMISE_CUE_PATTERNS = {
    "sl": re.compile(
        r"trenutek|preverim\b|preverjam\b|preveril|urejam\b|uredila\b|rezerviram\b"
        r"|poskušam\b|počakajte|pogledam\b|vpisujem\b|vas vpišem\b",
        re.IGNORECASE,
    ),
    "en": re.compile(
        r"\b(?:one|just a|give me a) moment\b|\bhold on\b|\bplease wait\b"
        r"|\blet me (?:check|look|see)\b|\bi'?ll check\b|\bcheck(?:ing)?\b"
        r"|\bi'?m (?:checking|processing|trying)\b|\bprocessing\b|\btrying\b",
        re.IGNORECASE,
    ),
}
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
    p for phrases in STILL_THERE_PHRASES.values() for p in phrases
)

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
FAREWELL_PATTERNS = {
    "sl": re.compile(
        r"nasvidenje|lep dan|se slišimo|se vidimo|hvala za klic|adijo",
        re.IGNORECASE,
    ),
    "en": re.compile(
        r"\bgoodbye\b|\bbye\b|have a (?:great|good|nice|lovely) day"
        r"|\btalk soon\b|\btake care\b|thanks for calling",
        re.IGNORECASE,
    ),
}
ACKNOWLEDGEMENT_WORDS = {
    "sl": frozenset(
        "ja jaa jap ok okej okay oki hvala lepa najlepša vam tudi enako prav "
        "v redu dobro super velja odlično adijo adio čao nasvidenje lep dan "
        "mhm aha no sem rekel rekla že se slišimo vidimo prosim".split()
    ),
    "en": frozenset(
        "yes yeah yep ok okay thanks thank you bye goodbye cheers great "
        "alright sure have a good nice great lovely day too same to take care "
        "mhm i said right perfect wonderful".split()
    ),
}
# Farewell turns (with only acknowledgements in between) after which a pure
# acknowledgement is no longer answered: 1 = the farewell itself, 2 = the one
# short "Prosim, lep dan!" the prompt allows in reply to the first "hvala".
FAREWELLS_BEFORE_SILENCE = 2


def _is_pure_acknowledgement(text: str, language: str) -> bool:
    words = re.findall(r"[^\W\d_]+", text.lower())
    return bool(words) and all(
        w in ACKNOWLEDGEMENT_WORDS[language] for w in words
    )


# How long to wait for update_agent() to finish handing the session from the
# bootstrap placeholder to the real ReceptionistAgent. The handoff first waits
# for queued speech (the disclosure) to finish playing, so this has to cover
# the disclosure's remaining playout, not just the swap itself.
AGENT_SWAP_TIMEOUT_SEC = 15.0

TIMEZONE = ZoneInfo("Europe/Ljubljana")

_RELATIVE_DAY_LABELS = {0: "DANES", 1: "JUTRI", 2: "POJUTRIŠNJEM"}
_RELATIVE_DAY_LABELS_EN = {0: "TODAY", 1: "TOMORROW", 2: "DAY AFTER TOMORROW"}

def _build_date_context(language: str = DEFAULT_LANGUAGE) -> str:
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
    # always lands within the 14-row table (max is 7, when today is Monday).
    next_monday_offset = 7 - now.weekday()
    next_week_offsets = range(next_monday_offset, next_monday_offset + 7)

    if language == "en":
        weekday_names = ENGLISH_WEEKDAYS
        relative_labels = _RELATIVE_DAY_LABELS_EN
        next_week_tag = "NEXT WEEK"
    else:
        weekday_names = SLOVENIAN_WEEKDAYS
        relative_labels = _RELATIVE_DAY_LABELS
        next_week_tag = "NASLEDNJI TEDEN"

    rows = []
    for offset in range(14):
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

    if language == "en":
        return (
            f"Current time is {time_str}, timezone Europe/Ljubljana. Below is "
            f"a pre-computed table of the next 14 days: ISO date, weekday, and "
            f"the exact spoken date phrase. The first three rows are tagged "
            f"[TODAY], [TOMORROW], and [DAY AFTER TOMORROW] directly — use "
            f"these tags as-is to resolve those specific words, do NOT count "
            f"rows or compute the offset yourself. Every row that falls in "
            f"next calendar week is additionally tagged [NEXT WEEK].\n\n"
            f"{table}\n\n"
            f"To resolve any relative date (tomorrow, on Monday, as soon as "
            f"possible, next week), look up the matching entry in this table "
            f"— do NOT calculate the date or weekday yourself. For a COMPOUND "
            f"reference that combines a week qualifier with a weekday name "
            f"(e.g. \"next week on Thursday\"), find the table row tagged "
            f"[NEXT WEEK] whose weekday matches the one requested, and use "
            f"that row's exact date and phrase — never compute which date "
            f"that is yourself, even approximately.\n"
            f"  - Bad: caller says \"next week on Thursday\" → mentally "
            f"counting forward some number of days and speaking whatever "
            f"date that lands on, whether or not it's actually a Thursday.\n"
            f"  - Good: caller says \"next week on Thursday\" → scan the "
            f"table for the row tagged [NEXT WEEK] with weekday Thursday, "
            f"and use exactly that row's date/phrase.\n"
            f"When speaking a date aloud, use the exact phrase from the "
            f"table verbatim (e.g. \"August 3rd\") — do NOT construct the "
            f"ordinal yourself, since generating it live is error-prone. "
            f"When confirming a weekday together with its date, prefer a "
            f"natural connector over a flat statement.\n"
            f"  - Not \"On Thursday it is September third.\" (awkward)\n"
            f"  - Good: \"On Thursday, so September third.\"\n"
        # 2026-09-21: "torek, osemindvajsetega septembra" (the 28th was a
        # Monday). The table was right; the model never looked a Tuesday
        # up — it glued the caller's weekday onto the first date of the
        # get_slots result it was reading. These two rules close the gaps
        # that let that through: nothing tied a spoken weekday to the same
        # row as its date, and the NEXT WEEK rule only covered a qualifier
        # and weekday said in one breath.
        "A weekday and its date must ALWAYS come from ONE place: one row of "
        "this table, or one label from a tool result (get_slots' "
        "\"day_labels\", or the \"day_label\" from check_slots/"
        "create_booking, e.g. \"Tuesday, September 29th\"). Never put a "
        "weekday you heard from the caller next to a date you took from "
        "somewhere else.\n"
        "  - Bad (observed 2026-09-21, Slovenian call): caller says "
        "\"Tuesday\" → \"So Tuesday, September twenty-eighth\" (the "
        "caller's weekday glued to the first date of a get_slots result — "
        "the 28th was a Monday, and the booking ended up on Monday)\n"
        "  - Good: find the Tuesday row/label first → \"So Tuesday, "
        "September twenty-ninth\"\n"
        "A week qualifier the caller said EARLIER still applies when they "
        "later name only a weekday. If they asked about \"next week\" and "
        "a few turns later say \"Tuesday\", that means the row tagged both "
        "Tuesday and [NEXT WEEK] — not this week's Tuesday, and not "
        "whichever date you were just looking at. Only if nothing earlier "
        "narrows it down and the weekday matches two rows, ask which one "
        "they mean (\"This Tuesday, or Tuesday next week?\").\n"
            f"If a caller's requested date falls outside this table, tell "
            f"them you'll pass the request on so someone calls back to "
            f"confirm (take_message) — never guess."
        )

    return (
        f"Current time is {time_str}, timezone Europe/Ljubljana. Below is a "
        f"pre-computed table of the next 14 days: ISO date, Slovenian weekday, "
        f"and the exact spoken date phrase. The first three rows are tagged "
        f"[DANES] (today), [JUTRI] (tomorrow), and [POJUTRIŠNJEM] (the day "
        f"after tomorrow) directly — use these tags as-is to resolve those "
        f"specific words, do NOT count rows or compute the offset yourself. "
        f"Every row that falls in next calendar week is additionally tagged "
        f"[NASLEDNJI TEDEN].\n\n"
        f"{table}\n\n"
        f"To resolve any relative date (jutri, v ponedeljek, čim prej, "
        f"naslednji teden), look up the matching entry in this table — do NOT "
        f"calculate the date or weekday yourself. For a COMPOUND reference "
        f"that combines a week qualifier with a weekday name (e.g. "
        f"\"naslednji teden v četrtek\"), find the table row tagged "
        f"[NASLEDNJI TEDEN] whose weekday matches the one requested, and use "
        f"that row's exact date and phrase — never compute which date that "
        f"is yourself, even approximately.\n"
        f"  - Bad: caller says \"naslednji teden v četrtek\" → mentally "
        f"counting forward some number of days and speaking whatever date "
        f"that lands on, whether or not it's actually a Thursday.\n"
        f"  - Good: caller says \"naslednji teden v četrtek\" → scan the "
        f"table for the row tagged [NASLEDNJI TEDEN] with weekday četrtek, "
        f"and use exactly that row's date/phrase.\n"
        f"When speaking a date aloud, use the exact phrase from the table "
        f"verbatim (e.g. \"tretjega avgusta\") — do NOT construct the "
        f"ordinal number yourself, since generating it live is error-prone. "
        f"When confirming a weekday together with its date, prefer a natural "
        f"connector over a flat statement.\n"
        f"  - Not \"V četrtek je to tretjega septembra.\" (awkward)\n"
        f"  - Good: \"V četrtek, torej tretjega septembra.\"\n"
        # See the matching note in the English branch above.
        "A weekday and its date must ALWAYS come from ONE place: one row of "
        "this table, or one label from a tool result (get_slots' "
        "\"day_labels\", or the "
        "\"day_label\" from check_slots/create_booking, e.g. \"torek, "
        "devetindvajsetega septembra\"). Never put a weekday you heard from "
        "the caller next to a date you took from somewhere else. You may put "
        "\"v\"/\"za\" in front and adjust the weekday's ending (sreda → v "
        "sredo), but never change which day it is.\n"
        "  - Bad (observed 2026-09-21): caller says \"torek\" → \"Torej "
        "torek, osemindvajsetega septembra\" (the caller's weekday glued to "
        "the first date of a get_slots result — the 28th was a Monday, and "
        "the booking ended up on Monday)\n"
        "  - Good: find the torek row/label first → \"Torej v torek, "
        "devetindvajsetega septembra\"\n"
        "A week qualifier the caller said EARLIER still applies when they "
        "later name only a weekday. If they asked about \"naslednji teden\" "
        "and a few turns later say \"torek\", that means the row tagged "
        "both torek and [NASLEDNJI TEDEN] — not this week's torek, and not "
        "whichever date you were just looking at. Only if nothing earlier "
        "narrows it down and the weekday matches two rows, ask which one "
        "they mean (\"Mislite ta torek ali torek naslednji teden?\").\n"
        f"If a caller's requested date falls outside this table, tell them "
        f"you'll pass the request on so someone calls back to confirm "
        f"(take_message) — never guess."
    )


def _build_system_prompt(company_data: dict, language: str = DEFAULT_LANGUAGE) -> str:
    static_prompt = STATIC_PROMPT_EN if language == "en" else STATIC_PROMPT_SL
    logger.info(
        "language pack loaded: language=%s static_prompt=%s",
        language,
        "STATIC_PROMPT_EN" if language == "en" else "STATIC_PROMPT_SL",
    )
    return (
        static_prompt
        + "\n\n"
        + _build_date_context(language)
        + "\n\n"
        + render_company_prompt(company_data, language)
    )


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
        logger.info("bootstrap: caller spoke before the agent was ready: %r (no reply)", text)
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
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._language = language
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

    @property
    def has_said_goodbye(self) -> bool:
        return self._farewell_count > 0

    def note_assistant_message(self, text: str) -> None:
        if FAREWELL_PATTERNS[self._language].search(text):
            self._farewell_count += 1

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: ChatMessage
    ) -> None:
        text = new_message.text_content or ""
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
        phrases = REDIRECT_PHRASES[self._language]
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


def build_receptionist_agent(
    company_data: dict,
    language: str,
    *,
    booking_tools: BookingTools,
    notes: CallNotes,
    extra_tools: list | tuple = (),
    on_suppressed_user_turn=None,
) -> "ReceptionistAgent":
    """The receptionist for one call. Shared by entrypoint() and the eval
    harness (evals/run_evals.py), so both run the same prompt and tools.
    extra_tools carries the ones bound to the live session (end_call_abusive).
    """
    return ReceptionistAgent(
        language=language,
        on_suppressed_user_turn=on_suppressed_user_turn,
        instructions=_build_system_prompt(company_data, language),
        tools=[
            booking_tools.get_slots,
            booking_tools.check_slots,
            booking_tools.create_booking,
            notes.take_message,
            *extra_tools,
        ],
    )


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
        model="tts-rt-v1",
        language=language,
        # 1.0 (default) read as noticeably slow to callers; 1.1 read as too
        # fast in testing (2026-08-30). 1.04 is a subtler nudge (plugin
        # range [0.7, 1.3]) chosen to not reopen the pronunciation-quality
        # tuning already settled on voice/model/language.
        speed=1.04,
    )


def _build_stt(settings: Settings, language: str = DEFAULT_LANGUAGE) -> soniox.STT:
    return soniox.STT(
        api_key=settings.soniox_api_key,
        params=soniox.STTOptions(language_hints=[language]),
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


def _render_greeting(custom: str | None, language: str, company_name: str) -> str:
    """The greeting spoken after the disclosure.

    The dashboard's greeting_text wins when set (it was saved but never used
    until 2026-10-16); "{name}" or "{podjetje}" in it is replaced with the
    company name. Plain str.replace, not str.format: owners type free text,
    and a stray "{" must not crash the call. The AI disclosure is NOT part
    of this and can't be removed by it — it is always spoken first, in code.
    """
    text = (custom or "").strip()
    if not text:
        return GREETING_TEMPLATE[language].format(name=company_name)
    return text.replace("{name}", company_name).replace("{podjetje}", company_name)


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
        await _speak_and_close(ctx, settings, DEFAULT_LANGUAGE, NO_CREDITS_MESSAGE[DEFAULT_LANGUAGE])
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
    language = (settings_row or {}).get("language") or DEFAULT_LANGUAGE
    if language not in SUPPORTED_LANGUAGES:
        logger.warning(
            "unsupported language %r for company_slug=%s, defaulting to %s",
            language,
            company_slug,
            DEFAULT_LANGUAGE,
        )
        language = DEFAULT_LANGUAGE

    async def _say_technical_difficulty(target_session: AgentSession) -> None:
        """Play the pre-rendered apology clip for Slovenian (bypasses TTS
        synthesis entirely, so it still works if the TTS provider itself is
        the thing that's down); English has no equivalent asset yet (a
        deliberate scope decision), so it falls back to plain session.say()
        text, which won't survive a TTS-provider outage but is otherwise
        fine.
        """
        message = TECHNICAL_DIFFICULTY_MESSAGE[language]
        try:
            if language == "sl":
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
        await _speak_and_close(ctx, settings, language, NO_CREDITS_MESSAGE[language])
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
    logger.info("call time limit: %ss (balance=%s)", call_limit_sec, balance)

    session = AgentSession(
        stt=_build_stt(settings, language),
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
        },
    )

    # Set by the end_call_abusive tool so _on_shutdown can log the call as
    # "ended_abusive" rather than the duration/booking-derived outcome.
    call_end_state = {"abusive": False, "logged_directly": False}

    # Message / callback request for the owner: set by the take_message tool
    # and by every code path that tells the caller "we'll call you back", so
    # that promise always reaches the owner (a trigger on receptionist_calls
    # notifies them; see migrations/phase6).
    notes = CallNotes()

    @function_tool
    async def end_call_abusive(context: RunContext) -> None:
        """End this call because the caller remained abusive after a warning.

        Only call this after you have already given the caller ONE polite
        warning and the abusive behaviour continued anyway. Never call it as
        a first response, and never for a caller who is merely frustrated,
        complaining, or unhappy with the company — only for genuine abuse
        directed at you, such as personal insults, sexual harassment, or
        threats.

        You do not need to say goodbye first: a short closing line is spoken
        automatically. Say nothing after calling this.
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
        await session.say(ABUSE_CLOSING_MESSAGE[language], allow_interruptions=False)

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
            phrases = STILL_THERE_PHRASES[language]
            logger.info("caller silent %.0fs — checking the line", CALLER_SILENCE_SEC)
            session.say(phrases[0])
            return
        logger.info("caller silent again — closing the call")
        try:
            await asyncio.wait_for(
                session.say(SILENCE_CLOSING_MESSAGE[language], allow_interruptions=False),
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
                and PROMISE_CUE_PATTERNS[language].search(text)
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

    async def _on_shutdown() -> None:
        if call_end_state["logged_directly"]:
            return
        booking_tools = booking_ref["tools"]
        created_termin_id = booking_tools.created_termin_id if booking_tools else None
        if booking_tools is not None and booking_tools.owner_followup_reason:
            notes.request_callback(booking_tools.owner_followup_reason)
        ended_at = call_ended_at.get("value") or datetime.now(timezone.utc)
        duration_sec = duration_seconds(started_at, ended_at)

        billed_credits, outcome = bill_call(
            duration_sec,
            created_termin_id=created_termin_id,
            abusive=call_end_state["abusive"],
            message_taken=notes.has_note,
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
                    "message": notes.message,
                    "callback_requested": notes.callback_requested,
                }
            )
        except Exception:
            logger.exception("failed to insert receptionist_calls row for call_id=%s", call_id)

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
                session.say(TIME_LIMIT_CLOSING_MESSAGE[language], allow_interruptions=False),
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
    disclosure_handle = session.say(
        AI_DISCLOSURE[language].strip(), allow_interruptions=False
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

    booking_tools = BookingTools(company_slug, company_data, language=language)
    booking_ref["tools"] = booking_tools

    # Caller hung up while the disclosure was playing / init was running.
    # _on_shutdown has logged (or will log) the call from the "close" event's
    # timestamp; don't go on to hand over and greet into a closed session —
    # that only ends in the swap timeout and a failed say().
    if "value" in call_ended_at:
        logger.info("bootstrap: call ended before the agent was ready t=+%.3fs", _t())
        return

    receptionist_agent = build_receptionist_agent(
        company_data,
        language,
        booking_tools=booking_tools,
        notes=notes,
        extra_tools=[end_call_abusive],
        on_suppressed_user_turn=_record_suppressed_user_turn,
    )
    agent_ref["receptionist"] = receptionist_agent

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

    greeting = _render_greeting(
        (settings_row or {}).get("greeting_text"),
        language,
        company_data["company"]["naziv"],
    )
    try:
        await session.say(greeting)
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
            agent_name="receptionistplus-worker",
        )
    )
