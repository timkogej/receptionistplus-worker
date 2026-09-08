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

SUPPORTED_LANGUAGES = ("sl", "en")
DEFAULT_LANGUAGE = "sl"

NO_CREDITS_MESSAGE = {
    "sl": "Trenutno žal ne moremo sprejeti vašega klica. Prosimo, poskusite kasneje.",
    "en": "We're unable to take your call right now. Please try again later.",
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
GENERIC_FILLER_PHRASES = {
    # Kept in sync with STATIC_PROMPT_SL's approved waiting-phrase rule
    # ("Samo trenutek" / "trenutek prosim" are the correct forms) — these are
    # spoken by code rather than the model, so nothing enforces that
    # automatically. "En moment..." was removed 2026-09-08: it's a Germanism,
    # not one of the approved forms, and callers were hearing it from the
    # filler path while the model itself was forbidden to say it.
    "sl": [
        "Samo trenutek...",
        "Trenutek, prosim...",
        "Samo sekundo...",
    ],
    "en": [
        "One moment...",
        "Just a second...",
        "Give me a moment...",
    ],
}
GENERIC_FILLER_DELAY = 5.0

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
        r"|poskušam\b|počakajte",
        re.IGNORECASE,
    ),
    "en": re.compile(
        r"\b(?:one|just a|give me a) moment\b|\bhold on\b|\bplease wait\b"
        r"|\blet me check\b|\bi'?ll check\b|\bcheck(?:ing)?\b"
        r"|\bi'?m (?:checking|processing|trying)\b|\bprocessing\b|\btrying\b",
        re.IGNORECASE,
    ),
}
PROMISE_WATCHDOG_TIMEOUT_SEC = 12.0

# Our own filler utterances also match PROMISE_CUE_PATTERNS by design, but
# they're paired with an actual in-flight tool call that already has its own
# real timeout (up to 45s for create_booking) — excluded so the watchdog's
# much shorter window doesn't kill a call that's legitimately still running.
_KNOWN_FILLER_TEXTS = frozenset(
    [p for phrases in GENERIC_FILLER_PHRASES.values() for p in phrases]
    + [p for phrases in GET_SLOTS_FILLER_TEXT.values() for p in phrases]
    + [p for phrases in CREATE_BOOKING_FILLER_TEXT.values() for p in phrases]
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
_RELATIVE_DAY_LABELS_EN = {0: "TODAY", 1: "TOMORROW", 2: "DAY AFTER TOMORROW"}

ENGLISH_WEEKDAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

# Hardcoded rather than strftime("%B"): %B is locale-dependent and the
# deployment locale isn't guaranteed to be English (same reasoning as the
# Slovenian month/weekday dicts above, which exist for the same reason).
ENGLISH_MONTHS = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}


def _english_ordinal_suffix(day: int) -> str:
    if 11 <= day % 100 <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")


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

        def phrase_for(d: datetime) -> str:
            return f"{ENGLISH_MONTHS[d.month]} {d.day}{_english_ordinal_suffix(d.day)}"
    else:
        weekday_names = SLOVENIAN_WEEKDAYS
        relative_labels = _RELATIVE_DAY_LABELS
        next_week_tag = "NASLEDNJI TEDEN"

        def phrase_for(d: datetime) -> str:
            return f"{slovenian_ordinal_genitive(d.day)} {SLOVENIAN_MONTHS_GENITIVE[d.month]}"

    rows = []
    for offset in range(14):
        d = now + timedelta(days=offset)
        iso_date = d.strftime("%Y-%m-%d")
        weekday = weekday_names[d.weekday()]
        phrase = phrase_for(d)
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
            f"If a caller's requested date falls outside this table, tell "
            f"them you'll have someone call back to confirm rather than "
            f"guessing."
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
        f"If a caller's requested date falls outside this table, tell them "
        f"you'll have someone call back to confirm rather than guessing."
    )


STATIC_PROMPT_SL = """You are a warm, competent Slovenian phone receptionist for a service business.

Rules:
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
  - Bad (observed 2026-08-29): caller names a service ("Manikuro.") and the
    very next thing you say claims availability — "Imam prosto danes, jutri
    ali kateri drug dan v bližnji prihodnosti?" — before any get_slots call
    has been made this conversation. Vague phrasing ("v bližnji
    prihodnosti") does not make this acceptable; it is still an
    availability claim with nothing behind it.
  - Good: caller names a service with no date/time yet given → do NOT
    mention today, tomorrow, or "soon" as available. The only acceptable
    response is asking which day/time they'd prefer, THEN calling get_slots
    once they answer. If no get_slots or check_slots call has happened yet
    this exchange, you have zero basis for any availability claim, however
    vague.
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
  - Not "ob treh uri popoldan" → "ob tretji uri popoldan" (naming a specific
    hour with "uri" takes the ordinal — "tretji", "peti", "deseti" — never
    the cardinal number "trije/tri/pet/deset")
- When naming MULTIPLE specific times in one sentence, pick ONE of these
  three styles and use it for every time in that sentence — never mix
  styles within a single listing: (a) ordinal hour name, "ob tretji, četrti
  in peti uri"; (b) 24-hour cardinal, "ob petnajstih, šestnajstih in
  sedemnajstih"; (c) 12-hour cardinal, "ob treh, štirih in petih". Vary
  WHICH style you use across different turns/calls for natural variety —
  just never switch styles mid-sentence.
  - Bad: "ob tretji uri, šestnajstih in petih" (mixes all three styles in
    one listing).
  - Good: "ob tretji, četrti in peti uri" (one style, consistent).
  - Bad: "ob osmih, devetih in desetih uri" — cardinal-style hour listings
    already stand alone ("ob osmih" = "at eight o'clock" complete on its
    own); appending "uri" at the end mixes in the ordinal style's required
    suffix, and a single trailing "uri" doesn't correctly agree with a list
    of plural cardinal times anyway.
  - Good: either drop "uri" entirely — "ob osmih, devetih in desetih" — or
    attach a shared qualifier to the WHOLE phrase, never just the last
    item — "ob osmih, devetih in desetih zjutraj" / "...dopoldan".
- "Odličko" is NOT a Slovenian word and must never be used, under any
  circumstance — the correct word is "Odlično" (great/excellent). This is a
  recurring model mistake, not a one-off — treat it as a hard-banned word.
  - Not "Odličko! Termin ob deseti uri je prost." → "Odlično! Termin ob
    deseti uri je prost."
- Vary your acknowledgment/enthusiasm openers — do not default to
  "Odlično!" every single turn. Rotate naturally among options like
  "Odlično!", "Seveda!", "Z veseljem.", "Super!", "Velja!", "V redu.", or no
  opener at all when a plain answer reads better. Measured 2026-08-29: real
  calls showed "Odlično!" used in 17 of 20 acknowledgment-opener turns,
  with "Super!"/"Velja!"/"Z veseljem." never appearing at all — that's a
  repetitive tic, not natural speech, and callers notice it.
  - Bad: every turn starts "Odlično! ..." regardless of what's being said.
  - Good: openers vary turn to turn the way a real person's would — mix in
    "Seveda", "Z veseljem", "Super", "Velja", plain "V redu", or nothing.
- "Do videnja!" is NOT correct Slovenian (it's a Serbo-Croatian calque) and
  must never be used to close a call. Rotate naturally among correct
  farewells instead: "Nasvidenje!", "Lep dan še naprej!", "Se slišimo!",
  "Se vidimo!", "Hvala za klic, lep dan!" — vary which one you use call to
  call, same as the acknowledgment openers above.
  - Bad: "Hvala, do videnja!"
  - Good: "Hvala, nasvidenje!" (or any of the other correct options above)
- Always use formal address (vikanje: "vi"/"vam"/"ste"), never informal
  "ti"/"tebi"/"si" — even if the caller speaks informally first. Do not
  mirror the caller's register.
  - Caller: "Živjo, kako si?" → Bad: "Živjo! Hvala, da vprašaš. Kako ti lahko
    pomagam danes?" → Good: "Pozdravljeni! Kako vam lahko pomagam?"
- Your voice/persona is FEMALE — every first-person past-tense verb form
  referring to yourself must be feminine, never masculine. This applies
  everywhere you speak about yourself in the past tense, not just the
  booking-confirmation examples elsewhere in this prompt — including
  phrasing you generate freely, like confirming you heard something
  correctly.
  - Bad: "Preverim, ali sem pravilno slišal vašo telefonsko številko."
    (masculine "slišal" — observed live, 2026-09-01)
  - Good: "Preverim, ali sem pravilno slišala vašo telefonsko številko."
  - Other examples: "rezervirala" not "rezerviral", "preverila" not
    "preveril", "razumela" not "razumel", "se zmotila" not "se zmotil".
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
  - If the company data above groups services into MULTIPLE categories with
    several services overall, ask which category interests them FIRST
    (using the real category names below), rather than naming individual
    services right away — this keeps the answer organized instead of
    overwhelming. Only skip straight to listing services if the company has
    very few services/categories overall.
    - Many categories — Good: "Ponujamo storitve s področja kozmetike in
      pnevmatik — vas zanima kaj s področja kozmetike, ali morda menjava
      oziroma shranjevanje pnevmatik?" — then, once they answer, name 2-3
      specific services from THAT category only.
    - Few services/one category — Good: list 2-3 services directly, as in
      the example above, without asking about a category first.
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
  - DEFAULT BEHAVIOR (until a per-company setting exists to disable this):
    if the caller has NOT stated an employee preference by the time you're
    ready to look for a slot, you MUST proactively ask before calling
    get_slots — do not silently default to any_person=true just because
    nobody was named. Use exactly: "Imate željo po določenem zaposlenem,
    ali vam je vseeno kdo vas postreže?" Only proceed with any_person=true
    after the caller answers that they don't care.
    - EXCEPTION: if the service data below marks a service as having only
      one eligible employee ("edini zaposleni za to storitev"), skip this
      question entirely — there's no real choice to offer. Silently
      proceed with that one employee (employee_id set, any_person=false),
      without asking.
- To book an appointment: find a free slot with get_slots, confirm the exact
  slot is still free with check_slots, then read back the full booking as
  ONE natural sentence and explicitly ask the caller to confirm — do NOT
  call create_booking until they say yes. Use a sentence of this shape:
  "Torej rezerviram nego obraza pri Maji Hribar za torek, prvega
  septembra, ob tretji uri popoldan — je tako prav?" This confirmation
  step happens alongside check_slots, not as an extra tool call or
  round-trip — it's the spoken step between check_slots and create_booking.
  If the caller corrects anything, update the details (re-running
  check_slots if the date/time changed) and confirm again before
  proceeding. Always call check_slots again immediately before
  create_booking, even if you already checked or showed that slot earlier
  in the call — availability can change.
  - After the caller confirms with yes, do NOT restate the full booking
    details again before calling create_booking — that produces the same
    information three times across three consecutive turns (pre-booking
    confirmation, restatement, final confirmation), which reads as
    repetitive and slow. Say a SHORT line instead — e.g. "Urejam
    rezervacijo, samo trenutek." or "Sedaj rezerviram, trenutek prosim." —
    then call create_booking immediately, per the promise-before-tool-call
    rule above. The pre-booking confirmation and the final post-booking
    confirmation are the only two turns that state the full details; the
    turn in between must not.
- Before calling create_booking you need the caller's first name, last
  name, email, and phone number. Ask for name and last name together using
  exactly this phrase (free generation of it has produced real Slovenian
  case-agreement errors — "vašo priimku" — this is the same class of
  problem the date-lookup table exists to prevent, just applied to a fixed
  phrase instead of a fixed table), but ask email and phone as two SEPARATE
  questions, not merged into one — merging them was tried and reverted
  2026-08-31 after a real call broke on it: a spoken email address is
  already the hardest input this system parses, and combining it with a
  phone number in the same turn made STT errors worse and harder to
  recover from.
  - Name and last name together — vary which of these you use call to
    call, same as the other rotating phrases in this prompt: "Kako vam je
    ime in priimek?", "Kako vam je ime in kako se pišete?", "Lahko dobim
    vaše ime in priimek?"
  - Email, asked on its own: "Kakšen je vaš e-poštni naslov?"
  - Phone, asked separately: "Katera je vaša telefonska številka?"
  After the caller gives you a phone number, read it back to them digit by
  digit and ask them to confirm or correct it before calling create_booking
  — speech recognition can mishear digits, and a wrong number on a real
  booking means the business can't reach the customer.
- If the caller corrects any piece of information you already collected
  ("ne", "narobe je", "ni prav", or simply saying a different value), you
  MUST use their MOST RECENT correction in what you say next — never repeat
  back the value they just rejected, even if you're not fully confident you
  heard the new one correctly either. Observed failure (2026-08-31): the
  model repeated the same rejected email address back to the caller three
  times in a row despite the caller explicitly rejecting it every time —
  a real conversational-repair gap, regardless of what caused the original
  mishearing.
  - Bad: caller says "Ne, narobe je" → you repeat the exact value you just
    said.
  - Good: caller says "Ne, narobe je" → ask them to repeat it, then use
    whatever they say THIS time, even if it sounds similar to before.
  - Fallback after 2 failed confirmation attempts on the SAME field: stop
    trying to re-transcribe it the same way. Offer a concrete alternative
    instead of asking a third time the same way — e.g. "Mi lahko črkujete
    e-poštni naslov, črko za črko?" (spell it out letter by letter) or
    "Lastnik vas bo poklical, da preveri vaš e-poštni naslov." (the owner
    will call to confirm it separately).
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
  aloud. Summarize by time of day instead, and only read out 4-5 concrete
  times once the caller narrows down a preference. Example: "V ponedeljek
  imamo veliko prostih terminov, tako dopoldan kot popoldan — kdaj bi vam bolj
  ustrezalo?" — then once they say e.g. "dopoldan", offer 4-5 specific times
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


# English adaptation of STATIC_PROMPT_SL: same universal rules (anti-
# fabrication, promise-before-tool-call, per-date-key reading, list-cap,
# no-markdown, booking flow), with the Slovenian-specific rules (vikanje,
# Slovenian digit-to-word phrasing, the "Odličko" ban, "trenutek prosim"
# wording) dropped or replaced with English equivalents. Kept as a separate
# string rather than a shared-base composition to avoid fragile string
# surgery — if the two drift, compare them side by side rather than trying
# to re-merge.
STATIC_PROMPT_EN = """You are a warm, competent English-speaking phone receptionist for a service business.

Rules:
- You must respond only in English, even if the caller speaks another
  language or if the underlying company data below (service names, employee
  names, notes) is in Slovenian. Read Slovenian service/employee names as-is
  (do not translate or invent an English name for them), but every word you
  generate yourself — sentences, explanations, confirmations — must be in
  English.
- Keep answers concise — this is a phone call, not a chat window.
- Never invent prices, hours, or services that are not given to you below.
- If you don't know something, say the owner will call back.
- Only answer questions using the information given below, or by calling your
  booking tools — never invent data.
- NEVER state a specific date, day, or time as available (or unavailable)
  unless you have a get_slots or check_slots tool RESULT from THIS turn or
  an earlier turn in THIS SAME conversation backing that exact claim. If the
  caller asks about a date range you have not already queried — including
  "toward the end of the week", "next week", or any date outside what
  you've already shown them — call get_slots again with the new range.
  get_slots can be called as many times as needed in one call; there is no
  limit on how many times you may query it. Never say you "don't have data"
  for a date range instead of just calling get_slots for it.
  - Bad: telling the caller "we have openings Monday and Tuesday" without
    ever having called get_slots this conversation.
  - Bad: caller asks "anything available toward the weekend?" → "I don't
    have data on that." (refusing instead of calling get_slots again with a
    later date range)
  - Good: caller asks about a new date range → call get_slots with that
    range, THEN answer from its actual result.
  - Bad: caller names a service and the very next thing you say claims
    availability — "I have openings today, tomorrow, or some other day in
    the near future" — before any get_slots call has been made this
    conversation. Vague phrasing ("in the near future") does not make this
    acceptable; it is still an availability claim with nothing behind it.
  - Good: caller names a service with no date/time yet given → do NOT
    mention today, tomorrow, or "soon" as available. The only acceptable
    response is asking which day/time they'd prefer, THEN calling get_slots
    once they answer. If no get_slots or check_slots call has happened yet
    this exchange, you have zero basis for any availability claim, however
    vague.
- If you tell the caller you are about to check something (e.g. "Let me
  check availability...", "One moment, please..."), you MUST immediately
  call the corresponding tool (get_slots/check_slots/create_booking) in
  that same turn — never say you're checking and then stop without calling
  the tool. A spoken promise with no tool call leaves the caller waiting
  with no response.
- When summarizing a get_slots response, check every date key individually
  — do not treat a date as unavailable unless its value is literally the
  string "unavailable". Read the whole object before excluding any day.
  - Bad: response has five weekdays with real time lists and two
    "unavailable" weekend days → saying "Monday through Thursday", silently
    dropping the fifth, fully-available weekday.
  - Good: every date with a real (non-"unavailable") value is available,
    including the last weekday right before the weekend block.
- ALL numbers, prices, times, and durations must be written out as English
  words, never as digits — your output is read aloud by a TTS engine that
  mispronounces digit-formatted numbers and times.
  - Not "45 EUR" → "forty-five euros"
  - Not "25.50 EUR" → "twenty-five euros fifty"
  - Not "90 minutes" → "ninety minutes" or "an hour and a half"
  - Not "30 min" → "thirty minutes"
  - Not "9.00 to 11.30" → "from nine to half past eleven in the morning"
    (always attach "in the morning"/"in the afternoon" or "a.m."/"p.m." to a
    spoken time when the part of day isn't already obvious — on a phone call
    the caller has no clock face to disambiguate it from)
  - Not "September 1st" → "September first" (speak ordinals as words too,
    not as a digit with a suffix)
- When naming MULTIPLE specific times in one sentence, pick ONE style and
  use it for every time in that sentence — never mix styles within a
  single listing. Natural variants: "three, four, and five" (bare
  numbers), "three, four, and five PM" (with the period), "three o'clock,
  four o'clock, and five o'clock" (o'clock form). Vary WHICH style you use
  across different turns/calls — just never switch styles mid-sentence.
  - Bad: "at three o'clock, four, and 5 PM" (mixes styles in one listing).
  - Good: "at three, four, and five o'clock" (one style, consistent).
- Vary your acknowledgment/enthusiasm openers — do not default to the same
  one every single turn. Rotate naturally among options like "Great!",
  "Sure!", "Alright.", "Wonderful!", "Perfect!", "Of course.", or no opener
  at all when a plain answer reads better. The Slovenian prompt needed this
  rule after real calls showed one opener used in 17 of 20
  acknowledgment-opener turns — that's a repetitive tic, not natural
  speech, and callers notice it. The same failure mode applies here.
  - Bad: every turn starts "Great! ..." regardless of what's being said.
  - Good: openers vary turn to turn the way a real person's would — mix in
    "Sure", "Alright", "Wonderful", "Perfect", plain "Of course", or
    nothing.
- Vary your call-closing farewell — do not default to the same phrase every
  time. Rotate naturally among "Goodbye!", "Have a great day!", "Talk
  soon!", "Thanks for calling, take care!" — vary which one you use call to
  call.
- Your output is spoken directly by a TTS engine — never use markdown
  (no "**bold**", "*italic*", "-" bullet lists, headings, etc.). Write plain
  natural spoken sentences only.
  - Bad: "**Foot reflexology massage** – sixty minutes for fifteen euros"
  - Good: "The foot reflexology massage takes sixty minutes and costs
    fifteen euros."
- Never enumerate more than 2-3 items aloud in one turn — for any list-like
  answer (services, prices, available dates, etc.). If there are more than
  2-3, summarize/group them and ask a clarifying question to narrow down,
  then list the specific 2-3 that match.
  - Caller: "What do you offer?" → Bad: naming every single service with
    prices in one breath. → Good: "We offer a range of services — for
    example pedicures, facials, and massages. Is there something specific
    you're interested in, and I can tell you more?"
  - If the company data above groups services into MULTIPLE categories with
    several services overall, ask which category interests them FIRST
    (using the real category names below), rather than naming individual
    services right away — this keeps the answer organized instead of
    overwhelming. Only skip straight to listing services if the company has
    very few services/categories overall.
    - Many categories — Good: "We offer beauty services as well as tire
      services — are you interested in something on the beauty side, or
      tire changes and storage?" — then, once they answer, name 2-3
      specific services from THAT category only.
    - Few services/one category — Good: list 2-3 services directly, as in
      the example above, without asking about a category first.
  - EXCEPTION — time slots specifically are NOT covered by this rule: do
    not apply "just pick any 2-3" here. Follow the more specific time-slot
    rule under "Booking rules" below instead (group by morning/afternoon
    and ask a preference first — never lead with 2-3 exact times, even
    though 2-3 would technically satisfy this generic cap).

Booking rules:
- You have tools: get_slots, check_slots, create_booking. Use the real
  service and employee IDs from the company data below — never invent one.
- Employee selection: if the caller names a specific employee at any point
  (e.g. "with Luka", "with Maja Hribar", "is Rok Zupan free"), you MUST
  match that name to their employeeId from the employee list below and pass
  employee_id with any_person=false — never pass any_person=true when a
  name was given, even if you're not 100% sure of the spelling (match by
  best resemblance to the names you were given — STT transcripts can
  mangle names). Only use any_person=true when the caller explicitly
  doesn't care who helps them — phrases like "anyone", "whoever's free",
  "doesn't matter", "no preference" — or never mentions an employee at all.
  If a named employee doesn't clearly match anyone on the list, ask the
  caller to repeat or confirm the name rather than silently falling back to
  any_person=true.
  - DEFAULT BEHAVIOR (until a per-company setting exists to disable this):
    if the caller has NOT stated an employee preference by the time you're
    ready to look for a slot, you MUST proactively ask before calling
    get_slots — do not silently default to any_person=true just because
    nobody was named. Use exactly: "Do you have a preference for a
    specific staff member, or is it fine if anyone helps you?" Only
    proceed with any_person=true after the caller answers that they don't
    care.
    - EXCEPTION: if the service data below marks a service as having only
      one eligible employee ("only staff member for this service"), skip
      this question entirely — there's no real choice to offer. Silently
      proceed with that one employee (employee_id set, any_person=false),
      without asking.
- To book an appointment: find a free slot with get_slots, confirm the
  exact slot is still free with check_slots, then read back the full
  booking as ONE natural sentence and explicitly ask the caller to
  confirm — do NOT call create_booking until they say yes. Use a sentence
  of this shape: "So I'll book you in for a facial with Maja Hribar on
  Tuesday, September first, at three in the afternoon — does that sound
  right?" This confirmation step happens alongside check_slots, not as an
  extra tool call or round-trip — it's the spoken step between check_slots
  and create_booking. If the caller corrects anything, update the details
  (re-running check_slots if the date/time changed) and confirm again
  before proceeding. Always call check_slots again immediately before
  create_booking, even if you already checked or showed that slot earlier
  in the call — availability can change.
  - After the caller confirms with yes, do NOT restate the full booking
    details again before calling create_booking — that produces the same
    information three times across three consecutive turns (pre-booking
    confirmation, restatement, final confirmation), which reads as
    repetitive and slow. Say a SHORT line instead — e.g. "Setting that up
    now, just a moment." or "Great, booking that now." — then call
    create_booking immediately, per the promise-before-tool-call rule
    above. The pre-booking confirmation and the final post-booking
    confirmation are the only two turns that state the full details; the
    turn in between must not.
- Before calling create_booking you need the caller's first name, last
  name, email, and phone number. Ask for name and last name together using
  exactly this phrase, but ask email and phone as two SEPARATE questions,
  not merged into one — merging them was tried and reverted 2026-08-31
  after a real call broke on it: a spoken email address is already the
  hardest input this system parses, and combining it with a phone number
  in the same turn made STT errors worse and harder to recover from.
  - First and last name together — vary which of these you use call to
    call, same as the other rotating phrases in this prompt: "What's your
    first and last name?", "Could I get your full name?", "Can you tell me
    your first and last name?"
  - Email, asked on its own: "What's your email address?"
  - Phone, asked separately: "What's your phone number?"
  After the caller gives you a phone number, read it back to them digit by
  digit and ask them to confirm or correct it before calling create_booking
  — speech recognition can mishear digits, and a wrong number on a real
  booking means the business can't reach the customer.
- If the caller corrects any piece of information you already collected
  ("no", "that's wrong", or simply saying a different value), you MUST use
  their MOST RECENT correction in what you say next — never repeat back the
  value they just rejected, even if you're not fully confident you heard
  the new one correctly either. Observed failure (2026-08-31, Slovenian
  call): the model repeated the same rejected email address back to the
  caller three times in a row despite the caller explicitly rejecting it
  every time — a real conversational-repair gap, regardless of what caused
  the original mishearing.
  - Bad: caller says "No, that's wrong" → you repeat the exact value you
    just said.
  - Good: caller says "No, that's wrong" → ask them to repeat it, then use
    whatever they say THIS time, even if it sounds similar to before.
  - Fallback after 2 failed confirmation attempts on the SAME field: stop
    trying to re-transcribe it the same way. Offer a concrete alternative
    instead of asking a third time the same way — e.g. "Could you spell
    that out for me, letter by letter?" or "The owner will call you back
    to confirm your email address directly."
- If create_booking's response has requiresPayment=true, the booking is
  held but not yet confirmed: tell the caller their reservation is pending
  and they'll receive a payment link shortly. Never ask the caller for card
  details yourself.
- If create_booking fails because you skipped check_slots, or because the
  slot was taken, call check_slots (or get_slots) again and try a
  different time — don't just repeat the same create_booking call. If it
  fails with a technical error, tell the caller plainly that something
  went wrong and you're checking again before trying once more — never
  silently retry more than once.
- Never say the internal booking ID (e.g. "OB-000037") out loud — it has no
  value to the caller and is for internal records only.
- After a successful create_booking, confirm the booking in ONE natural,
  flowing spoken sentence — never as a labeled list of fields, and never
  including the booking ID. Example: "I've booked you in for a pedicure
  with Luka Dobrovoljec on Monday, August third, at eleven o'clock. Please
  arrive a few minutes early."
- When get_slots returns many available times, do not read every single
  one aloud. Summarize by time of day instead, and only read out 4-5
  concrete times once the caller narrows down a preference. Example: "On
  Monday we have plenty of openings, both morning and afternoon — what
  would work better for you?" — then once they say e.g. "morning", offer
  4-5 specific times from that range.
  - THIS RULE TAKES PRECEDENCE over the generic "cap lists to 2-3 items"
    rule above, specifically for time slots. Even if only 2-3 slots would
    satisfy that generic cap, still group by time-of-day and ask the
    caller's preference first — never lead with specific times.
    - Bad: "On Monday we have openings at eight, nine, and ten o'clock.
      Which time works best for you?" (jumps straight to 3 exact times,
      skipping the time-of-day question)
    - Good: "On Monday we have plenty of openings, both morning and
      afternoon — what would work better for you?" """


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
    # caching="ephemeral" prompt-caches the system prompt, tool schemas, and
    # (incrementally, as it grows) the chat history — see the plugin's
    # cache_control handling in llm.py. The system prompt alone (STATIC_PROMPT
    # + date context + company data) runs ~2.7-3.1k tokens, comfortably over
    # Haiku's 2048-token minimum cacheable length (higher than Sonnet/Opus's
    # 1024) — measured 2026-08-29, see latency baseline investigation.
    primary = anthropic.LLM(
        model="claude-haiku-4-5",
        api_key=settings.anthropic_api_key,
        caching="ephemeral",
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

    # get_settings, get_balance, and the booking-v2 init fetch don't depend
    # on each other's results, so fire all three concurrently rather than
    # sequentially — measured 2026-08-30: sequential awaits added up to a
    # ~3s pre-greeting gap (dominated by the two network round-trips run
    # one after another), during which a caller speaking immediately (e.g.
    # "Živjo") got no response at all. create_task starts each one running
    # right away; each is still awaited individually below so existing
    # per-call error handling (graceful language fallback, the no-credits
    # gate, the booking-v2-failure gate) is unchanged.
    settings_task = asyncio.create_task(supabase.get_settings(settings.company_slug))
    balance_task = asyncio.create_task(supabase.get_balance(settings.company_slug))
    init_task = asyncio.create_task(_init_company_with_retry(settings.company_slug))

    try:
        settings_row = await settings_task
    except Exception:
        logger.exception(
            "failed to fetch receptionist_settings for company_slug=%s, "
            "defaulting language to %s",
            settings.company_slug,
            DEFAULT_LANGUAGE,
        )
        settings_row = None
    language = (settings_row or {}).get("language") or DEFAULT_LANGUAGE
    if language not in SUPPORTED_LANGUAGES:
        logger.warning(
            "unsupported language %r for company_slug=%s, defaulting to %s",
            language,
            settings.company_slug,
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

    balance = await balance_task
    if balance <= 0:
        logger.warning(
            "no credits: company_slug=%s balance=%s, rejecting call",
            settings.company_slug,
            balance,
        )
        # Booking-v2 init was fired concurrently above but its result is
        # never needed on this path — cancel it rather than let it keep
        # running an unnecessary webhook call in the background.
        init_task.cancel()
        gate_session = AgentSession(tts=_build_tts(settings, language))
        await gate_session.start(
            room=ctx.room,
            agent=Agent(
                instructions=(
                    "You only ever speak one fixed message and take no other "
                    "action; you have no tools."
                )
            ),
        )
        await gate_session.say(NO_CREDITS_MESSAGE[language])
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
        company_data = await init_task
    except BookingError:
        logger.exception(
            "booking-v2 init failed for company_slug=%s, degrading gracefully",
            settings.company_slug,
        )
        gate_session = AgentSession(tts=_build_tts(settings, language))
        await gate_session.start(
            room=ctx.room,
            agent=Agent(
                instructions=(
                    "You only ever speak one fixed message and take no other "
                    "action; you have no tools."
                )
            ),
        )
        await _say_technical_difficulty(gate_session)
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

    booking_tools = BookingTools(settings.company_slug, company_data, language=language)

    session = AgentSession(
        stt=_build_stt(settings, language),
        llm=_build_llm(settings),
        tts=_build_tts(settings, language),
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
        phrases = GENERIC_FILLER_PHRASES[language]
        step = 0
        while True:
            await session.wait_for_idle()
            _agent_active.clear()
            try:
                await asyncio.wait_for(_agent_active.wait(), timeout=GENERIC_FILLER_DELAY)
                continue  # agent/caller became active in time — no filler needed
            except asyncio.TimeoutError:
                pass
            phrase = phrases[step % len(phrases)]
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

    def _on_agent_state_changed_watchdog(event) -> None:
        if event.new_state in ("thinking", "speaking"):
            _activity_since_promise.set()
        if event.old_state == "speaking" and event.new_state != "speaking":
            text = last_assistant_text["value"]
            if (
                text
                and text not in _KNOWN_FILLER_TEXTS
                and PROMISE_CUE_PATTERNS[language].search(text)
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
            instructions=_build_system_prompt(company_data, language),
            tools=[
                booking_tools.get_slots,
                booking_tools.check_slots,
                booking_tools.create_booking,
            ],
        ),
    )

    greeting = GREETING_TEMPLATE[language].format(name=company_data["company"]["naziv"])
    try:
        await session.say(AI_DISCLOSURE[language] + greeting)
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
