"""Pure call-level business rules: which company a call is for, how long it
may run, and how it is billed.

Kept free of livekit imports (like static_prompt.py) so tests/ can exercise
it with the stdlib alone. worker.py wires these into the live call.
"""

from __future__ import annotations

import json
import math
import re

# Calls shorter than this are treated as accidental hangups/misdials, not
# billable conversations.
ABANDONED_CALL_THRESHOLD_SEC = 10

# Hard ceiling on one call, whatever the balance. Before this there was no
# limit at all: a caller who never hung up (or a stuck line) ran until the
# SIP side gave up, billing every minute. A real booking call runs 2-5 min.
MAX_CALL_SEC = 15 * 60

# How far a call may run past the credits the company had when it started.
# Credits are only checked at the start of a call, so without a cap a
# company with 1 credit left could hold a 15-minute call. The grace keeps a
# caller with a nearly empty balance from being cut off mid-sentence the
# moment the balance would cross zero.
CREDIT_OVERRUN_GRACE_SEC = 60

# Never cut a call shorter than this on credits alone: below it, the caller
# would hear the disclosure and greeting and then be dropped.
MIN_CALL_SEC = 120

# How long before the limit the agent starts wrapping up (spoken notice).
WRAP_UP_NOTICE_SEC = 30

_E164_RE = re.compile(r"^\+[1-9]\d{6,14}$")


def normalize_phone(raw: str | None, default_country_code: str = "386") -> str | None:
    """Best-effort E.164 for a SIP number ("+38640123456").

    Telnyx/LiveKit normally already send E.164; this also accepts the forms
    that show up in SIP headers and logs: "sip:+386...@host", "00386...",
    and a national "040 123 456" (assumed Slovenian). Returns None for
    anything that doesn't end up looking like a phone number, e.g.
    "anonymous" for a withheld caller ID.
    """
    if not raw:
        return None
    s = raw.strip()
    if s.lower().startswith(("sip:", "tel:")):
        s = s.split(":", 1)[1]
    s = s.split("@", 1)[0]
    s = s.split(";", 1)[0]
    plus = s.startswith("+")
    digits = re.sub(r"\D", "", s)
    if not digits:
        return None
    if plus:
        candidate = "+" + digits
    elif digits.startswith("00"):
        candidate = "+" + digits[2:]
    elif digits.startswith("0"):
        candidate = "+" + default_country_code + digits[1:]
    else:
        candidate = "+" + digits
    return candidate if _E164_RE.match(candidate) else None


def company_slug_from_job_metadata(metadata: str | None) -> str | None:
    """A company_slug set on the dispatch rule's agent metadata, if any.

    Lets a LiveKit dispatch rule pin a number to a company without a
    database row: metadata '{"company_slug": "salon-x"}'. Anything that
    isn't a JSON object with a non-empty string there is ignored.
    """
    if not metadata:
        return None
    try:
        data = json.loads(metadata)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    slug = data.get("company_slug")
    if isinstance(slug, str) and slug.strip():
        return slug.strip()
    return None


def call_time_limit_sec(balance_credits: float) -> int:
    """Seconds this call may run, given the balance it started with.

    One credit is one minute (see bill_call). The limit is the balance plus a
    grace period, floored at MIN_CALL_SEC and capped at MAX_CALL_SEC.
    """
    by_credits = int(max(0.0, balance_credits) * 60) + CREDIT_OVERRUN_GRACE_SEC
    return max(MIN_CALL_SEC, min(MAX_CALL_SEC, by_credits))


def bill_call(
    duration_sec: int,
    *,
    created_termin_id: str | None,
    abusive: bool = False,
    transferred: bool = False,
    message_taken: bool = False,
) -> tuple[float, str]:
    """(billed_credits, outcome) for a finished call.

    One credit per minute, to two decimals. Outcome precedence: a call the
    agent ended for abuse is reported as such whatever else happened (a
    booking made before that is still kept in created_termin_id); then a
    booking; then a transfer to a person; then a message/callback request;
    otherwise info_only. Calls
    under ABANDONED_CALL_THRESHOLD_SEC are free and reported as abandoned.
    Billing never depends on the outcome — only on how long the line was
    open — so the agent's own judgement (e.g. calling someone abusive) can't
    change what a company pays.
    """
    if duration_sec < ABANDONED_CALL_THRESHOLD_SEC:
        return 0.0, "abandoned"
    billed = round(duration_sec / 60.0, 2)
    if abusive:
        return billed, "ended_abusive"
    if created_termin_id:
        return billed, "booked"
    if transferred:
        return billed, "transferred"
    if message_taken:
        return billed, "message_taken"
    return billed, "info_only"


def duration_seconds(started_at, ended_at) -> int:
    return max(0, math.ceil((ended_at - started_at).total_seconds()))
