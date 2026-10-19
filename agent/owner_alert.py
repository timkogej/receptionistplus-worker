"""The owner's SMS alert after a call that needs a follow-up.

One short text per such call (callback request, message, unfinished
booking, booking problem), sent through the same n8n SMS flow as the
caller's confirmation (kind "owner_alert"), so it counts against the
company's SMS quota like any other. Text in the owner's language: Slovenian,
or English for a company whose language isn't Slovenian.

No livekit imports, so tests use it directly.
"""

from __future__ import annotations

from datetime import datetime

from agent.dates import ENGLISH_WEEKDAYS, SLOVENIAN_WEEKDAYS_ACCUSATIVE
from agent.sms_client import normalize_sms

_LEAD = {
    "sl": {
        "callback": "prosi za povratni klic",
        "message": "sporocilo",
        "unfinished": "nedokoncana rezervacija",
        "trouble": "preverite rezervacijo",
    },
    "en": {
        "callback": "callback requested",
        "message": "message",
        "unfinished": "unfinished booking",
        "trouble": "check a booking",
    },
}
_WITHHELD = {"sl": "skrita stevilka", "en": "number withheld"}


def owner_language(company_language: str | None) -> str:
    return "sl" if (company_language or "sl") == "sl" else "en"


def unfinished_booking_message(interest: dict, company_data: dict, language: str) -> str:
    """The note for a caller who looked for (or picked) a time but hung up
    before the booking was made: what they wanted, for the owner's call
    back. Also lands in the call log like any message."""
    names = {str(s.get("id")): s.get("naziv") for s in company_data.get("services", [])}
    services = ", ".join(n for n in (names.get(str(i)) for i in interest.get("service_ids") or []) if n)
    when = ""
    if interest.get("date"):
        d = datetime.strptime(interest["date"], "%Y-%m-%d").date()
        if language == "sl":
            when = f"za {SLOVENIAN_WEEKDAYS_ACCUSATIVE[d.weekday()]}, {d.day}. {d.month}."
            if interest.get("time"):
                when += f" ob {interest['time']}"
        else:
            when = f"for {ENGLISH_WEEKDAYS[d.weekday()]} {d.day}/{d.month}"
            if interest.get("time"):
                when += f" at {interest['time']}"
    if language == "sl":
        what = " ".join(p for p in (services or "termin", when) if p)
        return f"Klicatelj je iskal termin ({what}), a je odložil pred rezervacijo. Pokličite ga nazaj."
    what = " ".join(p for p in (services or "an appointment", when) if p)
    return f"The caller looked for {what} but hung up before booking. Please call them back."


def owner_alert_text(
    *,
    message: str | None,
    callback_requested: bool,
    kind: str | None,
    caller_phone: str | None,
    language: str,
) -> str:
    lang = language if language in _LEAD else "en"
    kind = kind or ("callback" if callback_requested else "message")
    lead = _LEAD[lang].get(kind, _LEAD[lang]["message"])
    who = caller_phone or _WITHHELD[lang]
    body = (message or "").strip()
    return normalize_sms(f"Receptionist+: {lead} ({who}). {body}".strip())
