"""SMS to the caller through the n8n workflow "Receptionist+ – SMS stranki"
(webhook receptionist-sms).

That workflow does what every Jedro+ SMS path does — subscription and quota
check, quota reservation, message_outbox row, sending via the shared
sms-send sub-workflow (BulkGate), finalising usage — so SMS sent from a call
count against the company's SMS quota like any other. Requests are signed
with the same HMAC as booking-v2, and unlike booking-v2 the signature is
enforced: an unsigned request is rejected.

Two kinds:
- confirmation: after create_booking; text built here (booking_confirmation_text).
- manage_link: a one-hour portal login link for cancelling/rescheduling;
  the workflow mints the token. Needs the customer's email, since portal
  login is keyed by email.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import unicodedata
from datetime import datetime

import requests

from agent.booking_client import _HMAC_SECRET_ENV, _sign
from agent.dates import WEEKDAYS
from agent.languages import JEDROPLUS_LANGUAGE_CODES, TO_JEDROPLUS_CODE

logger = logging.getLogger("receptionistplus-worker.sms")

SMS_WEBHOOK_URL = "https://n8n.jedroplus.com/webhook/receptionist-sms"

_CONFIRMATION_TEMPLATES = {
    "sl": "{company}: potrjujemo vaš termin — {services}, {weekday}, {d}. {m}. {y}, ob {time}. Lep pozdrav!",
    "en": "{company}: your booking is confirmed — {services}, {weekday} {d}/{m}/{y} at {time}. See you then!",
    "de": "{company}: Ihr Termin ist bestätigt — {services}, {weekday}, {d}.{m}.{y}, um {time} Uhr. Bis bald!",
    "hr": "{company}: potvrđujemo vaš termin — {services}, {weekday}, {d}. {m}. {y}., u {time}. Lijep pozdrav!",
    "it": "{company}: confermiamo il suo appuntamento — {services}, {weekday} {d}/{m}/{y} alle {time}. A presto!",
}
_DEFAULT_SERVICE_WORD = {"sl": "termin", "en": "appointment", "de": "Termin", "hr": "termin", "it": "appuntamento"}


def booking_confirmation_text(
    *,
    company_name: str,
    service_names: list[str],
    date: str,
    time_hhmm: str,
    language: str,
) -> str:
    """Short SMS text in the call's language. Digits are fine here — this is
    read, not spoken."""
    lang = language if language in _CONFIRMATION_TEMPLATES else "en"
    d = datetime.strptime(date, "%Y-%m-%d").date()
    services = ", ".join(n for n in service_names if n) or _DEFAULT_SERVICE_WORD[lang]
    # Same SMS rules as the company template (normalize_sms, defined below).
    return normalize_sms(_CONFIRMATION_TEMPLATES[lang].format(
        company=company_name,
        services=services,
        weekday=WEEKDAYS[lang][d.weekday()],
        d=d.day,
        m=d.month,
        y=d.year,
        time=time_hhmm,
    ))


# --- The company's own confirmation template ("Podatki podjetij") ----------
# Same template, placeholders and SMS rules as the Jedro+ booking
# confirmation (n8n "JEDROPLUS - Vpisovanje podatkov", node "Render SMS iz
# predloge1"), so a caller who booked by phone gets the text the company
# wrote. lastna_predloga_potrdilo is in the company's language;
# predloge_potrdilo_prevodi holds optional versions per Jedro+ language code
# ("eng", "de", "it", "hr").

SMS_MAX_CHARS = 160
_PLACEHOLDER = re.compile(r"\{\{\s*([^}]+?)\s*\}\}")
_LINK = re.compile(r"[a-z0-9][a-z0-9.-]*\.[a-z]{2,}/\S*", re.IGNORECASE)
# Not in the GSM alphabet: one of these turns the whole SMS into UCS-2,
# which halves how much fits in one message.
_NON_GSM = {"\u2014": "-", "\u2013": "-", "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"', "\u2026": "..."}


def _strip_diacritics(text: str) -> str:
    for a, b in _NON_GSM.items():
        text = text.replace(a, b)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.replace("đ", "d").replace("Đ", "D")


def _clamp_keep_link(text: str, limit: int = SMS_MAX_CHARS) -> str:
    """Cut to the limit at a word boundary, keeping a trailing link whole:
    the link is usually last and usually the point of the message."""
    if len(text) <= limit:
        return text
    m = _LINK.search(text)
    if m:
        link = m.group(0)
        rest = (text[: m.start()] + text[m.end():]).strip()
        room = limit - len(link) - 1
        if room >= 20:
            head = rest[:room].rstrip()
            end = max(head.rfind("."), head.rfind("!"), head.rfind("?"))
            if end >= 20:
                head = head[: end + 1]
            elif " " in head:
                head = head[: head.rfind(" ")]
            return f"{head.strip()} {link}".strip()
    head = text[:limit].rstrip()
    return head[: head.rfind(" ")].strip() if " " in head else head


def normalize_sms(text: str) -> str:
    """One line, no diacritics or non-GSM punctuation, at most 160 chars."""
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return _clamp_keep_link(_strip_diacritics(text))


def _translations(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            value = json.loads(raw)
        except ValueError:
            return {}
        return value if isinstance(value, dict) else {}
    return {}


def choose_confirmation_template(company_row: dict | None, language: str) -> str | None:
    """The company's template for this call's language, or None for the
    built-in text. Only when the company chose its own template ("LP"); in
    AI mode Jedro+ writes the text with a model, which the call doesn't."""
    row = company_row or {}
    if str(row.get("sms_type_potrdilo") or "AI").strip().upper() != "LP":
        return None
    company_lang = JEDROPLUS_LANGUAGE_CODES.get(str(row.get("language") or "slo").strip().lower(), "sl")
    if language == company_lang:
        template = str(row.get("lastna_predloga_potrdilo") or "").strip()
    else:
        code = TO_JEDROPLUS_CODE.get(language, "")
        template = str(_translations(row.get("predloge_potrdilo_prevodi")).get(code) or "").strip()
    # No version in the caller's language: the built-in text in it beats
    # the company's text in a language the caller didn't speak.
    return template or None


def render_confirmation_template(
    template: str,
    *,
    company_row: dict | None,
    service_names: list[str],
    date: str,
    time_hhmm: str,
    first_name: str = "",
    last_name: str = "",
    employee_name: str = "",
) -> str:
    row = company_row or {}
    d = datetime.strptime(date, "%Y-%m-%d").date()
    link = re.sub(r"^https?://", "", str(row.get("appt_management_link") or "").strip(), flags=re.I)
    company = str(row.get("Naziv Podjetja") or "").strip()
    phone = str(row.get("Kontaktni telefon") or "").strip()
    values = {
        "datum": f"{d.day:02d}.{d.month:02d}.{d.year}",
        "cas": time_hhmm,
        "čas": time_hhmm,
        "Čas": time_hhmm,
        "storitev": ", ".join(n for n in service_names if n),
        "ime_izvajalca": employee_name,
        "ime": first_name,
        "priimek": last_name,
        "ime_podjetja": company,
        "naslov": str(row.get("Naslov podjetja") or "").strip(),
        "telefon_podjetja": phone,
        "email_podjetja": str(row.get("Kontaktni_email") or row.get("Kontaktni email") or "").strip(),
        "povezava_prenarocanje": link,
        "leto": str(datetime.now().year),
        "salon": company,
        "telefon": phone,
    }
    filled = _PLACEHOLDER.sub(lambda m: str(values.get(m.group(1).strip(), "")), template)
    return normalize_sms(filled)


def _post(payload: dict, timeout: float) -> dict:
    secret = os.environ.get(_HMAC_SECRET_ENV)
    if not secret:
        return {"success": False, "error": "hmac_secret_missing_in_worker"}
    raw = json.dumps(payload, separators=(",", ":")).encode()
    ts = str(int(time.time()))
    response = requests.post(
        SMS_WEBHOOK_URL,
        data=raw,
        headers={
            "Content-Type": "application/json",
            "X-Signature": _sign(raw, secret=secret, timestamp=ts),
            "X-Timestamp": ts,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


async def send_sms(
    *,
    kind: str,
    company_slug: str,
    phone: str,
    text: str = "",
    email: str = "",
    termin_id: str | None = None,
    call_id: str | None = None,
    language: str = "sl",
    timeout: float = 20.0,
) -> dict:
    """{"success": bool, "error": str?}. Never raises: an SMS that doesn't go
    out must not break the call."""
    payload = {
        "kind": kind,
        "companySlug": company_slug,
        "phone": phone,
        "text": text,
        "email": email,
        "terminId": termin_id,
        "callId": call_id,
        # Lines the workflow adds itself (portal link text) follow it.
        "language": language,
    }
    try:
        result = await asyncio.to_thread(_post, payload, timeout)
    except Exception as exc:
        logger.exception("SMS request failed (kind=%s)", kind)
        return {"success": False, "error": f"request_failed: {exc}"}
    if not result.get("success"):
        logger.warning("SMS not sent (kind=%s): %s", kind, result.get("error"))
    else:
        logger.info("SMS sent (kind=%s)", kind)
    return result
