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
import time
from datetime import datetime

import requests

from agent.booking_client import _HMAC_SECRET_ENV, _sign
from agent.dates import WEEKDAYS

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
    return _CONFIRMATION_TEMPLATES[lang].format(
        company=company_name,
        services=services,
        weekday=WEEKDAYS[lang][d.weekday()],
        d=d.day,
        m=d.month,
        y=d.year,
        time=time_hhmm,
    )


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
