"""Client for the n8n booking-v2 webhook.

Real per-company data and booking actions, replacing the Phase 1 hardcoded
`test_company.py`. The webhook's contract was reverse-engineered from the live
n8n workflow (action names, field names, and response shapes) since no written
spec exists yet — see the "init"/"slots"/"check-slots"/"create" handlers below.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time

import requests

logger = logging.getLogger("receptionistplus-worker.booking")

BOOKING_WEBHOOK_URL = "https://n8n.jedroplus.com/webhook/booking-v2"

# Phase 5 Part A prep: HMAC-signs requests once the n8n side verifies them.
# NOT ACTIVE YET — BOOKING_V2_HMAC_SECRET is intentionally unset in every
# environment right now. n8n is being rolled out in log-only (non-enforcing)
# mode first, to surface any callers of booking-v2 other than this worker
# before signatures become mandatory. Do not set this env var until that
# rollout is confirmed complete and we're ready to flip signing on here too.
# The n8n Code node rejects timestamps more than 300s old — keep in sync.
_HMAC_SECRET_ENV = "BOOKING_V2_HMAC_SECRET"


class BookingError(RuntimeError):
    """Raised when the booking webhook is unreachable or returns malformed JSON."""


class BookingTimeoutError(BookingError):
    """Raised when the webhook call times out client-side.

    This is deliberately NOT treated as "the action definitely failed": the
    "create" action's workflow does real work (insert the appointment) early,
    then a long tail (confirmation email via an LLM node, SMS queueing, ...)
    that can push total latency well past our timeout. A timeout here means
    "unknown outcome," not "nothing happened" — callers of create_booking
    must not treat it the same as a clean connection failure.
    """


def _sign(raw_body: bytes, *, secret: str, timestamp: str) -> str:
    """HMAC-SHA256 over "<timestamp>." + raw_body, matching the n8n Code node."""
    message = f"{timestamp}.".encode() + raw_body
    return hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()


def _post(payload: dict, *, timeout: float) -> dict:
    logger.info("booking webhook request: %s", payload)

    secret = os.environ.get(_HMAC_SECRET_ENV)
    if secret:
        # Compact, deterministic serialization — the exact bytes we sign are
        # the exact bytes sent, so the n8n side never has to reproduce our
        # JSON serialization to verify the signature (it just HMACs the raw
        # body it received via the Webhook node's "Raw Body" option).
        raw_body = json.dumps(payload, separators=(",", ":")).encode()
        timestamp = str(int(time.time()))
        signature = _sign(raw_body, secret=secret, timestamp=timestamp)
        response = requests.post(
            BOOKING_WEBHOOK_URL,
            data=raw_body,
            headers={
                "Content-Type": "application/json",
                "X-Signature": signature,
                "X-Timestamp": timestamp,
            },
            timeout=timeout,
        )
    else:
        response = requests.post(BOOKING_WEBHOOK_URL, json=payload, timeout=timeout)

    response.raise_for_status()
    data = response.json()
    logger.info("booking webhook response: %s", data)
    return data


async def call_booking(payload: dict, *, timeout: float = 15) -> dict:
    try:
        return await asyncio.to_thread(_post, payload, timeout=timeout)
    except requests.Timeout as exc:
        raise BookingTimeoutError(f"booking webhook call timed out: {exc}") from exc
    except requests.RequestException as exc:
        raise BookingError(f"booking webhook call failed: {exc}") from exc


async def init_company(company_slug: str) -> dict:
    """Fetch a company's services/employees/settings via the "init" action."""
    data = await call_booking({"action": "init", "companySlug": company_slug})
    if "error" in data or "company" not in data:
        raise BookingError(f"init failed for companySlug={company_slug!r}: {data}")
    return data


async def get_slots(
    *,
    company_slug: str,
    service_ids: list[str],
    start_date: str,
    end_date: str,
    employee_id: str | None,
    any_person: bool,
    eligible_employee_ids: list[str] | None,
) -> dict:
    """Call the "slots" action: which dates/times are available over a range."""
    payload: dict = {
        "action": "slots",
        "companySlug": company_slug,
        "serviceIds": service_ids,
        "startDate": start_date,
        "endDate": end_date,
        "any_person": any_person,
    }
    if any_person:
        payload["eligibleEmployeeIds"] = eligible_employee_ids or []
    else:
        payload["employeeId"] = employee_id
    return await call_booking(payload)


async def check_slots(
    *,
    company_slug: str,
    service_ids: list[str],
    date: str,
    time: str,
    employee_id: str | None,
    any_person: bool,
    eligible_employee_ids: list[str] | None,
) -> dict:
    """Call the "check-slots" action: is this exact date/time still free right now."""
    payload: dict = {
        "action": "check-slots",
        "companySlug": company_slug,
        "serviceIds": service_ids,
        "date": date,
        "time": time,
        "any_person": any_person,
    }
    if any_person:
        payload["eligibleEmployeeIds"] = eligible_employee_ids or []
    else:
        payload["employeeId"] = employee_id
    return await call_booking(payload)


async def create_booking(
    *,
    company_slug: str,
    service_ids: list[str],
    date: str,
    time: str,
    first_name: str,
    last_name: str,
    email: str,
    phone: str,
    employee_id: str | None,
    any_person: bool,
    eligible_employee_ids: list[str] | None,
    notes: str,
) -> dict:
    """Call the "create" action: actually reserve the appointment."""
    payload: dict = {
        "action": "create",
        "companySlug": company_slug,
        "serviceIds": service_ids,
        "date": date,
        "time": time,
        "firstName": first_name,
        "lastName": last_name,
        "email": email,
        "phone": phone,
        "notes": notes,
        "privacy_consent": True,
        "any_person": any_person,
        # Additive field, not part of the original contract — safe for the
        # workflow to ignore if it doesn't already store it.
        "source": "receptionist",
    }
    if any_person:
        payload["eligibleEmployeeIds"] = eligible_employee_ids or []
    else:
        payload["employeeId"] = employee_id

    # Longer timeout than other actions: the workflow's tail (confirmation
    # email via an LLM node, SMS queueing) can push latency well past what
    # slots/check-slots ever take, even though the appointment itself is
    # usually inserted within a couple of seconds.
    return await call_booking(payload, timeout=45)
