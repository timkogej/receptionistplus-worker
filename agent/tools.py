"""Booking tools exposed to the LLM: get_slots, check_slots, create_booking.

These wrap the booking-v2 webhook (see booking_client.py). Two pieces of
business logic live here rather than only in the prompt, so they can't be
skipped by an LLM that "forgets" an instruction:

- check_slots must be called immediately before create_booking for the exact
  same service/employee/date/time, or create_booking refuses and tells the
  model to check first. This prevents the double-booking race where two
  callers are quoted the same free slot.
- if the company doesn't support `multiple_services_online`, create_booking
  refuses more than one service per call (the webhook itself doesn't enforce
  this).

"any employee" handling: the caller only ever says whether they care who does
the work (`any_person`); which employees are actually eligible for the
selected service(s) is computed here from the company's `init` data, not left
to the LLM to reconstruct. If a service has exactly one eligible employee,
`_resolve_person` pins the booking to that employee (`any_person=False`)
whatever the LLM passed — see its docstring.

Known upstream caveat (booking-v2 workflow, not fixable from here): a create
call can report failure to us (e.g. a downstream notification-insert error)
even though the appointment row itself was already committed. Our
check-slots-immediately-before-create gate below is what actually protects
against a blind retry re-booking the *same* slot — after any create_booking
attempt (success or failure) `_last_check_key` is cleared, so a retry is
forced to re-run check_slots, which will see the slot as taken by the
already-committed row. It cannot undo an upstream partial success (e.g. an
orphaned appointment row left behind if a later step failed) — that requires
a workflow-level fix (transactional writes, ID generation via a real
sequence instead of max()+1).

Filler speech for create_booking: since a normal (non-timeout) create call can
still take several seconds — the workflow's confirmation-email/SMS tail runs
before responding — create_booking wraps the webhook call in
`context.with_filler(...)`, the framework's built-in mechanism for exactly
this: if the session has been continuously idle for 4.5s, it speaks a filler
line ("Super, urejam rezervacijo, samo trenutek prosim.") without blocking the underlying
call. This does not change the 45s timeout — it just fills dead air.

Timeout handling for create_booking specifically: a client-side timeout on
"create" does NOT mean the booking failed — the workflow inserts the
appointment early, then does a long tail (confirmation email via an LLM node,
SMS queueing) that can outlast our timeout even though the booking itself
already went through. So a timeout is treated as an ambiguous outcome, not a
plain failure: we tell the LLM to say the reservation may already exist and
that it's checking, rather than the ordinary "technical error, trying again"
line — which previously led to a confusing "sorry, error, retrying" followed
immediately by "oh, that slot's taken" when the original attempt had in fact
succeeded.

any_person retries after an ambiguous outcome are handled DIFFERENTLY from
specific-employee ones (see _ambiguous_outcome_result) — this is
intentional, not an oversight to "clean up" later. A specific employee_id
retry is safe because check_slots on that same employee/slot correctly
reflects whether an earlier attempt already succeeded. An any_person retry
is NOT safe: it re-runs "is ANY eligible employee free", which can still
say yes via a genuinely-free DIFFERENT employee even though the original
attempt already succeeded with someone else — producing a real duplicate
booking for the same customer/time with two different staff members. This
happened for real during Phase 5 testing (two live Termini rows for the
same customer/slot, different employees) and was root-caused to exactly
this retry pattern, not to any staleness in n8n's check-slots query.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from zoneinfo import ZoneInfo

from livekit.agents import RunContext, function_tool

from agent import booking_client
from agent.booking_client import (
    BookingError,
    BookingMalformedResponseError,
    BookingTimeoutError,
)

logger = logging.getLogger("receptionistplus-worker.tools")

_TIMEZONE = ZoneInfo("Europe/Ljubljana")

_MARKUP_PATTERN = re.compile(r"</?[A-Za-z][\w:.-]*(?:\s+[^<>]*)?/?>")


def _sanitize_free_text(value: str) -> str:
    """Strip markup-like fragments from freeform, optional tool-call args.

    Observed 2026-08-14: `notes` came back from the LLM containing a stray
    tag resembling internal tool-call markup (`</antml parameter>`) with no
    actual caller-provided content. Root cause unconfirmed, but whatever
    produces it has no business ending up in a booking record the owner
    reads — strip it defensively rather than passing it through verbatim.
    """
    return _MARKUP_PATTERN.sub("", value).strip()


# Caller-facing phrasing suggested to the model for the two ambiguous
# create_booking outcomes. Language-keyed the same way as worker.py's
# NO_CREDITS_MESSAGE / TECHNICAL_DIFFICULTY_MESSAGE — before 2026-09-08 these
# were hardcoded Slovenian, so an English call got Slovenian suggested wording
# at the single most delicate moment in the flow. The surrounding instructions
# stay in English: those are addressed to the model, not spoken to the caller.
# Slovenian uses feminine first-person past forms ("naredila") per the FEMALE
# persona rule in STATIC_PROMPT_SL.
_AMBIGUOUS_PHRASE_RETRY_UNSAFE = {
    "sl": (
        "Rezervacijo sem morda že naredila, lastnik bo preveril in vas "
        "kontaktiral, če bo karkoli narobe"
    ),
    "en": (
        "I may have already made the reservation — the owner will check and "
        "get in touch if anything needs correcting"
    ),
}
_AMBIGUOUS_PHRASE_CHECKING = {
    "sl": "Rezervacijo sem morda že naredila, preverjam, prosim počakajte",
    "en": "I may have already made the reservation — let me check, one moment please",
}


def _ambiguous_outcome_result(
    any_person: bool, *, error: str, language: str = "sl"
) -> dict:
    """Build the LLM-facing result for an ambiguous create_booking outcome
    (BookingTimeoutError or BookingMalformedResponseError) — the appointment
    may already have been created server-side.

    DELIBERATELY ASYMMETRIC — do not "simplify" this back to one shared
    retry message for both any_person values. Root-caused 2026-08-17: an
    any_person=True booking has no stable identity across a retry. For a
    specific employee_id, check_slots on that same employee/slot correctly
    reflects whether an earlier attempt already succeeded, so the normal
    "check_slots, then maybe retry" pattern is safe. For any_person=True,
    a retry's check_slots/create_booking re-runs "is ANY eligible employee
    free" — if the original attempt already succeeded with employee A,
    a retry can still find employee B genuinely free and create a second,
    real, independent booking for the same customer/time with different
    staff. This happened for real (two live Termini rows, OB-000046 and
    OB-000047, same customer/slot, different employees — see the
    check-slots "staleness" investigation; check-slots itself was never
    stale, it was correctly answering an any-person query that retried).
    So any_person=True must never retry create_booking after an ambiguous
    outcome — take the message-taken path instead.

    Note that single-eligible-employee services can no longer reach the
    any_person branch at all: _resolve_person pins them to that one employee
    with any_person=False, which is the safe-to-retry path — and genuinely
    safe, since there is no second employee for a retry to land on.
    """
    if any_person:
        return {
            "success": False,
            "error": f"{error}_any_person",
            "message": (
                "The booking call's outcome is unknown, and any_person was "
                "True — retrying is NOT safe here, even via check_slots "
                "first. A fresh 'any free employee' query could succeed "
                "with a DIFFERENT employee than an earlier attempt that "
                "may have already gone through, creating a real duplicate "
                "booking. Do NOT call check_slots or create_booking again "
                "for this request. Tell the caller something like "
                f"'{_AMBIGUOUS_PHRASE_RETRY_UNSAFE[language]}' and end this "
                "booking attempt — do not try again this call."
            ),
        }
    return {
        "success": False,
        "error": error,
        "message": (
            "The booking call's outcome is unknown, but the appointment "
            "may already have been created server-side — this is NOT a "
            "plain failure. Tell the caller something like "
            f"'{_AMBIGUOUS_PHRASE_CHECKING[language]}' (do not "
            "say a plain error occurred). Then call check_slots for this "
            "exact slot/employee. If it now shows the slot as taken, tell "
            "the caller their booking is very likely confirmed and the "
            "owner will follow up if anything needs correcting — do not "
            "call create_booking again for this slot. Only call "
            "create_booking again if check_slots shows the slot is still "
            "free."
        ),
    }


def _filter_elapsed_times_today(slots_by_date: dict) -> None:
    """Drop already-elapsed HH:MM times from today's entry, in place.

    get_slots returns each open day's full schedule regardless of the
    current time of day — without this, a caller asking about "today" late
    in the day could be offered (and, in the worst case, actually book) a
    time slot that has already passed. Only touches today's date key; other
    days are untouched, and a day already marked closed (e.g. "unavailable")
    is left as-is.
    """
    now = datetime.now(_TIMEZONE)
    today_slots = slots_by_date.get(now.strftime("%Y-%m-%d"))
    if not isinstance(today_slots, list):
        return
    current_hhmm = now.strftime("%H:%M")
    slots_by_date[now.strftime("%Y-%m-%d")] = [
        t for t in today_slots if t >= current_hhmm
    ]


# Named (not inline) so worker.py's promise-follow-up watchdog can recognize
# and exclude these specific utterances — they're paired with an actual
# in-flight tool call that already has its own real timeout, so the
# watchdog's shorter window must not treat them as an unfulfilled promise.
GET_SLOTS_FILLER_TEXT = {
    "sl": [
        "Super, preverjam proste termine, samo trenutek prosim.",
        "Seveda, preverjam proste termine, samo trenutek prosim.",
        "V redu, preverjam proste termine, samo trenutek prosim.",
    ],
    "en": [
        "Great, checking available times, just a moment please.",
        "Sure, checking available times, just a moment please.",
        "Alright, checking available times, just a moment please.",
    ],
}
CREATE_BOOKING_FILLER_TEXT = {
    "sl": [
        "Super, urejam rezervacijo, samo trenutek prosim.",
        "Odlično, urejam rezervacijo, samo trenutek prosim.",
        "Z veseljem urejam rezervacijo, samo trenutek prosim.",
    ],
    "en": [
        "Great, I'm processing your booking, just a moment please.",
        "Wonderful, processing your booking now, just a moment please.",
        "Perfect, I'm setting up your booking, just a moment please.",
    ],
}


class BookingTools:
    def __init__(self, company_slug: str, company_data: dict, language: str = "sl") -> None:
        self._company_slug = company_slug
        self._company_data = company_data
        self._language = language
        self._last_check_key: tuple | None = None
        # Set on the first successful create_booking this session, for
        # receptionist_calls.created_termin_id / outcome (Phase 3 call logging).
        self.created_termin_id: str | None = None
        # Round-robin step counters for filler-text variety within one call
        # (same pattern as worker.py's _generic_filler_loop) — get_slots and
        # check_slots share one counter since they share GET_SLOTS_FILLER_TEXT.
        self._get_slots_filler_step = 0
        self._create_booking_filler_step = 0

    def _next_get_slots_filler(self) -> str:
        variants = GET_SLOTS_FILLER_TEXT[self._language]
        text = variants[self._get_slots_filler_step % len(variants)]
        self._get_slots_filler_step += 1
        return text

    def _next_create_booking_filler(self) -> str:
        variants = CREATE_BOOKING_FILLER_TEXT[self._language]
        text = variants[self._create_booking_filler_step % len(variants)]
        self._create_booking_filler_step += 1
        return text

    def _eligible_employee_ids(self, service_ids: list[str]) -> list[str]:
        by_service = self._company_data.get("employeesByServiceId", {})
        sets = [set(by_service.get(sid, [])) for sid in service_ids]
        if not sets:
            return []
        eligible = set.intersection(*sets) if len(sets) > 1 else sets[0]
        return sorted(eligible)

    def _resolve_person(
        self, service_ids: list[str], employee_id: str | None, any_person: bool
    ) -> tuple[str | None, bool, list[str]]:
        """Resolve the effective (employee_id, any_person, eligible_ids).

        Returns the EFFECTIVE values, which may differ from what the LLM
        passed — callers must rebind all three, not just employee_id.

        Single-eligible-employee services are pinned here rather than left to
        the prompt (2026-09-08). STATIC_PROMPT_* tells the model to skip the
        "do you have a staff preference?" question when a service lists only
        one eligible employee, but that's a conversational instruction: an
        LLM that follows it can still pass any_person=True (or an outright
        wrong employee_id) to the tool, and nothing downstream corrected it.
        Enforcing it here follows the same principle as the
        multiple_services_online check in create_booking — business rules the
        booking depends on are validated in code, not trusted to the model.

        Pinning also removes the ambiguous-outcome duplicate-booking hazard
        for these services: with any_person=False there is no "any free
        employee" requery for a retry to land on, and with exactly one
        eligible employee there is no second person it could pick anyway.
        """
        eligible_ids = self._eligible_employee_ids(service_ids)

        if len(eligible_ids) == 1:
            only_employee = eligible_ids[0]
            if any_person or employee_id != only_employee:
                logger.info(
                    "single eligible employee for services=%s: overriding "
                    "model-supplied employee_id=%r any_person=%s -> "
                    "employee_id=%r any_person=False",
                    service_ids,
                    employee_id,
                    any_person,
                    only_employee,
                )
            return only_employee, False, []

        if any_person:
            return None, True, eligible_ids
        return employee_id, False, []

    @function_tool
    async def get_slots(
        self,
        context: RunContext,
        service_ids: list[str],
        start_date: str,
        end_date: str,
        employee_id: str | None = None,
        any_person: bool = False,
    ) -> dict:
        """Look up which dates/times are available for one or more services.

        Args:
            service_ids: IDs of the services to book together (from the
                company data given to you). Only pass more than one if the
                company allows booking multiple services in one call.
            start_date: First date to check, as YYYY-MM-DD.
            end_date: Last date to check, as YYYY-MM-DD. Same as start_date
                for a single day.
            employee_id: Specific employee ID, if the caller asked for
                someone by name. Omit this and set any_person=True if the
                caller doesn't care who helps them.
            any_person: True if the caller is fine with any qualified
                employee, in which case employee_id is ignored. If the
                service has only one eligible employee, that employee is
                used automatically whatever you pass here.
        """
        employee_id, any_person, eligible = self._resolve_person(
            service_ids, employee_id, any_person
        )
        try:
            async with context.with_filler(
                self._next_get_slots_filler(), delay=2.5
            ):
                result = await booking_client.get_slots(
                    company_slug=self._company_slug,
                    service_ids=service_ids,
                    start_date=start_date,
                    end_date=end_date,
                    employee_id=employee_id,
                    any_person=any_person,
                    eligible_employee_ids=eligible,
                )
        except BookingError as exc:
            logger.error("get_slots failed: %s", exc)
            return {"success": False, "error": "technical_error", "message": str(exc)}
        if isinstance(result.get("slots"), dict):
            _filter_elapsed_times_today(result["slots"])
        return result

    @function_tool
    async def check_slots(
        self,
        context: RunContext,
        service_ids: list[str],
        date: str,
        time: str,
        employee_id: str | None = None,
        any_person: bool = False,
    ) -> dict:
        """Check whether one exact date/time is still free right now.

        You must call this immediately before create_booking, with the exact
        same service_ids/employee_id/any_person/date/time, every single time
        — never call create_booking without having just called this first for
        the same slot, even if you already showed the caller this slot via
        get_slots earlier in the conversation. Availability can change
        between calls.

        Args:
            service_ids: Same service IDs you intend to pass to
                create_booking.
            date: YYYY-MM-DD.
            time: HH:MM, 24-hour.
            employee_id: Same employee ID you intend to pass to
                create_booking, if any_person is False.
            any_person: Same value you intend to pass to create_booking.
        """
        employee_id, any_person, eligible = self._resolve_person(
            service_ids, employee_id, any_person
        )
        try:
            async with context.with_filler(
                self._next_get_slots_filler(), delay=2.5
            ):
                result = await booking_client.check_slots(
                    company_slug=self._company_slug,
                    service_ids=service_ids,
                    date=date,
                    time=time,
                    employee_id=employee_id,
                    any_person=any_person,
                    eligible_employee_ids=eligible,
                )
        except BookingError as exc:
            logger.error("check_slots failed: %s", exc)
            self._last_check_key = None
            return {"success": False, "error": "technical_error", "message": str(exc)}
        if result.get("available"):
            self._last_check_key = (
                tuple(service_ids),
                employee_id,
                any_person,
                tuple(eligible),
                date,
                time,
            )
        else:
            self._last_check_key = None
        return result

    @function_tool
    async def create_booking(
        self,
        context: RunContext,
        service_ids: list[str],
        date: str,
        time: str,
        first_name: str,
        email: str,
        phone: str,
        last_name: str = "",
        employee_id: str | None = None,
        any_person: bool = False,
        notes: str = "",
    ) -> dict:
        """Actually reserve the appointment. Requires a just-completed check_slots.

        If the response has requiresPayment=true, the booking is held but NOT
        confirmed — tell the caller their reservation is pending, and that
        they'll receive a payment link shortly (a real SMS/payment link is
        not sent yet in this phase). Do NOT ask the caller for card details
        yourself.

        Args:
            service_ids: Service ID(s) to book. Only pass more than one if
                the company data says multiple services can be booked online
                in one call — otherwise book one and offer to note any
                additional ones as a message.
            date: YYYY-MM-DD.
            time: HH:MM, 24-hour.
            first_name: Caller's first name.
            email: Caller's email address.
            phone: Caller's phone number.
            last_name: Caller's last name, if given.
            employee_id: Employee ID, if any_person is False.
            any_person: True if the caller is fine with any qualified
                employee. If the service has only one eligible employee,
                that employee is used automatically whatever you pass here.
            notes: Any extra note from the caller for this booking.
        """
        if (
            not self._company_data["company"].get("multiple_services_online")
            and len(service_ids) > 1
        ):
            return {
                "success": False,
                "error": "multiple_services_not_supported",
                "message": (
                    "This company only allows one service per online booking. "
                    "Book the first service now and offer to leave the rest as "
                    "a note for the owner."
                ),
            }

        last_name = _sanitize_free_text(last_name)
        notes = _sanitize_free_text(notes)

        employee_id, any_person, eligible = self._resolve_person(
            service_ids, employee_id, any_person
        )
        key = (
            tuple(service_ids),
            employee_id,
            any_person,
            tuple(eligible),
            date,
            time,
        )
        if self._last_check_key != key:
            return {
                "success": False,
                "error": "must_check_slots_first",
                "message": (
                    "You must call check_slots with these exact same "
                    "parameters right before create_booking. Call check_slots "
                    "now, then retry create_booking."
                ),
            }

        self._last_check_key = None  # single use — force a fresh check per booking
        try:
            async with context.with_filler(
                self._next_create_booking_filler(), delay=4.5
            ):
                result = await booking_client.create_booking(
                    company_slug=self._company_slug,
                    service_ids=service_ids,
                    date=date,
                    time=time,
                    first_name=first_name,
                    last_name=last_name,
                    email=email,
                    phone=phone,
                    employee_id=employee_id,
                    any_person=any_person,
                    eligible_employee_ids=eligible,
                    notes=notes,
                )
        except BookingTimeoutError as exc:
            logger.error("create_booking timed out (ambiguous outcome): %s", exc)
            return _ambiguous_outcome_result(
                any_person, error="ambiguous_timeout", language=self._language
            )
        except BookingMalformedResponseError as exc:
            logger.error(
                "create_booking got an empty/unparseable response (ambiguous "
                "outcome): %s",
                exc,
            )
            return _ambiguous_outcome_result(
                any_person,
                error="ambiguous_malformed_response",
                language=self._language,
            )
        except BookingError as exc:
            logger.error("create_booking failed: %s", exc)
            return {
                "success": False,
                "error": "technical_error",
                "message": (
                    "The booking call failed. Tell the caller plainly that "
                    "something went wrong, then call check_slots again for "
                    "this slot before trying create_booking again — do not "
                    "retry blindly, the appointment may already exist."
                ),
            }
        logger.info("create_booking outcome: %s", result)
        if result.get("success") and result.get("terminId"):
            self.created_termin_id = result["terminId"]
        return result
