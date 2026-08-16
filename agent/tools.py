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
to the LLM to reconstruct.

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
line ("Samo trenutek, urejam rezervacijo...") without blocking the underlying
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
GET_SLOTS_FILLER_TEXT = "Samo trenutek, preverjam proste termine..."
CREATE_BOOKING_FILLER_TEXT = "Samo trenutek, urejam rezervacijo..."


class BookingTools:
    def __init__(self, company_slug: str, company_data: dict) -> None:
        self._company_slug = company_slug
        self._company_data = company_data
        self._last_check_key: tuple | None = None
        # Set on the first successful create_booking this session, for
        # receptionist_calls.created_termin_id / outcome (Phase 3 call logging).
        self.created_termin_id: str | None = None

    def _eligible_employee_ids(self, service_ids: list[str]) -> list[str]:
        by_service = self._company_data.get("employeesByServiceId", {})
        sets = [set(by_service.get(sid, [])) for sid in service_ids]
        if not sets:
            return []
        eligible = set.intersection(*sets) if len(sets) > 1 else sets[0]
        return sorted(eligible)

    def _resolve_person(
        self, service_ids: list[str], employee_id: str | None, any_person: bool
    ) -> tuple[str | None, list[str]]:
        if any_person:
            return None, self._eligible_employee_ids(service_ids)
        return employee_id, []

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
                employee, in which case employee_id is ignored.
        """
        employee_id, eligible = self._resolve_person(
            service_ids, employee_id, any_person
        )
        try:
            async with context.with_filler(
                GET_SLOTS_FILLER_TEXT, delay=2.5
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
        employee_id, eligible = self._resolve_person(
            service_ids, employee_id, any_person
        )
        try:
            async with context.with_filler(
                GET_SLOTS_FILLER_TEXT, delay=2.5
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
                employee.
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

        employee_id, eligible = self._resolve_person(
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
                CREATE_BOOKING_FILLER_TEXT, delay=4.5
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
            return {
                "success": False,
                "error": "ambiguous_timeout",
                "message": (
                    "The booking call timed out, but the appointment may "
                    "already have been created server-side — this is NOT a "
                    "plain failure. Tell the caller something like "
                    "'Rezervacijo sem morda že naredila, preverjam, prosim "
                    "počakajte' (do not say a plain error occurred). Then "
                    "call check_slots for this exact slot. If it now shows "
                    "the slot as taken, tell the caller their booking is "
                    "very likely confirmed and the owner will follow up if "
                    "anything needs correcting — do not call create_booking "
                    "again for this slot. Only call create_booking again if "
                    "check_slots shows the slot is still free."
                ),
            }
        except BookingMalformedResponseError as exc:
            logger.error(
                "create_booking got an empty/unparseable response (ambiguous "
                "outcome): %s",
                exc,
            )
            return {
                "success": False,
                "error": "ambiguous_malformed_response",
                "message": (
                    "The booking call returned an empty/invalid response, but "
                    "the appointment may already have been created "
                    "server-side — this is NOT a plain failure. Tell the "
                    "caller something like 'Rezervacijo sem morda že "
                    "naredila, preverjam, prosim počakajte' (do not say a "
                    "plain error occurred). Then call check_slots for this "
                    "exact slot. If it now shows the slot as taken, tell the "
                    "caller their booking is very likely confirmed and the "
                    "owner will follow up if anything needs correcting — do "
                    "not call create_booking again for this slot. Only call "
                    "create_booking again if check_slots shows the slot is "
                    "still free."
                ),
            }
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
