"""Booking tools exposed to the LLM: get_slots, check_slots, create_booking.

These wrap the booking-v2 webhook (see booking_client.py). Two pieces of
business logic live here rather than only in the prompt, so they can't be
skipped by an LLM that "forgets" an instruction:

- check_slots must be called immediately before create_booking for the exact
  same service/employee/date/time, or create_booking refuses and tells the
  model to check first. This prevents the double-booking race where two
  callers are quoted the same free slot. "Immediately" is enforced as a
  maximum age (CHECK_SLOTS_MAX_AGE_SEC), not only as "the last check
  matched": before 2026-09-21 the prompt had the model check_slots, THEN
  collect the caller's name/email/phone, THEN create_booking, and since
  nothing expired the check, a two-minute-old availability result was
  accepted as if it were fresh. The prompt now puts check_slots after data
  collection; the age limit makes a drift back to the old order fail
  safely (a forced re-check) instead of silently booking on stale data.
- if the company doesn't support `multiple_services_online`, create_booking
  refuses more than one service per call (the webhook itself doesn't enforce
  this).

"any employee" handling: the caller only ever says whether they care who does
the work (`any_person`); which employees are actually eligible for the
selected service(s) is computed here from the company's `init` data, not left
to the LLM to reconstruct. If a service has exactly one eligible employee,
`_resolve_person` pins the booking to that employee (`any_person=False`)
whatever the LLM passed — see its docstring. With several eligible
employees, `_employee_selection_error` refuses a call that names someone who
doesn't perform the service, or makes no staff choice at all.

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
line ("Super, vas vpisujem, samo trenutek.") without blocking the underlying
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

import asyncio
import logging
import re
import time as time_module
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from livekit.agents import RunContext, function_tool

from agent import booking_client
from agent.business_info import booking_window_days
from agent.company_prompt import spoken_employee_names
from agent.dates import spoken_date_label
from agent.languages import pack
from agent.booking_client import (
    BookingError,
    BookingMalformedResponseError,
    BookingTimeoutError,
)

logger = logging.getLogger("receptionistplus-worker.tools")

_TIMEZONE = ZoneInfo("Europe/Ljubljana")

# How old a successful check_slots may be when create_booking consumes it.
# In the intended flow check_slots and create_booking run back-to-back in one
# LLM turn (seconds apart), so this only bites when a caller-facing exchange —
# typically data collection — happened in between. Generous enough that a
# slow webhook plus a spoken filler never trips it on the intended path.
CHECK_SLOTS_MAX_AGE_SEC = 60.0

# Time-of-day buckets for get_slots' "time_of_day" summary. Evening starts at
# 18:00 — what a Slovenian speaker calls "zvečer"; 17:xx is still "popoldan".
_AFTERNOON_START = "12:00"
_EVENING_START = "18:00"

# "/" is allowed inside the tag name: "</antml/parameter>" (observed in a
# real booking's notes 2026-09-22) slipped past the earlier [\w:.-] class.
_MARKUP_PATTERN = re.compile(r"</?[A-Za-z][\w:./-]*(?:\s+[^<>]*)?/?>")


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
# create_booking outcomes: LanguagePack.ambiguous_retry_unsafe /
# ambiguous_checking (agent/languages.py). Before 2026-09-08 these were
# hardcoded Slovenian, so an English call got Slovenian suggested wording at
# the single most delicate moment in the flow. The surrounding instructions
# stay in English: those are addressed to the model, not spoken to the caller.


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
                f"'{pack(language).ambiguous_retry_unsafe}' and end this "
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
            f"'{pack(language).ambiguous_checking}' (do not "
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


def _time_of_day_summary(slots_by_date: dict) -> dict:
    """Count each open day's free times per part of day.

    Returned to the model alongside the raw slots so that "dopoldan,
    popoldan ali zvečer?" is decided from counts computed here, not from
    the model scanning a list of HH:MM strings — and so "zvečer" is only
    ever offered when the day really has evening times. Days whose value
    isn't a list (e.g. "unavailable") are left out.
    """
    summary = {}
    for date, times in slots_by_date.items():
        if not isinstance(times, list):
            continue
        counts = {"morning": 0, "afternoon": 0, "evening": 0}
        for t in times:
            if t < _AFTERNOON_START:
                counts["morning"] += 1
            elif t < _EVENING_START:
                counts["afternoon"] += 1
            else:
                counts["evening"] += 1
        summary[date] = counts
    return summary


def _day_label(iso_date: str, language: str) -> str | None:
    try:
        return spoken_date_label(datetime.strptime(iso_date, "%Y-%m-%d").date(), language)
    except (TypeError, ValueError):
        return None


def _day_labels(dates, language: str) -> dict:
    """Map each YYYY-MM-DD to its weekday+date as ONE spoken string.

    Added 2026-09-21 after the model confirmed "torek, osemindvajsetega
    septembra" — the caller's weekday glued onto the first date of a
    get_slots result it was reading, when the 28th was a Monday. Raw ISO
    keys carry no weekday, so whatever weekday the model spoke next to one
    came from somewhere else. With the label computed here, a weekday and
    its date arrive as one string, and the prompt tells the model to copy
    both from it.
    """
    labels = {}
    for d in dates:
        label = _day_label(d, language)
        if label:
            labels[d] = label
    return labels


# Filler lines, spoken only if a tool call is still running after its delay
# (context.with_filler). Rewritten 2026-10-16 to sound like a person at a
# desk rather than a status message: they say what is being looked at ("v
# sredo") instead of announcing a process, and the booking one uses the
# receptionist's own verb ("vpisujem vas"). Each spoken line is recorded in
# BookingTools.spoken_fillers so worker.py's promise watchdog can tell them
# apart from a model turn that promised a check and then stopped — the line
# is paired with a real in-flight call that has its own timeout.
#
# The lines themselves are per language in agent/languages.py
# (get_slots_day_fillers / get_slots_range_fillers / check_slots_fillers /
# create_booking_fillers). check_slots runs as the last step before
# create_booking, right after the model's own "Samo še preverim, da je termin
# še prost." line — so its filler (only heard if the check takes >2.5s)
# continues that thought rather than repeating it, and must talk about
# CHECKING, never booking (2026-09-22: callers heard "rezerviram" while only
# the availability check was running).


def _weekday_for_filler(iso_date: str, language: str) -> str | None:
    try:
        d = datetime.strptime(iso_date, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None
    return pack(language).weekdays_for_filler[d.weekday()]


class BookingTools:
    def __init__(
        self,
        company_slug: str,
        company_data: dict,
        language: str = "sl",
        *,
        on_booked=None,
    ) -> None:
        self._company_slug = company_slug
        self._company_data = company_data
        self._language = language
        # Same window as the online booking page (max_dnevi_rezervacija):
        # get_slots won't look past it, so the phone can't book further
        # ahead than the web.
        self._window_days = booking_window_days(company_data.get("company"))
        self._last_check_key: tuple | None = None
        self._last_check_at: float = 0.0
        # Set on the first successful create_booking this session, for
        # receptionist_calls.created_termin_id / outcome (Phase 3 call logging).
        self.created_termin_id: str | None = None
        # Set when the caller was told the owner will check/follow up (an
        # ambiguous create_booking outcome). worker.py turns it into a
        # callback request on the call row, so the owner actually hears.
        self.owner_followup_reason: str | None = None
        # Round-robin step for filler variety within one call, and every
        # filler line actually handed to with_filler (see the note above
        # languages.py).
        self._filler_step = 0
        self.spoken_fillers: set[str] = set()
        # async callback(booking: dict) after a successful create_booking —
        # worker.py sends the SMS confirmation from it. Runs in the
        # background so the caller hears the confirmation straight away.
        self._on_booked = on_booked
        self.background_tasks: set = set()

    def set_language(self, language: str) -> None:
        """Called when the call switches language (worker.py): fillers, day
        labels and suggested phrasings follow from then on."""
        self._language = language

    def _next_filler(self, variants, **fmt) -> str:
        text = variants[self._filler_step % len(variants)].format(**fmt)
        self._filler_step += 1
        self.spoken_fillers.add(text)
        return text

    def _get_slots_filler(self, start_date: str, end_date: str) -> str:
        day = _weekday_for_filler(start_date, self._language) if start_date == end_date else None
        if day:
            return self._next_filler(pack(self._language).get_slots_day_fillers, day=day)
        return self._next_filler(pack(self._language).get_slots_range_fillers)

    def _note_ambiguous_booking(
        self, date: str, time: str, first_name: str, last_name: str
    ) -> None:
        name = " ".join(p for p in (first_name, last_name) if p).strip()
        self.owner_followup_reason = (
            f"Rezervacija za {name or 'stranko'} ({date} ob {time}) morda ni bila "
            "potrjena — rezervacijski sistem ni vrnil jasnega odgovora. Preverite "
            "v koledarju in po potrebi pokličite stranko."
        )

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

    def _employee_selection_error(
        self, service_ids: list[str], employee_id: str | None, any_person: bool
    ) -> dict | None:
        """Refuse a staff choice that doesn't fit the service, before it's used.

        Only relevant when several employees perform the service: with one,
        _resolve_person pins the booking to them whatever was passed; with
        none listed there is nothing to validate against. Two cases:

        - No decision at all (2026-09-21): neither employee_id nor
          any_person=True. Observed when a caller switched from the one
          single-employee service ("masaža glave. Ne, masaža stopal.") to a
          multi-employee one and the model skipped the staff question for the
          new service, apparently still applying the "don't ask" note from the
          one just abandoned. This can't prove the question was asked (the
          model could still pass any_person=True unprompted) — it catches the
          silent-default case, which is the one observed.
        - An employee who doesn't perform the service (2026-09-22): OB-000057
          booked Refleksna masaža stopal with Luka, who isn't in that
          service's employeesByServiceId list, and the webhook accepted it —
          nothing on either side checked. Refused here with the names of who
          DOES perform it, so the model can put that choice to the caller.
          Enforced in get_slots and check_slots as well as create_booking, so
          it's caught before any availability is quoted for that employee,
          not after the caller's details have been collected.
        """
        eligible = self._eligible_employee_ids(service_ids)
        if len(eligible) <= 1 or any_person:
            return None
        if employee_id is None:
            return {
                "success": False,
                "error": "employee_preference_required",
                "message": (
                    "Several staff members perform this service and you "
                    "passed neither employee_id nor any_person=True. Ask the "
                    "caller the staff-preference question from your "
                    "instructions first, then call again with employee_id "
                    "(if they name someone) or any_person=True (if they "
                    "don't mind)."
                ),
            }
        if str(employee_id) not in {str(e) for e in eligible}:
            employees = self._company_data.get("employees_ui", [])
            names = spoken_employee_names(employees)
            requested = names.get(employee_id) or names.get(str(employee_id)) or str(employee_id)
            performers = ", ".join(
                f"{names.get(e, e)} (staff ID {e})" for e in eligible
            )
            logger.info(
                "refusing employee_id=%r for services=%s: not eligible (eligible=%s)",
                employee_id,
                service_ids,
                eligible,
            )
            return {
                "success": False,
                "error": "employee_not_eligible",
                "message": (
                    f"{requested} does not perform this service. It is "
                    f"performed by: {performers}. Tell the caller that "
                    f"{requested} doesn't do this service, and ask whether "
                    "one of these suits them or whether anyone is fine — "
                    "then call again with that person's employee_id, or "
                    "any_person=True. Do not book it with "
                    f"{requested}."
                ),
            }
        return None

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
        """Free dates and times for one or more services over a date range.

        Args:
            service_ids: Service IDs from the company data (more than one
                only if the company allows it).
            start_date: First date, YYYY-MM-DD.
            end_date: Last date, YYYY-MM-DD (same as start_date for one day).
            employee_id: Staff ID, if the caller asked for someone.
            any_person: True if the caller doesn't mind who.

        The result has "day_labels" (weekday and spoken date per date) and
        "time_of_day" (free times per part of the day).
        """
        refusal = self._employee_selection_error(service_ids, employee_id, any_person)
        if refusal:
            return refusal
        window_note = None
        last_day = (
            datetime.now(_TIMEZONE).date() + timedelta(days=self._window_days - 1)
        ).isoformat()
        if start_date > last_day:
            return {
                "success": False,
                "error": "beyond_booking_window",
                "message": (
                    f"Bookings are only possible up to {self._window_days} days "
                    f"ahead (last bookable day {last_day}). Tell the caller and "
                    "offer either a date within that window or to pass the "
                    "request on to the owner with take_message."
                ),
            }
        if end_date > last_day:
            end_date = last_day
            window_note = (
                f"Only dates up to {last_day} were checked: bookings are open "
                f"{self._window_days} days ahead."
            )
        employee_id, any_person, eligible = self._resolve_person(
            service_ids, employee_id, any_person
        )
        try:
            async with context.with_filler(
                self._get_slots_filler(start_date, end_date), delay=2.5
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
            result["time_of_day"] = _time_of_day_summary(result["slots"])
            result["day_labels"] = _day_labels(result["slots"], self._language)
        if window_note and isinstance(result, dict):
            result["note"] = window_note
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
        """Is this exact slot still free? Call it last, right before
        create_booking in the same turn, with the same arguments.

        Args:
            service_ids: Same as for create_booking.
            date: YYYY-MM-DD.
            time: HH:MM, 24-hour.
            employee_id: Same as for create_booking.
            any_person: Same as for create_booking.
        """
        refusal = self._employee_selection_error(service_ids, employee_id, any_person)
        if refusal:
            return refusal
        employee_id, any_person, eligible = self._resolve_person(
            service_ids, employee_id, any_person
        )
        try:
            async with context.with_filler(
                self._next_filler(pack(self._language).check_slots_fillers), delay=2.5
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
            self._last_check_at = time_module.monotonic()
        else:
            self._last_check_key = None
        label = _day_label(date, self._language)
        if label and isinstance(result, dict):
            result["day_label"] = label
        return result

    @function_tool
    async def create_booking(
        self,
        context: RunContext,
        service_ids: list[str],
        date: str,
        time: str,
        first_name: str,
        phone: str,
        email: str = "",
        last_name: str = "",
        employee_id: str | None = None,
        any_person: bool = False,
        notes: str = "",
    ) -> dict:
        """Reserve the appointment, right after check_slots for the same slot.

        Args:
            service_ids: Service ID(s) (more than one only if the company
                allows it).
            date: YYYY-MM-DD.
            time: HH:MM, 24-hour.
            first_name: Caller's first name.
            phone: Phone number for the booking.
            email: The caller's email — or the stored one for a confirmed
                existing customer (always pass it then). Empty only when
                confirmations go by SMS and none was given.
            last_name: Caller's last name, if given.
            employee_id: Staff ID, when any_person is False.
            any_person: True if the caller doesn't mind who.
            notes: Extra note from the caller.
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
                    "Book the first service now and offer to write the rest "
                    "down for the owner, phrased naturally as in the company "
                    "data — don't describe the mechanics to the caller."
                ),
            }

        last_name = _sanitize_free_text(last_name)
        notes = _sanitize_free_text(notes)

        refusal = self._employee_selection_error(service_ids, employee_id, any_person)
        if refusal:
            return refusal

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
        check_age = time_module.monotonic() - self._last_check_at
        if self._last_check_key == key and check_age > CHECK_SLOTS_MAX_AGE_SEC:
            logger.info(
                "create_booking: matching check_slots is %.0fs old (max %.0fs) "
                "— forcing a fresh check",
                check_age,
                CHECK_SLOTS_MAX_AGE_SEC,
            )
            self._last_check_key = None
        if self._last_check_key != key:
            return {
                "success": False,
                "error": "must_check_slots_first",
                "message": (
                    "You must call check_slots with these exact same "
                    "parameters right before create_booking (a check older "
                    "than a minute doesn't count). Call check_slots now, then "
                    "retry create_booking — no need to ask the caller "
                    "anything first."
                ),
            }

        self._last_check_key = None  # single use — force a fresh check per booking
        try:
            async with context.with_filler(
                self._next_filler(pack(self._language).create_booking_fillers), delay=4.5
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
                    language=self._language,
                )
        except BookingTimeoutError as exc:
            logger.error("create_booking timed out (ambiguous outcome): %s", exc)
            self._note_ambiguous_booking(date, time, first_name, last_name)
            return _ambiguous_outcome_result(
                any_person, error="ambiguous_timeout", language=self._language
            )
        except BookingMalformedResponseError as exc:
            logger.error(
                "create_booking got an empty/unparseable response (ambiguous "
                "outcome): %s",
                exc,
            )
            self._note_ambiguous_booking(date, time, first_name, last_name)
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
        logger.info(
            "create_booking outcome: success=%s terminId=%s error=%s",
            result.get("success"),
            result.get("terminId"),
            result.get("error"),
        )
        if result.get("success") and result.get("terminId"):
            self.created_termin_id = result["terminId"]
            # A retry after an ambiguous outcome went through cleanly.
            self.owner_followup_reason = None
            if self._on_booked is not None:
                task = asyncio.create_task(
                    self._on_booked(
                        {
                            "terminId": result["terminId"],
                            "service_ids": list(service_ids),
                            "date": date,
                            "time": time,
                            "phone": phone,
                            "email": email,
                        }
                    ),
                    name="on_booked",
                )
                self.background_tasks.add(task)
                task.add_done_callback(self.background_tasks.discard)
        # Same label as get_slots/check_slots, for the final confirmation.
        label = _day_label(date, self._language)
        if label and isinstance(result, dict):
            result["day_label"] = label
        return result
