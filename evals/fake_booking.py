"""In-memory stand-in for the booking-v2 webhook, for evals.

Patches agent.booking_client's get_slots / check_slots / create_booking so
an eval run never touches the real calendar. The calendar is deterministic:
open Monday-Friday 08:00-18:00 and Saturday 08:00-12:00 on the hour, closed
Sunday, with a fixed set of taken slots, within the company's booking window.
"""

from __future__ import annotations

import contextlib
from datetime import date, datetime, timedelta
from unittest import mock
from zoneinfo import ZoneInfo

from agent import booking_client

TZ = ZoneInfo("Europe/Ljubljana")
WINDOW_DAYS = 14


def _open_times(d: date) -> list[str] | str:
    wd = d.weekday()
    if wd == 6:
        return "unavailable"
    last = 12 if wd == 5 else 18
    # Every half hour, so half-hour times ("ob pol dveh") get spoken too.
    times = [f"{h:02d}:{m}" for h in range(8, last) for m in ("00", "30")]
    # A few taken slots, different per weekday, so "is X free?" has real answers.
    taken = {0: {"09:00", "10:00"}, 1: {"14:00"}, 2: {"08:00", "16:00"}, 3: set(), 4: {"11:00"}, 5: {"08:00"}}
    return [t for t in times if t not in taken.get(wd, set())]


class FakeBooking:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.created: list[dict] = []
        self._booked: set[tuple[str, str]] = set()
        self._next_id = 1

    def _free(self, iso: str) -> list[str] | str:
        d = date.fromisoformat(iso)
        today = datetime.now(TZ).date()
        if d < today or d > today + timedelta(days=WINDOW_DAYS - 1):
            return "unavailable"
        times = _open_times(d)
        if isinstance(times, str):
            return times
        return [t for t in times if (iso, t) not in self._booked]

    async def get_slots(self, **kw) -> dict:
        self.calls.append(("get_slots", kw))
        start = date.fromisoformat(kw["start_date"])
        end = date.fromisoformat(kw["end_date"])
        slots = {}
        d = start
        while d <= end and len(slots) < 31:
            slots[d.isoformat()] = self._free(d.isoformat())
            d += timedelta(days=1)
        return {"success": True, "slots": slots}

    async def check_slots(self, **kw) -> dict:
        self.calls.append(("check_slots", kw))
        free = self._free(kw["date"])
        return {"success": True, "available": isinstance(free, list) and kw["time"] in free}

    async def create_booking(self, **kw) -> dict:
        self.calls.append(("create_booking", kw))
        self._booked.add((kw["date"], kw["time"]))
        termin_id = f"OB-EVAL-{self._next_id:03d}"
        self._next_id += 1
        self.created.append({**kw, "terminId": termin_id})
        return {"success": True, "terminId": termin_id}

    @contextlib.contextmanager
    def installed(self):
        with mock.patch.object(booking_client, "get_slots", self.get_slots), mock.patch.object(
            booking_client, "check_slots", self.check_slots
        ), mock.patch.object(booking_client, "create_booking", self.create_booking):
            yield self


class FakeAppointmentStore:
    """Stands in for the SupabaseClient methods agent/appointments.py uses:
    the caller's own appointments, and every change made to them."""

    def __init__(self, rows: list[dict]) -> None:
        self.rows = [dict(r) for r in rows]
        self.updates: list[tuple[str, dict]] = []
        self.history: list[tuple[str, str]] = []
        self.links: list[str] = []

    async def get_customer_appointments(self, company_id, customer_id, from_date):
        return [dict(r) for r in self.rows if r["Datum"] >= from_date and r["Status"] == "scheduled"]

    async def update_appointment(self, company_id, row_id, fields):
        for r in self.rows:
            if r["id"] == row_id:
                r.update(fields)
                self.updates.append((r["ID termina"], dict(fields)))
                return dict(r)
        return None

    async def add_appointment_history(self, company_id, termin_id, actor, action):
        self.history.append((termin_id, action))


def eval_day(days_ahead: int, avoid_sunday: bool = True) -> str:
    """A date days_ahead from today (moved off Sunday, when the fake
    calendar is closed)."""
    d = datetime.now(TZ).date() + timedelta(days=days_ahead)
    if avoid_sunday and d.weekday() == 6:
        d += timedelta(days=1)
    return d.isoformat()
