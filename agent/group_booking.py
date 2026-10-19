"""Several people in one call (mum and daughter, a couple's massage).

Times for everyone are found BEFORE anything is booked, so the caller never
ends up with half a booking:

- one after another ("zaporedno"): each next person starts at the first free
  time at or after the previous one ends (at most MAX_GAP_MIN later);
- at the same time ("hkrati"): one start time with a different employee for
  each person, and enough of every shared resource (room, bed) for all.

book_group then re-checks every slot and books them in order; if one fails,
the ones already made are cancelled again. The caller's details go on the
first booking; the others are booked under their own first name (with the
caller's phone), or under the caller with a note when no name was given.
One SMS confirms them all.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta

from livekit.agents import RunContext, function_tool

from agent import booking_client
from agent.booking_client import BookingError
from agent.call_policy import normalize_phone
from agent.company_prompt import spoken_employee_names
from agent.tools import _day_label, _filter_elapsed_times_today, _same_name, _sanitize_free_text

logger = logging.getLogger("receptionistplus-worker.group")

MAX_PEOPLE = 4
MAX_GAP_MIN = 30
MAX_OPTIONS = 5


def _min(t: str) -> int:
    h, m = str(t)[:5].split(":")
    return int(h) * 60 + int(m)


def _hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


_SL_HOURS = {1: "enih", 2: "dveh", 3: "treh", 4: "štirih", 5: "petih", 6: "šestih", 7: "sedmih",
             8: "osmih", 9: "devetih", 10: "desetih", 11: "enajstih", 12: "dvanajstih"}
_SL_HALF = {1: "pol enih", 2: "pol dveh", 3: "pol treh", 4: "pol štirih", 5: "pol petih", 6: "pol šestih",
            7: "pol sedmih", 8: "pol osmih", 9: "pol devetih", 10: "pol desetih", 11: "pol enajstih",
            12: "pol dvanajstih"}


def spoken_sl_time(hhmm: str) -> str | None:
    """"14:00" -> "ob dveh popoldan", "09:30" -> "ob pol desetih dopoldan":
    the 12-hour form with the part of the day (R-TIME-LIST-STYLE). None for
    quarter hours, which the model then words itself."""
    h, m = int(hhmm[:2]), int(hhmm[3:5])
    part = "dopoldan" if h < 12 else ("popoldan" if h < 18 else "zvečer")
    if h == 12 and m == 0:
        return "ob dvanajstih"
    if m == 0:
        return f"ob {_SL_HOURS[h % 12 or 12]} {part}"
    if m == 30:
        return f"ob {_SL_HALF[(h + 1) % 12 or 12]} {part}"
    return None


def _spread(options: list[dict], limit: int = MAX_OPTIONS) -> list[dict]:
    """A few options across the day rather than the first five in a row."""
    if len(options) <= limit:
        return options
    step = (len(options) - 1) / (limit - 1)
    return [options[round(i * step)] for i in range(limit)]


class GroupBooking:
    def __init__(self, booking_tools, *, cancel_created=None, on_group_booked=None) -> None:
        self._bt = booking_tools
        # async (terminRowId) -> bool: undo a booking when a later one fails.
        self._cancel_created = cancel_created
        # async (bookings: list[dict]) -> None: the one confirmation SMS.
        self._on_group_booked = on_group_booked
        self._options: list[dict] = []
        self._date: str | None = None
        self._people: list[dict] = []
        self.created: list[dict] = []

    # --- helpers -----------------------------------------------------------

    def _duration(self, service_id: str) -> int:
        for s in self._bt._company_data.get("services", []):
            if str(s.get("id")) == str(service_id):
                try:
                    return max(5, int(s.get("trajanjeMin") or 0))
                except (TypeError, ValueError):
                    break
        return 30

    def _service_name(self, service_id: str) -> str:
        for s in self._bt._company_data.get("services", []):
            if str(s.get("id")) == str(service_id):
                return s.get("naziv") or ""
        return ""

    def _capacity(self) -> dict[int, int]:
        caps = {}
        for r in self._bt._company_data.get("resursi") or []:
            try:
                caps[int(r["id"])] = max(1, int(r.get("kolicina") or 1) * int(r.get("kapaciteta") or 1))
            except (KeyError, TypeError, ValueError):
                continue
        return caps

    async def _free_times(self, service_id: str, date: str, employee_id: str | None, eligible: list[str]) -> list[str]:
        result = await booking_client.get_slots(
            company_slug=self._bt._company_slug,
            service_ids=[service_id],
            start_date=date,
            end_date=date,
            employee_id=employee_id,
            any_person=employee_id is None,
            eligible_employee_ids=eligible if employee_id is None else None,
        )
        slots = result.get("slots") if isinstance(result, dict) else None
        if not isinstance(slots, dict):
            return []
        _filter_elapsed_times_today(slots)
        day = slots.get(date)
        return sorted(day, key=_min) if isinstance(day, list) else []

    # --- tools -------------------------------------------------------------

    @function_tool
    async def find_group_times(
        self,
        context: RunContext,
        service_ids: list[str],
        employee_ids: list[str],
        date: str,
        together: bool,
    ) -> dict:
        """Find times on one day for SEVERAL people booked in this call,
        before booking anything.

        Args:
            service_ids: One service ID per person, the caller first.
            employee_ids: One entry per person in the same order: the
                employee ID they asked for, or "" if anyone may do it.
            date: YYYY-MM-DD.
            together: True for the same time (needs a different employee per
                person), False for one after another.
        """
        if len(employee_ids) != len(service_ids):
            return {"success": False, "error": "bad_people",
                    "message": "Give one employee_ids entry per service_ids entry (\"\" for anyone)."}
        people = [{"service_id": s, "employee_id": e} for s, e in zip(service_ids, employee_ids)]
        if not 2 <= len(people) <= MAX_PEOPLE:
            return {"success": False, "error": "bad_people", "message": f"Use 2 to {MAX_PEOPLE} people; one person: get_slots."}
        resolved = []
        for p in people:
            sid = str(p.get("service_id") or "")
            if not sid or not self._duration(sid):
                return {"success": False, "error": "bad_service"}
            eligible = self._bt._eligible_employee_ids([sid])
            emp = str(p.get("employee_id") or "").strip() or None
            if emp and emp not in eligible:
                return {"success": False, "error": "employee_not_eligible",
                        "message": f"That employee doesn't do service {sid}; check the service's line."}
            if not emp and len(eligible) == 1:
                emp = eligible[0]
            resolved.append({"service_id": sid, "employee_id": emp, "eligible": eligible,
                             "duration": self._duration(sid)})

        try:
            if together:
                options = await self._together(resolved, date)
            else:
                options = await self._one_after_another(resolved, date)
        except BookingError as exc:
            logger.error("find_group_times failed: %s", exc)
            return {"success": False, "error": "technical_error"}

        self._options, self._date, self._people = options, date, resolved
        names = spoken_employee_names(self._bt._company_data.get("employees_ui", []))
        return {
            "success": True,
            "day_label": _day_label(date, self._bt._language),
            "options": [
                {
                    "option": i,
                    "people": [
                        {
                            "time": slot["time"],
                            "say": spoken_sl_time(slot["time"]) if self._bt._language == "sl" else None,
                            "employee": names.get(slot["employee_id"]) if slot["employee_id"] else None,
                        }
                        for slot in opt["slots"]
                    ],
                }
                for i, opt in enumerate(options)
            ],
            "message": (
                "Nothing fits everyone that day; offer another day or one after another / together instead."
                if not options else
                "Offer one or two of these as ONE sentence covering everyone, saying each time as in "
                "\"say\" when given; when they pick, read it back for everyone, collect the details, "
                "then book_group with that option."
            ),
        }

    async def _one_after_another(self, people: list[dict], date: str) -> list[dict]:
        frees = await asyncio.gather(*(
            self._free_times(p["service_id"], date, p["employee_id"], p["eligible"]) for p in people
        ))
        options = []
        for start in frees[0]:
            slots, t = [], _min(start)
            ok = True
            for i, p in enumerate(people):
                if i == 0:
                    pick = t
                else:
                    later = [_min(x) for x in frees[i] if t <= _min(x) <= t + MAX_GAP_MIN]
                    if not later:
                        ok = False
                        break
                    pick = later[0]
                slots.append({"time": _hhmm(pick), "employee_id": p["employee_id"]})
                t = pick + p["duration"]
            if ok:
                options.append({"slots": slots})
        return _spread(options)

    async def _together(self, people: list[dict], date: str) -> list[dict]:
        # Shared resources: everyone at once needs that many units.
        need: dict[int, int] = {}
        for p in people:
            for rid in self._bt._resources([p["service_id"]]):
                need[rid] = need.get(rid, 0) + 1
        caps = self._capacity()
        if any(n > caps.get(rid, 1) for rid, n in need.items()):
            return []
        # Free times per (person, employee): who exactly is free when.
        pairs = []
        for i, p in enumerate(people):
            for emp in ([p["employee_id"]] if p["employee_id"] else p["eligible"]):
                pairs.append((i, emp))
        frees = await asyncio.gather(*(
            self._free_times(people[i]["service_id"], date, emp, []) for i, emp in pairs
        ))
        free_by = {}
        for (i, emp), times in zip(pairs, frees):
            for t in times:
                free_by.setdefault(t, {}).setdefault(i, []).append(emp)
        options = []
        for t in sorted(free_by, key=_min):
            choice = self._assign(free_by[t], len(people))
            if choice:
                options.append({"slots": [{"time": t, "employee_id": e} for e in choice]})
        return _spread(options)

    @staticmethod
    def _assign(candidates: dict[int, list[str]], n: int, i: int = 0, used=()) -> list[str] | None:
        if i == n:
            return []
        for emp in candidates.get(i, []):
            if emp in used:
                continue
            rest = GroupBooking._assign(candidates, n, i + 1, (*used, emp))
            if rest is not None:
                return [emp, *rest]
        return None

    @function_tool
    async def book_group(
        self,
        context: RunContext,
        option: int,
        caller_first_name: str,
        caller_last_name: str,
        phone: str,
        other_names: list[str],
        email: str = "",
    ) -> dict:
        """Book everyone at the option from find_group_times, only after the
        caller confirmed it and the caller's details are collected.

        Args:
            option: The option number from find_group_times.
            caller_first_name: The caller's first name.
            caller_last_name: The caller's last name.
            phone: The caller's phone number (used for everyone).
            other_names: One entry per other person, in order: their name
                as given ("Eva", "Eva Novak"), or "" if not given.
            email: The caller's email, if needed for the confirmation.
        """
        if not 0 <= option < len(self._options) or self._date is None:
            return {"success": False, "error": "unknown_option", "message": "Call find_group_times first."}
        slots = self._options[option]["slots"]
        if len(other_names) != len(slots) - 1:
            return {"success": False, "error": "bad_names",
                    "message": f"Give exactly {len(slots) - 1} other_names (\"\" when unknown)."}
        phone = normalize_phone(phone) or phone
        caller_first = _sanitize_free_text(caller_first_name)
        caller_last = _sanitize_free_text(caller_last_name)
        existing = await self._bt._existing_customer(phone)
        if existing:
            if not _same_name(existing.get("Ime") or "", existing.get("Priimek") or "", caller_first, caller_last):
                full = " ".join(p for p in ((existing.get("Ime") or "").strip(), (existing.get("Priimek") or "").strip()) if p)
                return {"success": False, "error": "existing_customer",
                        "first_name": existing.get("Ime"), "last_name": existing.get("Priimek"),
                        "email": existing.get("Email stranke") or "",
                        "message": f"This number belongs to {full}. Ask \"Ali ste {full}?\"; if yes, book_group again with those details."}
            if not email:
                email = (existing.get("Email stranke") or "").strip()

        date = self._date
        # Re-check everything first: nothing is booked unless all still fit.
        for p, slot in zip(self._people, slots):
            try:
                check = await booking_client.check_slots(
                    company_slug=self._bt._company_slug,
                    service_ids=[p["service_id"]],
                    date=date,
                    time=slot["time"],
                    employee_id=slot["employee_id"],
                    any_person=slot["employee_id"] is None,
                    eligible_employee_ids=p["eligible"] if slot["employee_id"] is None else None,
                    resursi_ids=self._bt._resources([p["service_id"]]),
                )
            except BookingError:
                return {"success": False, "error": "technical_error"}
            if not check.get("available"):
                return {"success": False, "error": "slot_no_longer_free",
                        "message": "One of these times was just taken. Say so and call find_group_times again."}

        made: list[dict] = []
        caller_full = " ".join(x for x in (caller_first, caller_last) if x)
        for i, (p, slot) in enumerate(zip(self._people, slots)):
            if i == 0:
                first, last, mail, note = caller_first, caller_last, email, ""
            else:
                other = _sanitize_free_text(other_names[i - 1])
                if other:
                    parts = other.split()
                    first, last = parts[0], " ".join(parts[1:])
                    note = f"Rezerviral(a) {caller_full} po telefonu."
                else:
                    first, last = caller_first, caller_last
                    note = f"Termin za spremljevalca/-ko osebe {caller_full} (ime ni bilo podano)."
                mail = ""
            try:
                result = await booking_client.create_booking(
                    company_slug=self._bt._company_slug,
                    service_ids=[p["service_id"]],
                    date=date,
                    time=slot["time"],
                    first_name=first,
                    last_name=last,
                    email=mail,
                    phone=phone,
                    employee_id=slot["employee_id"],
                    any_person=slot["employee_id"] is None,
                    eligible_employee_ids=p["eligible"] if slot["employee_id"] is None else None,
                    notes=note,
                    language=self._bt._language,
                    send_email_confirmation=False if i else self._bt._email_confirmation,
                    resursi_ids=self._bt._resources([p["service_id"]]),
                )
            except BookingError as exc:
                logger.error("group create failed: %s", exc)
                result = {"success": False, "error": "technical_error"}
            if not (result.get("success") and result.get("terminId")):
                await self._undo(made)
                return {"success": False, "error": result.get("error") or "booking_failed",
                        "message": ("Booking everyone didn't work, so nothing was booked. Say so plainly and "
                                    "offer other times (find_group_times) or take_message.")}
            made.append({
                "terminId": result["terminId"], "terminRowId": result.get("terminRowId"),
                "service_id": p["service_id"], "service": self._service_name(p["service_id"]),
                "date": date, "time": slot["time"], "name": first,
                "employee_name": result.get("oseba") or "",
            })

        self.created = made
        if self._bt.created_termin_id is None:
            self._bt.created_termin_id = made[0]["terminId"]
        self._bt.booking_interest = None
        if self._on_group_booked is not None:
            task = asyncio.create_task(self._on_group_booked(made, phone), name="on_group_booked")
            self._bt.background_tasks.add(task)
            task.add_done_callback(self._bt.background_tasks.discard)
        logger.info("group booked: %s", [m["terminId"] for m in made])
        return {
            "success": True,
            "day_label": _day_label(date, self._bt._language),
            "bookings": [{"name": m["name"], "service": m["service"], "time": m["time"]} for m in made],
            "message": "All booked. Confirm everyone in ONE sentence and close the call as after a single booking.",
        }

    async def _undo(self, made: list[dict]) -> None:
        failed = []
        for m in made:
            ok = False
            if self._cancel_created is not None and m.get("terminRowId") is not None:
                try:
                    ok = await self._cancel_created(m["terminRowId"])
                except Exception:
                    logger.exception("group rollback failed for %s", m["terminId"])
            if not ok:
                failed.append(m["terminId"])
        if failed:
            self._bt.owner_followup_reason = (
                "Skupinska rezervacija po telefonu ni uspela v celoti; preverite in po potrebi "
                f"odpovejte termine: {', '.join(failed)}."
            )
