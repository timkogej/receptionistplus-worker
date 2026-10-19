"""Cancel or move the caller's own appointment during the call.

Only for a caller recognised by caller ID (the number on file for an
existing customer): the same trust a receptionist gives a number she knows.
Anyone else gets a portal link texted to the number on file instead
(worker.py, send_manage_link), so a third party can't change someone's
appointment.

The rules are the customer portal's (n8n "Customer portal – Cancel
appointment" and "– Confirm reschedule"): the appointment must belong to
this customer and company and still be active; cancelling needs at least
cancel_appt_days, moving reschedule_appt_days before the appointment; a move
keeps the employee and the duration. Changes are only allowed for
appointments this call has listed, and only after the caller said yes
(the prompt asks for it; the tool says so too).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from livekit.agents import RunContext, function_tool

from agent import booking_client
from agent.booking_client import BookingError
from agent.company_prompt import spoken_employee_names
from agent.tools import _day_label

logger = logging.getLogger("receptionistplus-worker.appointments")

_TIMEZONE = ZoneInfo("Europe/Ljubljana")


def _days(value) -> int:
    try:
        days = int(str(value).strip())
    except (TypeError, ValueError):
        return 0
    return max(0, days)


def _hhmm(value) -> str:
    return str(value or "")[:5]


def _start(row: dict) -> datetime | None:
    try:
        return datetime.strptime(f"{row['Datum'][:10]} {_hhmm(row['Čas'])}", "%Y-%m-%d %H:%M").replace(
            tzinfo=_TIMEZONE
        )
    except (KeyError, TypeError, ValueError):
        return None


def _duration_min(row: dict) -> int | None:
    try:
        start = datetime.strptime(_hhmm(row["Čas"]), "%H:%M")
        end = datetime.strptime(_hhmm(row["Konec"]), "%H:%M")
    except (KeyError, TypeError, ValueError):
        return None
    minutes = int((end - start).total_seconds() // 60)
    return minutes if minutes > 0 else None


class AppointmentTools:
    def __init__(
        self,
        *,
        supabase,
        company_id: str,
        company_slug: str,
        customer: dict,
        profile: dict | None,
        company_data: dict,
        notes,
        language: str,
        resources_for=None,
        notify_customer=None,
        now=None,
    ) -> None:
        self._supabase = supabase
        self._company_id = company_id
        self._company_slug = company_slug
        self._customer_id = str(customer.get("ID stranke") or "")
        self._actor = " ".join(
            p for p in ((customer.get("Ime") or "").strip(), (customer.get("Priimek") or "").strip()) if p
        ) or "Receptionist+ (telefon)"
        self._cancel_days = _days((profile or {}).get("cancel_appt_days"))
        self._reschedule_days = _days((profile or {}).get("reschedule_appt_days"))
        self._names = spoken_employee_names(company_data.get("employees_ui", []))
        self._notes = notes
        self._language = language
        # service ids -> resource ids (BookingTools._resources), so a move
        # checks the room/bed like a new booking does.
        self._resources_for = resources_for or (lambda ids: [])
        # async (kind, row, new_date, new_time) -> None: the caller's SMS.
        self._notify_customer = notify_customer
        self._now = now or (lambda: datetime.now(_TIMEZONE))
        self._listed: dict[str, dict] = {}

    def set_language(self, language: str) -> None:
        self._language = language

    def _hours_until(self, row: dict) -> float | None:
        start = _start(row)
        return None if start is None else (start - self._now()).total_seconds() / 3600

    @function_tool
    async def list_my_appointments(self, context: RunContext) -> dict:
        """The caller's upcoming appointments with this business (they were
        recognised by their phone number). Call it when they want to cancel,
        move or check an appointment."""
        today = self._now().date().isoformat()
        try:
            rows = await self._supabase.get_customer_appointments(self._company_id, self._customer_id, today)
        except Exception:
            logger.exception("list_my_appointments failed")
            return {"success": False, "error": "technical_error"}
        now = self._now()
        upcoming = [r for r in rows if (_start(r) or now) > now]
        self._listed = {str(r["ID termina"]): r for r in upcoming}
        return {
            "success": True,
            "appointments": [
                {
                    "appointment_id": r["ID termina"],
                    "service": r.get("Storitev"),
                    "service_id": str(r.get("ID storitve") or ""),
                    "employee": self._names.get(str(r.get("ID Osebe")), (r.get("Oseba") or "").split(" ")[0]),
                    "employee_id": str(r.get("ID Osebe") or ""),
                    "date": r["Datum"][:10],
                    "day_label": _day_label(r["Datum"][:10], self._language),
                    "time": _hhmm(r["Čas"]),
                }
                for r in upcoming
            ],
            "message": (
                "No upcoming appointments." if not upcoming else
                "Say which one you found (service, day, time) and ask what they'd like to do. "
                "Never read out an appointment_id."
            ),
        }

    @function_tool
    async def cancel_appointment(self, context: RunContext, appointment_id: str) -> dict:
        """Cancel one of the caller's appointments from list_my_appointments,
        only after they clearly said yes to cancelling that exact one.

        Args:
            appointment_id: The appointment_id from list_my_appointments.
        """
        row = self._listed.get(str(appointment_id))
        if row is None:
            return {"success": False, "error": "unknown_appointment",
                    "message": "Call list_my_appointments first and use one of its appointment_ids."}
        hours = self._hours_until(row)
        if hours is not None and hours < self._cancel_days * 24:
            return {
                "success": False,
                "error": "cancel_window_expired",
                "message": (
                    f"Too late to cancel by phone: the business needs at least {self._cancel_days} "
                    "day(s) notice. Say so kindly and offer to pass a message to the owner (take_message)."
                ),
            }
        try:
            updated = await self._supabase.update_appointment(self._company_id, row["id"], {"Status": "cancelled"})
        except Exception:
            logger.exception("cancel_appointment failed")
            return {"success": False, "error": "technical_error",
                    "message": "Couldn't cancel it. Say so and offer to pass it to the owner (take_message)."}
        if not updated:
            return {"success": False, "error": "technical_error"}
        await self._supabase.add_appointment_history(self._company_id, row["ID termina"], self._actor, "odpovedan")
        self._listed.pop(str(appointment_id), None)
        self._notes.add_change(
            f"Stranka je po telefonu odpovedala termin {row.get('Storitev')} "
            f"{row['Datum'][:10]} ob {_hhmm(row['Čas'])} ({row['ID termina']})."
        )
        if self._notify_customer is not None:
            await self._notify_customer("cancelled", row, None, None)
        logger.info("appointment cancelled by phone: %s", row["ID termina"])
        return {"success": True, "message": "Cancelled. Confirm it briefly; a text is on its way if SMS is on."}

    @function_tool
    async def reschedule_appointment(
        self, context: RunContext, appointment_id: str, new_date: str, new_time: str
    ) -> dict:
        """Move one of the caller's appointments to a new time with the same
        employee, only after they said yes to that exact new time. Find
        free times first with get_slots, using the appointment's service_id
        and employee_id.

        Args:
            appointment_id: The appointment_id from list_my_appointments.
            new_date: YYYY-MM-DD.
            new_time: HH:MM, 24-hour.
        """
        row = self._listed.get(str(appointment_id))
        if row is None:
            return {"success": False, "error": "unknown_appointment",
                    "message": "Call list_my_appointments first and use one of its appointment_ids."}
        hours = self._hours_until(row)
        if hours is not None and hours < self._reschedule_days * 24:
            return {
                "success": False,
                "error": "reschedule_window_expired",
                "message": (
                    f"Too late to move it by phone: the business needs at least {self._reschedule_days} "
                    "day(s) notice. Say so kindly and offer to pass a message to the owner (take_message)."
                ),
            }
        duration = _duration_min(row)
        try:
            start = datetime.strptime(f"{new_date} {new_time}", "%Y-%m-%d %H:%M").replace(tzinfo=_TIMEZONE)
        except ValueError:
            return {"success": False, "error": "bad_date_or_time", "message": "Use YYYY-MM-DD and HH:MM."}
        if duration is None or start <= self._now():
            return {"success": False, "error": "bad_date_or_time"}
        service_id = str(row.get("ID storitve") or "")
        employee_id = str(row.get("ID Osebe") or "")
        try:
            check = await booking_client.check_slots(
                company_slug=self._company_slug,
                service_ids=[service_id],
                date=new_date,
                time=new_time,
                employee_id=employee_id,
                any_person=False,
                eligible_employee_ids=None,
                resursi_ids=self._resources_for([service_id]),
            )
        except BookingError:
            logger.exception("reschedule check failed")
            return {"success": False, "error": "technical_error"}
        if not check.get("available"):
            return {"success": False, "error": "slot_no_longer_free",
                    "message": "That time isn't free (any more). Offer other times from get_slots."}
        end = start + timedelta(minutes=duration)
        try:
            updated = await self._supabase.update_appointment(
                self._company_id,
                row["id"],
                {"Datum": new_date, "Čas": f"{new_time}:00", "Konec": end.strftime("%H:%M:00")},
            )
        except Exception:
            logger.exception("reschedule_appointment failed")
            return {"success": False, "error": "technical_error",
                    "message": "Couldn't move it. Say so and offer to pass it to the owner (take_message)."}
        if not updated:
            return {"success": False, "error": "technical_error"}
        await self._supabase.add_appointment_history(self._company_id, row["ID termina"], self._actor, "prestavljen")
        self._listed[str(appointment_id)] = updated
        self._notes.add_change(
            f"Stranka je po telefonu prestavila termin {row.get('Storitev')} "
            f"z {row['Datum'][:10]} ob {_hhmm(row['Čas'])} na {new_date} ob {new_time} ({row['ID termina']})."
        )
        if self._notify_customer is not None:
            await self._notify_customer("moved", row, new_date, new_time)
        logger.info("appointment moved by phone: %s", row["ID termina"])
        return {
            "success": True,
            "day_label": _day_label(new_date, self._language),
            "message": "Moved. Confirm the new day and time in one sentence; a text is on its way if SMS is on.",
        }
