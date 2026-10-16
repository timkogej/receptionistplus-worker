"""Booking business rules enforced in code (agent/tools.py).

Needs the worker's runtime dependencies (livekit-agents): run with the
project venv. Skipped, not failed, without them, so the stdlib-only tests
still run anywhere. The booking-v2 webhook is never called — every test
swaps booking_client's functions for in-memory fakes.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import unittest
from unittest import mock

try:
    from agent import tools as T
except ImportError:  # pragma: no cover - no livekit in this interpreter
    T = None

COMPANY = {
    "company": {"naziv": "Test salon", "multiple_services_online": False},
    "employees_ui": [
        {"id": "e1", "label": "Luka Novak"},
        {"id": "e2", "label": "Maja Kranjc"},
        {"id": "e3", "label": "Nina Zupan"},
    ],
    "employeesByServiceId": {"solo": ["e1"], "multi": ["e2", "e3"]},
}


class _Ctx:
    """Just enough of RunContext for the tools: with_filler() as a no-op."""

    @contextlib.asynccontextmanager
    async def _filler(self, *_a, **_kw):
        yield

    def with_filler(self, *a, **kw):
        return self._filler(*a, **kw)


def _run(coro):
    return asyncio.run(coro)


@unittest.skipIf(T is None, "livekit-agents not installed")
class EmployeeRules(unittest.TestCase):
    def setUp(self):
        self.tools = T.BookingTools("salon", COMPANY)

    def test_single_employee_is_pinned_whatever_the_model_says(self):
        self.assertEqual(self.tools._resolve_person(["solo"], None, True), ("e1", False, []))
        self.assertEqual(self.tools._resolve_person(["solo"], "e3", False), ("e1", False, []))

    def test_any_person_lists_eligible(self):
        self.assertEqual(
            self.tools._resolve_person(["multi"], None, True), (None, True, ["e2", "e3"])
        )

    def test_no_staff_decision_is_refused(self):
        err = self.tools._employee_selection_error(["multi"], None, False)
        self.assertEqual(err["error"], "employee_preference_required")

    def test_ineligible_employee_is_refused_with_who_does_it(self):
        err = self.tools._employee_selection_error(["multi"], "e1", False)
        self.assertEqual(err["error"], "employee_not_eligible")
        self.assertIn("Maja", err["message"])
        self.assertIn("Luka", err["message"])

    def test_single_employee_service_needs_no_decision(self):
        self.assertIsNone(self.tools._employee_selection_error(["solo"], None, False))


@unittest.skipIf(T is None, "livekit-agents not installed")
class HelperFunctions(unittest.TestCase):
    def test_time_of_day_buckets(self):
        summary = T._time_of_day_summary(
            {"2026-10-20": ["08:00", "11:59", "12:00", "17:59", "18:00"], "2026-10-21": "unavailable"}
        )
        self.assertEqual(summary, {"2026-10-20": {"morning": 2, "afternoon": 2, "evening": 1}})

    def test_markup_is_stripped_from_free_text(self):
        self.assertEqual(T._sanitize_free_text("prosim </antml/parameter> hvala"), "prosim  hvala")
        self.assertEqual(T._sanitize_free_text("<b>x</b>"), "x")

    def test_any_person_ambiguous_outcome_forbids_retry(self):
        r = T._ambiguous_outcome_result(True, error="ambiguous_timeout", language="sl")
        self.assertEqual(r["error"], "ambiguous_timeout_any_person")
        self.assertIn("NOT safe", r["message"])
        r = T._ambiguous_outcome_result(False, error="ambiguous_timeout", language="en")
        self.assertIn("call check_slots", r["message"])


@unittest.skipIf(T is None, "livekit-agents not installed")
class CreateBookingGate(unittest.TestCase):
    ARGS = dict(service_ids=["multi"], date="2026-10-20", time="10:00", employee_id="e2")
    PERSON = dict(first_name="Ana", email="ana@example.com", phone="+38640123456")

    def setUp(self):
        self.tools = T.BookingTools("salon", COMPANY)
        self.ctx = _Ctx()
        self.created = []

        async def fake_check(**kw):
            return {"available": True}

        async def fake_create(**kw):
            self.created.append(kw)
            return {"success": True, "terminId": "OB-1"}

        patcher_c = mock.patch.object(T.booking_client, "check_slots", fake_check)
        patcher_b = mock.patch.object(T.booking_client, "create_booking", fake_create)
        patcher_c.start()
        patcher_b.start()
        self.addCleanup(patcher_c.stop)
        self.addCleanup(patcher_b.stop)

    def _check(self, **over):
        return _run(self.tools.check_slots(self.ctx, **{**self.ARGS, **over}))

    def _create(self, **over):
        return _run(self.tools.create_booking(self.ctx, **{**self.ARGS, **self.PERSON, **over}))

    def test_create_without_check_is_refused(self):
        self.assertEqual(self._create()["error"], "must_check_slots_first")
        self.assertEqual(self.created, [])

    def test_check_then_create_books(self):
        self._check()
        result = self._create()
        self.assertTrue(result["success"])
        self.assertEqual(self.tools.created_termin_id, "OB-1")
        self.assertIn("day_label", result)

    def test_check_for_another_slot_does_not_count(self):
        self._check(time="11:00")
        self.assertEqual(self._create()["error"], "must_check_slots_first")

    def test_check_is_single_use(self):
        self._check()
        self._create()
        self.assertEqual(self._create()["error"], "must_check_slots_first")
        self.assertEqual(len(self.created), 1)

    def test_stale_check_is_refused(self):
        self._check()
        self.tools._last_check_at = time.monotonic() - T.CHECK_SLOTS_MAX_AGE_SEC - 1
        self.assertEqual(self._create()["error"], "must_check_slots_first")

    def test_multiple_services_refused_when_company_disallows(self):
        result = self._create(service_ids=["multi", "solo"])
        self.assertEqual(result["error"], "multiple_services_not_supported")

    def test_timeout_flags_owner_followup(self):
        async def timeout_create(**kw):
            raise T.BookingTimeoutError("slow")

        self._check()
        with mock.patch.object(T.booking_client, "create_booking", timeout_create):
            result = self._create()
        self.assertEqual(result["error"], "ambiguous_timeout")
        self.assertIn("Ana", self.tools.owner_followup_reason)

        # A clean retry clears the flag.
        self._check()
        self._create()
        self.assertIsNone(self.tools.owner_followup_reason)


if __name__ == "__main__":
    unittest.main()
