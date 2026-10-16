"""Call routing, time limit and billing rules (agent/call_policy.py).

stdlib only, like test_prompt_parity.py.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from agent import call_policy as cp


class NormalizePhone(unittest.TestCase):
    def test_e164_passes_through(self):
        self.assertEqual(cp.normalize_phone("+38640123456"), "+38640123456")

    def test_sip_uri(self):
        self.assertEqual(
            cp.normalize_phone("sip:+38640123456@sip.telnyx.com;user=phone"),
            "+38640123456",
        )

    def test_international_00_prefix(self):
        self.assertEqual(cp.normalize_phone("0038640123456"), "+38640123456")

    def test_national_slovenian_number(self):
        self.assertEqual(cp.normalize_phone("040 123 456"), "+38640123456")

    def test_withheld_or_garbage(self):
        for raw in (None, "", "anonymous", "+12", "abc"):
            self.assertIsNone(cp.normalize_phone(raw), raw)


class JobMetadata(unittest.TestCase):
    def test_slug_from_metadata(self):
        self.assertEqual(
            cp.company_slug_from_job_metadata('{"company_slug": " salon-x "}'),
            "salon-x",
        )

    def test_ignores_anything_else(self):
        for raw in (None, "", "salon-x", "[]", '{"company_slug": ""}', '{"company_slug": 5}', "{"):
            self.assertIsNone(cp.company_slug_from_job_metadata(raw), raw)


class CallTimeLimit(unittest.TestCase):
    def test_capped_at_max(self):
        self.assertEqual(cp.call_time_limit_sec(1000), cp.MAX_CALL_SEC)

    def test_follows_balance_plus_grace(self):
        self.assertEqual(cp.call_time_limit_sec(5), 5 * 60 + cp.CREDIT_OVERRUN_GRACE_SEC)

    def test_floor_for_tiny_or_negative_balance(self):
        self.assertEqual(cp.call_time_limit_sec(0.2), cp.MIN_CALL_SEC)
        self.assertEqual(cp.call_time_limit_sec(-3), cp.MIN_CALL_SEC)


class BillCall(unittest.TestCase):
    def test_short_call_is_free_whatever_happened(self):
        self.assertEqual(
            cp.bill_call(9, created_termin_id="OB-1", abusive=True, message_taken=True),
            (0.0, "abandoned"),
        )

    def test_one_credit_per_minute(self):
        self.assertEqual(cp.bill_call(90, created_termin_id=None), (1.5, "info_only"))

    def test_outcome_precedence(self):
        self.assertEqual(cp.bill_call(60, created_termin_id="OB-1")[1], "booked")
        self.assertEqual(
            cp.bill_call(60, created_termin_id="OB-1", abusive=True)[1], "ended_abusive"
        )
        self.assertEqual(
            cp.bill_call(60, created_termin_id=None, message_taken=True)[1], "message_taken"
        )
        self.assertEqual(
            cp.bill_call(60, created_termin_id="OB-1", message_taken=True)[1], "booked"
        )

    def test_outcome_never_changes_the_price(self):
        prices = {
            cp.bill_call(125, created_termin_id=t, abusive=a, message_taken=m)[0]
            for t in (None, "OB-1")
            for a in (False, True)
            for m in (False, True)
        }
        self.assertEqual(prices, {round(125 / 60, 2)})

    def test_duration_rounds_up_and_never_negative(self):
        t0 = datetime(2026, 10, 16, 10, 0, tzinfo=timezone.utc)
        self.assertEqual(cp.duration_seconds(t0, t0 + timedelta(seconds=10.2)), 11)
        self.assertEqual(cp.duration_seconds(t0, t0 - timedelta(seconds=5)), 0)


if __name__ == "__main__":
    unittest.main()
