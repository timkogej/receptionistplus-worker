"""Opening hours, owner's notes and caller lines (agent/business_info.py).
stdlib only."""

from __future__ import annotations

import json
import unittest
from datetime import datetime

from agent import business_info as B

URNIK = json.dumps({
    "Ponedeljek": {"enabled": True, "intervals": [{"start": "08:00", "end": "12:00"}, {"start": "13:00", "end": "16:00"}]},
    "Torek": {"enabled": True, "intervals": [{"start": "08:00", "end": "16:00"}]},
    "Sreda": {"enabled": False, "intervals": [{"start": "08:00", "end": "16:00"}]},
    "Sobota": {"enabled": True, "intervals": [{"start": "09:00", "end": "08:00"}]},
})


class Urnik(unittest.TestCase):
    def test_parse(self):
        s = B.parse_urnik(URNIK)
        self.assertEqual(s[0], [("08:00", "12:00"), ("13:00", "16:00")])
        self.assertEqual(s[2], [])          # disabled
        self.assertEqual(s[5], [])          # end before start is dropped
        self.assertEqual(s[6], [])          # missing day

    def test_unusable_schedules_are_none(self):
        for raw in (None, "", "{", "[]", json.dumps({"Ponedeljek": {"enabled": False}})):
            self.assertIsNone(B.parse_urnik(raw), raw)

    def test_is_open(self):
        s = B.parse_urnik(URNIK)
        monday = datetime(2026, 10, 19)
        self.assertTrue(B.is_open(s, monday.replace(hour=9)))
        self.assertFalse(B.is_open(s, monday.replace(hour=12, minute=30)))  # lunch
        self.assertFalse(B.is_open(s, monday.replace(hour=16)))             # end is exclusive
        self.assertFalse(B.is_open(s, datetime(2026, 10, 21, 10)))          # Wednesday closed
        self.assertFalse(B.is_open(None, monday))


class Blocks(unittest.TestCase):
    def test_business_block_uses_receptionist_notes_over_chatbot(self):
        block, hours = B.render_business_block(
            {"urnik": URNIK, "Navodila chatbot": "chatbot text", "Naslov podjetja": "Glavna 1"},
            "Parkiranje je brezplačno.",
            "sl",
        )
        self.assertTrue(hours)
        self.assertIn("Parkiranje je brezplačno.", block)
        self.assertNotIn("chatbot text", block)
        self.assertIn("Sreda: zaprto", block)
        self.assertIn("Naslov: Glavna 1", block)

    def test_falls_back_to_chatbot_notes_and_caps_length(self):
        block, hours = B.render_business_block({"Navodila chatbot": "x " * 5000}, None, "en")
        self.assertFalse(hours)
        self.assertLess(len(block), B.MAX_INSTRUCTIONS_CHARS + 600)

    def test_empty_profile_gives_nothing(self):
        self.assertEqual(B.render_business_block(None, None, "sl"), ("", False))

    def test_caller_lines(self):
        self.assertIn("skrita", B.render_caller_block(None, None, "sl"))
        known = B.render_caller_block(
            "+38641555123",
            {"Ime": "Ana", "Priimek": "Novak", "Email stranke": "ana@primer.si"},
            "sl",
        )
        self.assertIn("+38641555123", known)
        self.assertIn("Ana Novak", known)
        self.assertNotIn("z Ana?", known)  # the name must be declined (z Ano)
        self.assertIn("nikoli ne beri na glas", known)
        unknown = B.render_caller_block("+38641555123", None, "en")
        self.assertNotIn("existing customer", unknown)

    def test_booking_window(self):
        self.assertEqual(B.booking_window_days({"max_dnevi_rezervacija": 30}), 30)
        self.assertEqual(B.booking_window_days({"max_dnevi_rezervacija": 365}), 60)
        self.assertEqual(B.booking_window_days({"max_dnevi_rezervacija": "x"}), 14)
        self.assertEqual(B.booking_window_days(None), 14)


if __name__ == "__main__":
    unittest.main()
