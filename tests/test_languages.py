"""Language packs and switching rules (agent/languages.py). stdlib only."""

from __future__ import annotations

import unittest
from dataclasses import fields

from agent import languages as L

ALL = list(L.SUPPORTED_LANGUAGES)


class Packs(unittest.TestCase):
    def test_every_language_has_every_field_filled(self):
        for code in ALL:
            p = L.pack(code)
            for f in fields(p):
                value = getattr(p, f.name)
                self.assertTrue(value, f"{code}.{f.name} is empty")

    def test_variants_and_weekdays(self):
        for code in ALL:
            p = L.pack(code)
            self.assertEqual(len(p.weekdays_for_filler), 7, code)
            for line in p.get_slots_day_fillers:
                self.assertIn("{day}", line, code)
            self.assertGreaterEqual(len(p.redirect_phrases), 2, code)

    def test_farewell_and_promise_patterns_match_own_lines(self):
        for code in ALL:
            p = L.pack(code)
            self.assertTrue(p.farewells.search(p.silence_closing) or p.farewells.search(p.abuse_closing), code)
            self.assertTrue(p.promise_cues.search(p.transfer_message) or p.promise_cues.search(p.get_slots_range_fillers[0]), code)

    def test_unknown_language_falls_back_to_english(self):
        self.assertEqual(L.pack("fr").code, "en")
        self.assertEqual(L.pack(None).code, "en")


class Normalize(unittest.TestCase):
    def test_codes(self):
        cases = {"slo": "sl", "SL": "sl", "sl-SI": "sl", "eng": "en", "en_US": "en",
                 "de": "de", "hr": "hr", "it": "it", "fr": None, "": None, None: None}
        for raw, want in cases.items():
            self.assertEqual(L.normalize_language(raw), want, raw)


class DetectSwitch(unittest.TestCase):
    def ds(self, detected, words, previous=None, current="sl"):
        return L.detect_switch(current=current, detected=detected, words=words,
                               previous_detected=previous, allowed=ALL)

    def test_long_foreign_turn_switches(self):
        self.assertEqual(self.ds("en", 5), "en")

    def test_short_turn_needs_a_repeat(self):
        self.assertIsNone(self.ds("en", 2))
        self.assertEqual(self.ds("en", 2, previous="en"), "en")

    def test_single_word_never(self):
        self.assertIsNone(self.ds("en", 1, previous="en"))

    def test_slovenian_croatian_needs_two_long_turns(self):
        self.assertIsNone(self.ds("hr", 12))
        self.assertIsNone(self.ds("hr", 4, previous="hr"))
        self.assertEqual(self.ds("hr", 6, previous="hr"), "hr")
        self.assertIsNone(self.ds("sl", 12, current="hr"))

    def test_same_or_unsupported_or_not_allowed(self):
        self.assertIsNone(self.ds("sl", 9))
        self.assertIsNone(self.ds("fr", 9))
        self.assertIsNone(L.detect_switch(current="sl", detected="de", words=9,
                                          previous_detected=None, allowed=["sl", "en"]))


if __name__ == "__main__":
    unittest.main()
