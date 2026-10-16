"""Spoken dates (agent/dates.py). stdlib only."""

from __future__ import annotations

import unittest
from datetime import date

from agent import dates


class SlovenianOrdinals(unittest.TestCase):
    def test_all_days(self):
        expected = {
            1: "prvega", 2: "drugega", 3: "tretjega", 7: "sedmega", 8: "osmega",
            10: "desetega", 13: "trinajstega", 20: "dvajsetega",
            21: "enaindvajsetega", 28: "osemindvajsetega",
            29: "devetindvajsetega", 30: "tridesetega", 31: "enaintridesetega",
        }
        for day, word in expected.items():
            self.assertEqual(dates.slovenian_ordinal_genitive(day), word, day)
        for day in range(1, 32):
            dates.slovenian_ordinal_genitive(day)  # no gaps

    def test_out_of_range(self):
        for day in (0, 32):
            with self.assertRaises(ValueError):
                dates.slovenian_ordinal_genitive(day)


class Labels(unittest.TestCase):
    def test_weekday_and_date_come_from_the_same_date(self):
        # 2026-09-28 is a Monday — the 2026-09-21 incident glued "torek" to it.
        self.assertEqual(
            dates.spoken_date_label(date(2026, 9, 28), "sl"),
            "ponedeljek, osemindvajsetega septembra",
        )
        self.assertEqual(
            dates.spoken_date_label(date(2026, 9, 29), "en"),
            "Tuesday, September 29th",
        )

    def test_english_suffixes(self):
        for day, suffix in {1: "st", 2: "nd", 3: "rd", 4: "th", 11: "th",
                            12: "th", 13: "th", 21: "st", 22: "nd", 23: "rd"}.items():
            self.assertTrue(
                dates.spoken_date_phrase(date(2026, 10, day), "en").endswith(f"{day}{suffix}")
            )


if __name__ == "__main__":
    unittest.main()
