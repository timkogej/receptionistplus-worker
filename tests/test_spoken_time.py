import unittest

from agent.call_summary import booking_facts
from agent.spoken_text import SpokenTextFilter, drop_internal_sentences
from agent.spoken_time import spoken_time, spoken_time_list, spoken_time_map
from agent.tools import _say_times


class SpokenTime(unittest.TestCase):
    def test_slovenian_half_hours_count_to_the_next_hour(self):
        self.assertEqual(spoken_time("13:30"), "ob pol dveh popoldan")
        self.assertEqual(spoken_time("14:30"), "ob pol treh popoldan")
        self.assertEqual(spoken_time("12:30"), "ob pol enih popoldan")
        self.assertEqual(spoken_time("11:30"), "ob pol dvanajstih dopoldan")

    def test_slovenian_parts_of_day(self):
        self.assertEqual(spoken_time("08:00"), "ob osmih zjutraj")
        self.assertEqual(spoken_time("10:00"), "ob desetih dopoldan")
        self.assertEqual(spoken_time("12:00"), "ob dvanajstih")
        self.assertEqual(spoken_time("19:00"), "ob sedmih zvečer")

    def test_quarters_and_others(self):
        self.assertEqual(spoken_time("14:15"), "ob četrt čez dve popoldan")
        self.assertEqual(spoken_time("14:45"), "ob tri četrt na tri popoldan")
        self.assertIsNone(spoken_time("14:10"))
        self.assertIsNone(spoken_time("14:30", "de"))

    def test_english(self):
        self.assertEqual(spoken_time("13:30", "en"), "at half past one in the afternoon")
        self.assertEqual(spoken_time("14:45", "en"), "at quarter to three in the afternoon")

    def test_list(self):
        # The live call 2026-10-26 said "dveh in pol" for 14:30.
        self.assertEqual(
            spoken_time_list(["12:00", "13:00", "14:00", "14:30", "15:30"]),
            "ob dvanajstih, enih, dveh, pol treh ali pol štirih",
        )
        self.assertEqual(spoken_time_list(["09:00"]), "ob devetih")
        self.assertEqual(spoken_time_list(["13:00", "13:30"], "en"), "at one or half past one")
        self.assertIsNone(spoken_time_list(["13:00", "13:10"]))

    def test_map_and_say_times(self):
        self.assertEqual(spoken_time_map(["13:30", "13:30", "13:10"]), {"13:30": "ob pol dveh popoldan"})
        offer = {"2026-10-29": {"afternoon": ["13:00", "13:30"]}}
        self.assertEqual(_say_times(offer, "sl"), {"2026-10-29": {"afternoon": "ob enih ali pol dveh"}})


class CallerFacingText(unittest.TestCase):
    def test_withheld_number_is_never_mentioned(self):
        out = drop_internal_sentences("Hvala. Na katero številko vas lahko pokličemo? Številka je skrita, zato jo prosim navedite.")
        self.assertEqual(out.strip(), "Hvala. Na katero številko vas lahko pokličemo?")
        self.assertEqual(drop_internal_sentences("Your number is withheld. What number can we reach you on?").strip(),
                         "What number can we reach you on?")

    def test_short_readback_lead_in_is_cut(self):
        f = SpokenTextFilter([], "sl")
        out = f.feed("Preberem nazaj: nič šest osem, šest šest tri, štiri ena nič — je tako prav? ") + f.flush()
        self.assertEqual(out.strip(), "Nič šest osem, šest šest tri, štiri ena nič — je tako prav?")


class SummaryFacts(unittest.TestCase):
    def test_booked_time_comes_from_the_system(self):
        facts = booking_facts(
            [{"date": "2026-10-29", "time": "13:30", "service_ids": ["41"], "employee": "Maja Hribar"}],
            {"services": [{"id": 41, "naziv": "Klasična nega obraza"}]},
            "sl",
        )
        self.assertIn("Klasična nega obraza, četrtek 29. 10. 13:30, Maja Hribar", facts)
        self.assertEqual(booking_facts([], None, "sl"), "")


if __name__ == "__main__":
    unittest.main()


class ReadbackTime(unittest.TestCase):
    def test_time_follows_the_confirmed_read_back(self):
        from types import SimpleNamespace

        from agent.tools import BookingTools

        def msg(text):
            return SimpleNamespace(type="message", role="assistant", text_content=text)

        bt = BookingTools.__new__(BookingTools)
        bt._language = "sl"
        bt._last_slots = {"2026-10-29": ["12:00", "13:30", "14:30"]}
        ctx = SimpleNamespace(session=SimpleNamespace(history=SimpleNamespace(items=[
            msg("V četrtek popoldan imam prosto ob dvanajstih, pol dveh ali pol treh."),
            msg("Torej nega obraza v četrtek, devetindvajsetega oktobra, ob pol dveh popoldan — je tako prav?"),
            msg("Nič šest osem, šest šest tri, štiri ena nič — je tako prav?"),
        ])))
        self.assertEqual(bt._readback_time(ctx, "2026-10-29", "14:30"), "13:30")
        self.assertEqual(bt._readback_time(ctx, "2026-10-29", "13:30"), "13:30")
        self.assertEqual(bt._readback_time(ctx, "2026-10-30", "14:30"), "14:30")
