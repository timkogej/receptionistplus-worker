"""The opening line (agent/greeting.py) and the owner's booking extras
(agent/booking_extras.py). Stdlib only."""

from __future__ import annotations

import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from agent.booking_extras import BookingExtras, parse_items, render_extras_block
from agent.greeting import STYLES, build_greeting, spoken_business_name, time_of_day

TZ = ZoneInfo("Europe/Ljubljana")
LANGS = ("sl", "en", "de", "hr", "it")


def at(hour: int) -> datetime:
    return datetime(2026, 10, 23, hour, 30, tzinfo=TZ)


class BusinessName(unittest.TestCase):
    def test_legal_forms_are_not_spoken(self):
        cases = {
            "Jedroplus d.o.o.": "Jedroplus",
            "Salon Lepota d.o.o": "Salon Lepota",
            "Frizerstvo Maja, Maja Novak s.p.": "Frizerstvo Maja",
            "Kozmetika Nina d. o. o.": "Kozmetika Nina",
            "Beauty Lounge GmbH": "Beauty Lounge",
            "Studio Bella S.r.l.": "Studio Bella",
            "  Salon   Ana  ": "Salon Ana",
        }
        for legal, spoken in cases.items():
            self.assertEqual(spoken_business_name(legal), spoken, legal)

    def test_empty(self):
        self.assertEqual(spoken_business_name(None), "")


class Greeting(unittest.TestCase):
    def test_classic_slovenian(self):
        g = build_greeting(language="sl", style="classic", business_name="Salon Maja", now=at(14))
        self.assertEqual(g.opening, "Salon Maja, dober dan! Sem digitalna asistentka.")
        self.assertEqual(g.invitation, "Kako vam lahko pomagam?")

    def test_time_of_day(self):
        self.assertEqual(time_of_day("sl", at(8)), "dobro jutro")
        self.assertEqual(time_of_day("sl", at(18)), "dober večer")
        self.assertEqual(time_of_day("en", at(12)), "good afternoon")
        self.assertEqual(time_of_day("it", at(17)), "buonasera")

    def test_every_style_and_language_names_the_business_and_the_ai_first(self):
        notice = {"sl": "digitalna asistentka", "en": "digital assistant", "de": "digitale Assistentin",
                  "hr": "digitalna asistentica", "it": "assistente digitale"}
        for lang in LANGS:
            for style in ("classic", "sales", "short"):
                g = build_greeting(language=lang, style=style, business_name="Salon Maja", now=at(14))
                self.assertTrue(g.opening.startswith("Salon Maja, "), (lang, style))
                # The AI notice is in the non-interruptible part.
                self.assertIn(notice[lang], g.opening, (lang, style))
                self.assertTrue(g.invitation, (lang, style))
                self.assertLess(len(g.text), 140, (lang, style))

    def test_assistant_name_is_optional(self):
        g = build_greeting(language="sl", style="classic", business_name="X", assistant_name="Ana", now=at(9))
        self.assertEqual(g.opening, "X, dobro jutro! Sem Ana, digitalna asistentka.")
        g = build_greeting(language="sl", style="short", business_name="X", assistant_name="Ana", now=at(9))
        self.assertEqual(g.text, "X, dobro jutro, tukaj Ana, digitalna asistentka. Izvolite?")

    def test_without_a_business_name(self):
        g = build_greeting(language="sl", style="classic", business_name="", now=at(14))
        self.assertEqual(g.opening, "Dober dan! Sem digitalna asistentka.")

    def test_custom_keeps_the_notice_and_falls_back_without_text(self):
        g = build_greeting(language="sl", style="custom", business_name="Salon Maja",
                           custom_text="Pozdravljeni v {name}!", disclosure="NOTICE.")
        self.assertEqual((g.opening, g.invitation, g.style), ("NOTICE.", "Pozdravljeni v Salon Maja!", "custom"))
        g = build_greeting(language="sl", style="custom", business_name="S", custom_text="  ", now=at(14))
        self.assertEqual(g.style, "classic")

    def test_unknown_style_is_classic(self):
        self.assertEqual(build_greeting(language="sl", style="loud", business_name="S").style, "classic")
        self.assertIn("custom", STYLES)


class Extras(unittest.TestCase):
    def test_bad_rows_are_skipped(self):
        items = parse_items([{"text": "  A?  "}, {"text": ""}, 5, "B?", {"text": "C?", "service_ids": "x"}])
        self.assertEqual([i.text for i in items], ["A?", "B?", "C?"])
        self.assertEqual(parse_items(None), ())

    def test_block_names_services(self):
        extras = BookingExtras.from_settings({
            "booking_questions": [{"text": "Alergije?", "service_ids": ["18"]}],
            "booking_advice": [{"text": "Brez ličil.", "service_ids": []}],
        })
        block = render_extras_block(extras, {"services": [{"id": "18", "naziv": "Pedikura"}]})
        self.assertIn('- For Pedikura: "Alergije?"', block)
        self.assertIn('- For every service: "Brez ličil."', block)
        self.assertEqual(render_extras_block(BookingExtras(), {}), "")


if __name__ == "__main__":
    unittest.main()
