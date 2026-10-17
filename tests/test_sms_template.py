import unittest

from agent.sms_client import (
    booking_confirmation_text,
    choose_confirmation_template,
    normalize_sms,
    render_confirmation_template,
)

ROW = {
    "sms_type_potrdilo": "LP",
    "lastna_predloga_potrdilo": "Pozdravljeni {{ime}}! Termin {{datum}} ob {{cas}}. Lep pozdrav, {{ime_podjetja}}",
    "lastna_predloga_potrdilo_en": "Hi {{ime}}! Your appointment is on {{datum}} at {{cas}}. {{ime_podjetja}}",
    "language": "slo",
    "Naziv Podjetja": "Jedro Kozmetika",
    "Naslov podjetja": "Ljubljanska cesta 5",
    "Kontaktni telefon": "+38640111222",
    "Kontaktni_email": "info@example.com",
    "appt_management_link": "https://portal.jedroplus.com/manage/jedro",
}


class ChooseTemplate(unittest.TestCase):
    def test_company_language_uses_main_template(self):
        self.assertTrue(choose_confirmation_template(ROW, "sl").startswith("Pozdravljeni"))

    def test_translation_for_other_language(self):
        self.assertTrue(choose_confirmation_template(ROW, "en").startswith("Hi"))

    def test_missing_translation_falls_back_to_built_in(self):
        self.assertIsNone(choose_confirmation_template(ROW, "de"))

    def test_ai_mode_uses_built_in(self):
        self.assertIsNone(choose_confirmation_template({**ROW, "sms_type_potrdilo": "AI"}, "sl"))

    def test_no_english_version_falls_back_to_built_in(self):
        self.assertIsNone(choose_confirmation_template({**ROW, "lastna_predloga_potrdilo_en": ""}, "en"))

    def test_jezik_posiljanja_wins(self):
        row = {**ROW, "jezik posiljanja": "eng"}
        self.assertTrue(choose_confirmation_template(row, "en").startswith("Pozdravljeni"))

    def test_english_date_format(self):
        text = render_confirmation_template(
            "{{datum}}", company_row=ROW, service_names=[], date="2026-10-20",
            time_hhmm="14:00", language="en",
        )
        self.assertEqual(text, "20/10/2026")

    def test_no_row(self):
        self.assertIsNone(choose_confirmation_template(None, "sl"))


class RenderTemplate(unittest.TestCase):
    def render(self, template, **kw):
        args = dict(
            company_row=ROW,
            service_names=["Pedikura"],
            date="2026-10-20",
            time_hhmm="14:00",
            first_name="Tim",
            last_name="Kogej",
            employee_name="Maja",
        )
        args.update(kw)
        return render_confirmation_template(template, **args)

    def test_placeholders_and_diacritics(self):
        text = self.render(ROW["lastna_predloga_potrdilo"].replace("Termin", "Vaš termin"))
        self.assertEqual(
            text, "Pozdravljeni Tim! Vas termin 20.10.2026 ob 14:00. Lep pozdrav, Jedro Kozmetika"
        )

    def test_all_placeholders(self):
        text = self.render(
            "{{storitev}} {{ime_izvajalca}} {{priimek}} {{naslov}} {{telefon_podjetja}} "
            "{{email_podjetja}} {{čas}} {{ neznano }}"
        )
        self.assertEqual(
            text,
            "Pedikura Maja Kogej Ljubljanska cesta 5 +38640111222 info@example.com 14:00",
        )

    def test_long_text_keeps_link(self):
        text = self.render("Pozdravljeni {{ime}}. " + "Zelo dolgo besedilo. " * 12 + "{{povezava_prenarocanje}}")
        self.assertLessEqual(len(text), 160)
        self.assertTrue(text.endswith("portal.jedroplus.com/manage/jedro"))

    def test_newlines_collapse(self):
        self.assertEqual(self.render("A\n\nB {{ime}}"), "A B Tim")


class BuiltIn(unittest.TestCase):
    def test_built_in_is_gsm_safe(self):
        text = booking_confirmation_text(
            company_name="Jedro Kozmetika",
            service_names=["Pedikura"],
            date="2026-10-20",
            time_hhmm="14:00",
            language="sl",
        )
        self.assertNotIn("—", text)
        self.assertNotIn("š", text)
        self.assertIn("ob 14:00", text)

    def test_normalize_german(self):
        self.assertEqual(normalize_sms("Grüße – bis bald"), "Gruße - bis bald")


if __name__ == "__main__":
    unittest.main()
