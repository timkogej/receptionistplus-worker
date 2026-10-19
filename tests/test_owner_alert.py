import unittest

from agent.owner_alert import owner_alert_text, owner_language, unfinished_booking_message

COMPANY = {"services": [{"id": "18", "naziv": "Pedikura"}, {"id": "19", "naziv": "Masaža"}]}


class OwnerAlert(unittest.TestCase):
    def test_unfinished_with_slot(self):
        msg = unfinished_booking_message(
            {"service_ids": ["18"], "date": "2026-10-21", "time": "14:00"}, COMPANY, "sl"
        )
        self.assertIn("Pedikura za sredo, 21. 10. ob 14:00", msg)

    def test_unfinished_without_date(self):
        msg = unfinished_booking_message({"service_ids": ["19"], "date": None, "time": None}, COMPANY, "sl")
        self.assertIn("(Masaža)", msg)

    def test_unfinished_english(self):
        msg = unfinished_booking_message(
            {"service_ids": ["18"], "date": "2026-10-21", "time": "14:00"}, COMPANY, "en"
        )
        self.assertIn("Pedikura for Wednesday 21/10 at 14:00", msg)

    def test_callback_alert(self):
        text = owner_alert_text(
            message="Ana Novak sprašuje za ceno paketa.",
            callback_requested=True,
            kind=None,
            caller_phone="+38641555123",
            language="sl",
        )
        self.assertEqual(
            text, "Receptionist+: prosi za povratni klic (+38641555123). Ana Novak sprasuje za ceno paketa."
        )

    def test_message_withheld(self):
        text = owner_alert_text(
            message="Zamujala bo deset minut.", callback_requested=False, kind=None,
            caller_phone=None, language="sl",
        )
        self.assertTrue(text.startswith("Receptionist+: sporocilo (skrita stevilka)."))

    def test_trouble_kind_and_length(self):
        text = owner_alert_text(
            message="x" * 300, callback_requested=True, kind="trouble",
            caller_phone="+38641555123", language="sl",
        )
        self.assertTrue(text.startswith("Receptionist+: preverite rezervacijo"))
        self.assertLessEqual(len(text), 160)

    def test_owner_language(self):
        self.assertEqual(owner_language("sl"), "sl")
        self.assertEqual(owner_language("de"), "en")
        self.assertEqual(owner_language(None), "sl")


if __name__ == "__main__":
    unittest.main()
