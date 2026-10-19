import unittest

from agent.spoken_text import SpokenTextFilter, fix_slovenian_hours


def run(parts, emails=(), language="sl"):
    f = SpokenTextFilter(set(emails), language)
    return "".join(f.feed(p) for p in parts) + f.flush()


class SpokenText(unittest.TestCase):
    def test_stored_email_split_across_chunks(self):
        out = run(["Potrdilo pošljemo na ana.", "novak@primer.si. Hvala ", "za klic!"], ["Ana.Novak@primer.si"])
        self.assertEqual(out, "Potrdilo pošljemo na vaš e-naslov. Hvala za klic!")

    def test_other_email_is_kept(self):
        out = run(["Vaš e-naslov je marko@primer.si, je tako prav?"], ["ana@primer.si"])
        self.assertIn("marko@primer.si", out)

    def test_english_wording(self):
        out = run(["We'll send it to ana@primer.si."], ["ana@primer.si"], "en")
        self.assertEqual(out, "We'll send it to your email address.")

    def test_lead_in_in_one_sentence(self):
        out = run(["Hvala, preberem vam ", "številko nazaj: ena dva tri — je tako prav?"])
        self.assertEqual(out, "Hvala, ena dva tri — je tako prav?")

    def test_lead_in_as_own_sentence(self):
        out = run(["Hvala. Preberem vam številko nazaj.", " Ena dva tri — je tako prav?"])
        self.assertEqual(out, "Hvala. Ena dva tri — je tako prav?")

    def test_lead_in_with_colon(self):
        out = run(["Hvala. Preberem vam številko, da preverim: ", "ena dva tri — je tako prav?"])
        self.assertEqual(out, "Hvala. Ena dva tri — je tako prav?")

    def test_ordinary_text_untouched(self):
        text = "Ob dveh popoldan je prosto. Torej rezerviram pedikuro — je tako prav?"
        self.assertEqual(run([text]), text)

    def test_lead_in_left_alone_in_english(self):
        text = "I'll read your number back: one two three."
        self.assertEqual(run([text], language="en"), text)


class SlovenianHours(unittest.TestCase):
    def test_24_hour_list_becomes_12_hour_with_part_of_day(self):
        self.assertEqual(
            fix_slovenian_hours("Imamo prosto ob trinajstih, štirinajstih in petnajstih."),
            "Imamo prosto ob enih, dveh in treh popoldan.",
        )

    def test_evening(self):
        self.assertEqual(fix_slovenian_hours("ob devetnajstih"), "ob sedmih zvečer")

    def test_ordinal_without_uri(self):
        self.assertEqual(fix_slovenian_hours("ob deseti dopoldan"), "ob desetih dopoldan")

    def test_24_hour_ordinal_keeps_uri(self):
        self.assertEqual(fix_slovenian_hours("Ob štirinajsti uri je prosto."), "Ob drugi uri popoldan je prosto.")

    def test_correct_forms_untouched(self):
        for text in (
            "ob tretji, četrti in peti uri",
            "ob osmih, devetih in desetih dopoldan",
            "ob deseti uri",
            "Termin ob dveh popoldan je prost.",
            "v petnajstih minutah",
        ):
            self.assertEqual(fix_slovenian_hours(text), text)

    def test_filter_applies_it_in_slovenian_only(self):
        self.assertEqual(run(["Prosto ob štirinajstih. "], language="sl"), "Prosto ob dveh popoldan. ")
        self.assertEqual(run(["ob štirinajstih "], language="en"), "ob štirinajstih ")

    def test_cyrillic_to_latin(self):
        self.assertEqual(run(["Termin термин je prost."]), "Termin termin je prost.")


if __name__ == "__main__":
    unittest.main()
