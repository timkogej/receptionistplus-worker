import unittest

from agent.spoken_text import SpokenTextFilter, caller_digits, fix_phone_readback, fix_slovenian_hours, spoken_sl_digits


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


class InternalData(unittest.TestCase):
    def test_sentence_with_id_dropped(self):
        out = run(["Z veseljem. Pedikura je ID 18. Imate željo po zaposlenem?"])
        self.assertEqual(" ".join(out.split()), "Z veseljem. Imate željo po zaposlenem?")

    def test_booking_id_dropped(self):
        self.assertEqual(run(["Rezervirala sem termin OB-000063. Lep dan!"]).strip(), "Lep dan!")

    def test_normal_text_kept(self):
        text = "Pedikura traja petinštirideset minut."
        self.assertEqual(run([text]), text)


class PhoneReadback(unittest.TestCase):
    HEARD = "041222333"

    def test_caller_digits_from_numerals_and_words(self):
        self.assertEqual(caller_digits("Moja številka je 041 222 333."), self.HEARD)
        self.assertEqual(caller_digits("nič štiri ena, dva dva dva, tri tri tri"), self.HEARD)
        self.assertIsNone(caller_digits("Ob dveh popoldan."))

    def test_wrong_order_is_replaced(self):
        self.assertEqual(
            fix_phone_readback("Štiri ena nič, dva dva dva, tri tri tri — je tako prav?", self.HEARD),
            "Nič štiri ena, dva dva dva, tri tri tri — je tako prav?",
        )

    def test_right_reading_untouched(self):
        text = "Nič štiri ena, dva dva dva, tri tri tri — je tako prav?"
        self.assertEqual(fix_phone_readback(text, self.HEARD), text)

    def test_nothing_heard_nothing_changed(self):
        text = "Štiri ena nič, dva dva dva, tri tri tri"
        self.assertEqual(fix_phone_readback(text, None), text)

    def test_grouping(self):
        self.assertEqual(spoken_sl_digits("0412223334"), "nič štiri ena, dva dva dva, tri tri, tri štiri")

    def test_through_the_filter(self):
        f = SpokenTextFilter(set(), "sl", heard_digits=self.HEARD)
        out = f.feed("Hvala. Štiri ena nič, dva dva dva, tri tri tri — je tako prav?") + f.flush()
        self.assertEqual(out, "Hvala. Nič štiri ena, dva dva dva, tri tri tri — je tako prav?")


class ThinkingAloud(unittest.TestCase):
    def _run(self, chunks, lang="sl"):
        f = SpokenTextFilter(set(), lang)
        out = "".join(f.feed(c) for c in chunks)
        return out + f.flush()

    def test_self_answered_question_is_dropped(self):
        self.assertEqual(
            self._run(["Z veseljem. Naslednji torek je sedemindvajsetega oktobra? ",
                       "Ne, naslednji torek v naslednjem tednu je tretjega novembra. Preverim termine."]),
            "Z veseljem. Naslednji torek v naslednjem tednu je tretjega novembra. Preverim termine.",
        )
        self.assertEqual(
            self._run(["Naslednji petek je tridesetega oktobra? Ne, pozor: naslednji petek je šestega novembra."]),
            "Naslednji petek je šestega novembra.",
        )

    def test_real_questions_stay(self):
        self.assertEqual(self._run(["Je tako prav?"]), "Je tako prav?")
        self.assertEqual(self._run(["Imate željo? ", "Ali vam je vseeno?"]), "Imate željo? Ali vam je vseeno?")
        self.assertEqual(self._run(["Ste že bili pri nas? Ne skrbite, uredim."]),
                         "Ste že bili pri nas? Ne skrbite, uredim.")

    def test_english(self):
        self.assertEqual(self._run(["Next Tuesday is the 27th? No, it's the 3rd of November."], "en"),
                         "It's the 3rd of November.")



class PhoneFirst(unittest.TestCase):
    def test_name_question_becomes_phone_question(self):
        f = SpokenTextFilter(set(), "sl", phone_first=True)
        out = f.feed("Super. Mi lahko poveste svoje ime in priimek? ") + f.flush()
        self.assertEqual(out.strip(), "Super. Na katero številko vas lahko pokličemo?")

    def test_off_leaves_it(self):
        f = SpokenTextFilter(set(), "sl")
        out = f.feed("Super. Mi lahko poveste svoje ime in priimek? ") + f.flush()
        self.assertEqual(out.strip(), "Super. Mi lahko poveste svoje ime in priimek?")



class InternalCode(unittest.TestCase):
    def test_service_code_sentence_dropped(self):
        f = SpokenTextFilter(set(), "sl")
        out = f.feed("Pedikura ima šifro 18. V sredo imamo proste termine. ") + f.flush()
        self.assertEqual(out.strip(), "V sredo imamo proste termine.")


if __name__ == "__main__":
    unittest.main()
