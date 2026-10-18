import unittest

from agent.spoken_text import SpokenTextFilter


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


if __name__ == "__main__":
    unittest.main()
