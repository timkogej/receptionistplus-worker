"""Spoken-date helpers shared by the prompt's date table (worker.py) and the
booking tools' per-date labels (tools.py).

Both places must produce the exact same words for a given date: the tools'
labels exist so the model can copy a weekday+date pair from ONE string
instead of combining a weekday from one place with a date from another
(observed 2026-09-21: "torek, osemindvajsetega septembra" — the 28th was a
Monday). Keeping one implementation guarantees the table row and the tool
label for a date never disagree.
"""

from __future__ import annotations

from datetime import date

SLOVENIAN_WEEKDAYS = [
    "ponedeljek",
    "torek",
    "sreda",
    "četrtek",
    "petek",
    "sobota",
    "nedelja",
]

# "v <weekday>" takes the accusative: v ponedeljek, v sredo, v soboto.
SLOVENIAN_WEEKDAYS_ACCUSATIVE = [
    "ponedeljek",
    "torek",
    "sredo",
    "četrtek",
    "petek",
    "soboto",
    "nedeljo",
]

SLOVENIAN_MONTHS_GENITIVE = {
    1: "januarja",
    2: "februarja",
    3: "marca",
    4: "aprila",
    5: "maja",
    6: "junija",
    7: "julija",
    8: "avgusta",
    9: "septembra",
    10: "oktobra",
    11: "novembra",
    12: "decembra",
}

_ORDINAL_ONES = {
    1: "prvega",
    2: "drugega",
    3: "tretjega",
    4: "četrtega",
    5: "petega",
    6: "šestega",
    7: "sedmega",
    8: "osmega",
    9: "devetega",
}

_ORDINAL_TEENS = {
    10: "desetega",
    11: "enajstega",
    12: "dvanajstega",
    13: "trinajstega",
    14: "štirinajstega",
    15: "petnajstega",
    16: "šestnajstega",
    17: "sedemnajstega",
    18: "osemnajstega",
    19: "devetnajstega",
}

_COMPOUND_PREFIX = {
    1: "enain",
    2: "dvain",
    3: "triin",
    4: "štiriin",
    5: "petin",
    6: "šestin",
    7: "sedemin",
    8: "osemin",
    9: "devetin",
}


def slovenian_ordinal_genitive(day: int) -> str:
    """Genitive masculine ordinal for a day-of-month, 1-31 (e.g. 3 -> "tretjega").

    Used for spoken dates ("tretjega avgusta"). Verified against all 31 values,
    including the irregular teens (sedmega/osmega, not "sedemega"/"osemega")
    and the "X-in-Y-deseto" compound forms (21-29, 31).
    """
    if day in _ORDINAL_ONES:
        return _ORDINAL_ONES[day]
    if day in _ORDINAL_TEENS:
        return _ORDINAL_TEENS[day]
    if day == 20:
        return "dvajsetega"
    if day == 30:
        return "tridesetega"
    if 21 <= day <= 29:
        return _COMPOUND_PREFIX[day - 20] + "dvajsetega"
    if day == 31:
        return _COMPOUND_PREFIX[1] + "tridesetega"
    raise ValueError(f"day out of range 1-31: {day}")


ENGLISH_WEEKDAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

# Hardcoded rather than strftime("%B"): %B is locale-dependent and the
# deployment locale isn't guaranteed to be English (same reasoning as the
# Slovenian month/weekday dicts above, which exist for the same reason).
ENGLISH_MONTHS = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}


def _english_ordinal_suffix(day: int) -> str:
    if 11 <= day % 100 <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")


# German: ordinal after "am"/"den" (weak declension, -ten).
_DE_ORDINAL = {
    1: "ersten", 2: "zweiten", 3: "dritten", 4: "vierten", 5: "fünften",
    6: "sechsten", 7: "siebten", 8: "achten", 9: "neunten", 10: "zehnten",
    11: "elften", 12: "zwölften", 13: "dreizehnten", 14: "vierzehnten",
    15: "fünfzehnten", 16: "sechzehnten", 17: "siebzehnten", 18: "achtzehnten",
    19: "neunzehnten", 20: "zwanzigsten", 30: "dreißigsten",
}
_DE_UNITS = {1: "ein", 2: "zwei", 3: "drei", 4: "vier", 5: "fünf", 6: "sechs",
             7: "sieben", 8: "acht", 9: "neun"}
GERMAN_WEEKDAYS = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]
GERMAN_MONTHS = {1: "Januar", 2: "Februar", 3: "März", 4: "April", 5: "Mai", 6: "Juni",
                 7: "Juli", 8: "August", 9: "September", 10: "Oktober", 11: "November",
                 12: "Dezember"}


def german_ordinal(day: int) -> str:
    if day in _DE_ORDINAL:
        return _DE_ORDINAL[day]
    if 21 <= day <= 29:
        return _DE_UNITS[day - 20] + "und" + "zwanzigsten"
    if day == 31:
        return "einunddreißigsten"
    raise ValueError(f"day out of range 1-31: {day}")


# Croatian: masculine ordinal in the genitive ("dvadeset prvog listopada").
_HR_ORDINAL = {
    1: "prvog", 2: "drugog", 3: "trećeg", 4: "četvrtog", 5: "petog", 6: "šestog",
    7: "sedmog", 8: "osmog", 9: "devetog", 10: "desetog", 11: "jedanaestog",
    12: "dvanaestog", 13: "trinaestog", 14: "četrnaestog", 15: "petnaestog",
    16: "šesnaestog", 17: "sedamnaestog", 18: "osamnaestog", 19: "devetnaestog",
    20: "dvadesetog", 30: "tridesetog",
}
CROATIAN_WEEKDAYS = ["ponedjeljak", "utorak", "srijeda", "četvrtak", "petak", "subota", "nedjelja"]
CROATIAN_MONTHS_GENITIVE = {1: "siječnja", 2: "veljače", 3: "ožujka", 4: "travnja",
                            5: "svibnja", 6: "lipnja", 7: "srpnja", 8: "kolovoza",
                            9: "rujna", 10: "listopada", 11: "studenoga", 12: "prosinca"}


def croatian_ordinal_genitive(day: int) -> str:
    if day in _HR_ORDINAL:
        return _HR_ORDINAL[day]
    if 21 <= day <= 29:
        return "dvadeset " + _HR_ORDINAL[day - 20]
    if day == 31:
        return "trideset prvog"
    raise ValueError(f"day out of range 1-31: {day}")


# Italian: cardinal numbers, except "primo" for the 1st.
_IT_DAY = ["", "primo", "due", "tre", "quattro", "cinque", "sei", "sette", "otto",
           "nove", "dieci", "undici", "dodici", "tredici", "quattordici", "quindici",
           "sedici", "diciassette", "diciotto", "diciannove", "venti", "ventuno",
           "ventidue", "ventitré", "ventiquattro", "venticinque", "ventisei",
           "ventisette", "ventotto", "ventinove", "trenta", "trentuno"]
ITALIAN_WEEKDAYS = ["lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica"]
ITALIAN_MONTHS = {1: "gennaio", 2: "febbraio", 3: "marzo", 4: "aprile", 5: "maggio",
                  6: "giugno", 7: "luglio", 8: "agosto", 9: "settembre", 10: "ottobre",
                  11: "novembre", 12: "dicembre"}

WEEKDAYS = {
    "sl": SLOVENIAN_WEEKDAYS,
    "en": ENGLISH_WEEKDAYS,
    "de": GERMAN_WEEKDAYS,
    "hr": CROATIAN_WEEKDAYS,
    "it": ITALIAN_WEEKDAYS,
}


def spoken_date_phrase(d: date, language: str = "sl") -> str:
    """Spoken day-of-month + month, e.g. "devetindvajsetega septembra" /
    "September 29th" / "neunundzwanzigsten September" / "dvadeset devetog
    rujna" / "ventinove settembre"."""
    if language == "en":
        return f"{ENGLISH_MONTHS[d.month]} {d.day}{_english_ordinal_suffix(d.day)}"
    if language == "de":
        return f"{german_ordinal(d.day)} {GERMAN_MONTHS[d.month]}"
    if language == "hr":
        return f"{croatian_ordinal_genitive(d.day)} {CROATIAN_MONTHS_GENITIVE[d.month]}"
    if language == "it":
        return f"{_IT_DAY[d.day]} {ITALIAN_MONTHS[d.month]}"
    return f"{slovenian_ordinal_genitive(d.day)} {SLOVENIAN_MONTHS_GENITIVE[d.month]}"


def spoken_date_label(d: date, language: str = "sl") -> str:
    """Weekday and date as ONE string, e.g. "torek, devetindvajsetega
    septembra" / "Tuesday, September 29th" / "Dienstag, den
    neunundzwanzigsten September"."""
    weekday = WEEKDAYS.get(language, SLOVENIAN_WEEKDAYS)[d.weekday()]
    if language == "de":
        return f"{weekday}, den {spoken_date_phrase(d, language)}"
    if language == "it":
        return f"{weekday} {spoken_date_phrase(d, language)}"
    return f"{weekday}, {spoken_date_phrase(d, language)}"
