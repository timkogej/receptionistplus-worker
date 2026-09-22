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


def spoken_date_phrase(d: date, language: str = "sl") -> str:
    """Spoken day-of-month + month, e.g. "devetindvajsetega septembra" /
    "September 29th"."""
    if language == "en":
        return f"{ENGLISH_MONTHS[d.month]} {d.day}{_english_ordinal_suffix(d.day)}"
    return f"{slovenian_ordinal_genitive(d.day)} {SLOVENIAN_MONTHS_GENITIVE[d.month]}"


def spoken_date_label(d: date, language: str = "sl") -> str:
    """Weekday and date as ONE string, e.g. "torek, devetindvajsetega
    septembra" / "Tuesday, September 29th"."""
    weekdays = ENGLISH_WEEKDAYS if language == "en" else SLOVENIAN_WEEKDAYS
    return f"{weekdays[d.weekday()]}, {spoken_date_phrase(d, language)}"
