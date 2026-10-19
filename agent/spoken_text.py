"""Last pass over the receptionist's reply text before it is spoken.

Kept free of livekit imports so tests can use it directly.
"""

from __future__ import annotations

import re

_SPOKEN_EMAIL = {"sl": "vaš e-naslov", "en": "your email address", "de": "Ihre E-Mail-Adresse", "hr": "vašu e-adresu", "it": "il suo indirizzo email"}


# "Preberem vam številko nazaj:" before a phone read-back. R-PHONE-READBACK
# forbids it, and the model still wrote it in most evals (2026-10-18).
_READBACK_LEAD_IN = re.compile(
    r"\b(?:da\s+)?(?:vam\s+)?preberem\s+(?:vam\s+)?(?:še\s+)?(?:vašo\s+)?"
    r"(?:telefonsko\s+)?številko(?:\s+nazaj)?(?:,?\s*da\s+(?:jo\s+)?preverim)?\s*[:.,\u2014-]+\s*",
    re.IGNORECASE,
)
_SPOKEN_SEGMENT_END = re.compile(r"[.!?\u2026:](?=\s)")


# --- Slovenian clock hours (R-SL-HOUR-URI, R-TIME-LIST-STYLE) ------------
# The prompt asks for "ob dveh popoldan" (12-hour, with the part of the day)
# and for "uri" after an ordinal hour; the model still wrote "ob
# štirinajstih" and "ob deseti dopoldan" now and then (eval 2026-10-19).
# A run of hours after "ob" is rewritten here when it breaks those rules.

_CARD = {
    "enih": 1, "dveh": 2, "treh": 3, "štirih": 4, "petih": 5, "šestih": 6, "sedmih": 7,
    "osmih": 8, "devetih": 9, "desetih": 10, "enajstih": 11, "dvanajstih": 12,
    "trinajstih": 13, "štirinajstih": 14, "petnajstih": 15, "šestnajstih": 16,
    "sedemnajstih": 17, "osemnajstih": 18, "devetnajstih": 19, "dvajsetih": 20,
    "enaindvajsetih": 21,
}
_ORD = {
    "prvi": 1, "drugi": 2, "tretji": 3, "četrti": 4, "peti": 5, "šesti": 6, "sedmi": 7,
    "osmi": 8, "deveti": 9, "deseti": 10, "enajsti": 11, "dvanajsti": 12,
    "trinajsti": 13, "štirinajsti": 14, "petnajsti": 15, "šestnajsti": 16,
    "sedemnajsti": 17, "osemnajsti": 18, "devetnajsti": 19, "dvajseti": 20,
    "enaindvajseti": 21,
}
_CARD_BY_N = {n: w for w, n in _CARD.items() if n <= 12}
_ORD_BY_N = {n: w for w, n in _ORD.items() if n <= 12}
_HOUR_WORD = "|".join(sorted(map(re.escape, [*_CARD, *_ORD]), key=len, reverse=True))
_HOUR_RUN = re.compile(
    rf"\b(ob)\s+((?:{_HOUR_WORD})(?:\s*(?:,|\bin\b|\bali\b)\s*(?:{_HOUR_WORD}))*)\b"
    rf"(\s+uri\b)?(\s+(?:dopoldan|popoldan|zvečer|zjutraj|opoldne)\b)?",
    re.IGNORECASE,
)
_HOUR_ITEM = re.compile(rf"\b({_HOUR_WORD})\b", re.IGNORECASE)


def _fix_hour_run(m: re.Match) -> str:
    ob, run, uri, part = m.group(1), m.group(2), m.group(3) or "", m.group(4) or ""
    words = [w.lower() for w in _HOUR_ITEM.findall(run)]
    ordinal = [w in _ORD for w in words]
    values = [_ORD.get(w) or _CARD[w] for w in words]
    has_24h = any(v > 12 for v in values)
    ordinal_without_uri = any(ordinal) and not uri
    mixed = any(ordinal) and not all(ordinal)
    if not (has_24h or ordinal_without_uri or mixed):
        return m.group(0)
    keep_ordinal = all(ordinal) and bool(uri)

    def word(match: re.Match) -> str:
        n = _ORD.get(match.group(1).lower()) or _CARD[match.group(1).lower()]
        n12 = n % 12 or 12
        return _ORD_BY_N[n12] if keep_ordinal else _CARD_BY_N[n12]

    new_run = _HOUR_ITEM.sub(word, run)
    if not part:
        late = [v for v in values if v > 12]
        if late:
            part = " zvečer" if min(late) >= 18 else " popoldan"
    return f"{ob} {new_run}{uri if keep_ordinal else ''}{part}"


def fix_slovenian_hours(text: str) -> str:
    return _HOUR_RUN.sub(_fix_hour_run, text)


# Cyrillic letters in a Latin-script reply ("termin" came out once as
# "термин"); TTS can't read them as intended.
_CYRILLIC = str.maketrans({
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ж": "ž", "з": "z", "и": "i",
    "й": "j", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s",
    "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "c", "ч": "č", "ш": "š", "ы": "i", "э": "e",
    "ј": "j", "љ": "lj", "њ": "nj", "ћ": "ć", "ђ": "đ", "џ": "dž", "ь": "", "ъ": "",
    "А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D", "Е": "E", "Ж": "Ž", "З": "Z", "И": "I",
    "К": "K", "Л": "L", "М": "M", "Н": "N", "О": "O", "П": "P", "Р": "R", "С": "S", "Т": "T",
    "У": "U", "Ф": "F", "Х": "H", "Ц": "C", "Ч": "Č", "Ш": "Š",
})


# --- Phone number read-back (R-PHONE-READBACK) ------------------------------
# The model once read "041 222 333" back as "štiri ena nič, dva dva dva, …";
# a caller who says "ja" to that gets the wrong number on the booking. A run
# of six or more Slovenian digit words is compared with the digits the
# caller last said, and replaced with the right reading when they differ.

_SL_DIGIT = {
    "nič": "0", "nula": "0", "ena": "1", "en": "1", "eden": "1", "dva": "2", "dve": "2",
    "tri": "3", "trije": "3", "štiri": "4", "štirje": "4", "pet": "5", "šest": "6",
    "sedem": "7", "osem": "8", "devet": "9",
}
_SL_DIGIT_WORD = {"0": "nič", "1": "ena", "2": "dva", "3": "tri", "4": "štiri", "5": "pet",
                  "6": "šest", "7": "sedem", "8": "osem", "9": "devet"}
_DIGIT_ALT = "|".join(sorted(_SL_DIGIT, key=len, reverse=True))
_DIGIT_RUN = re.compile(rf"\b(?:{_DIGIT_ALT})(?:[\s,]+(?:{_DIGIT_ALT})){{5,}}\b", re.IGNORECASE)


def caller_digits(text: str) -> str | None:
    """The digits in what the caller said (numerals or digit words), if it
    looks like a phone number (6 to 15 digits)."""
    digits = "".join(re.findall(r"\d", text or ""))
    if len(digits) < 6:
        words = re.findall(rf"\b(?:{_DIGIT_ALT})\b", (text or "").lower())
        digits = "".join(_SL_DIGIT[w] for w in words)
    return digits if 6 <= len(digits) <= 15 else None


def spoken_sl_digits(digits: str) -> str:
    groups = [digits[i:i + 3] for i in range(0, len(digits), 3)]
    if len(groups) > 1 and len(groups[-1]) == 1:  # no lone last digit
        groups[-2:] = [groups[-2][:2], groups[-2][2:] + groups[-1]]
    return ", ".join(" ".join(_SL_DIGIT_WORD[d] for d in g) for g in groups)


def fix_phone_readback(text: str, heard: str | None) -> str:
    if not heard:
        return text

    def swap(m: re.Match) -> str:
        said = "".join(_SL_DIGIT[w.lower()] for w in re.findall(rf"\b(?:{_DIGIT_ALT})\b", m.group(0), re.IGNORECASE))
        if said == heard or abs(len(said) - len(heard)) > 1:
            return m.group(0)
        right = spoken_sl_digits(heard)
        return right[0].upper() + right[1:] if m.group(0)[0].isupper() else right

    return _DIGIT_RUN.sub(swap, text)


# Internal data the model sometimes says out loud mid-reply ("Pedikura je
# ID 18."); the sentence carrying it is dropped (any language).
_INTERNAL_DATA = re.compile(
    r"\bID\s*\d+\b|\b(?:service|employee|appointment)_id\b|\bany_person\b|\bOB-\d{3,}\b"
)
_SENTENCE = re.compile(r"[^.!?\u2026]*[.!?\u2026]+\s*|[^.!?\u2026]+$")


def drop_internal_sentences(text: str) -> str:
    if not _INTERNAL_DATA.search(text):
        return text
    kept = [m.group(0) for m in _SENTENCE.finditer(text) if not _INTERNAL_DATA.search(m.group(0))]
    return "".join(kept)


class SpokenTextFilter:
    """Last pass over reply text before TTS, a sentence at a time (TTS
    speaks by sentence anyway): stored email addresses become "vaš
    e-naslov", the Slovenian read-back lead-in is cut, Slovenian clock
    hours are put right, and Cyrillic letters become Latin ones."""

    def __init__(self, emails, language: str, heard_digits: str | None = None) -> None:
        # The digits of the phone number the caller last said, if any.
        self._heard_digits = heard_digits
        self._emails = {e.strip().lower() for e in emails if e and "@" in e}
        self._say = _SPOKEN_EMAIL.get(language, _SPOKEN_EMAIL["en"])
        self._language = language
        self._held = ""
        self._after_cut: str | None = None

    def _clean(self, text: str) -> str:
        if self._emails:
            def swap(m: re.Match) -> str:
                word = m.group(0)
                core = word.rstrip(".,;:!?")
                return self._say + word[len(core):] if core.lower() in self._emails else word
            text = re.sub(r"\S+@\S+", swap, text)
        if self._after_cut is not None:
            # Right after a cut lead-in: one space, and a capital if the cut
            # left a sentence end (or nothing) before it.
            body = text.lstrip()
            if body and self._after_cut in ("", ".", "!", "?"):
                body = body[0].upper() + body[1:]
            text = (" " if self._after_cut else "") + body
            self._after_cut = None
        text = drop_internal_sentences(text)
        if re.search(r"[\u0400-\u04FF]", text):
            text = text.translate(_CYRILLIC)
        if self._language == "sl":
            text = fix_slovenian_hours(text)
            text = fix_phone_readback(text, self._heard_digits)
            stripped = _READBACK_LEAD_IN.sub("", text)
            if stripped != text:
                stripped = stripped.rstrip()
                if stripped and _READBACK_LEAD_IN.search(text).end() < len(text):
                    # Lead-in and digits in one piece: fix the join here.
                    i = _READBACK_LEAD_IN.search(text).start()
                    head, tail = text[:i].rstrip(), text[_READBACK_LEAD_IN.search(text).end():]
                    if tail and (not head or head[-1] in ".!?"):
                        tail = tail[0].upper() + tail[1:]
                    stripped = (head + " " + tail).strip() if head else tail
                else:
                    # Nothing left: treat as a sentence end (space + capital).
                    self._after_cut = stripped[-1:] if stripped else "."
                text = stripped
        return text

    def feed(self, text: str) -> str:
        self._held += text
        ends = list(_SPOKEN_SEGMENT_END.finditer(self._held))
        if not ends:
            return ""
        cut = ends[-1].end()
        out, self._held = self._held[:cut], self._held[cut:]
        return self._clean(out)

    def flush(self) -> str:
        out, self._held = self._held, ""
        return self._clean(out)
