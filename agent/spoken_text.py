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
    r"(?:telefonsko\s+)?številko(?:\s+nazaj)?(?:,?\s*da\s+(?:jo\s+)?preverim)?\s*[:.,\u2014-]+\s*"
    # "Preberem nazaj: nič šest osem ..." (live call 2026-10-26)
    r"|\b(?:vam\s+)?preberem\s+(?:vam\s+)?nazaj\s*[:.,\u2014-]+\s*",
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
    # "Pedikura ima šifro 18." (eval 2026-10-26)
    r"|\b(?:šifr[aoei]\w*|številk[aoi] storitve)\s+\d+\b"
    # "Številka je skrita, zato jo prosim navedite." (live call 2026-10-26):
    # how the line reached us is none of the caller's business.
    r"|\bštevilk\w*\s+(?:vam\s+)?(?:je\s+)?(?:skrit|zakrit|prikrit)\w*|\b(?:skrit|zakrit)\w*\s+številk\w*"
    r"|\bnumber\s+(?:is\s+)?(?:withheld|hidden|private)\b|\b(?:withheld|hidden)\s+number\b"
    # Reasoning about the prompt after a tool call: "according to the table,
    # the torek in NASLEDNJI TEDEN is 2026-11-03" (eval 2026-10-26).
    r"|\b\d{4}-\d{2}-\d{2}\b|\baccording to the\b|\bthe caller (?:says|said|wants|means)\b|\brow tagged\b"
    r"|(?-i:\bNASLEDNJI TEDEN\b|\bTA TEDEN\b)"
    # "Next Tuesday: Pedikura is service 18, any person." (eval 2026-10-26)
    r"|\bservice\s+\d+\b|\bany[ _]person\b|^\s*(?:next|this)\s+\w+day\s*:",
    re.IGNORECASE,
)
_SENTENCE = re.compile(r"[^.!?\u2026]*[.!?\u2026]+\s*|[^.!?\u2026]+$")


def drop_internal_sentences(text: str) -> str:
    if not _INTERNAL_DATA.search(text):
        return text
    kept = [m.group(0) for m in _SENTENCE.finditer(text) if not _INTERNAL_DATA.search(m.group(0))]
    return "".join(kept)


# --- Thinking aloud (R-NO-THINKING-ALOUD) ---------------------------------
# The model sometimes asks itself a question and answers it: "Naslednji
# torek je sedemindvajsetega oktobra? Ne, naslednji torek v naslednjem tednu
# je tretjega novembra." (evals 2026-10-26, about 1 in 15 date replies, even
# with the prompt rule). The question is dropped and the reply goes on from
# the correction.
_SELF_CORRECTION = {
    "sl": r"(?:ne|oziroma|pravzaprav)",
    "en": r"(?:no|actually|wait)",
    "de": r"(?:nein|eigentlich|moment)",
    "hr": r"(?:ne|zapravo|odnosno)",
    "it": r"(?:no|anzi|in realtà)",
}


def _self_correction_start(language: str) -> re.Pattern:
    word = _SELF_CORRECTION.get(language, _SELF_CORRECTION["en"])
    return re.compile(rf"^\s*{word}\b\s*[,.:!]\s*(?:(?:pozor|wait|warte)\s*[,.:!]\s*)?", re.I)


# --- Phone number first on a withheld number (caller block, R-ASK-DETAILS) --
# With the number withheld, the phone number has to come first so
# find_customer can match an existing customer before a misheard name is
# taken down (live call 2026-10-17). The prompt says so, and the model still
# asked for the name first in about half the eval calls (2026-10-26). Until
# the caller has said a phone number, a question for the name is replaced
# with the one for the number.
_NAME_QUESTION = {
    "sl": r"\b(?:ime in priimek|vaše ime|svoje ime|kako vam je ime|kako se pišete)\b",
    "en": r"\b(?:your (?:first and last |full )?name|first and last name)\b",
    "de": r"\b(?:Ihren (?:vollen )?Namen|Vor- und Nachnamen|wie heißen Sie)\b",
    "hr": r"\b(?:ime i prezime|vaše ime|kako se zovete)\b",
    "it": r"\b(?:nome e cognome|il suo nome|come si chiama)\b",
}
_PHONE_QUESTION = {
    "sl": "Na katero številko vas lahko pokličemo?",
    "en": "What number can we reach you on?",
    "de": "Darf ich zuerst Ihre Telefonnummer haben?",
    "hr": "Možete li mi najprije reći svoj broj telefona?",
    "it": "Può dirmi prima il suo numero di telefono?",
}


PHONE_FIRST_QUESTIONS = _PHONE_QUESTION


def ask_phone_before_name(text: str, language: str) -> str:
    """`text` with a question for the caller's name replaced by the one for
    their phone number (once; any further name question is dropped)."""
    pattern = _NAME_QUESTION.get(language)
    if not pattern or not re.search(pattern, text, re.I):
        return text
    out, replaced = [], False
    for m in _SENTENCE.finditer(text):
        sentence = m.group(0)
        if sentence.rstrip().endswith("?") and re.search(pattern, sentence, re.I):
            if not replaced:
                lead = sentence[: len(sentence) - len(sentence.lstrip())]
                trail = sentence[len(sentence.rstrip()):]
                out.append(lead + _PHONE_QUESTION[language] + trail)
                replaced = True
            continue
        out.append(sentence)
    return "".join(out)


# --- The business speaks in the plural (R-CONCISE) -----------------------
# "V četrtek imam prosto ..." → "imamo": the receptionist speaks for the
# salon (Tim, 2026-10-26). Only availability phrases; "nimam podatka" stays.
_SL_SINGULAR_HAVE = re.compile(
    r"\b([Nn]?[Ii])mam(?=\s+(?:še\s+|žal\s+)?(?:prost|nekaj|kar|veliko|termin|samo|le\b|na voljo|ob\b))"
)


# "Pri nas imate manikuro" ("you have") for "imamo" (live call 2026-10-26).
_SL_PRI_NAS_IMATE = re.compile(r"\b([Pp]ri nas) imate\b")
# "V petek imamo proste termine tako dopoldan kot popoldan. Vam bolj ustreza
# dopoldan ali popoldan?" names the parts of the day twice (Tim: "čudno").
_SL_PARTS_TWICE = re.compile(
    r"\bimamo\s+(?:še\s+)?(?:proste\s+termine|prosto|prostih\s+terminov|termine)\s+"
    r"(?:tako\s+|in\s+)?(?:dopoldan|zjutraj)\w*\s*(?:,\s*)?(?:in|kot|ali)\s+(?:tudi\s+)?popoldan\w*"
    r"(?:\s*(?:,|in|ali)\s*(?:tudi\s+)?zvečer)?",
    re.I,
)


def business_plural(text: str) -> str:
    text = _SL_SINGULAR_HAVE.sub(lambda m: m.group(1) + "mamo", text)
    text = _SL_PRI_NAS_IMATE.sub(lambda m: m.group(1) + " imamo", text)
    return _SL_PARTS_TWICE.sub(
        lambda m: ("I" if m.group(0)[0] == "I" else "i") + "mamo še kar nekaj prostih terminov", text
    )


# --- Price and duration nobody asked for (R-CONCISE) ---------------------
# "Pedikura traja petinštirideset minut." on every service choice, live
# calls 2026-10-26, although the prompt says only when asked. Dropped unless
# the caller's last words asked about time or price.
_ASKED_PRICE_OR_TIME = re.compile(
    r"\btraja|\bdolgo\b|\bčasa\b|\bcen[aeiou]?\b|\bstane\b|\bkoliko\b|\bevr|\bplač"
    r"|\bhow long\b|\bprice|\bcost|\bhow much\b|\bminutes?\b",
    re.I,
)
_DURATION = r"(?:minut\w*|ur[oe]\b|uri\b|uro in pol|minutes?\b|hours?\b|an hour\b)"
_PRICE = r"(?:evr\w*|euros?\b)"
_CLAUSE = re.compile(
    rf",?\s*(?:ki|which|that)\s+(?:traja|takes|lasts|stane|costs)\b[^.?!]*?(?:{_DURATION}|{_PRICE})"
    rf"(?:\s+(?:in|and)\s+(?:stane|costs)\b[^.?!]*?{_PRICE})?(?=\s*,|\s*[.?!]|\s*$)"
    # "glave, ki traja trideset minut, in refleksno" -> "glave in refleksno"
    r"(?:,(?=\s+(?:in|ali|and|or)\b))?",
    re.I,
)
_STATEMENT = re.compile(
    rf"\b(?:traja|takes|lasts)\b[^.?!]*?{_DURATION}|\b(?:stane|costs)\b[^.?!]*?{_PRICE}", re.I
)


def caller_asked_price_or_time(text: str | None) -> bool:
    return bool(_ASKED_PRICE_OR_TIME.search(text or ""))


def drop_unasked_details(text: str) -> str:
    """Duration and price out of a reply nobody asked them for: a relative
    clause goes ("manikuro, ki traja šestdeset minut" -> "manikuro"), a
    sentence that is only about them goes whole."""
    if not _STATEMENT.search(text):
        return text
    text = _CLAUSE.sub("", text)
    text = re.sub(r",\s*,", ",", text)
    kept = [m.group(0) for m in _SENTENCE.finditer(text) if not _STATEMENT.search(m.group(0))]
    return "".join(kept)


# --- Audio tags (Soniox tts-rt-v2) ---------------------------------------
# "[warm]", "[chuckles]" before a sentence change how it is spoken. Only a
# few are allowed, and only when the TTS model reads them; every other
# bracketed word is dropped so it is never read aloud.
AUDIO_TAGS = ("warm", "happy", "reassuringly", "chuckles", "laughs")
_AUDIO_TAG = re.compile(r"\s*\[([a-z][a-z \-]{0,24})\]\s*", re.I)


def audio_tags(text: str, keep: bool) -> str:
    def swap(m: re.Match) -> str:
        if keep and m.group(1).lower() in AUDIO_TAGS:
            return f"[{m.group(1).lower()}] "
        # Dropped: keep one space between words, none at the start.
        return " " if m.start() > 0 and m.end() < len(m.string) else ""

    return _AUDIO_TAG.sub(swap, text)


def without_audio_tags(text: str) -> str:
    """For the stored transcript and the summary."""
    return audio_tags(text or "", keep=False).strip()


_PHONE_DIGITS = re.compile(r"(?<![\w.,])\+?\d{2,4}(?:[ /-]?\d{2,4}){2,3}(?![\w.,]\d)")


class SpokenTextFilter:
    """Last pass over reply text before TTS, a sentence at a time (TTS
    speaks by sentence anyway): stored email addresses become "vaš
    e-naslov", the Slovenian read-back lead-in is cut, Slovenian clock
    hours are put right, and Cyrillic letters become Latin ones."""

    def __init__(
        self,
        emails,
        language: str,
        heard_digits: str | None = None,
        *,
        phone_first: bool = False,
        keep_audio_tags: bool = False,
        details_asked: bool = True,
    ) -> None:
        self._keep_audio_tags = keep_audio_tags
        # False: the caller didn't ask about price or duration this turn.
        self._details_asked = details_asked
        # Number withheld and no phone number said yet: ask for it before the name.
        self._phone_first = phone_first
        # The digits of the phone number the caller last said, if any.
        self._heard_digits = heard_digits
        self._emails = {e.strip().lower() for e in emails if e and "@" in e}
        self._say = _SPOKEN_EMAIL.get(language, _SPOKEN_EMAIL["en"])
        self._language = language
        self._held = ""
        self._after_cut: str | None = None
        # A question at the end of what was let through so far, held until
        # the next words show whether the model answers it itself.
        self._question = ""
        self._correction = _self_correction_start(language)
        # Set when a dropped correction ends a chunk; becomes _after_cut once
        # this chunk is cleaned, so the next words start a sentence.
        self._start_next: str | None = None

    def _drop_thinking_aloud(self, text: str, *, final: bool) -> str:
        text = self._question + text
        self._question = ""
        sentences = [m.group(0) for m in _SENTENCE.finditer(text)]
        if "".join(sentences) != text:  # no clean split: leave it alone
            return text
        out: list[str] = []
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            if sentence.rstrip().endswith("?"):
                if i + 1 < len(sentences):
                    nxt = sentences[i + 1]
                    m = self._correction.match(nxt)
                    if m:
                        rest = nxt[m.end():]
                        if rest:
                            sentences[i + 1] = (" " if out else "") + rest[0].upper() + rest[1:]
                        else:
                            # The correction's own words are still to come:
                            # start them as a new sentence.
                            sentences[i + 1] = ""
                            self._start_next = "." if out else ""
                        i += 1
                        continue
                elif not final:
                    self._question = sentence
                    break
            out.append(sentence)
            i += 1
        return "".join(out)

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
        if "[" in text:
            text = audio_tags(text, self._keep_audio_tags)
        text = drop_internal_sentences(text)
        if not self._details_asked:
            text = drop_unasked_details(text)
        if self._phone_first:
            text = ask_phone_before_name(text, self._language)
        if re.search(r"[\u0400-\u04FF]", text):
            text = text.translate(_CYRILLIC)
        if self._language == "sl":
            # "na številko 041 555 123": digits are read badly by TTS.
            text = _PHONE_DIGITS.sub(lambda m: spoken_sl_digits(re.sub(r"\D", "", m.group(0))), text)
            text = business_plural(text)
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
        return self._clean_chunk(out, final=False)

    def flush(self) -> str:
        out, self._held = self._held, ""
        return self._clean_chunk(out, final=True)

    def _clean_chunk(self, text: str, *, final: bool) -> str:
        text = self._clean(self._drop_thinking_aloud(text, final=final))
        if self._start_next is not None:
            self._after_cut, self._start_next = self._start_next, None
        return text
