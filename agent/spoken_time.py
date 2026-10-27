"""Clock times the way people say them on the phone, made in code.

The model used to word HH:MM itself and got half hours wrong: "14:30" came
out as "ob dveh in pol" (English "half past two" in Slovenian words) where
Slovenian says "ob pol treh", and the caller's "ob pol dveh" was then heard
as 14:30 (live call 2026-10-26). The tools now hand the model ready phrases
("ob pol treh popoldan") and a time -> phrase map, so it only copies them.

Only Slovenian and English have phrases; for other languages the functions
return None and the model words the time itself, as before.
"""

from __future__ import annotations

import re

# Genitive plural after "ob" ("ob dveh", "ob pol treh").
_SL_AT = {1: "enih", 2: "dveh", 3: "treh", 4: "štirih", 5: "petih", 6: "šestih", 7: "sedmih",
          8: "osmih", 9: "devetih", 10: "desetih", 11: "enajstih", 12: "dvanajstih"}
# Nominative ("četrt čez dve", "tri četrt na tri").
_SL_NOM = {1: "ena", 2: "dve", 3: "tri", 4: "štiri", 5: "pet", 6: "šest", 7: "sedem",
           8: "osem", 9: "devet", 10: "deset", 11: "enajst", 12: "dvanajst"}
_EN = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
       8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}


def _parse(hhmm: str) -> tuple[int, int] | None:
    try:
        h, m = str(hhmm)[:5].split(":")
        h, m = int(h), int(m)
    except (ValueError, TypeError):
        return None
    return (h, m) if 0 <= h < 24 and 0 <= m < 60 else None


def _h12(h: int) -> int:
    return h % 12 or 12


def _sl_part(h: int) -> str:
    if h < 9:
        return "zjutraj"
    if h < 12:
        return "dopoldan"
    if h < 18:
        return "popoldan"
    return "zvečer"


def _en_part(h: int) -> str:
    if h < 12:
        return "in the morning"
    if h < 18:
        return "in the afternoon"
    return "in the evening"


def _sl_bare(h: int, m: int) -> str | None:
    """Without "ob" and the part of the day: "pol treh", "dveh"."""
    nxt = _h12(h + 1)
    if m == 0:
        return _SL_AT[_h12(h)]
    if m == 30:
        return f"pol {_SL_AT[nxt]}"
    if m == 15:
        return f"četrt čez {_SL_NOM[_h12(h)]}"
    if m == 45:
        return f"tri četrt na {_SL_NOM[nxt]}"
    return None


def _en_bare(h: int, m: int) -> str | None:
    if m == 0:
        return _EN[_h12(h)]
    if m == 30:
        return f"half past {_EN[_h12(h)]}"
    if m == 15:
        return f"quarter past {_EN[_h12(h)]}"
    if m == 45:
        return f"quarter to {_EN[_h12(h + 1)]}"
    return None


def spoken_time(hhmm: str, language: str = "sl") -> str | None:
    """"14:30" -> "ob pol treh popoldan" / "at half past two in the
    afternoon". None for times without a phrase (14:10) or other languages."""
    parsed = _parse(hhmm)
    if not parsed:
        return None
    h, m = parsed
    if language == "sl":
        bare = _sl_bare(h, m)
        if not bare:
            return None
        if h == 12 and m == 0:
            return "ob dvanajstih"
        return f"ob {bare} {_sl_part(h)}"
    if language == "en":
        bare = _en_bare(h, m)
        if not bare:
            return None
        if h == 12 and m == 0:
            return "at noon"
        return f"at {bare} {_en_part(h)}"
    return None


def spoken_time_list(times: list[str], language: str = "sl") -> str | None:
    """Several times of one part of the day as one phrase, the part of the
    day left out (the caller just chose it): "ob enih, pol dveh, dveh ali pol
    treh". None if any time has no phrase."""
    if language == "sl":
        words = [_sl_bare(*p) if (p := _parse(t)) else None for t in times]
        lead, last = "ob", "ali"
    elif language == "en":
        words = [_en_bare(*p) if (p := _parse(t)) else None for t in times]
        lead, last = "at", "or"
    else:
        return None
    if not words or any(w is None for w in words):
        return None
    if len(words) == 1:
        return f"{lead} {words[0]}"
    return f"{lead} {', '.join(words[:-1])} {last} {words[-1]}"


def spoken_time_map(times, language: str = "sl") -> dict[str, str]:
    """HH:MM -> phrase for every time that has one."""
    out = {}
    for t in sorted({str(t)[:5] for t in times}):
        phrase = spoken_time(t, language)
        if phrase:
            out[t] = phrase
    return out


def times_said(text: str, candidates, language: str = "sl") -> set[str]:
    """Which of `candidates` (HH:MM) `text` names. Full phrases first ("ob
    pol dveh popoldan"); without the part of the day, the bare form ("ob pol
    dveh"), which may match two candidates (1:30 and 13:30)."""
    low = " ".join(str(text or "").lower().replace("pól", "pol").split())
    cands = sorted({str(t)[:5] for t in candidates})
    found = {t for t in cands if (p := spoken_time(t, language)) and p.lower() in low}
    if found:
        return found
    lead = "ob" if language == "sl" else "at"
    bare = _sl_bare if language == "sl" else _en_bare if language == "en" else None
    if not bare:
        return set()
    for t in cands:
        parsed = _parse(t)
        word = bare(*parsed) if parsed else None
        if word and re.search(rf"\b{lead}\s+{re.escape(word)}\b", low):
            found.add(t)
    return found
