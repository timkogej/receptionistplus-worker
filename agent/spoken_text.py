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


class SpokenTextFilter:
    """Last pass over reply text before TTS, a sentence at a time (TTS
    speaks by sentence anyway): stored email addresses become "vaš
    e-naslov", and the Slovenian read-back lead-in is cut."""

    def __init__(self, emails, language: str) -> None:
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
        if self._language == "sl":
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
