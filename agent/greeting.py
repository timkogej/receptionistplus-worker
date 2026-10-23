"""The first thing a caller hears.

Callers decide in the first seconds whether to stay on the line, so the
opening works like a good receptionist's: the business name first (they
know they dialled the right place), a greeting for the time of day, who is
speaking, and an invitation, in under five seconds. Since 2026-10-23 this
replaces the separate "please note you are speaking with a digital voice
assistant, not a person" notice followed by "welcome to <legal name>".

The EU AI Act notice is the "I'm the digital assistant" part. It sits in
`opening`, which the worker speaks non-interruptibly, so a caller can't talk
over it; `invitation` (the question) is interruptible.

Styles, chosen by the owner in Jedro+ (receptionist_settings.greeting_style):
  classic  "Salon Maja, dober dan! Sem digitalna asistentka. Kako vam lahko pomagam?"
  sales    "... Lahko vam takoj uredim termin ali odgovorim na vprašanje."
  short    "Salon Maja, dober dan, tukaj digitalna asistentka. Izvolite?"
  custom   the owner's own text (greeting_text), after the standard notice,
           as before 2026-10-23.
An optional assistant name ("Sem Ana, digitalna asistentka.") is off by
default: without it no caller takes the assistant for a person called Ana.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

_TIMEZONE = ZoneInfo("Europe/Ljubljana")

STYLES = ("classic", "sales", "short", "custom")
DEFAULT_STYLE = "classic"

# (morning until, evening from) as hours; words per part of the day.
_TIME_OF_DAY: dict[str, tuple[int, int, tuple[str, str, str]]] = {
    "sl": (10, 18, ("dobro jutro", "dober dan", "dober večer")),
    "en": (12, 18, ("good morning", "good afternoon", "good evening")),
    "de": (10, 18, ("guten Morgen", "guten Tag", "guten Abend")),
    "hr": (10, 18, ("dobro jutro", "dobar dan", "dobra večer")),
    "it": (13, 17, ("buongiorno", "buongiorno", "buonasera")),
}


@dataclass(frozen=True)
class _Texts:
    # {tod} = time-of-day greeting; {who} = the "I'm the assistant" part.
    who: str            # without an assistant name
    who_named: str      # {assistant} = the name
    short_who: str
    short_who_named: str
    classic: str
    sales: str
    short: str


_TEXTS: dict[str, _Texts] = {
    "sl": _Texts(
        who="Sem digitalna asistentka.",
        who_named="Sem {assistant}, digitalna asistentka.",
        short_who="tukaj digitalna asistentka.",
        short_who_named="tukaj {assistant}, digitalna asistentka.",
        classic="Kako vam lahko pomagam?",
        sales="Lahko vam takoj uredim termin ali odgovorim na vprašanje.",
        short="Izvolite?",
    ),
    "en": _Texts(
        who="I'm the digital assistant.",
        who_named="I'm {assistant}, the digital assistant.",
        short_who="this is the digital assistant.",
        short_who_named="this is {assistant}, the digital assistant.",
        classic="How can I help you?",
        sales="I can book an appointment for you right away or answer a question.",
        short="How can I help?",
    ),
    "de": _Texts(
        who="Ich bin die digitale Assistentin.",
        who_named="Ich bin {assistant}, die digitale Assistentin.",
        short_who="hier spricht die digitale Assistentin.",
        short_who_named="hier spricht {assistant}, die digitale Assistentin.",
        classic="Wie kann ich Ihnen helfen?",
        sales="Ich kann Ihnen gleich einen Termin buchen oder eine Frage beantworten.",
        short="Was kann ich für Sie tun?",
    ),
    "hr": _Texts(
        who="Ja sam digitalna asistentica.",
        who_named="Ja sam {assistant}, digitalna asistentica.",
        short_who="ovdje digitalna asistentica.",
        short_who_named="ovdje {assistant}, digitalna asistentica.",
        classic="Kako vam mogu pomoći?",
        sales="Mogu vam odmah rezervirati termin ili odgovoriti na pitanje.",
        short="Izvolite?",
    ),
    "it": _Texts(
        who="Sono l'assistente digitale.",
        who_named="Sono {assistant}, l'assistente digitale.",
        short_who="qui l'assistente digitale.",
        short_who_named="qui {assistant}, l'assistente digitale.",
        classic="Come posso aiutarla?",
        sales="Posso fissarle subito un appuntamento o rispondere a una domanda.",
        short="Mi dica?",
    ),
}

# Legal forms said by nobody on the phone: "Jedroplus d.o.o." is "Jedroplus".
_LEGAL_FORM = re.compile(
    r"[\s,]+(?:"
    r"j\.?\s?d\.?\s?o\.?\s?o|d\.?\s?o\.?\s?o|d\.?\s?d|s\.?\s?p|k\.?\s?d|d\.?\s?n\.?\s?o"
    r"|gmbh|ag|e\.?\s?u|og|kg|ltd|llc|inc|s\.?\s?r\.?\s?l|s\.?\s?p\.?\s?a|s\.?\s?a\.?\s?s|obrt"
    r")\.?$",
    re.IGNORECASE,
)


def spoken_business_name(legal_name: str | None) -> str:
    """The name a receptionist would answer with: "Frizerstvo Maja, Maja
    Novak s.p." -> "Frizerstvo Maja", "Jedroplus d.o.o." -> "Jedroplus"."""
    name = re.sub(r"\s+", " ", str(legal_name or "")).strip()
    # The part after a comma is usually the owner and the legal form.
    head = name.split(",")[0].strip()
    if head:
        name = head
    for _ in range(3):
        stripped = _LEGAL_FORM.sub("", name).strip(" ,")
        if stripped == name or not stripped:
            break
        name = stripped
    return name


def time_of_day(language: str, now: datetime | None = None) -> str:
    morning_until, evening_from, words = _TIME_OF_DAY.get(language, _TIME_OF_DAY["sl"])
    hour = (now or datetime.now(_TIMEZONE)).astimezone(_TIMEZONE).hour
    if hour < morning_until:
        return words[0]
    if hour >= evening_from:
        return words[2]
    return words[1]


def normalize_style(value) -> str:
    style = str(value or "").strip().lower()
    return style if style in STYLES else DEFAULT_STYLE


@dataclass(frozen=True)
class Greeting:
    opening: str      # spoken whole, not interruptible (carries the AI notice)
    invitation: str   # interruptible
    style: str

    @property
    def text(self) -> str:
        return f"{self.opening} {self.invitation}".strip()


def _clean_assistant_name(value) -> str:
    name = re.sub(r"[^\w\s'-]", "", str(value or ""), flags=re.UNICODE)
    return re.sub(r"\s+", " ", name).strip()[:30]


def build_greeting(
    *,
    language: str,
    style: str | None,
    business_name: str | None,
    assistant_name: str | None = None,
    custom_text: str | None = None,
    disclosure: str = "",
    now: datetime | None = None,
) -> Greeting:
    """custom style: `disclosure` then the owner's text ("{name}" or
    "{podjetje}" become the business name); without a text, classic."""
    style = normalize_style(style)
    name = (business_name or "").strip()
    texts = _TEXTS.get(language, _TEXTS["en"])

    if style == "custom":
        text = (custom_text or "").strip()
        if text:
            # Plain replace, not format: owners type free text and a stray
            # "{" must not break the call.
            text = text.replace("{name}", name).replace("{podjetje}", name)
            return Greeting(opening=disclosure, invitation=text, style="custom")
        style = DEFAULT_STYLE
    tod = time_of_day(language, now)
    assistant = _clean_assistant_name(assistant_name)
    if style == "short":
        who = (texts.short_who_named if assistant else texts.short_who).format(assistant=assistant)
        head = f"{name}, {tod}" if name else tod[:1].upper() + tod[1:]
        opening = f"{head}, {who}"
    else:
        who = (texts.who_named if assistant else texts.who).format(assistant=assistant)
        head = f"{name}, {tod}!" if name else f"{tod[:1].upper()}{tod[1:]}!"
        opening = f"{head} {who}"
    invitation = texts.sales if style == "sales" else texts.short if style == "short" else texts.classic
    return Greeting(opening=opening, invitation=invitation, style=style)


def render_assistant_identity(assistant_name: str | None) -> str:
    """For the system prompt: the name the greeting used, if any, so "who am
    I speaking to?" gets the same answer."""
    name = _clean_assistant_name(assistant_name)
    if name:
        return (
            f"YOUR NAME: the business has named you {name}. If asked who you are, say you are "
            f"{name}, the business's digital assistant (never claim to be a person)."
        )
    return (
        "YOUR NAME: you have no personal name. If asked who you are, say you are the "
        "business's digital assistant (never claim to be a person)."
    )
