"""What the owner wants asked before a booking, and told after it.

Set in Jedro+ (Receptionist+ settings, receptionist_settings.booking_questions
and .booking_advice, phase11), each a list of
    {"text": "Ali imate alergijo na barvo za lase?", "service_ids": ["18", "21"]}
where an empty service_ids means every service.

Questions: create_booking refuses to book until every question for the
chosen services has an answer, and writes the answers into the
appointment's note for the owner. Advice ("Pridite z umitimi lasmi."): told
to the caller after the booking (create_booking returns it) and used to
answer "is there anything I should know before?".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

MAX_ITEMS = 10
MAX_TEXT = 300


def _clean(text) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()[:MAX_TEXT]


@dataclass(frozen=True)
class Item:
    text: str
    service_ids: frozenset = field(default_factory=frozenset)

    def applies_to(self, service_ids) -> bool:
        return not self.service_ids or bool(self.service_ids & {str(s) for s in service_ids})


def parse_items(raw) -> tuple[Item, ...]:
    """Tolerant of whatever the column holds: bad rows are skipped."""
    if not isinstance(raw, list):
        return ()
    items = []
    for row in raw[:MAX_ITEMS]:
        if isinstance(row, str):
            row = {"text": row}
        if not isinstance(row, dict):
            continue
        text = _clean(row.get("text"))
        if not text:
            continue
        ids = row.get("service_ids") or []
        ids = frozenset(str(i) for i in ids if str(i).strip()) if isinstance(ids, list) else frozenset()
        items.append(Item(text=text, service_ids=ids))
    return tuple(items)


@dataclass(frozen=True)
class BookingExtras:
    questions: tuple[Item, ...] = ()
    advice: tuple[Item, ...] = ()

    @classmethod
    def from_settings(cls, row: dict | None) -> "BookingExtras":
        row = row or {}
        return cls(parse_items(row.get("booking_questions")), parse_items(row.get("booking_advice")))

    def __bool__(self) -> bool:
        return bool(self.questions or self.advice)

    def questions_for(self, service_ids) -> list[str]:
        return [q.text for q in self.questions if q.applies_to(service_ids)]

    def advice_for(self, service_ids) -> list[str]:
        return [a.text for a in self.advice if a.applies_to(service_ids)]


def answers_note(answers: list[str]) -> str:
    """The answers as one line for the appointment note the owner reads."""
    parts = [_clean(a) for a in answers if _clean(a)]
    return ("Odgovori stranke: " + "; ".join(parts) + ".") if parts else ""


def _scope(item: Item, names: dict[str, str]) -> str:
    if not item.service_ids:
        return "every service"
    known = [names.get(i, "") for i in sorted(item.service_ids)]
    return ", ".join(n for n in known if n) or "certain services (none currently offered)"


def render_extras_block(extras: BookingExtras | None, company_data: dict) -> str:
    """The owner's questions and advice for the system prompt. The texts
    are in the owner's language; the model says them in the call's."""
    if not extras:
        return ""
    names = {str(s.get("id")): str(s.get("naziv") or "") for s in company_data.get("services", [])}
    lines: list[str] = []
    if extras.questions:
        lines += [
            "BEFORE-BOOKING QUESTIONS (set by the owner):",
            "Once the service, time and the caller's details are settled and before create_booking, "
            "ask the questions for the service being booked: one at a time, briefly, in the call's "
            "language, without saying the owner set them. Pass every answer to create_booking in "
            "`answers` as \"<question> — <answer>\" (\"ne ve\" / \"ni odgovora\" if the caller doesn't "
            "know or won't say). Never ask questions for services that aren't being booked.",
        ]
        lines += [f"- For {_scope(q, names)}: \"{q.text}\"" for q in extras.questions]
    if extras.advice:
        if lines:
            lines.append("")
        lines += [
            "PREPARATION ADVICE (set by the owner):",
            "After a successful booking create_booking returns the advice for that service: tell it "
            "to the caller in one or two short sentences. Also use it to answer questions such as "
            "\"is there anything I should know before?\".",
        ]
        lines += [f"- For {_scope(a, names)}: \"{a.text}\"" for a in extras.advice]
    return "\n".join(lines)
