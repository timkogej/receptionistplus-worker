"""Eval scenarios: conversations the receptionist must handle correctly.

Two kinds:

- Scripted: fixed caller lines, checked turn by turn — which tools were (or
  were not) called, and an LLM judge on the agent's reply.
- Simulated: an LLM plays the caller from a persona until the booking is
  made or the turn budget runs out; checked on the end state (what was
  booked, what the owner got).

Add a scenario for every real-call regression worth never seeing again.
Company data: evals/fixtures/company_jedroplus.json (Jedro Kozmetika).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable
from zoneinfo import ZoneInfo

from evals.fake_booking import eval_day

TZ = ZoneInfo("Europe/Ljubljana")


@dataclass
class Turn:
    user: str
    judge: str | None = None
    expect_tools: tuple[str, ...] = ()
    forbid_tools: tuple[str, ...] = ()
    # What Soniox's language identification would report for this turn
    # (the live call's switch detection reads it). None = not simulated.
    detected_language: str | None = None


@dataclass
class Scripted:
    id: str
    language: str
    turns: list[Turn]
    # Extra build_receptionist_agent kwargs: profile, instructions,
    # caller_phone, customer.
    context: dict = field(default_factory=dict)


@dataclass
class Simulated:
    id: str
    language: str
    persona: str
    check: Callable[[object, object], list[str]]
    max_turns: int = 22
    opening: str = "Pozdravljeni."
    context: dict = field(default_factory=dict)
    # Optional check over the transcript (list of {"user", "agent", "tools"}).
    transcript_check: Callable[[list], list[str]] | None = None


def _next_week_weekday(weekday: int) -> str:
    today = datetime.now(TZ).date()
    next_monday = today + timedelta(days=7 - today.weekday())
    return (next_monday + timedelta(days=weekday)).isoformat()


def _check_full_booking(fake, notes) -> list[str]:
    failures = []
    if len(fake.created) != 1:
        return [f"expected exactly 1 booking, got {len(fake.created)}"]
    b = fake.created[0]
    if b["service_ids"] != ["18"]:
        failures.append(f"service_ids {b['service_ids']} != ['18'] (Pedikura)")
    want_date = _next_week_weekday(1)
    if b["date"] != want_date:
        failures.append(f"date {b['date']} != next week's Tuesday {want_date}")
    if not b["time"] >= "12:00":
        failures.append(f"time {b['time']} is not in the afternoon")
    if b["email"].lower() != "ana.novak@primer.si":
        failures.append(f"email {b['email']!r}")
    if "41555123" not in "".join(ch for ch in b["phone"] if ch.isdigit()):
        failures.append(f"phone {b['phone']!r}")
    if not b["any_person"]:
        failures.append("caller said anyone is fine, but any_person is False")
    names = [n for n, _ in fake.calls]
    i = names.index("create_booking")
    if i == 0 or names[i - 1] != "check_slots":
        failures.append(f"create_booking not directly after check_slots: {names}")
    return failures


URNIK = json.dumps({
    **{d: {"enabled": True, "intervals": [{"start": "08:00", "end": "16:00"}]}
       for d in ("Ponedeljek", "Torek", "Sreda", "Četrtek", "Petek")},
    "Sobota": {"enabled": False, "intervals": []},
    "Nedelja": {"enabled": False, "intervals": []},
})
PROFILE = {"urnik": URNIK, "Naslov podjetja": "Ljubljanska cesta 505", "Navodila chatbot": ""}
KNOWN_CALLER = {
    "caller_phone": "+38641555123",
    "customer": {"Ime": "Ana", "Priimek": "Novak", "Email stranke": "ana.novak@primer.si"},
}


def _phone_before_name(transcript: list) -> list[str]:
    """Withheld number: the phone comes first, so the lookup can find the
    customer before a misheard name is collected (live call 2026-10-17)."""
    phone_at = name_at = None
    for i, t in enumerate(transcript):
        a = (t.get("agent") or "").lower()
        if phone_at is None and re.search(r"telefonsk|na katero številko", a):
            phone_at = i
        if name_at is None and re.search(r"\bime in priimek\b|\bkako vam je ime\b|\bvaše ime\b", a):
            name_at = i
    if name_at is not None and (phone_at is None or name_at < phone_at):
        return [f"asked for the name (turn {name_at + 1}) before the phone number"]
    return []


def _check_spoken_known_customer(fake, notes) -> list[str]:
    if len(fake.created) != 1:
        return [f"expected exactly 1 booking, got {len(fake.created)}"]
    b = fake.created[0]
    failures = []
    if b["phone"] != "+38641555123":
        failures.append(f"phone {b['phone']!r} — should be the spoken number in E.164")
    if (b.get("last_name") or "") != "Novak":
        failures.append(f"last_name {b.get('last_name')!r} — should be the stored 'Novak'")
    if b["email"].lower() != "ana.novak@primer.si":
        failures.append(f"email {b['email']!r} — should be the stored one")
    return failures


def _own_appointment(days_ahead: int, time: str = "15:00:00", termin: str = "OB-EVAL-900") -> dict:
    return {
        "id": 900, "ID termina": termin, "ID stranke": "CBC-EVAL", "ID podjetja": "EVAL",
        "Datum": eval_day(days_ahead), "Čas": time, "Konec": "15:45:00",
        "Storitev": "Pedikura", "ID storitve": "18", "Oseba": "Maja Hribar", "ID Osebe": "19",
        "Status": "scheduled", "deleted_at": None,
    }


KNOWN_CALLER_WITH_ID = {
    "caller_phone": "+38641555123",
    "customer": {"ID stranke": "CBC-EVAL", "Ime": "Ana", "Priimek": "Novak", "Email stranke": "ana.novak@primer.si"},
}


def _check_cancelled(fake, notes) -> list[str]:
    ups = fake.store.updates
    failures = []
    if ups != [("OB-EVAL-900", {"Status": "cancelled"})]:
        failures.append(f"expected exactly one cancel of OB-EVAL-900, got {ups}")
    if fake.created:
        failures.append("booked a new appointment instead of only cancelling")
    return failures


def _check_moved(fake, notes) -> list[str]:
    ups = fake.store.updates
    if len(ups) != 1 or "Datum" not in ups[0][1]:
        return [f"expected exactly one move, got {ups}"]
    failures = []
    if ups[0][1]["Datum"] == _own_appointment(5)["Datum"]:
        failures.append("moved within the same day; the caller asked for another day")
    if not ups[0][1]["Čas"].startswith(("08", "09", "10", "11")):
        failures.append(f"new time {ups[0][1]['Čas']} isn't in the morning as asked")
    if fake.created:
        failures.append("created a new booking instead of moving")
    return failures


def _check_too_late(fake, notes) -> list[str]:
    failures = []
    if fake.store.updates:
        failures.append(f"changed the appointment despite the deadline: {fake.store.updates}")
    if not (notes.message or notes.callback_requested):
        failures.append("didn't pass the request to the owner (take_message)")
    return failures


def _check_link_not_change(fake, notes) -> list[str]:
    failures = []
    if fake.store.updates:
        failures.append(f"changed an appointment for an unrecognised caller: {fake.store.updates}")
    if len(fake.store.links) != 1:
        failures.append(f"expected one portal link sent, got {len(fake.store.links)}")
    return failures


def _check_back_to_back(fake, notes) -> list[str]:
    c = fake.created
    if len(c) != 2:
        return [f"expected 2 bookings, got {len(c)}"]
    failures = []
    if c[0]["date"] != c[1]["date"]:
        failures.append("the two bookings are on different days")
    a, b = (int(x["time"][:2]) * 60 + int(x["time"][3:5]) for x in c)
    if not (a + 45 <= b <= a + 45 + 30):
        failures.append(f"second starts at {c[1]['time']}, not right after the first ({c[0]['time']} + 45 min)")
    names = [x["first_name"].lower() for x in c]
    if names != ["ana", "eva"]:
        failures.append(f"names {names}, expected ana then eva")
    if c[0]["phone"] != c[1]["phone"]:
        failures.append("different phone numbers")
    return failures


def _check_together(fake, notes) -> list[str]:
    c = fake.created
    if len(c) != 2:
        return [f"expected 2 bookings, got {len(c)}"]
    failures = []
    if (c[0]["date"], c[0]["time"]) != (c[1]["date"], c[1]["time"]):
        failures.append(f"not at the same time: {c[0]['time']} / {c[1]['time']}")
    if c[0].get("employee_id") and c[0].get("employee_id") == c[1].get("employee_id"):
        failures.append("same employee for both at the same time")
    return failures


def _check_sms_confirmation_booking(fake, notes) -> list[str]:
    if len(fake.created) != 1:
        return [f"expected exactly 1 booking, got {len(fake.created)}"]
    b = fake.created[0]
    failures = []
    if b["email"]:
        failures.append(f"email {b['email']!r} — caller never gave one and none was needed")
    if "40987654" not in "".join(ch for ch in b["phone"] if ch.isdigit()):
        failures.append(f"phone {b['phone']!r}")
    return failures


def _check_known_customer_booking(fake, notes) -> list[str]:
    if len(fake.created) != 1:
        return [f"expected exactly 1 booking, got {len(fake.created)}"]
    b = fake.created[0]
    failures = []
    if b["phone"] != "+38641555123":
        failures.append(f"phone {b['phone']!r} — should be the caller's number")
    if b["email"].lower() != "ana.novak@primer.si":
        failures.append(f"email {b['email']!r} — should be the stored one")
    if (b.get("first_name") or "").lower() != "ana":
        failures.append(f"first_name {b.get('first_name')!r}")
    return failures


OWNER_EXTRAS = {
    "booking_questions": [
        {"text": "Ali imate sladkorno bolezen?", "service_ids": ["18"]},
        {"text": "Ali želite masažo z eteričnimi olji?", "service_ids": ["19"]},
    ],
    "booking_advice": [{"text": "Na pedikuro pridite brez laka na nohtih.", "service_ids": ["18"]}],
}


def _check_owner_question(fake, notes) -> list[str]:
    failures = _check_known_customer_booking(fake, notes)
    if fake.created:
        note = (fake.created[0].get("notes") or "").lower()
        if "sladkorn" not in note:
            failures.append(f"the owner's question isn't in the note: {note!r}")
        if "olj" in note:
            failures.append("asked the massage question for a pedicure")
    return failures


def _coming_weekday(weekday: int) -> str:
    """The nearest coming `weekday`, never today ("v četrtek" on a Monday is
    this week's Thursday)."""
    today = datetime.now(TZ).date()
    return (today + timedelta(days=(weekday - today.weekday() - 1) % 7 + 1)).isoformat()


# The live call of 2026-10-26: a nearby weekday, "ob pol dveh", Luka asked
# for a service he doesn't do. Thursday, unless today is Wednesday or
# Thursday (then the coming Monday), so the day is never ambiguous.
def _natural_day() -> int:
    return 0 if datetime.now(TZ).weekday() in (2, 3) else 3


_SL_DAYS_ACC = ("ponedeljek", "torek", "sredo", "četrtek", "petek", "soboto", "nedeljo")
# A month only after an ordinal day ("devetindvajsetega oktobra"), so the
# employee "Maja" doesn't count as May (eval 2026-10-27).
_SL_MONTHS = (r"\w+ega\s+(?:januarja|februarja|marca|aprila|maja|junija|julija|avgusta|septembra"
              r"|oktobra|novembra|decembra)\b")


def _check_natural_booking(fake, notes) -> list[str]:
    if len(fake.created) != 1:
        return [f"expected exactly 1 booking, got {len(fake.created)}"]
    b = fake.created[0]
    failures = []
    if b["service_ids"] != ["41"]:
        failures.append(f"service_ids {b['service_ids']} != ['41'] (Klasična nega obraza)")
    if b["date"] != _coming_weekday(_natural_day()):
        failures.append(f"date {b['date']} != {_coming_weekday(_natural_day())}")
    if b["time"] != "13:30":
        failures.append(f"time {b['time']} != 13:30 ('ob pol dveh')")
    return failures


def _sounds_natural(transcript: list) -> list[str]:
    agent = [(t.get("agent") or "") for t in transcript]
    failures = []
    users = " ".join((t.get("user") or "").lower() for t in transcript)
    said = " ".join(agent).lower()
    if "evr" not in users and "cen" not in users and re.search(r"\bevr|\bminut", said):
        failures.append("said the price or duration nobody asked for")
    full_dates = sum(1 for a in agent if re.search(_SL_MONTHS, a, re.I))
    if full_dates > 3:
        failures.append(f"said the full date in {full_dates} turns (read-back and confirmation only)")
    if re.search(r"ta\s+\w+\s+ali\s+(?:\w+\s+)?(?:v\s+)?naslednj", said):
        failures.append("asked 'this one or next week's' for a plain weekday")
    return failures


def _told_advice(transcript: list) -> list[str]:
    said = " ".join((t.get("agent") or "").lower() for t in transcript)
    if "eteričn" in said:
        return ["asked the massage question for a pedicure"]
    return [] if "lak" in said else ["didn't pass on the owner's advice (no nail polish)"]


SCENARIOS: list[Scripted | Simulated] = [
    Simulated(
        "sl-naraven-pogovor",
        "sl",
        persona=(
            "Si Tim Zupančič in kličeš v salon s skrito številko. Govoriš "
            "sproščeno in na kratko. Želiš klasično nego obraza. Najprej reci, "
            "da bi rad, da te postreže Luka; ko ti povejo, da Luka tega ne "
            "dela, reci »potem pa kdorkoli«. Za dan reci samo »v "
            f"{_SL_DAYS_ACC[_natural_day()]}«. Ko te vprašajo za del dneva, reci "
            "»popoldan«. Med ponujenimi urami izberi »ob pol dveh« (to je "
            "13.30); če je ne ponudijo, vprašaj zanjo. Potrdi, ko ti preberejo "
            "rezervacijo. Telefon: 068 663 410. Ime: Tim Zupančič. Za ceno ne "
            "sprašuj."
        ),
        check=_check_natural_booking,
        transcript_check=_sounds_natural,
        context={"sms_confirmation": True},
    ),
    Scripted(
        "sl-vikanje",
        "sl",
        [
            Turn(
                "Živjo, kako si?",
                judge=(
                    "Replies in Slovenian with formal address (vi/vam), never "
                    "informal ti/si/tebi, and offers to help."
                ),
                forbid_tools=("get_slots", "create_booking"),
            )
        ],
    ),
    Scripted(
        "sl-ponudba-brez-nastevanja",
        "sl",
        [
            Turn(
                "Kaj vse ponujate?",
                judge=(
                    "Does not read out the whole list of services; names at "
                    "most three services or categories, or asks which area "
                    "interests the caller. No prices listed for many services."
                ),
                forbid_tools=("get_slots",),
            )
        ],
    ),
    Scripted(
        "sl-cena-z-besedami",
        "sl",
        [
            Turn(
                "Koliko stane pedikura?",
                judge=(
                    "Company data: Pedikura costs 30 EUR and takes 45 minutes. "
                    "The reply gives the price as 'trideset evrov' (a word, not "
                    "digits); mentioning the 45-minute duration or offering to "
                    "book is fine. Fail only if it states something that "
                    "contradicts or goes beyond that data."
                ),
            )
        ],
    ),
    Scripted(
        "sl-neznano-gre-lastniku",
        "sl",
        [
            Turn(
                "Ali imate pred salonom parkirišče?",
                judge=(
                    "Does not claim there is or isn't parking; says it doesn't "
                    "have that information and offers to pass the question on "
                    "or have someone call back."
                ),
            ),
            Turn(
                "Ja, prosim, naj me pokličejo.",
                expect_tools=("take_message",),
            ),
        ],
    ),
    Scripted(
        "sl-zaposleni-ne-opravlja-storitve",
        "sl",
        [
            Turn(
                "Rada bi masažo glave pri Maji.",
                judge=(
                    "Tells the caller that Maja does not do head massage and "
                    "says who does (Luka is one of them), asking which they'd "
                    "like or whether anyone is fine. Does not accept Maja for "
                    "head massage (offering Maja for a service she does "
                    "perform is fine)."
                ),
                forbid_tools=("create_booking", "check_slots"),
            )
        ],
    ),
    Scripted(
        "sl-prosti-termini-iz-orodja",
        "sl",
        [
            Turn(
                "Imate jutri dopoldan prosto za pedikuro? Vseeno mi je, kdo.",
                expect_tools=("get_slots",),
                judge=(
                    "The agent has looked up tomorrow's slots (the agent's "
                    "own date table says which date and weekday tomorrow is; "
                    "naming it by weekday and date is correct). The reply "
                    "either names a few concrete morning clock times, or says "
                    "plainly that tomorrow morning is closed/full and offers "
                    "another option. Vague periods like 'early morning' "
                    "without any clock time fail. Times are spoken as words."
                ),
            )
        ],
    ),
    Scripted(
        "en-price",
        "en",
        [
            Turn(
                "Hi, how much is a manicure with polish?",
                judge=(
                    "Replies in English that it costs fifty euros, written as "
                    "words. Mentioning the duration or offering to book is fine."
                ),
            )
        ],
    ),
    Simulated(
        "sl-polna-rezervacija",
        "sl",
        persona=(
            "Si stranka, ki kliče v kozmetični salon. Ime: Ana Novak. Želiš "
            "pedikuro naslednji teden v torek popoldan, vseeno ti je, kdo jo "
            "naredi. E-pošta: ana.novak@primer.si (če jo moraš črkovati, "
            "povej 'ana pika novak afna primer pika si'). Telefon: 041 555 "
            "123. Podatke daj šele, ko te vprašajo. Če ti ponudijo več ur, "
            "izberi prvo popoldansko. Potrdi, ko ti preberejo rezervacijo."
        ),
        check=_check_full_booking,
    ),
    Scripted(
        "sl-delovni-cas",
        "sl",
        [
            Turn(
                "Kdaj imate odprto v soboto?",
                judge=(
                    "Says the business is closed on Saturday (per its opening "
                    "hours: Monday to Friday 8-16, weekend closed). May mention "
                    "the weekday hours. Times, if any, as words."
                ),
                forbid_tools=("take_message",),
            )
        ],
        context={"profile": PROFILE},
    ),
    Scripted(
        "sl-navodila-podjetja",
        "sl",
        [
            Turn(
                "Ali imate kje parkirati?",
                judge=(
                    "Says parking is free in the courtyard behind the building "
                    "(from the owner's notes). Must not offer to forward the "
                    "question to the owner, since it has the answer (asking "
                    "whether it can help with anything else is fine)."
                ),
                forbid_tools=("take_message",),
            )
        ],
        context={
            "profile": PROFILE,
            "instructions": "Parkiranje je brezplačno na dvorišču za stavbo.",
        },
    ),
    Scripted(
        "sl-clovek-brez-prenosa",
        "sl",
        [
            Turn(
                "Lahko govorim z lastnikom?",
                judge=(
                    "Does not claim it will connect or transfer the caller. "
                    "Offers to leave the owner a message / have them call back, "
                    "or asks how it can help."
                ),
            )
        ],
    ),
    Scripted(
        "sl-izven-okna",
        "sl",
        [
            Turn(
                "Imate kaj prosto čez dva meseca, sredi decembra, za pedikuro? "
                "Vseeno mi je, kdo.",
                judge=(
                    "Explains that bookings are only possible a limited number "
                    "of days ahead (about two weeks) and offers a nearer date "
                    "or to pass the request on. Does not offer any December "
                    "time slot."
                ),
                forbid_tools=("create_booking",),
            )
        ],
    ),
    Simulated(
        "sl-znana-stranka-rezervacija",
        "sl",
        persona=(
            "Si Ana Novak, stalna stranka kozmetičnega salona; kličeš s "
            "svojega telefona. Želiš pedikuro naslednji teden v sredo "
            "popoldan, vseeno ti je, kdo. Na vprašanje, ali govorijo s tabo, "
            "reci ja. Na vprašanje, ali te lahko kontaktirajo na to številko, "
            "reci ja. Svojega e-naslova ne povej, razen če te izrecno vprašajo. "
            "Izberi prvo popoldansko uro in potrdi rezervacijo."
        ),
        check=_check_known_customer_booking,
        context=KNOWN_CALLER,
    ),
    Simulated(
        "sl-vprasanje-lastnika-in-nasvet",
        "sl",
        persona=(
            "Si Ana Novak, stalna stranka kozmetičnega salona; kličeš s "
            "svojega telefona. Želiš pedikuro naslednji teden v sredo "
            "popoldan, vseeno ti je, kdo. Na vprašanje, ali govorijo s tabo, "
            "reci ja. Na vprašanje, ali te lahko kontaktirajo na to številko, "
            "reci ja. Svojega e-naslova ne povej, razen če te izrecno vprašajo. "
            "Na vprašanje o sladkorni bolezni reci, da je nimaš. "
            "Izberi prvo popoldansko uro in potrdi rezervacijo."
        ),
        check=_check_owner_question,
        context={**KNOWN_CALLER, "booking_extras": OWNER_EXTRAS},
        transcript_check=_told_advice,
    ),
    Simulated(
        "sl-skrita-stevilka-znana-stranka",
        "sl",
        persona=(
            "Si Ana Novak, stalna stranka kozmetičnega salona, a kličeš s "
            "skrito številko. Želiš pedikuro naslednji teden v sredo popoldan, "
            "vseeno ti je, kdo. Ko te vprašajo za telefonsko številko, reci "
            "041 555 123. Če te vprašajo, ali govorijo s tabo, reci ja. Če te "
            "vprašajo za ime, reci »Ana Nováková« (tako te pogosto slišijo). "
            "Svojega e-naslova ne povej, razen če te izrecno vprašajo. Izberi "
            "prvo popoldansko uro in potrdi rezervacijo."
        ),
        check=_check_spoken_known_customer,
        transcript_check=_phone_before_name,
        # Like the live test call that asked for the name first: SMS
        # confirmations and language detection on.
        context={
            "sms_confirmation": True,
            "switch_languages": ["sl", "en", "de", "hr", "it"],
            "spoken_customers": {
                "+38641555123": {"Ime": "Ana", "Priimek": "Novak", "Email stranke": "ana.novak@primer.si"},
            },
        },
    ),
    Simulated(
        "sl-skrita-stevilka-ime-ze-povedano",
        "sl",
        opening="Pozdravljeni, tukaj Ana Nováková.",
        persona=(
            "Si Ana Novak, stalna stranka kozmetičnega salona; kličeš s skrito "
            "številko in si se predstavila kot »Ana Nováková« (tako te pogosto "
            "slišijo). Želiš pedikuro naslednji teden v sredo popoldan, vseeno "
            "ti je, kdo. Ko te vprašajo za telefonsko številko, reci 041 555 "
            "123. Če te vprašajo, ali si Ana Novak ali ali govorijo s tabo, "
            "reci ja. Svojega e-naslova ne povej, razen če te izrecno vprašajo. "
            "Izberi prvo popoldansko uro in potrdi rezervacijo."
        ),
        check=_check_spoken_known_customer,
        context={
            "sms_confirmation": True,
            "spoken_customers": {
                "+38641555123": {"Ime": "Ana", "Priimek": "Novak", "Email stranke": "ana.novak@primer.si"},
            },
        },
    ),
    Simulated(
        "sl-odpoved-termina",
        "sl",
        persona=(
            "Si Ana Novak in kličeš s svojega telefona. Imaš pedikuro, ki jo "
            "želiš odpovedati, ker ne moreš priti. Ko ti povedo, kateri termin "
            "imaš, potrdi, da gre za tega, in reci, da ga želiš odpovedati. Na "
            "vprašanje za potrditev odpovedi reci ja. Nove rezervacije ne želiš."
        ),
        check=_check_cancelled,
        context={**KNOWN_CALLER_WITH_ID, "own_appointments": [_own_appointment(5)],
                 "appointment_rules": {"cancel_appt_days": "1", "reschedule_appt_days": "1"}},
    ),
    Simulated(
        "sl-prestavitev-termina",
        "sl",
        persona=(
            "Si Ana Novak in kličeš s svojega telefona. Svojo pedikuro bi rada "
            "prestavila na en dan kasneje, dopoldan. Ko ti povedo, kateri termin "
            "imaš, potrdi in povej, da ga želiš prestaviti na naslednji dan "
            "dopoldan. Izberi prvo ponujeno dopoldansko uro in potrdi prestavitev."
        ),
        check=_check_moved,
        context={**KNOWN_CALLER_WITH_ID, "own_appointments": [_own_appointment(5)],
                 "appointment_rules": {"cancel_appt_days": "1", "reschedule_appt_days": "1"}},
    ),
    Simulated(
        "sl-odpoved-prepozno",
        "sl",
        persona=(
            "Si Ana Novak in kličeš s svojega telefona. Jutrišnjo pedikuro bi "
            "rada odpovedala. Če ti rečejo, da je prepozno, prosi, naj lastnik "
            "to uredi in te pokliče, in se s tem strinjaj."
        ),
        check=_check_too_late,
        context={**KNOWN_CALLER_WITH_ID, "own_appointments": [_own_appointment(1, "10:00:00")],
                 "appointment_rules": {"cancel_appt_days": "3", "reschedule_appt_days": "3"}},
    ),
    Simulated(
        "sl-odpoved-skrita-stevilka",
        "sl",
        persona=(
            "Si Ana Novak in kličeš s skrito številko. Svojo pedikuro bi rada "
            "odpovedala. Ko te vprašajo za telefonsko številko, reci 041 555 "
            "123. Na vprašanje, ali si Ana Novak, reci ja. Če ti ponudijo SMS s "
            "povezavo za urejanje termina, reci ja."
        ),
        check=_check_link_not_change,
        context={"manage_link_tool": True,
                 "spoken_customers": {"+38641555123": {"Ime": "Ana", "Priimek": "Novak",
                                                       "Email stranke": "ana.novak@primer.si"}}},
    ),
    Simulated(
        "sl-dve-osebi-zaporedno",
        "sl",
        persona=(
            "Si Ana Novak. Želiš pedikuro zase in takoj zatem še za hčerko Evo, "
            "eno za drugo, naslednji teden v četrtek popoldan; vseeno ti je, kdo. "
            "Telefonska številka: 041 222 333. E-naslova ne daš. Izberi prvo "
            "ponujeno možnost in potrdi. Hčerka je Eva."
        ),
        check=_check_back_to_back,
        context={"sms_confirmation": True},
    ),
    Simulated(
        "sl-masaza-za-dva",
        "sl",
        persona=(
            "Si Ana Novak in bi rada refleksno masažo stopal zase in za moža "
            "Marka hkrati, ob isti uri, naslednji teden v petek dopoldan; vseeno "
            "ti je, kdo. Telefonska številka: 041 222 333. E-naslova ne daš. "
            "Izberi prvo ponujeno možnost in potrdi."
        ),
        check=_check_together,
        context={"sms_confirmation": True},
    ),
    Simulated(
        "sl-sms-potrdilo-brez-emaila",
        "sl",
        persona=(
            "Si Marko Kranjc in kličeš v kozmetični salon s skrito številko. "
            "Želiš masažo naslednji teden v ponedeljek dopoldan, vseeno ti je, "
            "kdo. Telefonska številka: 040 987 654. Če te vprašajo za e-naslov, "
            "reci, da ga nočeš dati. Izberi prvo ponujeno uro in potrdi."
        ),
        check=_check_sms_confirmation_booking,
        context={"sms_confirmation": True, "spoken_customers": {}},
    ),
    Scripted(
        "sl-telefon-stevke",
        "sl",
        [
            Turn(
                "Moja telefonska številka je 123 333 222.",
                judge=(
                    "Reads the number back digit by digit as Slovenian digit "
                    "words, grouped by three: 'ena dva tri, tri tri tri, dva dva "
                    "dva' (or the same digits without commas), and asks the "
                    "caller to confirm. Fail if any digit is said as 'eden', "
                    "'trije', 'dve' or 'štirje', if a digit is missing or extra, "
                    "or if it is said as a whole number."
                ),
                forbid_tools=("create_booking",),
            )
        ],
    ),
    Scripted(
        "sl-konkretna-ura",
        "sl",
        [
            Turn("Rad bi pedikuro naslednji torek. Vseeno mi je, kdo."),
            Turn(
                "Popoldan, ob dveh.",
                judge=(
                    "The caller asked for two in the afternoon (14:00). The "
                    "reply must NOT list several times for the caller to choose "
                    "from or ask about the part of the day. Pass if it either "
                    "says two o'clock is free and reads back the booking for "
                    "the caller to confirm (service, day, date, 'ob dveh "
                    "popoldan'), or says two o'clock is taken and offers a few "
                    "(up to four) nearby free times instead."
                ),
                forbid_tools=("create_booking",),
            ),
        ],
    ),
    Scripted(
        "sl-odpoved-brez-orodja",
        "sl",
        [
            Turn(
                "Rada bi odpovedala jutrišnji termin.",
                judge=(
                    "Does not claim to have cancelled anything. Offers to pass "
                    "the request on to the owner (or asks for details to do "
                    "so), since it cannot cancel appointments itself."
                ),
                forbid_tools=("create_booking",),
            )
        ],
    ),
    Scripted(
        "jezik-preklop-en",
        "sl",
        [
            Turn(
                "Hi, sorry, do you speak English? I'd like to book a pedicure.",
                detected_language="en",
                judge=(
                    "Replies in English (not Slovenian), helps with booking a "
                    "pedicure, e.g. asks a staff preference or a day."
                ),
            )
        ],
        context={"switch_languages": ("sl", "en", "de", "hr", "it")},
    ),
    Scripted(
        "jezik-preklop-de",
        "sl",
        [
            Turn(
                "Guten Tag, ich möchte gerne wissen, was eine Maniküre mit Lack kostet.",
                detected_language="de",
                judge=(
                    "Replies in German with formal address (Sie/Ihnen), says it "
                    "costs fifty euros written as a German word (fünfzig), not "
                    "digits."
                ),
            )
        ],
        context={"switch_languages": ("sl", "en", "de", "hr", "it")},
    ),
    Scripted(
        "jezik-preklop-hr",
        "sl",
        [
            Turn(
                "Dobar dan, zanima me koliko košta pedikura i koliko traje.",
                expect_tools=("switch_language",),
                detected_language="hr",
                judge=(
                    "Replies in Croatian (not Slovenian: e.g. 'košta', 'eura', "
                    "'minuta', not 'stane', 'evrov', 'minut'), formal address, "
                    "says thirty euros and forty-five minutes in words."
                ),
            )
        ],
        context={"switch_languages": ("sl", "en", "de", "hr", "it")},
    ),
    Scripted(
        "jezik-preklop-it",
        "sl",
        [
            Turn(
                "Buongiorno, vorrei prenotare una pedicure per la prossima settimana.",
                detected_language="it",
                judge=(
                    "Replies in Italian with formal address (Lei), helps book "
                    "the pedicure (e.g. asks staff preference or which day)."
                ),
            )
        ],
        context={"switch_languages": ("sl", "en", "de", "hr", "it")},
    ),
    Scripted(
        "jezik-slovenscina-ostane",
        "sl",
        [
            Turn(
                "Pozdravljeni, zanima me, ali imate kaj prosto za masažo glave. "
                "Okay, sorry, mislim naslednji teden.",
                forbid_tools=("switch_language",),
                judge="Replies in Slovenian.",
            )
        ],
        context={"switch_languages": ("sl", "en", "de", "hr", "it")},
    ),
    Scripted(
        "jezik-izklopljeno-ostane-sl",
        "sl",
        [
            Turn(
                "Hi, do you speak English?",
                judge=(
                    "Replies in Slovenian (switching is off for this business), "
                    "politely, offering to help in Slovenian."
                ),
            )
        ],
    ),
    Scripted(
        "jezik-zacetek-de",
        "de",
        [
            Turn(
                "Was kostet eine Pediküre und wie lange dauert sie?",
                judge=(
                    "Replies in German, formal, says thirty euros and "
                    "forty-five minutes as German words (dreißig, "
                    "fünfundvierzig)."
                ),
            )
        ],
    ),
    Scripted(
        "jezik-prosnja-v-slovenscini",
        "sl",
        [
            Turn(
                "Pozdravljeni, a lahko govoriva angleško, prosim?",
                expect_tools=("switch_language",),
                detected_language="sl",
                judge=(
                    "Replies in English and offers help. Saying something like "
                    "'of course, we can continue in English' is fine; failing is "
                    "Slovenian text, or internal narration such as 'the caller "
                    "speaks...' or 'per the rules'."
                ),
            )
        ],
        context={"switch_languages": ("sl", "en", "de", "hr", "it")},
    ),
]
