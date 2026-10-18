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
        if phone_at is None and "telefonsk" in a:
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


SCENARIOS: list[Scripted | Simulated] = [
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
