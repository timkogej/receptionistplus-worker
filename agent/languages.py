"""Language packs: every fixed line the worker speaks or matches, per language.

Before 2026-10-16 these lived as {"sl": ..., "en": ...} dicts spread over
worker.py, tools.py and sms_client.py; adding a language meant finding all of
them. Now one LanguagePack per language holds them, and `pack(code)` is the
only lookup. The model-facing prompt is separate (static_prompt.py builds it
per language from the Slovenian prompt or the English core + a language note).

Supported: the five languages Jedro+ already stores per customer
("Stranke".language: slo/eng/de/it/hr — see JEDROPLUS_LANGUAGE_CODES).

Persona: female, formal address, in every language — same voice (Soniox
"Nina" speaks all five). Lines spoken about herself use feminine forms
(sl "zaključila", hr "završila", it "chiuderò" is neutral).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SUPPORTED_LANGUAGES = ("sl", "en", "de", "hr", "it")
DEFAULT_LANGUAGE = "sl"

# English names, used inside the (English) model instructions.
LANGUAGE_NAMES = {
    "sl": "Slovenian",
    "en": "English",
    "de": "German",
    "hr": "Croatian",
    "it": "Italian",
}

# Jedro+ "Stranke".language / jp_normalize_language codes <-> ours.
JEDROPLUS_LANGUAGE_CODES = {"slo": "sl", "eng": "en", "de": "de", "it": "it", "hr": "hr"}
TO_JEDROPLUS_CODE = {v: k for k, v in JEDROPLUS_LANGUAGE_CODES.items()}


def normalize_language(code: str | None) -> str | None:
    """Our two-letter code for anything Jedro+ or Soniox calls a language
    ("slo", "sl", "SL-si", "eng", "en-US"...), or None if unsupported."""
    if not code:
        return None
    c = str(code).strip().lower().replace("_", "-")
    if c in JEDROPLUS_LANGUAGE_CODES:
        return JEDROPLUS_LANGUAGE_CODES[c]
    c = c.split("-")[0]
    return c if c in SUPPORTED_LANGUAGES else None


@dataclass(frozen=True)
class LanguagePack:
    code: str
    # Gates and closings (worker.py)
    no_credits: str
    technical_difficulty: str
    ai_disclosure: str
    abuse_closing: str
    time_limit_closing: str
    transfer_message: str
    still_there: tuple[str, ...]
    silence_closing: str
    # Said when a reply is slow to start (ReceptionistAgent.llm_node), so the
    # caller doesn't sit in silence; rotated.
    thinking_fillers: tuple[str, ...]
    redirect_phrases: tuple[str, ...]
    # Matching (worker.py watchdogs)
    promise_cues: re.Pattern
    farewells: re.Pattern
    acknowledgements: frozenset
    # Tool fillers (tools.py); {day} = weekday as used after the filler's verb
    get_slots_day_fillers: tuple[str, ...]
    get_slots_range_fillers: tuple[str, ...]
    check_slots_fillers: tuple[str, ...]
    create_booking_fillers: tuple[str, ...]
    weekdays_for_filler: tuple[str, ...]  # Monday first
    ambiguous_retry_unsafe: str
    ambiguous_checking: str


_PACKS: dict[str, LanguagePack] = {}


def _register(p: LanguagePack) -> None:
    _PACKS[p.code] = p


def pack(code: str | None) -> LanguagePack:
    return _PACKS.get(normalize_language(code) or "", _PACKS["en"])


_register(LanguagePack(
    code="sl",
    thinking_fillers=("Trenutek.", "Samo trenutek."),
    no_credits="Trenutno žal ne moremo sprejeti vašega klica. Prosimo, poskusite kasneje.",
    technical_difficulty="Oprostite, trenutno imamo tehnične težave, lastnik vas bo poklical nazaj.",
    ai_disclosure=(
        "Prosimo, upoštevajte, da govorite z digitalnim glasovnim asistentom, "
        "ne z osebo."
    ),
    abuse_closing="Hvala za klic. Klic bom zdaj zaključila. Lep dan še naprej.",
    time_limit_closing=(
        "Oprostite, klic moram zdaj zaključiti. Če česa še nismo uredili, vas "
        "bomo poklicali nazaj. Lep dan še naprej."
    ),
    transfer_message="Povezujem vas, trenutek prosim.",
    still_there=("Ste še na liniji?", "Halo, me slišite?"),
    silence_closing="Žal vas ne slišim več. Pokličite nas, prosim, še enkrat. Lep dan.",
    redirect_phrases=(
        "Oprostite, da vas prekinem — kako vam lahko pomagam?",
        "Oprostite, da vas prekinem — mi lahko na kratko poveste, kaj potrebujete?",
        "Se opravičujem za prekinitev — povejte mi, prosim, s čim vam lahko pomagam.",
    ),
    promise_cues=re.compile(
        r"trenutek|preverim\b|preverjam\b|preveril|urejam\b|uredila\b|rezerviram\b"
        r"|poskušam\b|počakajte|pogledam\b|vpisujem\b|vas vpišem\b",
        re.IGNORECASE,
    ),
    farewells=re.compile(
        r"nasvidenje|lep dan|se slišimo|se vidimo|hvala za klic|adijo", re.IGNORECASE
    ),
    acknowledgements=frozenset(
        "ja jaa jap ok okej okay oki hvala lepa najlepša vam tudi enako prav "
        "v redu dobro super velja odlično adijo adio čao nasvidenje lep dan "
        "mhm aha no sem rekel rekla že se slišimo vidimo prosim".split()
    ),
    get_slots_day_fillers=(
        "Trenutek, pogledam, kaj imamo prosto v {day}.",
        "Samo trenutek, pogledam koledar za {day}.",
        "Pogledam {day}, trenutek prosim.",
    ),
    get_slots_range_fillers=(
        "Trenutek, pogledam, kaj imamo prosto.",
        "Samo trenutek, pogledam v koledar.",
        "Pogledam koledar, trenutek prosim.",
    ),
    check_slots_fillers=("Še trenutek, prosim.", "Samo še sekundo, prosim."),
    create_booking_fillers=(
        "Še malo, pa je urejeno.",
        "Samo še sekundo, prosim.",
        "Skoraj je urejeno.",
    ),
    # "v <dan>" = accusative
    weekdays_for_filler=("ponedeljek", "torek", "sredo", "četrtek", "petek", "soboto", "nedeljo"),
    ambiguous_retry_unsafe=(
        "Rezervacijo sem morda že naredila, lastnik bo preveril in vas "
        "kontaktiral, če bo karkoli narobe"
    ),
    ambiguous_checking="Rezervacijo sem morda že naredila, preverjam, prosim počakajte",
))

_register(LanguagePack(
    code="en",
    thinking_fillers=("One moment.", "Just a moment."),
    no_credits="We're unable to take your call right now. Please try again later.",
    technical_difficulty=(
        "Sorry, we're experiencing technical difficulties right now — the "
        "owner will call you back."
    ),
    ai_disclosure=(
        "Please note that you are speaking with a digital voice assistant, "
        "not a person."
    ),
    abuse_closing="Thank you for calling. I'll end the call here. Have a good day.",
    time_limit_closing=(
        "I'm sorry, I have to end the call now. If anything is still open, "
        "we'll call you back. Have a good day."
    ),
    transfer_message="I'm putting you through now, one moment please.",
    still_there=("Are you still there?", "Hello, can you hear me?"),
    silence_closing="I'm afraid I can't hear you anymore. Please call us again. Have a good day.",
    redirect_phrases=(
        "Sorry to jump in — how can I help you?",
        "Sorry to interrupt — could you tell me briefly what you need?",
        "Apologies for cutting in — please tell me what I can help you with.",
    ),
    promise_cues=re.compile(
        r"\b(?:one|just a|give me a) moment\b|\bhold on\b|\bplease wait\b"
        r"|\blet me (?:check|look|see)\b|\bi'?ll check\b|\bcheck(?:ing)?\b"
        r"|\bi'?m (?:checking|processing|trying)\b|\bprocessing\b|\btrying\b",
        re.IGNORECASE,
    ),
    farewells=re.compile(
        r"\bgoodbye\b|\bbye\b|have a (?:great|good|nice|lovely) day"
        r"|\btalk soon\b|\btake care\b|thanks for calling",
        re.IGNORECASE,
    ),
    acknowledgements=frozenset(
        "yes yeah yep ok okay thanks thank you bye goodbye cheers great "
        "alright sure have a good nice great lovely day too same to take care "
        "mhm i said right perfect wonderful".split()
    ),
    get_slots_day_fillers=(
        "One moment, let me look at {day}.",
        "Just a second, I'm checking {day}.",
        "Let me pull up {day}, one moment.",
    ),
    get_slots_range_fillers=(
        "One moment, let me see what we have.",
        "Just a second, I'm looking at the calendar.",
        "Let me check the calendar, one moment.",
    ),
    check_slots_fillers=("Just a moment longer.", "One more second, please."),
    create_booking_fillers=(
        "Almost done.",
        "Just one more second, please.",
        "Nearly there.",
    ),
    weekdays_for_filler=("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"),
    ambiguous_retry_unsafe=(
        "I may have already made the reservation — the owner will check and "
        "get in touch if anything needs correcting"
    ),
    ambiguous_checking="I may have already made the reservation — let me check, one moment please",
))

_register(LanguagePack(
    code="de",
    thinking_fillers=("Einen Moment.", "Einen Augenblick."),
    no_credits=(
        "Leider können wir Ihren Anruf im Moment nicht entgegennehmen. Bitte "
        "versuchen Sie es später noch einmal."
    ),
    technical_difficulty=(
        "Entschuldigung, wir haben gerade technische Probleme — der Inhaber "
        "ruft Sie zurück."
    ),
    ai_disclosure=(
        "Bitte beachten Sie, dass Sie mit einem digitalen Sprachassistenten "
        "sprechen, nicht mit einer Person."
    ),
    abuse_closing="Danke für Ihren Anruf. Ich beende das Gespräch jetzt. Einen schönen Tag noch.",
    time_limit_closing=(
        "Entschuldigung, ich muss das Gespräch jetzt beenden. Falls noch etwas "
        "offen ist, rufen wir Sie zurück. Einen schönen Tag noch."
    ),
    transfer_message="Ich verbinde Sie, einen Moment bitte.",
    still_there=("Sind Sie noch dran?", "Hallo, hören Sie mich?"),
    silence_closing=(
        "Ich kann Sie leider nicht mehr hören. Bitte rufen Sie uns noch einmal "
        "an. Einen schönen Tag."
    ),
    redirect_phrases=(
        "Entschuldigung, dass ich unterbreche — wie kann ich Ihnen helfen?",
        "Entschuldigung — können Sie mir kurz sagen, was Sie brauchen?",
        "Verzeihung, dass ich Sie unterbreche — womit kann ich Ihnen helfen?",
    ),
    promise_cues=re.compile(
        r"\bmoment\b|\bich (?:schaue|sehe|prüfe|buche|trage)\b|\bschau(?:e)? nach\b"
        r"|\bbitte warten\b|\bprüfe\b|\bnachsehen\b",
        re.IGNORECASE,
    ),
    farewells=re.compile(
        r"auf wiederhören|auf wiedersehen|schönen tag|tschüss|danke für ihren anruf",
        re.IGNORECASE,
    ),
    acknowledgements=frozenset(
        "ja jo jawohl ok okay gut danke dankeschön schön auch ihnen gleichfalls "
        "super prima perfekt tschüss tschau auf wiederhören wiedersehen schönen "
        "tag mhm aha alles klar genau".split()
    ),
    get_slots_day_fillers=(
        "Einen Moment, ich schaue, was am {day} frei ist.",
        "Einen Augenblick, ich sehe im Kalender für {day} nach.",
        "Ich schaue mir den {day} an, einen Moment bitte.",
    ),
    get_slots_range_fillers=(
        "Einen Moment, ich schaue, was frei ist.",
        "Einen Augenblick, ich sehe im Kalender nach.",
        "Ich schaue in den Kalender, einen Moment bitte.",
    ),
    check_slots_fillers=("Noch einen Moment, bitte.", "Nur noch eine Sekunde, bitte."),
    create_booking_fillers=(
        "Gut, ich trage Sie ein, einen Moment.",
        "Perfekt, ich trage den Termin ein, einen Moment bitte.",
        "Ich trage die Buchung ein, nur einen Moment.",
    ),
    weekdays_for_filler=("Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"),
    ambiguous_retry_unsafe=(
        "Die Buchung ist möglicherweise schon erfolgt — der Inhaber prüft das "
        "und meldet sich, falls etwas nicht stimmt"
    ),
    ambiguous_checking="Die Buchung ist möglicherweise schon erfolgt — ich prüfe das, einen Moment bitte",
))

_register(LanguagePack(
    code="hr",
    thinking_fillers=("Trenutak.", "Samo trenutak."),
    no_credits="Nažalost trenutno ne možemo primiti vaš poziv. Molimo, pokušajte kasnije.",
    technical_difficulty=(
        "Ispričavam se, trenutno imamo tehničkih poteškoća — vlasnik će vas "
        "nazvati."
    ),
    ai_disclosure=(
        "Molimo, imajte na umu da razgovarate s digitalnim glasovnim "
        "asistentom, a ne s osobom."
    ),
    abuse_closing="Hvala na pozivu. Sada ću završiti razgovor. Ugodan dan.",
    time_limit_closing=(
        "Ispričavam se, moram sada završiti razgovor. Ako nešto još nije "
        "riješeno, nazvat ćemo vas. Ugodan dan."
    ),
    transfer_message="Spajam vas, trenutak molim.",
    still_there=("Jeste li još na vezi?", "Halo, čujete li me?"),
    silence_closing="Nažalost vas više ne čujem. Molimo, nazovite nas ponovno. Ugodan dan.",
    redirect_phrases=(
        "Oprostite što vas prekidam — kako vam mogu pomoći?",
        "Oprostite što prekidam — možete li mi ukratko reći što trebate?",
        "Ispričavam se što prekidam — recite mi, molim vas, kako vam mogu pomoći.",
    ),
    promise_cues=re.compile(
        r"trenutak|provjerim\b|provjeravam\b|pogledam\b|upisujem\b|rezerviram\b"
        r"|pričekajte",
        re.IGNORECASE,
    ),
    farewells=re.compile(
        r"doviđenja|ugodan dan|hvala na pozivu|čujemo se|vidimo se|bok", re.IGNORECASE
    ),
    acknowledgements=frozenset(
        "da ok okej hvala lijepa vama također isto u redu dobro super odlično "
        "bok doviđenja ugodan dan mhm aha čujemo se vidimo se molim".split()
    ),
    get_slots_day_fillers=(
        "Trenutak, pogledat ću što imamo slobodno u {day}.",
        "Samo trenutak, gledam kalendar za {day}.",
        "Pogledat ću {day}, trenutak molim.",
    ),
    get_slots_range_fillers=(
        "Trenutak, pogledat ću što imamo slobodno.",
        "Samo trenutak, gledam kalendar.",
        "Gledam kalendar, trenutak molim.",
    ),
    check_slots_fillers=("Još trenutak, molim.", "Samo još sekundu, molim."),
    create_booking_fillers=(
        "Super, upisujem vas, samo trenutak.",
        "Odlično, upisujem vas, trenutak molim.",
        "Upisujem rezervaciju, samo trenutak.",
    ),
    # "u <dan>" = accusative
    weekdays_for_filler=("ponedjeljak", "utorak", "srijedu", "četvrtak", "petak", "subotu", "nedjelju"),
    ambiguous_retry_unsafe=(
        "Rezervaciju sam možda već napravila, vlasnik će provjeriti i javiti "
        "vam se ako nešto nije u redu"
    ),
    ambiguous_checking="Rezervaciju sam možda već napravila, provjeravam, pričekajte molim",
))

_register(LanguagePack(
    code="it",
    thinking_fillers=("Un attimo.", "Un momento."),
    no_credits=(
        "Purtroppo al momento non possiamo rispondere alla sua chiamata. La "
        "preghiamo di riprovare più tardi."
    ),
    technical_difficulty=(
        "Mi scusi, al momento abbiamo dei problemi tecnici — il titolare la "
        "richiamerà."
    ),
    ai_disclosure=(
        "La informiamo che sta parlando con un assistente vocale digitale, non "
        "con una persona."
    ),
    abuse_closing="Grazie per la chiamata. Ora chiudo la conversazione. Buona giornata.",
    time_limit_closing=(
        "Mi scusi, ora devo chiudere la chiamata. Se qualcosa è rimasto in "
        "sospeso, la richiameremo. Buona giornata."
    ),
    transfer_message="La metto in contatto, un momento per favore.",
    still_there=("È ancora in linea?", "Pronto, mi sente?"),
    silence_closing="Purtroppo non la sento più. La preghiamo di richiamarci. Buona giornata.",
    redirect_phrases=(
        "Mi scusi se la interrompo — come posso aiutarla?",
        "Mi scusi — può dirmi brevemente di cosa ha bisogno?",
        "Mi perdoni l'interruzione — mi dica pure come posso aiutarla.",
    ),
    promise_cues=re.compile(
        r"\bun momento\b|\bun attimo\b|\bcontrollo\b|\bverifico\b|\bguardo\b"
        r"|\bprenoto\b|\bsto (?:controllando|verificando|prenotando)\b|\battenda\b",
        re.IGNORECASE,
    ),
    farewells=re.compile(
        r"arrivederci|buona giornata|grazie per la chiamata|a presto|ciao", re.IGNORECASE
    ),
    acknowledgements=frozenset(
        "sì si ok okay va bene grazie mille anche a lei altrettanto perfetto "
        "ottimo arrivederci ciao buona giornata mhm certo esatto prego".split()
    ),
    get_slots_day_fillers=(
        "Un momento, guardo cosa abbiamo libero {day}.",
        "Un attimo, controllo il calendario per {day}.",
        "Guardo {day}, un momento per favore.",
    ),
    get_slots_range_fillers=(
        "Un momento, guardo cosa abbiamo libero.",
        "Un attimo, controllo il calendario.",
        "Guardo il calendario, un momento per favore.",
    ),
    check_slots_fillers=("Ancora un momento, per favore.", "Solo un secondo, per favore."),
    create_booking_fillers=(
        "Perfetto, la inserisco subito, un momento.",
        "Benissimo, registro la prenotazione, un attimo.",
        "Sto registrando la prenotazione, solo un momento.",
    ),
    weekdays_for_filler=("lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica"),
    ambiguous_retry_unsafe=(
        "Potrei aver già registrato la prenotazione — il titolare controllerà "
        "e la contatterà se qualcosa non va"
    ),
    ambiguous_checking="Potrei aver già registrato la prenotazione — controllo, attenda un momento",
))


# How sure we must be before switching on detected speech alone (the model
# can still switch earlier via the switch_language tool, e.g. when the
# caller asks "can we speak English?" in Slovenian). Soniox identifies the
# language per token; a short turn is easily a name, "okay", "sorry".
SWITCH_MIN_WORDS = 4
SWITCH_MIN_WORDS_REPEATED = 2
CLOSE_PAIRS = {frozenset(("sl", "hr"))}
CLOSE_PAIR_MIN_WORDS = 5


def detect_switch(
    *,
    current: str,
    detected: str | None,
    words: int,
    previous_detected: str | None,
    allowed: list[str] | tuple,
) -> str | None:
    """The language to switch to before replying to this turn, or None.

    Switch when the caller's turn is in another allowed language and is
    either long enough on its own (SWITCH_MIN_WORDS) or the second short
    turn in a row in that same language.
    """
    lang = normalize_language(detected)
    if not lang or lang == current or lang not in allowed:
        return None
    if frozenset((current, lang)) in CLOSE_PAIRS:
        # Slovenian and Croatian share much of their vocabulary; a
        # language identifier can label one as the other. Only a sustained
        # run counts: two long turns in a row.
        return lang if words >= CLOSE_PAIR_MIN_WORDS and normalize_language(previous_detected) == lang else None
    if words >= SWITCH_MIN_WORDS:
        return lang
    if words >= SWITCH_MIN_WORDS_REPEATED and normalize_language(previous_detected) == lang:
        return lang
    return None
