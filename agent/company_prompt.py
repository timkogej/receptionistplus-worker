"""Render a booking-v2 "init" response into the per-company prompt block.

Same static+injected split as Phase 1's test_company.py, just fed by real
webhook data instead of a hardcoded fake salon.
"""

from __future__ import annotations

from collections import Counter


def spoken_employee_names(employees: list[dict]) -> dict[str, str]:
    """Map employee ID -> the name the agent should say out loud.

    First name only ("Luka"), unless another employee shares that first
    name, in which case the full label ("Luka Dobrovoljec") so the caller
    can tell them apart. Computed here rather than left to the model:
    spotting a shared first name means comparing the whole list, which is
    the kind of cross-referencing the model has been unreliable at.
    """
    first_names = {emp["id"]: emp["label"].split()[0] for emp in employees if emp["label"].split()}
    counts = Counter(first_names.values())
    return {
        emp["id"]: (
            first_names[emp["id"]]
            if emp["id"] in first_names and counts[first_names[emp["id"]]] == 1
            else emp["label"]
        )
        for emp in employees
    }


def render_company_prompt(init_data: dict, language: str = "sl") -> str:
    if language == "en":
        return _render_company_prompt_en(init_data)
    return _render_company_prompt_sl(init_data)


def _render_company_prompt_sl(init_data: dict) -> str:
    company = init_data["company"]
    services = init_data.get("services", [])
    employees = init_data.get("employees_ui", [])
    employees_by_service = init_data.get("employeesByServiceId", {})
    employee_names = spoken_employee_names(employees)
    categories = init_data.get("categories", [])
    services_by_category = init_data.get("servicesByCategory", {})

    def service_line(svc: dict) -> str:
        # The single-employee marker carries its instruction INLINE, on the
        # service line itself, rather than only tagging the service and
        # stating the rule separately further down. Observed live 2026-09-08
        # (Masaža glave, the one single-employee service in this catalogue):
        # the model read the line correctly — it quoted the price from it —
        # and still asked "Imate željo po določenem zaposlenem?", because
        # acting on the marker required joining it to a rule ~30 lines below
        # and to another in STATIC_PROMPT_SL. Stating the consequence where
        # the model is already reading removes that join.
        #
        # Scoped 2026-09-21 to "only if this is the service finally chosen":
        # a caller said "masaža glave. Ne, masaža stopal." and the model
        # skipped the staff question for the massage they switched TO,
        # apparently still following this note on the one they abandoned.
        #
        # Multi-employee services list who performs them (first names), so
        # "who does this?" / "who works there?" is answered for the service
        # at hand instead of by reading out the whole staff list.
        note = ""
        eligible = employees_by_service.get(svc["id"], [])
        if len(eligible) == 1:
            name = employee_names.get(eligible[0], "?")
            note = (
                f" — to storitev opravlja SAMO {name}, zato NE sprašuj po "
                f"želji glede zaposlenega, ampak rezerviraj pri njem/njej. "
                f"To velja SAMO, če stranka na koncu izbere prav to "
                f"storitev — če se premisli za drugo, ravnaj po vrstici "
                f"tiste storitve"
            )
        elif eligible:
            names = ", ".join(employee_names.get(e, "?") for e in eligible)
            note = f" — opravljajo: {names}"
        return (
            f'- {svc["naziv"]} (ID storitve: {svc["id"]}): {svc["cena"]} EUR, '
            f'{svc["trajanjeMin"]} minut{note}'
        )

    lines = [f'Podatki o podjetju "{company["naziv"]}":']

    if categories and services_by_category:
        lines.append("\nStoritve po kategorijah:")
        for cat in categories:
            cat_services = services_by_category.get(cat["id"], [])
            if not cat_services:
                continue
            lines.append(f'\n{cat["name"]}:')
            for svc in cat_services:
                lines.append(service_line(svc))
    else:
        lines.append("\nStoritve:")
        for svc in services:
            lines.append(service_line(svc))

    lines.append("\nZaposleni:")
    for emp in employees:
        role = f' ({emp["subtitle"]})' if emp.get("subtitle") else ""
        lines.append(
            f'- {emp["label"]}{role} (ID zaposlenega: {emp["id"]}) — '
            f'v pogovoru: {employee_names[emp["id"]]}'
        )

    if company.get("multiple_services_online"):
        lines.append(
            "\nStranka lahko v enem klicu rezervira več storitev naenkrat."
        )
    else:
        # The offer is phrased for the caller here, with an explicit
        # bad/good pair, because the bare instruction ("offer to note the
        # rest as a message for the owner") got spoken back almost verbatim
        # — "lahko dodatne storitve tudi zabeležim za lastnika" (observed
        # 2026-09-21) — which describes our backend to the caller, and was
        # offered to someone who had only asked what services exist.
        lines.append(
            "\nStranka lahko v enem klicu rezervira samo eno storitev. Če "
            "želi več storitev, rezerviraj eno, za ostale pa ponudi, da jih "
            "zapišeš (v opombo pri rezervaciji). To ponudi SAMO, ko stranka "
            "res želi rezervirati več kot eno storitev — ne, ko le sprašuje, "
            "katere storitve imate. Povej naravno, brez opisovanja, kaj se "
            "zgodi v ozadju:\n"
            "  - Slabo: \"Lahko dodatne storitve tudi zabeležim za lastnika.\"\n"
            "  - Dobro: \"Danes lahko rezerviram eno storitev — za drugo pa z "
            "veseljem zapišem, da jo želite, in lastnik vam sporoči "
            "podrobnosti.\""
        )

    # Deliberately NOT "if the caller doesn't say, just book anyone
    # (any_person)" — that was written before the proactive
    # ask-for-employee-preference default existed in STATIC_PROMPT_SL, and
    # because this block is appended AFTER the static prompt it was winning
    # on recency and cancelling that rule out. The static rule is the one
    # that should win; this line now restates it instead of contradicting
    # it.
    lines.append(
        "\nČe stranka ne izrazi želje po določenem zaposlenem, jo vprašaj, "
        "ali ima željo po določenem zaposlenem — takoj, ko je storitev "
        "dokončno izbrana, in PREDEN vprašaš za dan. Šele ko pove, da ji je "
        "vseeno, rezerviraj pri kateremkoli zaposlenem, ki opravlja izbrano "
        "storitev (any_person). Izjema: če je pri KONČNO izbrani storitvi "
        "navedeno, da jo opravlja SAMO en zaposleni, tega vprašanja ne "
        "postavljaj — rezerviraj pri tistem zaposlenem. Če stranka zamenja "
        "storitev, to presodi znova za novo storitev."
    )

    lines.append(
        "\nZa delovni čas nimaš točnih podatkov — če te vprašajo po urniku, "
        "ponudi namesto tega preverjanje prostih terminov za konkreten dan, "
        "ali povej, da bo lastnik poklical nazaj."
    )

    return "\n".join(lines)


def _render_company_prompt_en(init_data: dict) -> str:
    company = init_data["company"]
    services = init_data.get("services", [])
    employees = init_data.get("employees_ui", [])
    employees_by_service = init_data.get("employeesByServiceId", {})
    employee_names = spoken_employee_names(employees)
    categories = init_data.get("categories", [])
    services_by_category = init_data.get("servicesByCategory", {})

    def service_line(svc: dict) -> str:
        # See the note in _render_company_prompt_sl: the instruction is inline
        # on the service line because tagging the service and stating the rule
        # elsewhere was not enough to stop the model asking the question.
        note = ""
        eligible = employees_by_service.get(svc["id"], [])
        if len(eligible) == 1:
            name = employee_names.get(eligible[0], "?")
            note = (
                f" — {name} is the ONLY staff member for this service, so do "
                f"NOT ask about a staff preference; book with them. This "
                f"applies ONLY if the caller ends up choosing this exact "
                f"service — if they switch to another, follow that "
                f"service's line instead"
            )
        elif eligible:
            names = ", ".join(employee_names.get(e, "?") for e in eligible)
            note = f" — performed by: {names}"
        return (
            f'- {svc["naziv"]} (service ID: {svc["id"]}): {svc["cena"]} EUR, '
            f'{svc["trajanjeMin"]} minutes{note}'
        )

    lines = [f'Company information for "{company["naziv"]}":']

    if categories and services_by_category:
        lines.append("\nServices by category:")
        for cat in categories:
            cat_services = services_by_category.get(cat["id"], [])
            if not cat_services:
                continue
            lines.append(f'\n{cat["name"]}:')
            for svc in cat_services:
                lines.append(service_line(svc))
    else:
        lines.append("\nServices:")
        for svc in services:
            lines.append(service_line(svc))

    lines.append("\nStaff:")
    for emp in employees:
        role = f' ({emp["subtitle"]})' if emp.get("subtitle") else ""
        lines.append(
            f'- {emp["label"]}{role} (staff ID: {emp["id"]}) — '
            f'say: {employee_names[emp["id"]]}'
        )

    if company.get("multiple_services_online"):
        lines.append(
            "\nThe caller can book more than one service in a single call."
        )
    else:
        # See the note in _render_company_prompt_sl.
        lines.append(
            "\nThe caller can only book one service per call. If they want "
            "more than one, book the first and offer to write the rest down "
            "(in the booking's notes). Only offer this when the caller "
            "actually wants to book more than one service — not when they "
            "are just asking what services exist. Say it naturally, without "
            "describing what happens behind the scenes:\n"
            "  - Bad: \"I can also log the additional services for the owner.\"\n"
            "  - Good: \"I can book one service for you today — for the other "
            "one, I'll happily make a note that you'd like it, and the owner "
            "will get back to you with the details.\""
        )

    # See the note in _render_company_prompt_sl: this deliberately no longer
    # tells the model to silently default to any_person, which contradicted
    # the proactive ask-for-employee-preference rule in STATIC_PROMPT_EN and
    # won on recency by being appended after it.
    lines.append(
        "\nIf the caller hasn't expressed a preference, ask whether they'd "
        "like a specific staff member — as soon as the service is final, "
        "and BEFORE asking about the day. Only once they say they don't "
        "mind should you book with any staff member who performs the chosen "
        "service (any_person). Exception: if the FINALLY chosen service is "
        "marked as having only ONE staff member, don't ask that question — "
        "book with that person. If the caller switches service, decide "
        "again for the new one."
    )

    lines.append(
        "\nYou don't have exact opening hours — if asked about the schedule, "
        "offer to check available slots for a specific day instead, or say "
        "the owner will call back."
    )

    return "\n".join(lines)
