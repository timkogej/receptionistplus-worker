"""Render a booking-v2 "init" response into the per-company prompt block.

Same static+injected split as Phase 1's test_company.py, just fed by real
webhook data instead of a hardcoded fake salon.
"""

from __future__ import annotations

from collections import Counter

from agent.dates import slovenian_euros, slovenian_minutes


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


def render_company_prompt(
    init_data: dict, language: str = "sl", *, hours_known: bool = False
) -> str:
    """hours_known: the business block (business_info.py) carries real
    opening hours, so the "you don't know the hours" line is left out."""
    if language != "sl":
        return _render_company_prompt_en(init_data, hours_known)
    return _render_company_prompt_sl(init_data, hours_known)


def _render_company_prompt_sl(init_data: dict, hours_known: bool = False) -> str:
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
        # The price and duration also as words, ready to copy: the model
        # otherwise built e.g. 45 as "štirideset pet" (eval, 2026-10-17).
        price_words = slovenian_euros(svc["cena"])
        minutes_words = slovenian_minutes(svc["trajanjeMin"])
        price = f'{svc["cena"]} EUR' + (f" ({price_words})" if price_words else "")
        duration = f'{svc["trajanjeMin"]} minut' + (f" ({minutes_words})" if minutes_words else "")
        return f'- {svc["naziv"]} (ID storitve: {svc["id"]}): {price}, {duration}{note}'

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

    # The "ask for a staff preference" paragraph that used to follow here
    # restated R-EMPLOYEE-PREFERENCE-ASK word for word; removed 2026-10-17
    # (the per-service "SAMO ..." notes above still carry the exception).

    if not hours_known:
        lines.append(
            "\nZa delovni čas nimaš točnih podatkov — če te vprašajo po urniku, "
            "ponudi namesto tega preverjanje prostih terminov za konkreten dan, "
            "ali ponudi, da vprašanje posreduješ lastniku (take_message)."
        )

    return "\n".join(lines)


def _render_company_prompt_en(init_data: dict, hours_known: bool = False) -> str:
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

    # See the note in _render_company_prompt_sl (paragraph removed 2026-10-17).

    if not hours_known:
        lines.append(
            "\nYou don't have exact opening hours — if asked about the schedule, "
            "offer to check available slots for a specific day instead, or say "
            "offer to pass the question on to the owner (take_message)."
        )

    return "\n".join(lines)
