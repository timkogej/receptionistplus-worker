"""Render a booking-v2 "init" response into the per-company prompt block.

Same static+injected split as Phase 1's test_company.py, just fed by real
webhook data instead of a hardcoded fake salon.
"""

from __future__ import annotations


def render_company_prompt(init_data: dict, language: str = "sl") -> str:
    if language == "en":
        return _render_company_prompt_en(init_data)
    return _render_company_prompt_sl(init_data)


def _render_company_prompt_sl(init_data: dict) -> str:
    company = init_data["company"]
    services = init_data.get("services", [])
    employees = init_data.get("employees_ui", [])
    employees_by_service = init_data.get("employeesByServiceId", {})
    employee_names = {emp["id"]: emp["label"] for emp in employees}
    categories = init_data.get("categories", [])
    services_by_category = init_data.get("servicesByCategory", {})

    def service_line(svc: dict) -> str:
        note = ""
        eligible = employees_by_service.get(svc["id"], [])
        if len(eligible) == 1:
            name = employee_names.get(eligible[0], "?")
            note = f" — edini zaposleni za to storitev: {name}"
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
        lines.append(f'- {emp["label"]}{role} (ID zaposlenega: {emp["id"]})')

    if company.get("multiple_services_online"):
        lines.append(
            "\nStranka lahko v enem klicu rezervira več storitev naenkrat."
        )
    else:
        lines.append(
            "\nStranka lahko v enem klicu rezervira samo eno storitev. Če želi "
            "več storitev, rezerviraj eno in ponudi, da dodatne zabeležiš kot "
            "sporočilo za lastnika."
        )

    # Deliberately NOT "if the caller doesn't say, just book anyone
    # (any_person)" — that was written before the proactive
    # ask-for-employee-preference default existed in STATIC_PROMPT_SL, and
    # because this block is appended AFTER the static prompt it was winning
    # on recency and cancelling that rule out. The static rule is the one
    # that should win; this line now restates it instead of contradicting
    # it.
    lines.append(
        "\nČe stranka ne izrazi želje po določenem zaposlenem, jo najprej "
        "vprašaj, ali ima željo po določenem zaposlenem — šele ko pove, da ji "
        "je vseeno, rezerviraj pri kateremkoli od zgoraj naštetih zaposlenih, "
        "ki opravljajo izbrano storitev (any_person). Izjema: če je pri "
        "storitvi naveden edini zaposleni za to storitev, tega vprašanja ne "
        "postavljaj."
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
    employee_names = {emp["id"]: emp["label"] for emp in employees}
    categories = init_data.get("categories", [])
    services_by_category = init_data.get("servicesByCategory", {})

    def service_line(svc: dict) -> str:
        note = ""
        eligible = employees_by_service.get(svc["id"], [])
        if len(eligible) == 1:
            name = employee_names.get(eligible[0], "?")
            note = f" — only staff member for this service: {name}"
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
        lines.append(f'- {emp["label"]}{role} (staff ID: {emp["id"]})')

    if company.get("multiple_services_online"):
        lines.append(
            "\nThe caller can book more than one service in a single call."
        )
    else:
        lines.append(
            "\nThe caller can only book one service per call. If they want "
            "more than one, book the first and offer to leave the rest as a "
            "note for the owner."
        )

    # See the note in _render_company_prompt_sl: this deliberately no longer
    # tells the model to silently default to any_person, which contradicted
    # the proactive ask-for-employee-preference rule in STATIC_PROMPT_EN and
    # won on recency by being appended after it.
    lines.append(
        "\nIf the caller hasn't expressed a preference, ask them first "
        "whether they'd like a specific staff member — only once they say "
        "they don't mind should you book with any of the staff listed above "
        "who perform the chosen service (any_person). Exception: if a "
        "service is marked with an only staff member for this service, don't "
        "ask that question."
    )

    lines.append(
        "\nYou don't have exact opening hours — if asked about the schedule, "
        "offer to check available slots for a specific day instead, or say "
        "the owner will call back."
    )

    return "\n".join(lines)
