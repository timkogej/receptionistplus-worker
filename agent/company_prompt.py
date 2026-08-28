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

    lines = [f'Podatki o podjetju "{company["naziv"]}":']

    lines.append("\nStoritve:")
    for svc in services:
        lines.append(
            f'- {svc["naziv"]} (ID storitve: {svc["id"]}): {svc["cena"]} EUR, '
            f'{svc["trajanjeMin"]} minut'
        )

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

    lines.append(
        "\nČe stranka ne pove, s katerim zaposlenim želi termin, lahko rezerviraš "
        "pri kateremkoli od zgoraj naštetih zaposlenih, ki opravljajo izbrano "
        "storitev (any_person)."
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

    lines = [f'Company information for "{company["naziv"]}":']

    lines.append("\nServices:")
    for svc in services:
        lines.append(
            f'- {svc["naziv"]} (service ID: {svc["id"]}): {svc["cena"]} EUR, '
            f'{svc["trajanjeMin"]} minutes'
        )

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

    lines.append(
        "\nIf the caller doesn't say which staff member they want, you can "
        "book with any of the staff listed above who perform the chosen "
        "service (any_person)."
    )

    lines.append(
        "\nYou don't have exact opening hours — if asked about the schedule, "
        "offer to check available slots for a specific day instead, or say "
        "the owner will call back."
    )

    return "\n".join(lines)
