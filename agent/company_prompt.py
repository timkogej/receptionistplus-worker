"""Render a booking-v2 "init" response into the per-company prompt block.

Same static+injected split as Phase 1's test_company.py, just fed by real
webhook data instead of a hardcoded fake salon.
"""

from __future__ import annotations


def render_company_prompt(init_data: dict) -> str:
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
