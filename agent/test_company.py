"""Hardcoded placeholder company data, standing in for the real `init` call.

Once the n8n `booking-v2` integration exists, `TEST_COMPANY` will be replaced by
the dict returned from that call (same shape), and `render_company_prompt`
will not need to change.
"""

from __future__ import annotations

TEST_COMPANY: dict = {
    "name": "Salon Lepote",
    "services": [
        {"name": "Striženje", "price_eur": 25, "duration_min": 30},
        {"name": "Barvanje", "price_eur": 45, "duration_min": 90},
        {"name": "Feniranje", "price_eur": 15, "duration_min": 20},
    ],
    "employees": ["Maja", "Špela"],
    "hours": {
        "Mon-Fri": "9:00-19:00",
        "Sat": "9:00-14:00",
        "Sun": "zaprto",
    },
    "faq": [
        {
            "q": "Kje lahko parkiram?",
            "a": "Parkirišče je na voljo za stavbo.",
        },
        {
            "q": "Kakšna je politika odpovedi termina?",
            "a": "Termin lahko brezplačno odpoveste do 24 ur pred obiskom.",
        },
    ],
}


def render_company_prompt(company: dict) -> str:
    """Render a company data dict into a prompt block the LLM can read."""
    lines = [f"Podatki o podjetju: {company['name']}", ""]

    lines.append("Storitve:")
    for svc in company["services"]:
        lines.append(
            f"- {svc['name']}: {svc['price_eur']} EUR, {svc['duration_min']} min"
        )
    lines.append("")

    lines.append("Zaposleni: " + ", ".join(company["employees"]))
    lines.append("")

    lines.append("Delovni čas:")
    for day, hours in company["hours"].items():
        lines.append(f"- {day}: {hours}")
    lines.append("")

    lines.append("Pogosta vprašanja:")
    for entry in company["faq"]:
        lines.append(f"- V: {entry['q']}")
        lines.append(f"  O: {entry['a']}")

    return "\n".join(lines)
