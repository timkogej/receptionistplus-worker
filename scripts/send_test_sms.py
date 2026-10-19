"""Send one real SMS through the n8n "Receptionist+ – SMS stranki" workflow.

Uses the company's SMS quota like any other SMS. For checking the workflow
end to end after a change:

    .venv/bin/python scripts/send_test_sms.py --phone +38640123456
    .venv/bin/python scripts/send_test_sms.py --phone +38640123456 \\
        --kind manage_link --email you@example.com
    .venv/bin/python scripts/send_test_sms.py --phone +38640123456 --kind owner_alert

Reads BOOKING_V2_HMAC_SECRET from .env (the workflow rejects unsigned calls).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.owner_alert import owner_alert_text  # noqa: E402
from agent.sms_client import booking_confirmation_text, send_sms  # noqa: E402


def _load_env() -> None:
    env = Path(__file__).resolve().parent.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key, value)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--phone", required=True)
    p.add_argument("--company", default="jedroplus-d-o-o")
    p.add_argument("--kind", choices=["confirmation", "manage_link", "owner_alert"], default="confirmation")
    p.add_argument("--email", default="")
    args = p.parse_args()
    _load_env()
    text = ""
    if args.kind == "confirmation":
        text = booking_confirmation_text(
            company_name="TEST", service_names=["Pedikura"], date="2026-10-21",
            time_hhmm="14:00", language="sl",
        )
    elif args.kind == "owner_alert":
        text = owner_alert_text(
            message="TEST: Ana Novak prosi za povratni klic glede cene paketa.",
            callback_requested=True, kind=None, caller_phone="+38641555123", language="sl",
        )
    result = asyncio.run(send_sms(
        kind=args.kind, company_slug=args.company, phone=args.phone,
        text=text, email=args.email,
    ))
    print(result)


if __name__ == "__main__":
    main()
