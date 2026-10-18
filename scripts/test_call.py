"""Talk to the production worker from the browser, without a phone number.

Prints a meet.livekit.io link. Opening it joins a fresh room and asks
LiveKit to send the receptionistplus-worker agent there (the worker uses
explicit dispatch, so the plain Agents Playground never gets an agent).

    .venv/bin/python scripts/test_call.py
    .venv/bin/python scripts/test_call.py --caller-phone +38640123456

It is a real call for that company: credits are billed, a booking lands in
its calendar, SMS confirmations really go out. --caller-phone pretends the
call came from that number (caller ID, known-customer lookup); without it
the caller is "withheld", as with a hidden number.

Reads LIVEKIT_URL, LIVEKIT_API_KEY and LIVEKIT_API_SECRET from .env. The
token in the link is valid for an hour — don't share it.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlencode

from livekit import api

AGENT_NAME = "receptionistplus-worker"


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
    p.add_argument("--company", default="jedroplus-d-o-o")
    p.add_argument("--caller-phone", default="", help="E.164, e.g. +38640123456")
    args = p.parse_args()
    _load_env()

    room = f"test-{secrets.token_hex(4)}"
    attributes = {"sip.phoneNumber": args.caller_phone} if args.caller_phone else {}
    token = (
        api.AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
        .with_identity(f"tester-{secrets.token_hex(3)}")
        .with_name("Test")
        .with_ttl(timedelta(hours=1))
        .with_attributes(attributes)
        .with_grants(api.VideoGrants(room_join=True, room=room))
        .with_room_config(
            api.RoomConfiguration(
                agents=[
                    api.RoomAgentDispatch(
                        agent_name=AGENT_NAME,
                        metadata=json.dumps({"company_slug": args.company}),
                    )
                ]
            )
        )
        .to_jwt()
    )
    url = "https://meet.livekit.io/custom?" + urlencode(
        {"liveKitUrl": os.environ["LIVEKIT_URL"], "token": token}
    )
    print(f"company={args.company} room={room} caller={args.caller_phone or 'withheld'}")
    print(url)


if __name__ == "__main__":
    main()
