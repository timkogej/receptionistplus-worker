"""Phase 5 Part B: GDPR transcript retention.

Anonymizes receptionist_calls for calls older than RETENTION_DAYS: the
transcript is cleared to an empty array, and the caller's phone number and
message (phase6 columns) to null. The rest of the row (billing, outcome,
duration) is kept — those aren't the PII this policy targets, and the
business still needs them for billing history.

Run daily via the receptionistplus-transcript-retention systemd timer
(deploy/receptionistplus-transcript-retention.{service,timer}). Safe to
re-run: PATCHing an already-empty transcript to [] is a no-op.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta, timezone

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("purge_old_transcripts")

RETENTION_DAYS = 365


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"missing required environment variable: {name}")
    return value


def main() -> None:
    supabase_url = _require_env("SUPABASE_URL").rstrip("/")
    api_key = _require_env("SUPABASE_SERVICE_ROLE_KEY")

    cutoff = datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)
    cutoff_iso = cutoff.isoformat()

    def _patch(body: dict) -> requests.Response:
        return requests.patch(
            f"{supabase_url}/rest/v1/receptionist_calls",
            headers={
                "apikey": api_key,
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Prefer": "return=representation",
            },
            params={
                "started_at": f"lt.{cutoff_iso}",
                "select": "id",
            },
            json=body,
            timeout=30,
        )

    response = _patch({"transcript": [], "caller_phone": None, "message": None})
    if response.status_code == 400:
        # phase6 columns not there yet — clear what exists.
        logger.warning("phase6 columns missing, clearing transcript only: %s", response.text[:200])
        response = _patch({"transcript": []})
    response.raise_for_status()
    purged = response.json()
    logger.info(
        "anonymized %d call(s) started before %s",
        len(purged),
        cutoff_iso,
    )


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as exc:
        logger.error("purge request failed: %s", exc)
        sys.exit(1)
