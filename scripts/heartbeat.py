"""Write this host's worker health to receptionist_worker_heartbeat.

Run every minute by the receptionistplus-heartbeat systemd timer — as its
own unit, so it still reports when the worker itself is down. Reads the
LiveKit worker's health server (prod default port 8081): "/" is 200 "OK"
only while the worker is connected to LiveKit, "/worker" has the number of
calls in progress.

The n8n workflow "Receptionist+ – Nadzor workerja" alerts from these rows;
a row that stops updating means the whole server is down.
"""

from __future__ import annotations

import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

HEALTH_URL = os.environ.get("WORKER_HEALTH_URL", "http://127.0.0.1:8081")
REPO = Path(__file__).resolve().parent.parent


def _check() -> tuple[bool, str, int | None]:
    try:
        r = requests.get(HEALTH_URL + "/", timeout=5)
    except requests.RequestException as exc:
        return False, f"health server unreachable: {exc.__class__.__name__}", None
    if r.status_code != 200:
        return False, f"health {r.status_code}: {r.text[:200]}", None
    jobs = None
    try:
        jobs = int(requests.get(HEALTH_URL + "/worker", timeout=5).json().get("active_jobs", 0))
    except (requests.RequestException, ValueError, TypeError):
        pass
    return True, "OK", jobs


def main() -> int:
    url = os.environ["SUPABASE_URL"].rstrip("/")
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    healthy, detail, jobs = _check()
    revision_file = REPO / "REVISION"
    row = {
        "id": socket.gethostname(),
        "healthy": healthy,
        "detail": detail,
        "active_jobs": jobs,
        "revision": revision_file.read_text().strip() if revision_file.exists() else None,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    resp = requests.post(
        f"{url}/rest/v1/receptionist_worker_heartbeat",
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            # Upsert on id; alerted/alerted_at are left as they are (not sent).
            "Prefer": "resolution=merge-duplicates,return=minimal",
        },
        data=json.dumps(row),
        timeout=10,
    )
    if resp.status_code >= 300:
        print(f"heartbeat write failed: {resp.status_code} {resp.text[:200]}", file=sys.stderr)
        return 1
    print(f"healthy={healthy} detail={detail} active_jobs={jobs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
