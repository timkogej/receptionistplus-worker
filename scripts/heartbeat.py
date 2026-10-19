"""Write this host's worker health to receptionist_worker_heartbeat.

Run every minute by the receptionistplus-heartbeat systemd timer — as its
own unit, so it still reports when the worker itself is down. Reads the
LiveKit worker's health server (prod default port 8081): "/" is 200 "OK"
only while the worker is connected to LiveKit, "/worker" has the number of
calls in progress.

The n8n workflow "Receptionist+ – Nadzor workerja" alerts from these rows;
a row that stops updating means the whole server is down.

That workflow runs in n8n on this same server, so it can't report the
server (or n8n itself) going down. HEALTHCHECK_PING_URL, when set, is an
outside dead man's switch (e.g. healthchecks.io): pinged every minute while
the worker and n8n are both up, pinged at /fail when either isn't, and when
the pings stop altogether the service alerts on its own.
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
# n8n runs booking-v2 and the SMS flows; without it calls can't book.
N8N_HEALTH_URL = os.environ.get("N8N_HEALTH_URL", "https://n8n.jedroplus.com/healthz")
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


def _n8n_ok() -> tuple[bool, str]:
    try:
        r = requests.get(N8N_HEALTH_URL, timeout=8)
    except requests.RequestException as exc:
        return False, f"n8n unreachable: {exc.__class__.__name__}"
    return (True, "OK") if r.status_code == 200 else (False, f"n8n health {r.status_code}")


def _ping_outside(ok: bool, detail: str) -> None:
    """Best effort: a failed ping must not stop the Supabase row."""
    ping = os.environ.get("HEALTHCHECK_PING_URL", "").strip().rstrip("/")
    if not ping:
        return
    try:
        requests.post(ping if ok else ping + "/fail", data=detail.encode()[:1000], timeout=10)
    except requests.RequestException as exc:
        print(f"outside ping failed: {exc.__class__.__name__}", file=sys.stderr)


def main() -> int:
    url = os.environ["SUPABASE_URL"].rstrip("/")
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    healthy, detail, jobs = _check()
    n8n_ok, n8n_detail = _n8n_ok()
    _ping_outside(
        healthy and n8n_ok,
        f"worker: {detail}; n8n: {n8n_detail}; active_jobs={jobs}; host={socket.gethostname()}",
    )
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
    print(f"healthy={healthy} detail={detail} active_jobs={jobs} n8n={n8n_detail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
