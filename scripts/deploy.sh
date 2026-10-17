#!/usr/bin/env bash
# Deploy the committed worker to the VPS and restart it.
#
#   scripts/deploy.sh            # deploy HEAD
#   scripts/deploy.sh --dry-run  # show what would be copied, change nothing
#   scripts/deploy.sh --units    # also install the systemd unit files
#
# What it does, in order — and stops at the first failure:
#   1. refuses a dirty working tree (what runs in production must be a commit)
#   2. runs the unit tests locally
#   3. rsyncs the repo to /opt/receptionist-plus (never .env, .venv, tts_out)
#   4. on the server: pip install -r requirements.txt, fix ownership,
#      optionally install unit files, restart, and check it came back up
#
# Restarting drains the worker: calls in progress finish first (systemd waits
# up to TimeoutStopSec — see deploy/receptionistplus-worker.service), but a
# call that comes in DURING the drain gets no agent. Deploy when it's quiet.
#
# Migrations are NOT applied by this script. If the commit adds a file under
# migrations/, run it in the Supabase SQL editor first.

set -euo pipefail

HOST="${DEPLOY_HOST:-root@178.105.39.138}"
DIR="/opt/receptionist-plus"
SERVICE="receptionistplus-worker"
DRY_RUN=0
UNITS=0

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --units) UNITS=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")/.."

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Working tree has uncommitted changes — commit them first." >&2
  exit 1
fi

REV="$(git rev-parse --short HEAD)"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
echo "Deploying $REV ($BRANCH) to $HOST:$DIR"

PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"
"$PY" -m unittest discover -s tests

RSYNC_FLAGS=(-az --itemize-changes
  --exclude .git/ --exclude .venv/ --exclude .env --exclude tts_out/
  --exclude __pycache__/ --exclude evals/results/ --exclude .DS_Store)
if [ "$DRY_RUN" -eq 1 ]; then
  rsync "${RSYNC_FLAGS[@]}" --dry-run ./ "$HOST:$DIR/"
  echo "(dry run — nothing copied, nothing restarted)"
  exit 0
fi

# No --delete on purpose: the server directory may hold files that were
# never in git (old assets, local notes). Removing a file from the repo
# means removing it on the server by hand.
rsync "${RSYNC_FLAGS[@]}" ./ "$HOST:$DIR/"

ssh "$HOST" bash -s -- "$DIR" "$SERVICE" "$REV" "$UNITS" <<'REMOTE'
set -euo pipefail
DIR="$1"; SERVICE="$2"; REV="$3"; UNITS="$4"
cd "$DIR"
echo "$REV" > REVISION
.venv/bin/pip install -q -r requirements.txt
chown -R receptionistplus:receptionistplus "$DIR"
if [ "$UNITS" = "1" ]; then
  cp deploy/*.service deploy/*.timer /etc/systemd/system/
  systemctl daemon-reload
  systemctl enable --now receptionistplus-transcript-retention.timer receptionistplus-heartbeat.timer
fi
systemctl restart "$SERVICE"
sleep 5
if systemctl is-active --quiet "$SERVICE"; then
  echo "OK: $SERVICE is running revision $REV"
  journalctl -u "$SERVICE" -n 15 --no-pager
else
  echo "FAILED: $SERVICE is not active" >&2
  journalctl -u "$SERVICE" -n 60 --no-pager >&2
  exit 1
fi
REMOTE
