# receptionistplus-worker

A [LiveKit Agents](https://docs.livekit.io/agents/) Python worker that answers inbound SIP
calls. Currently it does one thing on participant join: speak a fixed, configurable
greeting via TTS, then wait silently. No STT and no LLM are wired up yet.

- STT/TTS: [Soniox](https://docs.livekit.io/agents/integrations/stt/soniox/) (`livekit-plugins-soniox`)
- TTS (alternate): [ElevenLabs](https://docs.livekit.io/agents/integrations/tts/elevenlabs/) (`livekit-plugins-elevenlabs`)
- Runtime: Python 3.12

## Project layout

```
agent/
  worker.py   # entrypoint: connects to LiveKit, speaks greeting, waits
  config.py   # env-backed settings
deploy/
  receptionistplus-worker.service
.env.example
requirements.txt
```

## Run locally

Requires Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# fill in LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET and the
# TTS_PROVIDER's API key (SONIOX_API_KEY or ELEVENLABS_API_KEY)

set -a; source .env; set +a
python -m agent.worker dev
```

`dev` connects to LiveKit Cloud and prints logs to the console. Because the worker
registers with `agent_name="receptionistplus-worker"`, it uses **explicit dispatch** —
it will not join rooms until a dispatch rule/request targets that agent name (see
step 0.6 of the project plan, not yet configured).

## Deploy to the VPS (Ubuntu 24.04)

1. Install Python 3.12 and venv tooling:

   ```bash
   sudo apt update
   sudo apt install -y python3.12 python3.12-venv
   ```

2. Copy this repo to the server at `/opt/receptionist-plus`, then create the venv
   and install dependencies:

   ```bash
   cd /opt/receptionist-plus
   python3.12 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```

3. Create `/etc/receptionistplus/.env` from `.env.example` and fill in real credentials
   (this file is loaded by the systemd unit and should be `chmod 600`, owned by the
   service user):

   ```bash
   sudo mkdir -p /etc/receptionistplus
   sudo cp .env.example /etc/receptionistplus/.env
   sudo chmod 600 /etc/receptionistplus/.env
   sudo $EDITOR /etc/receptionistplus/.env
   ```

4. Install the systemd unit:

   ```bash
   sudo cp deploy/receptionistplus-worker.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable receptionistplus-worker
   ```

   The service runs as a dedicated `receptionistplus` user — create it first if it
   doesn't exist (`sudo useradd --system --no-create-home receptionistplus`) and make
   sure it owns `/opt/receptionist-plus`.

5. Start it once SIP dispatch is configured:

   ```bash
   sudo systemctl start receptionistplus-worker
   sudo systemctl status receptionistplus-worker
   journalctl -u receptionistplus-worker -f
   ```

## Configuration

All settings are read from the environment (see `.env.example`):

| Variable | Required | Description |
|---|---|---|
| `LIVEKIT_URL` | yes | LiveKit Cloud project URL (`wss://...`) |
| `LIVEKIT_API_KEY` | yes | LiveKit API key |
| `LIVEKIT_API_SECRET` | yes | LiveKit API secret |
| `SONIOX_API_KEY` | if `TTS_PROVIDER=soniox` | Soniox API key |
| `ELEVENLABS_API_KEY` | if `TTS_PROVIDER=elevenlabs` | ElevenLabs API key |
| `TTS_PROVIDER` | no (default `soniox`) | `soniox` or `elevenlabs` |
| `TTS_VOICE_ID` | no | Provider-specific voice id/name; falls back to provider default |

## Comparing TTS providers (`tools/tts_ab.py`)

A standalone script (no LiveKit runtime needed) that synthesizes a fixed set of
Slovenian test sentences through both Soniox TTS and ElevenLabs Flash v2.5, so you
can listen and compare quality on declension, dual forms, names, times, prices, and
digit read-out:

```bash
export SONIOX_API_KEY=...
export ELEVENLABS_API_KEY=...
python tools/tts_ab.py
```

Output goes to `tts_out/soniox/<n>.wav` and `tts_out/elevenlabs/<n>.mp3` (gitignored).
Optional overrides: `SONIOX_TTS_VOICE` (default `Maya`), `SONIOX_TTS_MODEL` (default
`tts-rt-v1`), `ELEVENLABS_VOICE_ID` (default Rachel, `21m00Tcm4TlvDq8ikWAM`).
