# receptionistplus-worker

The voice side of Receptionist+: a [LiveKit Agents](https://docs.livekit.io/agents/)
Python worker that answers inbound phone calls for Jedro+ businesses, answers
questions about their services, and books appointments.

Dashboard, Stripe checkout and the credits UI live in the main Jedro+ repo
(`app/api/receptionistplus/*`), not here.

## How a call works

1. **Routing.** The dialled number (`sip.trunkPhoneNumber`) decides the company:
   `company_slug` in the dispatch rule's job metadata, else a
   `receptionist_phone_numbers` row, else `COMPANY_SLUG` from the env.
2. **Gates.** `receptionist_settings.enabled` must be true and the credit balance
   above zero, or the caller hears a short "can't take your call" line.
3. **Disclosure first.** The EU AI Act notice plays (not interruptible) while the
   company's services and staff load from the n8n `booking-v2` webhook (`init`).
   Then the greeting: `receptionist_settings.greeting_text` if set, else
   "Pozdravljeni, dobrodošli v {company}...".
4. **Conversation.** Soniox STT → Claude (Haiku; GPT-4o-mini fallback) → Soniox TTS.
   Prompt = `agent/static_prompt.py` (per language) + a 14-day date table +
   the company block (`agent/company_prompt.py`).
5. **Tools.** `get_slots`, `check_slots`, `create_booking` (booking-v2), `take_message`
   (message / callback request for the owner), `end_call_abusive`.
6. **Limits.** A call runs at most 15 minutes, less if the balance can't cover it
   (`agent/call_policy.py`); the caller hears a closing line and the owner gets a
   callback request.
7. **After the call.** Credits are deducted atomically (`receptionist_deduct_credits`),
   and a `receptionist_calls` row is written with outcome, transcript, caller number
   and any message. A database trigger notifies the owner about messages.

## Layout

```
agent/
  worker.py          entrypoint: routing, gates, session, watchdogs, call logging
  call_policy.py     pure rules: phone normalisation, time limit, billing/outcome
  call_notes.py      take_message tool + callback requests from code paths
  tools.py           booking tools and the business rules enforced in code
  booking_client.py  n8n booking-v2 webhook client (HMAC-signed)
  supabase_client.py receptionist_* tables over PostgREST
  static_prompt.py   SL/EN system prompts with rule IDs (stripped before use)
  company_prompt.py  company data → prompt block
  dates.py           spoken Slovenian/English dates
evals/               conversation evals against a fake calendar (see below)
migrations/          SQL to run by hand in the Supabase SQL editor
scripts/             deploy.sh, purge_old_transcripts.py (GDPR, daily timer)
deploy/              systemd units
tests/               unit tests
tools/tts_ab.py      offline TTS A/B (Soniox / ElevenLabs / Azure)
```

## Run locally

Python 3.12 or 3.13.

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # fill in the keys
set -a; source .env; set +a
.venv/bin/python -m agent.worker dev
```

The worker registers as `receptionistplus-worker` (explicit dispatch): it only
joins rooms that a SIP dispatch rule sends to that agent name.

## Tests and evals

```bash
.venv/bin/python -m unittest discover -s tests      # free, no network
.venv/bin/python -m evals.run_evals                 # real LLM, fake calendar
.venv/bin/python -m evals.run_evals --model claude-haiku-4-5 --repeat 3
```

Evals run the real prompt and tools against `evals/fixtures/company_jedroplus.json`
with booking-v2 replaced by an in-memory calendar — nothing is booked for real. A
full run costs a few cents. Results go to `evals/results/` (gitignored) for
before/after comparisons. Every real-call regression worth never seeing again
belongs in `evals/scenarios.py`.

## Deploy

```bash
scripts/deploy.sh --dry-run
scripts/deploy.sh            # add --units when deploy/*.service changed
```

The script refuses uncommitted changes, runs the tests, rsyncs to
`root@178.105.39.138:/opt/receptionist-plus`, installs requirements, restarts
`receptionistplus-worker` and prints its log. Env lives in
`/etc/receptionistplus/.env` on the server. **Migrations are not applied by the
script** — run new files from `migrations/` in the Supabase SQL editor first.

## Configuration

| Variable | Required | |
|---|---|---|
| `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | yes | LiveKit Cloud project |
| `ANTHROPIC_API_KEY` | yes | primary LLM |
| `OPENAI_API_KEY` | no | GPT-4o-mini fallback if Anthropic fails |
| `SONIOX_API_KEY` | yes | STT, and TTS when `TTS_PROVIDER=soniox` |
| `ELEVENLABS_API_KEY` | if `TTS_PROVIDER=elevenlabs` | |
| `TTS_PROVIDER` | no (`soniox`) | `soniox` or `elevenlabs` |
| `TTS_VOICE_ID` | no | provider voice; Soniox default `Maya` |
| `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` | yes | receptionist_* tables |
| `BOOKING_V2_HMAC_SECRET` | yes in prod | signs booking-v2 requests |
| `COMPANY_SLUG` | no | fallback company for numbers with no mapping |
| `AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION` | no | only for `tools/tts_ab.py` |

To give a company a number, add a row (E.164) — or set `{"company_slug": "..."}`
as the agent metadata on that number's LiveKit dispatch rule:

```sql
insert into receptionist_phone_numbers (phone_number, company_slug)
values ('+38612345678', 'salon-x');
```
