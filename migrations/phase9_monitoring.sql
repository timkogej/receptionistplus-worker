-- Phase 9 (sklop 7): worker monitoring and per-call metrics.
-- Additive only. Apply via the Supabase SQL editor.

-- One row per worker host, written every minute by
-- scripts/heartbeat.py (systemd timer receptionistplus-heartbeat). The n8n
-- workflow "Receptionist+ – Nadzor workerja" reads it every 5 minutes and
-- texts an alert when the worker is unhealthy or the row goes stale (the
-- whole server is down). alerted/alerted_at are that workflow's state, so it
-- alerts once per outage and once on recovery.
create table if not exists receptionist_worker_heartbeat (
    id text primary key,                -- host name
    healthy boolean not null,
    detail text,                        -- "OK" or why not
    active_jobs integer,
    revision text,                      -- deployed git revision
    updated_at timestamptz not null default now(),
    alerted boolean not null default false,
    alerted_at timestamptz
);
alter table receptionist_worker_heartbeat enable row level security;
-- No policies: service role only (worker host, n8n).

-- Per-call latency and reliability numbers, written by the worker at the
-- end of each call: first-token/first-byte percentiles per stage, turns,
-- LLM fallbacks, provider errors, language switches. JSON so new numbers
-- need no migration.
alter table receptionist_calls add column if not exists metrics jsonb;
