-- Phase 3: credits, call logging, and the credit-gate.
-- Additive only — no changes to any existing table.
--
-- Keyed on company_slug (text), matching the real booking-v2 contract
-- (booking_client.py), not the idPodjetja/uuid duality assumed by the
-- original architecture doc.
--
-- Apply via the Supabase SQL editor (Project → SQL Editor → paste → Run).

create table if not exists receptionist_settings (
    company_slug text primary key,
    enabled boolean not null default false,
    low_balance_threshold numeric not null default 60, -- credits
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists receptionist_credits (
    company_slug text primary key,
    balance_credits numeric(12, 2) not null default 0,
    updated_at timestamptz not null default now()
);

create table if not exists receptionist_credit_transactions (
    id uuid primary key default gen_random_uuid(),
    company_slug text not null,
    delta_credits numeric(12, 2) not null,       -- + purchase/grant, - deduction
    balance_after numeric(12, 2) not null,
    type text not null,                           -- purchase | deduction | trial_grant | adjustment
    call_id uuid,                                 -- nullable, links to receptionist_calls.id for deductions
    stripe_checkout_session text,
    note text,
    created_at timestamptz not null default now()
);

create index if not exists idx_receptionist_credit_transactions_company_created
    on receptionist_credit_transactions (company_slug, created_at desc);

create table if not exists receptionist_calls (
    id uuid primary key default gen_random_uuid(),
    company_slug text not null,
    started_at timestamptz not null,
    ended_at timestamptz,
    duration_sec int,
    billed_credits numeric(12, 2),
    -- outcome is intentionally free text, not an enum or CHECK constraint, so
    -- the worker can add a value without a schema migration. The list below is
    -- documentation of what the worker actually emits (agent/worker.py) and is
    -- not enforced by the database.
    --   booked              - create_booking succeeded
    --   info_only           - answered questions, no booking made
    --   abandoned           - shorter than ABANDONED_CALL_THRESHOLD_SEC, unbilled
    --   ended_abusive       - agent closed the call after abuse continued past
    --                         one warning (still billed by duration; a booking
    --                         made earlier in the call is kept in
    --                         created_termin_id)
    --   no_credits          - credit gate refused the call
    --   booking_unavailable - booking-v2 init failed, call could not proceed
    -- ("message_taken" and "error" were listed here historically but the worker
    -- has never emitted either.)
    outcome text,
    transcript jsonb,                             -- [{role, text, ts}]
    created_termin_id text,                       -- e.g. "OB-000039" if a booking happened
    livekit_room text,
    created_at timestamptz not null default now()
);

create index if not exists idx_receptionist_calls_company_started
    on receptionist_calls (company_slug, started_at desc);
