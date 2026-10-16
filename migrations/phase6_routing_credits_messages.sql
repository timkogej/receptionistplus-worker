-- Phase 6 (sklop 1): per-number call routing, atomic credit deduction, and
-- caller messages / callback requests.
-- Additive only — no existing column changes meaning, and the worker keeps
-- working against the old schema until this is applied (see the fallbacks in
-- agent/supabase_client.py).
--
-- Apply via the Supabase SQL editor (Project → SQL Editor → paste → Run),
-- BEFORE deploying the worker version that calls receptionist_deduct_credits.

-- ---------------------------------------------------------------------------
-- 1) Which company answers which phone number.
--
-- Until now every call went to the one company named in COMPANY_SLUG (the
-- worker's env), whatever number was dialled. The worker now looks the
-- dialled number up here; COMPANY_SLUG stays only as a fallback for numbers
-- with no row, so the current single number keeps working unchanged.
--
-- phone_number is E.164 ("+38612345678"), the form Telnyx/LiveKit put in the
-- sip.trunkPhoneNumber attribute.
-- ---------------------------------------------------------------------------
create table if not exists receptionist_phone_numbers (
    phone_number text primary key check (phone_number ~ '^\+[1-9][0-9]{6,14}$'),
    company_slug text not null,
    note text,
    created_at timestamptz not null default now()
);

create index if not exists idx_receptionist_phone_numbers_company
    on receptionist_phone_numbers (company_slug);

alter table receptionist_phone_numbers enable row level security;
-- No policies: only the service role (worker, dashboard API routes) reads it.

-- ---------------------------------------------------------------------------
-- 2) Call log: who called, which number they dialled, and any message left.
--
-- caller_phone is needed to call the customer back; it is personal data and
-- is cleared together with the transcript by scripts/purge_old_transcripts.py.
-- ---------------------------------------------------------------------------
alter table receptionist_calls add column if not exists caller_phone text;
alter table receptionist_calls add column if not exists dialed_number text;
alter table receptionist_calls add column if not exists message text;
alter table receptionist_calls add column if not exists callback_requested boolean not null default false;

-- ---------------------------------------------------------------------------
-- 3) Atomic, idempotent credit deduction.
--
-- The worker used to read the balance, write a ledger row, then PATCH the new
-- balance — three separate requests, so two calls ending at the same moment
-- could both read the same balance and one deduction was lost. This does it
-- in one transaction with the credits row locked, and a second call for the
-- same call_id (worker retry) is a no-op that returns the current balance.
--
-- The low-balance trigger (jp_receptionist_nizko_stanje, Jedro+ repo) fires on
-- the UPDATE below exactly as it did on the old PATCH.
-- ---------------------------------------------------------------------------
create unique index if not exists uq_receptionist_credit_tx_call_deduction
    on receptionist_credit_transactions (call_id)
    where type = 'deduction' and call_id is not null;

create or replace function receptionist_deduct_credits(
    p_company_slug text,
    p_call_id uuid,
    p_amount numeric
)
returns numeric
language plpgsql
security definer
set search_path = public
as $fn$
declare
    v_balance numeric;
    v_new     numeric;
    v_tx_id   uuid;
begin
    if p_amount is null or p_amount <= 0 then
        raise exception 'p_amount must be positive, got %', p_amount;
    end if;

    -- Create the row if it is missing, so the lock below always has a target.
    insert into receptionist_credits (company_slug, balance_credits)
    values (p_company_slug, 0)
    on conflict (company_slug) do nothing;

    select balance_credits into v_balance
    from receptionist_credits
    where company_slug = p_company_slug
    for update;

    v_new := v_balance - p_amount;

    insert into receptionist_credit_transactions
        (company_slug, delta_credits, balance_after, type, call_id)
    values
        (p_company_slug, -p_amount, v_new, 'deduction', p_call_id)
    on conflict (call_id) where type = 'deduction' and call_id is not null
    do nothing
    returning id into v_tx_id;

    if v_tx_id is null then
        -- Already deducted for this call: return the balance as it stands.
        return v_balance;
    end if;

    update receptionist_credits
       set balance_credits = v_new,
           updated_at = now()
     where company_slug = p_company_slug;

    return v_new;
end;
$fn$;

revoke all on function receptionist_deduct_credits(text, uuid, numeric) from public;
revoke all on function receptionist_deduct_credits(text, uuid, numeric) from anon, authenticated;
grant execute on function receptionist_deduct_credits(text, uuid, numeric) to service_role;

-- ---------------------------------------------------------------------------
-- 4) Notify the owner when a caller leaves a message or asks for a callback.
--
-- Same mechanism as the low-balance warning: a trigger, so it fires however
-- the row is written. Owners and admins of the company get one notification
-- per call. Never blocks the insert.
-- ---------------------------------------------------------------------------
create or replace function receptionist_notify_message()
returns trigger
language plpgsql
security definer
set search_path = public
as $fn$
declare
    v_podjetje  uuid;
    v_prejemnik record;
    v_naslov    text;
    v_telo      text;
begin
    if coalesce(NEW.message, '') = '' and not NEW.callback_requested then
        return NEW;
    end if;

    select c.id into v_podjetje
    from public.companies c
    where c.slug = NEW.company_slug
    limit 1;

    if v_podjetje is null then
        return NEW;
    end if;

    v_naslov := case when NEW.callback_requested
                     then 'Receptionist+: stranka prosi za povratni klic'
                     else 'Receptionist+: novo sporočilo stranke' end;
    v_telo := concat_ws(E'\n',
                        nullif(NEW.message, ''),
                        case when NEW.caller_phone is not null
                             then 'Telefon: ' || NEW.caller_phone end);

    for v_prejemnik in
        select cm.user_id
        from public.company_members cm
        where cm.company_id = v_podjetje
          and lower(cm.role) in ('owner', 'admin')
    loop
        insert into public.notifications
            (company_id, recipient_user_id, type, severity, title, body,
             action_url, entity_type, entity_id, dedupe_key)
        values
            (v_podjetje, v_prejemnik.user_id, 'system', 'warning', v_naslov, v_telo,
             '/receptionist-plus', 'receptionist_call', NEW.id::text,
             'recept_sporocilo_' || NEW.id::text || '_' || v_prejemnik.user_id::text)
        on conflict do nothing;
    end loop;

    return NEW;
exception when others then
    return NEW;
end;
$fn$;

drop trigger if exists receptionist_notify_message_t on receptionist_calls;
create trigger receptionist_notify_message_t
    after insert on receptionist_calls
    for each row execute function receptionist_notify_message();
