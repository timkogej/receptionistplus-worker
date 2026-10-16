-- Phase 7 (sklop 4): instructions/FAQ, transfer to a person, call summaries.
-- Additive only. The worker runs against the old schema too (missing columns
-- are treated as "feature off"), so the order of migration vs deploy doesn't
-- matter — but the features only appear once this is applied.
--
-- Apply via the Supabase SQL editor.

-- What the business wants the receptionist to know beyond services/staff:
-- FAQ, parking, payment, cancellation policy, "always mention X". Free text,
-- written by the owner. When empty, the worker falls back to Chatbot+'s
-- "Navodila chatbot" in "Podatki podjetij", so a business that already wrote
-- instructions for the chatbot gets them on the phone too.
alter table receptionist_settings add column if not exists instructions text;

-- Transfer to a person. Only offered when enabled, a number is set, and the
-- business is open right now according to "Podatki podjetij".urnik.
alter table receptionist_settings add column if not exists transfer_enabled boolean not null default false;
alter table receptionist_settings add column if not exists transfer_number text
    check (transfer_number is null or transfer_number ~ '^\+[1-9][0-9]{6,14}$');

-- Booking confirmations from a call go by SMS (n8n "Receptionist+ – SMS
-- stranki", counted against the company's SMS quota) and the caller isn't
-- asked for an email. Switch off for businesses without SMS in their plan:
-- the receptionist then asks for an email and booking-v2 sends the usual
-- email confirmation.
alter table receptionist_settings add column if not exists sms_confirmation boolean not null default true;

-- One-to-three sentence summary of the call, written by the worker after
-- hang-up, for the owner's call list and notifications.
alter table receptionist_calls add column if not exists summary text;
-- outcome gains the value 'transferred' (free text column, no change needed).

-- Notifications for messages/callbacks now include the summary when there is
-- one: the owner sees what the call was about, not just the message line.
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
                             then 'Telefon: ' || NEW.caller_phone end,
                        case when coalesce(NEW.summary, '') <> ''
                             then 'Povzetek klica: ' || NEW.summary end);

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
