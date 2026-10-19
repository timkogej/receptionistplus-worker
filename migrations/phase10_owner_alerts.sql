-- Phase 10: SMS alert to the owner when a call needs a follow-up
-- (callback request, message, unfinished booking, booking problem).
-- Additive only. Apply via the Supabase SQL editor.
--
-- owner_alert_sms: on by default once this has run (Tim, 2026-10-19).
-- owner_alert_phone: E.164; null -> the company's "Kontaktni telefon".
-- The Jedro+ settings page shows the switch only after this migration.

alter table receptionist_settings add column if not exists owner_alert_sms boolean not null default true;
alter table receptionist_settings add column if not exists owner_alert_phone text;
