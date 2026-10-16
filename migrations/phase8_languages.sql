-- Phase 8 (sklop 5): languages.
-- Additive only; the worker writes the call row without this column until
-- it exists (agent/supabase_client.py retries without optional columns).
--
-- Apply via the Supabase SQL editor.

-- The language the call ENDED in (sl/en/de/hr/it) — after any switch. For
-- the dashboard call list and for seeing how often callers need another
-- language.
alter table receptionist_calls add column if not exists language text;

-- receptionist_settings.detect_caller_language already exists (dashboard).
-- Since sklop 5 the worker reads it: on = the call may switch between
-- Slovenian, English, German, Croatian and Italian, and a known customer
-- starts in their stored language ("Stranke".language).
