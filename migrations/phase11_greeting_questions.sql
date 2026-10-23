-- Receptionist+: the greeting style the owner picks, and what the assistant
-- asks before a booking and tells after it. Same as Jedro+'s
-- supabase/migrations/1791400000_receptionist_pozdrav_vprasanja.sql (run one
-- of them, once). Until it has run the worker reads the older columns
-- (agent/supabase_client.py) and greets in the classic style.
--
-- greeting_style: classic | sales | short | custom (custom = greeting_text,
--   after the standard AI notice). The texts are in the worker
--   (agent/greeting.py) in every receptionist language.
-- greeting_name: the business name the greeting starts with; null = the
--   company name without "d.o.o." / "s.p.".
-- assistant_name: optional ("Sem Ana, digitalna asistentka."); null = none.
-- booking_questions / booking_advice: [{"text": "...", "service_ids": ["18"]}],
--   service_ids empty = every service. Answers go into the appointment note.
--
-- Run manually via the Supabase SQL editor.

ALTER TABLE receptionist_settings ADD COLUMN IF NOT EXISTS greeting_style text NOT NULL DEFAULT 'classic';
ALTER TABLE receptionist_settings DROP CONSTRAINT IF EXISTS receptionist_settings_greeting_style_check;
ALTER TABLE receptionist_settings ADD CONSTRAINT receptionist_settings_greeting_style_check
  CHECK (greeting_style IN ('classic', 'sales', 'short', 'custom'));
ALTER TABLE receptionist_settings ADD COLUMN IF NOT EXISTS greeting_name text;
ALTER TABLE receptionist_settings ADD COLUMN IF NOT EXISTS assistant_name text;
ALTER TABLE receptionist_settings ADD COLUMN IF NOT EXISTS booking_questions jsonb NOT NULL DEFAULT '[]'::jsonb;
ALTER TABLE receptionist_settings ADD COLUMN IF NOT EXISTS booking_advice jsonb NOT NULL DEFAULT '[]'::jsonb;

-- Owners who already wrote their own greeting keep it.
UPDATE receptionist_settings SET greeting_style = 'custom'
WHERE greeting_text IS NOT NULL AND btrim(greeting_text) <> '' AND greeting_style = 'classic';

-- Rollback:
--   ALTER TABLE receptionist_settings
--     DROP COLUMN IF EXISTS greeting_style, DROP COLUMN IF EXISTS greeting_name,
--     DROP COLUMN IF EXISTS assistant_name, DROP COLUMN IF EXISTS booking_questions,
--     DROP COLUMN IF EXISTS booking_advice;
