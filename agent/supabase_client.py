"""Direct Supabase REST (PostgREST) client for the Phase 3 credit/call tables.

Everything in booking_client.py goes through the n8n booking-v2 webhook, which
already talks to Supabase on our behalf. The tables here (receptionist_settings,
receptionist_credits, receptionist_credit_transactions, receptionist_calls) are
owned by this worker, not by the booking-v2 workflow, so we talk to Supabase's
REST API directly instead of adding more surface to that webhook.

# TODO Phase 5: rotate SUPABASE_SERVICE_ROLE_KEY. The key currently in use was
# reused from the one already embedded in the n8n booking-v2 workflow's HTTP
# node headers (flagged during Phase 2 discovery as exposed in plaintext) —
# a deliberate, temporary reuse-for-now decision, not a discovery of a new leak.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import requests

logger = logging.getLogger("receptionistplus-worker.supabase")


# receptionist_calls columns added by later migrations (phase6, phase7). An
# insert that PostgREST rejects for an unknown column is retried without
# them, so a deploy that runs ahead of its migration still logs the call.
PHASE6_CALL_COLUMNS = frozenset(
    {"caller_phone", "dialed_number", "message", "callback_requested", "summary", "language", "metrics"}
)

_SETTINGS_BASE = (
    "company_slug,enabled,low_balance_threshold,language,greeting_text,detect_caller_language"
)
_SETTINGS_PHASE7 = (
    _SETTINGS_BASE + ",instructions,transfer_enabled,transfer_number,sms_confirmation"
)
_SETTINGS_PHASE10 = _SETTINGS_PHASE7 + ",owner_alert_sms,owner_alert_phone"
_SETTINGS_PHASE11 = (
    _SETTINGS_PHASE10
    + ",greeting_style,greeting_name,assistant_name,booking_questions,booking_advice"
)
# Newest first: a column a migration hasn't added yet makes PostgREST answer
# 400, and the next, older list is tried. The features those columns drive
# stay off until then.
_SETTINGS_SELECTS = (_SETTINGS_PHASE11, _SETTINGS_PHASE10, _SETTINGS_PHASE7, _SETTINGS_BASE)

_PROFILE_SELECT = (
    '"ID Podjetja","Naziv Podjetja","Naslov podjetja","Kontaktni telefon",'
    '"Spletna stran",urnik,"Navodila chatbot",cancel_appt_days,reschedule_appt_days'
)


class SupabaseError(RuntimeError):
    """Raised when a Supabase REST call fails or returns an unexpected shape."""


def _headers(api_key: str) -> dict:
    return {
        "apikey": api_key,
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _request(
    method: str,
    url: str,
    api_key: str,
    *,
    params: dict | None = None,
    json_body: dict | list | None = None,
    prefer: str | None = None,
) -> requests.Response:
    headers = _headers(api_key)
    if prefer:
        headers["Prefer"] = prefer
    response = requests.request(
        method, url, headers=headers, params=params, json=json_body, timeout=10
    )
    response.raise_for_status()
    return response


class SupabaseClient:
    def __init__(self, base_url: str, api_key: str) -> None:
        self._rest_url = base_url.rstrip("/") + "/rest/v1"
        self._api_key = api_key

    async def get_settings(self, company_slug: str) -> dict | None:
        """Fetch receptionist_settings for a company, or None if no row exists."""

        def _get(select: str) -> list:
            return _request(
                "GET",
                f"{self._rest_url}/receptionist_settings",
                self._api_key,
                params={"company_slug": f"eq.{company_slug}", "select": select},
            ).json()

        def _do() -> dict | None:
            for select in _SETTINGS_SELECTS[:-1]:
                try:
                    rows = _get(select)
                    break
                except requests.HTTPError as exc:
                    if exc.response is None or exc.response.status_code != 400:
                        raise
            else:
                rows = _get(_SETTINGS_SELECTS[-1])
            return rows[0] if rows else None

        try:
            return await asyncio.to_thread(_do)
        except requests.RequestException as exc:
            raise SupabaseError(f"get_settings failed for {company_slug!r}: {exc}") from exc

    async def get_balance(self, company_slug: str) -> float:
        """Fetch balance_credits for a company. Missing row is treated as 0."""

        def _do() -> float:
            resp = _request(
                "GET",
                f"{self._rest_url}/receptionist_credits",
                self._api_key,
                params={
                    "company_slug": f"eq.{company_slug}",
                    "select": "balance_credits",
                },
            )
            rows = resp.json()
            if not rows:
                return 0.0
            return float(rows[0]["balance_credits"])

        try:
            return await asyncio.to_thread(_do)
        except requests.RequestException as exc:
            raise SupabaseError(f"get_balance failed for {company_slug!r}: {exc}") from exc

    async def get_company_profile(self, company_slug: str) -> dict | None:
        """The company's row in "Podatki podjetij" (shared with the rest of
        Jedro+): its "ID Podjetja", address, phone, website, opening hours
        (urnik, JSON) and Chatbot+ instructions."""

        def _do() -> dict | None:
            rows = _request(
                "GET",
                f"{self._rest_url}/Podatki podjetij",
                self._api_key,
                params={"slug": f"eq.{company_slug}", "select": _PROFILE_SELECT, "limit": "1"},
            ).json()
            return rows[0] if rows else None

        try:
            return await asyncio.to_thread(_do)
        except requests.RequestException as exc:
            raise SupabaseError(f"get_company_profile failed for {company_slug!r}: {exc}") from exc

    async def get_confirmation_template_row(self, company_slug: str) -> dict | None:
        """The "Podatki podjetij" fields the SMS confirmation template needs.
        select=* on purpose: lastna_predloga_potrdilo_en only exists after the
        Jedro+ migration, and naming a missing column fails the request."""
        fields = (
            "sms_type_potrdilo", "lastna_predloga_potrdilo", "lastna_predloga_potrdilo_en",
            "jezik posiljanja", "language", "Naziv Podjetja", "Naslov podjetja", "Kontaktni telefon",
            "Kontaktni_email", "Kontaktni email", "appt_management_link",
        )

        def _do() -> dict | None:
            rows = _request(
                "GET",
                f"{self._rest_url}/Podatki podjetij",
                self._api_key,
                params={"slug": f"eq.{company_slug}", "select": "*", "limit": "1"},
            ).json()
            return {k: rows[0].get(k) for k in fields} if rows else None

        try:
            return await asyncio.to_thread(_do)
        except requests.RequestException as exc:
            raise SupabaseError(f"get_confirmation_template_row failed for {company_slug!r}: {exc}") from exc

    async def sms_quota_left(self, company_slug: str) -> int | None:
        """SMS the company can still send this billing period, by the same
        rules as every Jedro+ SMS path (n8n "Kvota" nodes): a live
        subscription that isn't SMS-blocked, a plan with SMS, quota =
        override, else plan + add-on, minus what this period already used
        or reserved. 0 when SMS isn't available at all; None if unknown."""

        def _get(table: str, params: dict) -> list:
            return _request("GET", f"{self._rest_url}/{table}", self._api_key, params=params).json()

        def _do() -> int:
            company = _get("companies", {"slug": f"eq.{company_slug}", "select": "id", "limit": "1"})
            if not company:
                return 0
            subs = _get(
                "company_subscriptions",
                {
                    "company_id": f"eq.{company[0]['id']}",
                    "sms_blocked": "eq.false",
                    "status": "in.(trialing,active,past_due,paused)",
                    "order": "current_period_end.desc",
                    "limit": "1",
                    "select": "plan_id,sms_quota_override,sms_addon_monthly,current_period_start",
                },
            )
            if not subs or not subs[0].get("plan_id"):
                return 0
            sub = subs[0]
            plans = _get("plans", {"id": f"eq.{sub['plan_id']}", "select": "sms_enabled,sms_quota_monthly", "limit": "1"})
            plan = plans[0] if plans else {}
            if not plan.get("sms_enabled"):
                return 0
            if sub.get("sms_quota_override") is not None:
                quota = int(sub["sms_quota_override"])
            else:
                quota = int(plan.get("sms_quota_monthly") or 0) + int(sub.get("sms_addon_monthly") or 0)
            usage = _get(
                "company_sms_usage",
                {
                    "company_id": f"eq.{company[0]['id']}",
                    "period_start": f"eq.{str(sub.get('current_period_start') or '')[:10]}",
                    "select": "used_count,reserved_count",
                    "limit": "1",
                },
            )
            used = sum(int((usage[0] if usage else {}).get(k) or 0) for k in ("used_count", "reserved_count"))
            return max(0, quota - used)

        try:
            return await asyncio.to_thread(_do)
        except (requests.RequestException, ValueError, TypeError, KeyError):
            logger.exception("SMS quota check failed for %s", company_slug)
            return None

    # --- The caller's own appointments (cancel / reschedule by phone) ------
    # Same rules as the customer portal's n8n flows ("Customer portal –
    # Cancel appointment" / "– Confirm reschedule"), keyed by "ID stranke"
    # instead of email: phone customers often have no email.

    _APPT_COLUMNS = 'id,"ID termina","ID stranke","ID podjetja",Datum,"Čas",Konec,Storitev,"ID storitve",Oseba,"ID Osebe",Status,deleted_at'

    async def get_customer_appointments(self, company_id: str, customer_id: str, from_date: str) -> list[dict]:
        def _do() -> list[dict]:
            return _request(
                "GET",
                f"{self._rest_url}/Termini",
                self._api_key,
                params={
                    "ID podjetja": f"eq.{company_id}",
                    "ID stranke": f"eq.{customer_id}",
                    "Datum": f"gte.{from_date}",
                    "Status": "in.(scheduled,pending_payment)",
                    "deleted_at": "is.null",
                    "select": self._APPT_COLUMNS,
                    "order": 'Datum.asc,"Čas".asc',
                    "limit": "10",
                },
            ).json()

        try:
            return await asyncio.to_thread(_do)
        except requests.RequestException as exc:
            raise SupabaseError(f"get_customer_appointments failed: {exc}") from exc

    async def update_appointment(self, company_id: str, row_id: int, fields: dict) -> dict | None:
        """PATCH one Termini row; returns the updated row."""

        def _do() -> dict | None:
            rows = _request(
                "PATCH",
                f"{self._rest_url}/Termini",
                self._api_key,
                params={"id": f"eq.{row_id}", "ID podjetja": f"eq.{company_id}"},
                json_body=fields,
                prefer="return=representation",
            ).json()
            return rows[0] if rows else None

        try:
            return await asyncio.to_thread(_do)
        except requests.RequestException as exc:
            raise SupabaseError(f"update_appointment failed: {exc}") from exc

    async def add_appointment_history(self, company_id: str, termin_id: str, actor: str, action: str) -> None:
        """jp_zgodovina_pripisi, as the portal flows call it ("odpovedan" /
        "prestavljen"). Best effort: history must not undo a change."""

        def _do() -> None:
            _request(
                "POST",
                f"{self._rest_url}/rpc/jp_zgodovina_pripisi",
                self._api_key,
                json_body={
                    "p_company": company_id,
                    "p_entiteta": termin_id,
                    "p_izvedel": actor,
                    "p_izvedel_tip": "stranka",
                    "p_akcija": action,
                },
            )

        try:
            await asyncio.to_thread(_do)
        except requests.RequestException:
            logger.exception("appointment history write failed (%s %s)", action, termin_id)

    async def find_customer_by_phone(self, company_id: str, phone_e164: str) -> dict | None:
        """An existing customer of this company with this number, or None.

        Only what the call needs: name, whether an email is on file, and
        their preferred language. Nothing from here is stored by the worker.
        """

        def _do() -> dict | None:
            rows = _request(
                "GET",
                f"{self._rest_url}/Stranke",
                self._api_key,
                params={
                    "ID Podjetja": f"eq.{company_id}",
                    "phone_e164": f"eq.{phone_e164}",
                    "select": '"ID stranke",Ime,Priimek,"Email stranke",language',
                    "order": '"Zadnja interakcija".desc.nullslast',
                    "limit": "1",
                },
            ).json()
            return rows[0] if rows else None

        try:
            return await asyncio.to_thread(_do)
        except requests.RequestException as exc:
            raise SupabaseError(f"find_customer_by_phone failed: {exc}") from exc

    async def update_customer_language(
        self, company_id: str, customer_id: str, jedroplus_code: str
    ) -> None:
        """Set "Stranke".language ("slo"/"eng"/"de"/"it"/"hr") for one
        customer of one company."""

        def _do() -> None:
            _request(
                "PATCH",
                f"{self._rest_url}/Stranke",
                self._api_key,
                params={
                    "ID Podjetja": f"eq.{company_id}",
                    "ID stranke": f"eq.{customer_id}",
                },
                json_body={"language": jedroplus_code},
                prefer="return=minimal",
            )

        try:
            await asyncio.to_thread(_do)
        except requests.RequestException as exc:
            raise SupabaseError(f"update_customer_language failed: {exc}") from exc

    async def lookup_company_by_number(self, phone_number: str) -> str | None:
        """company_slug for a dialled E.164 number, or None if unmapped.

        A missing receptionist_phone_numbers table (migration not applied
        yet) is treated as "no mapping" rather than an error, so the worker
        can be deployed before the migration without dropping calls.
        """

        def _do() -> str | None:
            try:
                resp = _request(
                    "GET",
                    f"{self._rest_url}/receptionist_phone_numbers",
                    self._api_key,
                    params={
                        "phone_number": f"eq.{phone_number}",
                        "select": "company_slug",
                    },
                )
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 404:
                    logger.warning(
                        "receptionist_phone_numbers missing (migration phase6 "
                        "not applied?) — no per-number routing"
                    )
                    return None
                raise
            rows = resp.json()
            return rows[0]["company_slug"] if rows else None

        try:
            return await asyncio.to_thread(_do)
        except requests.RequestException as exc:
            raise SupabaseError(f"lookup_company_by_number failed: {exc}") from exc

    async def deduct_credits(
        self, *, company_slug: str, call_id: str, billed_credits: float
    ) -> float:
        """Deduct a call's credits in one transaction. Returns the new balance.

        Calls the receptionist_deduct_credits SQL function (migration
        phase6), which locks the balance row, writes the ledger row and the
        new balance together, and is idempotent per call_id. The old
        read-then-write sequence lost a deduction whenever two calls ended
        at the same time.
        """

        def _do() -> float:
            try:
                resp = _request(
                    "POST",
                    f"{self._rest_url}/rpc/receptionist_deduct_credits",
                    self._api_key,
                    json_body={
                        "p_company_slug": company_slug,
                        "p_call_id": call_id,
                        "p_amount": billed_credits,
                    },
                )
            except requests.HTTPError as exc:
                if exc.response is None or exc.response.status_code != 404:
                    raise
                # Function not created yet (migration phase6 not applied):
                # bill the old way rather than not at all.
                logger.warning(
                    "receptionist_deduct_credits missing — falling back to "
                    "non-atomic deduction; apply migrations/phase6"
                )
                return self._deduct_credits_legacy(company_slug, call_id, billed_credits)
            return float(resp.json())

        try:
            return await asyncio.to_thread(_do)
        except requests.RequestException as exc:
            raise SupabaseError(f"deduct_credits failed for {company_slug!r}: {exc}") from exc

    async def update_call_summary(self, call_id: str, summary: str) -> None:
        """Add the post-call summary to an already written call row. A
        missing summary column (phase7 not applied) is ignored."""

        def _do() -> None:
            try:
                _request(
                    "PATCH",
                    f"{self._rest_url}/receptionist_calls",
                    self._api_key,
                    params={"id": f"eq.{call_id}"},
                    json_body={"summary": summary},
                    prefer="return=minimal",
                )
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 400:
                    logger.warning("summary column missing (phase7 not applied?)")
                    return
                raise

        try:
            await asyncio.to_thread(_do)
        except requests.RequestException as exc:
            raise SupabaseError(f"update_call_summary failed: {exc}") from exc

    def _deduct_credits_legacy(
        self, company_slug: str, call_id: str, billed_credits: float
    ) -> float:
        """Pre-phase6 read-then-write deduction. Not atomic; only used until
        the migration is applied."""
        resp = _request(
            "GET",
            f"{self._rest_url}/receptionist_credits",
            self._api_key,
            params={"company_slug": f"eq.{company_slug}", "select": "balance_credits"},
        )
        rows = resp.json()
        current_balance = float(rows[0]["balance_credits"]) if rows else 0.0
        new_balance = current_balance - billed_credits
        _request(
            "POST",
            f"{self._rest_url}/receptionist_credit_transactions",
            self._api_key,
            json_body={
                "company_slug": company_slug,
                "delta_credits": -billed_credits,
                "balance_after": new_balance,
                "type": "deduction",
                "call_id": call_id,
            },
            prefer="return=minimal",
        )
        _request(
            "PATCH",
            f"{self._rest_url}/receptionist_credits",
            self._api_key,
            params={"company_slug": f"eq.{company_slug}"},
            json_body={
                "balance_credits": new_balance,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            prefer="return=minimal",
        )
        return new_balance

    async def insert_call(self, call_row: dict) -> None:
        """Insert one receptionist_calls row.

        If the phase6 columns aren't there yet (PostgREST rejects unknown
        columns with a 400), retry without them: losing the caller's number
        is better than losing the whole call record, which billing history
        depends on.
        """

        def _post(row: dict) -> None:
            _request(
                "POST",
                f"{self._rest_url}/receptionist_calls",
                self._api_key,
                json_body=row,
                prefer="return=minimal",
            )

        def _do() -> None:
            try:
                _post(call_row)
            except requests.HTTPError as exc:
                if exc.response is None or exc.response.status_code != 400:
                    raise
                reduced = {k: v for k, v in call_row.items() if k not in PHASE6_CALL_COLUMNS}
                if reduced == call_row:
                    raise
                logger.warning(
                    "insert_call rejected (%s) — retrying without phase6 columns",
                    exc.response.text[:200],
                )
                _post(reduced)

        try:
            await asyncio.to_thread(_do)
        except requests.RequestException as exc:
            raise SupabaseError(f"insert_call failed: {exc}") from exc
