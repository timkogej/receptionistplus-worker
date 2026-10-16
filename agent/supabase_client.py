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
    {"caller_phone", "dialed_number", "message", "callback_requested", "summary"}
)

_SETTINGS_BASE = "company_slug,enabled,low_balance_threshold,language,greeting_text"
_SETTINGS_PHASE7 = _SETTINGS_BASE + ",instructions,transfer_enabled,transfer_number"

_PROFILE_SELECT = (
    '"ID Podjetja","Naziv Podjetja","Naslov podjetja","Kontaktni telefon",'
    '"Spletna stran",urnik,"Navodila chatbot"'
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
            try:
                rows = _get(_SETTINGS_PHASE7)
            except requests.HTTPError as exc:
                if exc.response is None or exc.response.status_code != 400:
                    raise
                # phase7 columns not there yet: the features they drive stay off.
                rows = _get(_SETTINGS_BASE)
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
