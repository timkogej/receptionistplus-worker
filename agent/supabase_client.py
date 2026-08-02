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

        def _do() -> dict | None:
            resp = _request(
                "GET",
                f"{self._rest_url}/receptionist_settings",
                self._api_key,
                params={
                    "company_slug": f"eq.{company_slug}",
                    "select": "company_slug,enabled,low_balance_threshold",
                },
            )
            rows = resp.json()
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

    async def deduct_credits(
        self, *, company_slug: str, call_id: str, billed_credits: float
    ) -> float:
        """Ledger a deduction and update the running balance. Returns the new balance.

        Order matches the append-only ledger design: write the transaction row
        first (with the balance it produces), then update the balance itself.
        """

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

        try:
            return await asyncio.to_thread(_do)
        except requests.RequestException as exc:
            raise SupabaseError(f"deduct_credits failed for {company_slug!r}: {exc}") from exc

    async def insert_call(self, call_row: dict) -> None:
        def _do() -> None:
            _request(
                "POST",
                f"{self._rest_url}/receptionist_calls",
                self._api_key,
                json_body=call_row,
                prefer="return=minimal",
            )

        try:
            await asyncio.to_thread(_do)
        except requests.RequestException as exc:
            raise SupabaseError(f"insert_call failed: {exc}") from exc
