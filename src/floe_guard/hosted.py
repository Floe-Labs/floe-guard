"""Hosted Floe upgrade hook — reads server-side remaining budget.

The local :class:`~floe_guard.BudgetGuard` is *estimate-based*: it prices tokens
from a vendored cost map in your own process. A determined agent (or a bug) can
run around it, and it only sees the one vendor you instrumented.

Hosted Floe is the upgrade: enforcement lives server-side against a real credit
line, so the ceiling is **un-bypassable** and spans **every vendor** (LLM tokens
*and* paid x402 tool calls) under one budget, with team budgets and analytics.

This module is the read side of that upgrade. When ``FLOE_API_KEY`` is set,
:func:`hosted_enforcement_available` reports it and :func:`hosted_remaining_usd`
queries the live Floe endpoint for the agent's remaining server-side budget.

Honest framing: this client only **reads** the remaining budget. The actual
un-bypassable, cross-vendor *enforcement* is performed server-side by hosted Floe
— not by this code. Use the returned number to inform a local ceiling; the
server is the source of truth.

    Upgrade: https://dev-dashboard.floelabs.xyz  ·  https://floelabs.xyz
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request

from .errors import HostedEnforcementError

FLOE_API_KEY_ENV = "FLOE_API_KEY"
FLOE_API_BASE_URL_ENV = "FLOE_API_BASE_URL"
DEFAULT_BASE_URL = "https://credit-api.floelabs.xyz"
CREDIT_REMAINING_PATH = "/v1/agents/credit-remaining"

# USDC has 6 implied decimals; raw integer strings divide by this to get USD.
_USDC_DECIMALS = 1_000_000


def hosted_enforcement_available() -> bool:
    """True if a Floe API key is present in the environment.

    Presence of a key is the signal that a caller *could* read the hosted budget.
    It does not itself perform any network call.
    """
    return bool(os.environ.get(FLOE_API_KEY_ENV, "").strip())


def hosted_remaining_usd(
    api_key: str | None = None,
    *,
    base_url: str | None = None,
    timeout: float = 10.0,
) -> float:
    """Read the agent's remaining server-side budget from hosted Floe, in USD.

    GETs ``/v1/agents/credit-remaining`` with ``Authorization: Bearer <key>`` and
    returns the remaining budget as a USD float. "Remaining" is the **minimum** of
    ``headroomToAutoBorrow`` and (when present) ``sessionSpendRemaining`` — both
    are raw USDC strings (6 decimals) divided by 1e6.

    This is a *read* only. Enforcement still happens server-side in Floe.

    Args:
        api_key: agent key (``floe_<hex>``). Defaults to ``FLOE_API_KEY`` env var.
        base_url: API base. Defaults to ``FLOE_API_BASE_URL`` env var, else the
            production host.
        timeout: socket timeout in seconds.

    Raises:
        HostedEnforcementError: missing key, non-200 (401/403/404 surfaced with
            the server's ``error`` field), network/timeout, or malformed JSON.
    """
    key = (api_key or os.environ.get(FLOE_API_KEY_ENV, "")).strip()
    if not key:
        raise HostedEnforcementError(
            f"No Floe API key: pass api_key= or set {FLOE_API_KEY_ENV}."
        )

    env_base = os.environ.get(FLOE_API_BASE_URL_ENV, "").strip()
    base = ((base_url or "").strip() or env_base or DEFAULT_BASE_URL).rstrip("/")
    parsed = urllib.parse.urlparse(base)
    if parsed.scheme != "https" or not parsed.netloc:
        # The request carries the Floe agent key as a bearer token — never send it
        # over a non-https or malformed URL where it could leak to an arbitrary host.
        raise HostedEnforcementError(
            f"Refusing to send the Floe API key to {base!r}: "
            "the base URL must be an https:// URL with a host."
        )
    url = f"{base}{CREDIT_REMAINING_PATH}"

    request = urllib.request.Request(
        url,
        method="GET",
        headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        raise HostedEnforcementError(_describe_http_error(exc)) from exc
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise HostedEnforcementError(
            f"Could not reach hosted Floe at {url}: {exc}"
        ) from exc

    try:
        payload = json.loads(body)
    except (ValueError, TypeError) as exc:
        raise HostedEnforcementError(
            f"Malformed JSON from hosted Floe at {url}: {exc}"
        ) from exc

    return _remaining_usd_from_payload(payload, url)


def _remaining_usd_from_payload(payload: object, url: str) -> float:
    if not isinstance(payload, dict):
        raise HostedEnforcementError(
            f"Unexpected response shape from hosted Floe at {url}: expected an object."
        )

    headroom = _usd_from_raw(payload.get("headroomToAutoBorrow"), "headroomToAutoBorrow", url)

    session_raw = payload.get("sessionSpendRemaining")
    if session_raw is None:
        return headroom

    session = _usd_from_raw(session_raw, "sessionSpendRemaining", url)
    return min(headroom, session)


def _usd_from_raw(value: object, field: str, url: str) -> float:
    try:
        return int(str(value)) / _USDC_DECIMALS
    except (ValueError, TypeError) as exc:
        raise HostedEnforcementError(
            f"Invalid {field!r} ({value!r}) from hosted Floe at {url}: {exc}"
        ) from exc


def _describe_http_error(exc: urllib.error.HTTPError) -> str:
    server_error = _server_error_field(exc)
    detail = f" ({server_error})" if server_error else ""
    if exc.code == 401:
        return f"Hosted Floe rejected the API key (401 unauthorized){detail}."
    if exc.code == 403:
        return f"Hosted Floe agent is closed or suspended (403 forbidden){detail}."
    if exc.code == 404:
        return f"Hosted Floe agent has no credit limit / not provisioned (404){detail}."
    return f"Hosted Floe returned HTTP {exc.code}{detail}."


def _server_error_field(exc: urllib.error.HTTPError) -> str | None:
    try:
        data = json.loads(exc.read())
    except Exception:
        return None
    if isinstance(data, dict):
        value = data.get("error")
        if isinstance(value, str):
            return value
    return None
