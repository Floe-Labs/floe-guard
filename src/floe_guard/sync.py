"""Opt-in ledger sync — push the local spend ledger to Floe's Reconcile Mode.

**Zero-telemetry is the default, and stays the default.** Nothing in this module
runs unless the caller *explicitly* opts in with a Floe API key — via
:meth:`~floe_guard.BudgetGuard.enable_sync` + :meth:`~floe_guard.BudgetGuard.sync`,
or the ``floe-guard push`` CLI. There is no implicit enablement and no background
send: the ledger leaves your process only when you call one of those. Your ledger,
your key, your choice.

**Why sync at all.** Floe's gateway can't see spend it never routed — BYOK,
self-hosted, or off-path LLM/tool calls. Pushing your local ledger into Reconcile
Mode is how that spend lands on the ledger and your **Coverage Score** becomes
computable. Budget, not balance: this reports *what you already spent* for
coverage/attribution; it does not move money or change any wallet balance.

**What leaves the process — exactly the** :meth:`~floe_guard.BudgetGuard.export_log`
**JSONL** and nothing else: one line per spend event, each a priced-cost record —
``timestamp``, ``kind`` (``"llm"``/``"tool"``), ``model_or_tool``,
``prompt_tokens``, ``completion_tokens``, ``cost_usd``, and the optional ``label``
/ ``reserved`` you set. **No prompts, no message content, no identifiers** beyond a
``label`` you choose.

    Opt in: https://dev-dashboard.floelabs.xyz  ·  https://floelabs.xyz
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request

from .errors import LedgerSyncError

FLOE_API_KEY_ENV = "FLOE_API_KEY"
FLOE_API_BASE_URL_ENV = "FLOE_API_BASE_URL"
DEFAULT_BASE_URL = "https://credit-api.floelabs.xyz"
LEDGER_SYNC_PATH = "/v1/agents/ledger/sync"


def push_ledger(
    jsonl: str,
    api_key: str | None = None,
    *,
    base_url: str | None = None,
    timeout: float = 30.0,
) -> int:
    """POST an :meth:`~floe_guard.BudgetGuard.export_log` JSONL ledger to Reconcile
    Mode and return the number of events the server accepted.

    This is the ONLY function that sends the ledger, and it sends only when called.
    An empty ledger is a no-op (returns ``0`` without any network call).

    Args:
        jsonl: the ledger as newline-delimited JSON — ``export_log()`` output.
        api_key: Floe agent key (``floe_<hex>``, ``read_write``). Defaults to the
            ``FLOE_API_KEY`` env var.
        base_url: API base. Defaults to ``FLOE_API_BASE_URL``, else the prod host.
        timeout: socket timeout in seconds.

    Raises:
        LedgerSyncError: missing key, a non-2xx response, network/timeout, or a
            malformed response body.
    """
    # An empty ledger never touches the network — nothing to send.
    if not jsonl.strip():
        return 0

    key = (api_key or os.environ.get(FLOE_API_KEY_ENV, "")).strip()
    if not key:
        raise LedgerSyncError(
            f"No Floe API key: pass api_key= or set {FLOE_API_KEY_ENV}. "
            "Sync is opt-in and needs your key."
        )

    env_base = os.environ.get(FLOE_API_BASE_URL_ENV, "").strip()
    base = ((base_url or "").strip() or env_base or DEFAULT_BASE_URL).rstrip("/")
    parsed = urllib.parse.urlparse(base)
    if parsed.scheme != "https" or not parsed.netloc:
        # The request carries the Floe agent key as a bearer token AND your spend
        # ledger — never send either over a non-https or malformed URL.
        raise LedgerSyncError(
            f"Refusing to send the ledger to {base!r}: "
            "the base URL must be an https:// URL with a host."
        )
    url = f"{base}{LEDGER_SYNC_PATH}"

    request = urllib.request.Request(
        url,
        data=jsonl.encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/x-ndjson",
            "Accept": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        raise LedgerSyncError(_describe_http_error(exc)) from exc
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise LedgerSyncError(f"Could not reach Floe at {url}: {exc}") from exc

    try:
        payload = json.loads(body)
    except (ValueError, TypeError) as exc:
        raise LedgerSyncError(f"Malformed JSON from Floe at {url}: {exc}") from exc

    if not isinstance(payload, dict):
        raise LedgerSyncError(f"Unexpected response shape from Floe at {url}.")
    # "synced" is the count the server accepted (new rows); "duplicates" (already
    # ingested, idempotent) are not counted here. Absent → 0.
    synced = payload.get("synced", 0)
    try:
        return int(synced)
    except (ValueError, TypeError):
        return 0


def _describe_http_error(exc: urllib.error.HTTPError) -> str:
    detail = ""
    try:
        data = json.loads(exc.read())
        if isinstance(data, dict) and isinstance(data.get("error"), str):
            detail = f" ({data['error']})"
    except Exception:
        pass
    if exc.code == 401:
        return f"Floe rejected the API key (401 unauthorized){detail}."
    if exc.code == 403:
        return (
            f"Floe refused the sync (403 forbidden){detail} — the agent may be "
            "closed/suspended, or the key is read-only (a read_write key is required)."
        )
    return f"Floe returned HTTP {exc.code}{detail}."
