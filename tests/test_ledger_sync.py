"""Opt-in ledger sync — zero-telemetry stays the default.

The load-bearing test here is that a guard which never opted in makes **zero**
network calls, ever (``urllib.request.urlopen`` is asserted un-called). Sync sends
only the ``export_log()`` JSONL, only under the explicit flag, only with a key.
"""

from __future__ import annotations

import io
import json
import urllib.error
from unittest import mock

import pytest

from floe_guard import BudgetGuard, LedgerSyncError, push_ledger

_LEDGER_KEYS = {
    "timestamp",
    "kind",
    "model_or_tool",
    "prompt_tokens",
    "completion_tokens",
    "cost_usd",
    "label",
    "reserved",
}


def _ok(payload: dict[str, object]) -> mock.MagicMock:
    resp = mock.MagicMock()
    resp.read.return_value = json.dumps(payload).encode()
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


def _guard_with_spend() -> BudgetGuard:
    guard = BudgetGuard(limit_usd=10.0)
    guard.record_tool("api", 0.05)  # exactly one ledger event
    return guard


# ── AC1: zero egress without opt-in ───────────────────────────────────────────


def test_no_network_without_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    # A full guard lifecycle with NO enable_sync must never touch the network,
    # and sync() must raise (not silently send) — the zero-telemetry default.
    monkeypatch.delenv("FLOE_API_KEY", raising=False)
    with mock.patch("urllib.request.urlopen") as urlopen:
        guard = BudgetGuard(limit_usd=10.0)
        guard.record_tool("api", 0.05)
        guard.record("gpt-4o", 1200, 350)
        _ = guard.advisory()
        _ = guard.export_log()
        with pytest.raises(RuntimeError, match="not enabled"):
            guard.sync()
    urlopen.assert_not_called()


def test_disable_sync_revokes_no_network() -> None:
    guard = _guard_with_spend()
    guard.enable_sync(api_key="floe_abc")
    guard.disable_sync()
    with mock.patch("urllib.request.urlopen") as urlopen:
        with pytest.raises(RuntimeError, match="not enabled"):
            guard.sync()
    urlopen.assert_not_called()


def test_empty_ledger_syncs_nothing_no_network() -> None:
    guard = BudgetGuard(limit_usd=10.0)  # no spend recorded
    guard.enable_sync(api_key="floe_abc")
    with mock.patch("urllib.request.urlopen") as urlopen:
        assert guard.sync() == 0
    urlopen.assert_not_called()


# ── AC2: opt-in send is exactly export_log(), under the key ───────────────────


def test_sync_posts_export_log_under_key() -> None:
    guard = _guard_with_spend()
    guard.enable_sync(api_key="floe_abc")
    with mock.patch("urllib.request.urlopen", return_value=_ok({"synced": 1})) as urlopen:
        assert guard.sync() == 1
    request = urlopen.call_args.args[0]
    assert request.get_header("Authorization") == "Bearer floe_abc"
    assert request.method == "POST"
    assert request.full_url.endswith("/v1/agents/ledger/sync")
    assert request.get_header("Content-type") == "application/x-ndjson"
    # The body is EXACTLY export_log() — no prompts, no content, no extra fields.
    assert request.data.decode("utf-8") == guard.export_log()
    for line in guard.export_log().splitlines():
        assert set(json.loads(line)) <= _LEDGER_KEYS


def test_sync_requires_enable_first_no_network() -> None:
    guard = _guard_with_spend()
    with mock.patch("urllib.request.urlopen") as urlopen:
        with pytest.raises(RuntimeError, match="enable_sync"):
            guard.sync()
    urlopen.assert_not_called()


# ── push_ledger fail-closed ───────────────────────────────────────────────────


def test_push_ledger_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FLOE_API_KEY", raising=False)
    with mock.patch("urllib.request.urlopen") as urlopen:
        with pytest.raises(LedgerSyncError, match="No Floe API key"):
            push_ledger('{"cost_usd":0.01}\n')
    urlopen.assert_not_called()


def test_push_ledger_empty_is_noop_no_network() -> None:
    with mock.patch("urllib.request.urlopen") as urlopen:
        assert push_ledger("   ") == 0
    urlopen.assert_not_called()


@pytest.mark.parametrize(
    "bad_base",
    ["http://credit-api.floelabs.xyz", "file:///etc/passwd", "ftp://x", "not-a-url", "https://"],
)
def test_push_ledger_refuses_non_https(bad_base: str) -> None:
    # The request carries the key AND the ledger — never over a non-https/bad URL.
    with mock.patch("urllib.request.urlopen") as urlopen:
        with pytest.raises(LedgerSyncError, match="https"):
            push_ledger('{"cost_usd":0.01}\n', api_key="floe_abc", base_url=bad_base)
    urlopen.assert_not_called()


@pytest.mark.parametrize(("code", "needle"), [(401, "401"), (403, "403")])
def test_push_ledger_http_error_raises(code: int, needle: str) -> None:
    err = urllib.error.HTTPError(
        "https://credit-api.floelabs.xyz/v1/agents/ledger/sync",
        code,
        "err",
        None,  # type: ignore[arg-type]
        io.BytesIO(b'{"error":"x"}'),
    )
    with mock.patch("urllib.request.urlopen", side_effect=err):
        with pytest.raises(LedgerSyncError, match=needle):
            push_ledger('{"cost_usd":0.01}\n', api_key="floe_abc")


# ── CLI ───────────────────────────────────────────────────────────────────────


def test_cli_push_reads_file_and_posts(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from floe_guard.__main__ import main

    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        '{"timestamp":1.0,"kind":"tool","model_or_tool":"api",'
        '"prompt_tokens":null,"completion_tokens":null,"cost_usd":0.05}\n'
    )
    with mock.patch("urllib.request.urlopen", return_value=_ok({"synced": 1})) as urlopen:
        rc = main(["push", str(ledger), "--key", "floe_abc"])
    assert rc == 0
    assert urlopen.call_args.args[0].get_header("Authorization") == "Bearer floe_abc"


def test_cli_push_no_key_returns_1_no_network(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    from floe_guard.__main__ import main

    monkeypatch.delenv("FLOE_API_KEY", raising=False)
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text('{"timestamp":1.0,"kind":"tool","model_or_tool":"api","cost_usd":0.05}\n')
    with mock.patch("urllib.request.urlopen") as urlopen:
        rc = main(["push", str(ledger)])  # non-empty ledger, no key → LedgerSyncError → rc 1
    assert rc == 1
    urlopen.assert_not_called()
