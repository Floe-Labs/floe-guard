"""Opt-in ledger sync — zero-telemetry stays the default.

The load-bearing test here is that a guard which never opted in makes **zero**
network calls, ever (``floe_guard.sync._OPENER.open`` is asserted un-called). Sync sends
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

# One schema-valid ledger line, for tests isolating the key/https/network paths.
_ONE_EVENT = '{"timestamp":1.0,"kind":"tool","model_or_tool":"api","cost_usd":0.01}\n'


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
    with mock.patch("floe_guard.sync._OPENER.open") as urlopen:
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
    with mock.patch("floe_guard.sync._OPENER.open") as urlopen:
        with pytest.raises(RuntimeError, match="not enabled"):
            guard.sync()
    urlopen.assert_not_called()


def test_empty_ledger_syncs_nothing_no_network() -> None:
    guard = BudgetGuard(limit_usd=10.0)  # no spend recorded
    guard.enable_sync(api_key="floe_abc")
    with mock.patch("floe_guard.sync._OPENER.open") as urlopen:
        assert guard.sync() == 0
    urlopen.assert_not_called()


# ── AC2: opt-in send is exactly export_log(), under the key ───────────────────


def test_sync_posts_export_log_under_key() -> None:
    guard = _guard_with_spend()
    guard.enable_sync(api_key="floe_abc")
    with mock.patch("floe_guard.sync._OPENER.open", return_value=_ok({"synced": 1})) as urlopen:
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
    with mock.patch("floe_guard.sync._OPENER.open") as urlopen:
        with pytest.raises(RuntimeError, match="enable_sync"):
            guard.sync()
    urlopen.assert_not_called()


# ── push_ledger fail-closed ───────────────────────────────────────────────────


def test_push_ledger_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FLOE_API_KEY", raising=False)
    with mock.patch("floe_guard.sync._OPENER.open") as urlopen:
        with pytest.raises(LedgerSyncError, match="No Floe API key"):
            push_ledger(_ONE_EVENT)
    urlopen.assert_not_called()


def test_push_ledger_missing_key_points_to_key_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The missing-key error is the OSS→hosted handoff — it must say where keys come from.
    monkeypatch.delenv("FLOE_API_KEY", raising=False)
    with pytest.raises(LedgerSyncError, match=r"dev-dashboard\.floelabs\.xyz/keys"):
        push_ledger(_ONE_EVENT)


def test_push_ledger_empty_is_noop_no_network() -> None:
    with mock.patch("floe_guard.sync._OPENER.open") as urlopen:
        assert push_ledger("   ") == 0
    urlopen.assert_not_called()


@pytest.mark.parametrize(
    "bad_base",
    ["http://credit-api.floelabs.xyz", "file:///etc/passwd", "ftp://x", "not-a-url", "https://"],
)
def test_push_ledger_refuses_non_https(bad_base: str) -> None:
    # The request carries the key AND the ledger — never over a non-https/bad URL.
    with mock.patch("floe_guard.sync._OPENER.open") as urlopen:
        with pytest.raises(LedgerSyncError, match="https"):
            push_ledger(_ONE_EVENT, api_key="floe_abc", base_url=bad_base)
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
    with mock.patch("floe_guard.sync._OPENER.open", side_effect=err):
        with pytest.raises(LedgerSyncError, match=needle):
            push_ledger(_ONE_EVENT, api_key="floe_abc")


# ── CLI ───────────────────────────────────────────────────────────────────────


def test_cli_push_reads_file_and_posts(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from floe_guard.__main__ import main

    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        '{"timestamp":1.0,"kind":"tool","model_or_tool":"api",'
        '"prompt_tokens":null,"completion_tokens":null,"cost_usd":0.05}\n'
    )
    with mock.patch("floe_guard.sync._OPENER.open", return_value=_ok({"synced": 1})) as urlopen:
        rc = main(["push", str(ledger), "--key", "floe_abc"])
    assert rc == 0
    assert urlopen.call_args.args[0].get_header("Authorization") == "Bearer floe_abc"


def test_cli_push_no_key_returns_1_no_network(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    from floe_guard.__main__ import main

    monkeypatch.delenv("FLOE_API_KEY", raising=False)
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text('{"timestamp":1.0,"kind":"tool","model_or_tool":"api","cost_usd":0.05}\n')
    with mock.patch("floe_guard.sync._OPENER.open") as urlopen:
        rc = main(["push", str(ledger)])  # non-empty ledger, no key → LedgerSyncError → rc 1
    assert rc == 1
    urlopen.assert_not_called()


def test_cli_push_success_points_to_dashboard(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    # The first-sync → hosted handoff: a successful push must point the user at
    # the dashboard, not just "Synced N events."
    from floe_guard.__main__ import main

    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        '{"timestamp":1.0,"kind":"tool","model_or_tool":"api",'
        '"prompt_tokens":null,"completion_tokens":null,"cost_usd":0.05}\n'
    )
    with mock.patch("floe_guard.sync._OPENER.open", return_value=_ok({"synced": 3})):
        rc = main(["push", str(ledger), "--key", "floe_abc"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Synced 3" in out
    assert "dev-dashboard.floelabs.xyz" in out


def test_cli_push_all_duplicates_still_points_to_dashboard(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    # synced=0 on a NON-empty ledger means every event was already ingested —
    # hosted coverage exists, so the pointer still applies. (An idempotent
    # re-push must not dead-end the user who already synced successfully.)
    from floe_guard.__main__ import main

    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        '{"timestamp":1.0,"kind":"tool","model_or_tool":"api",'
        '"prompt_tokens":null,"completion_tokens":null,"cost_usd":0.05}\n'
    )
    with mock.patch("floe_guard.sync._OPENER.open", return_value=_ok({"synced": 0})):
        rc = main(["push", str(ledger), "--key", "floe_abc"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "all duplicates" in out
    assert "dev-dashboard.floelabs.xyz" in out


def test_cli_push_empty_ledger_no_dashboard_pointer(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    # A truly empty ledger never touches the network — nothing was ever pushed,
    # so there is no hosted coverage to point at.
    from floe_guard.__main__ import main

    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("")
    with mock.patch("floe_guard.sync._OPENER.open") as urlopen:
        rc = main(["push", str(ledger), "--key", "floe_abc"])
    assert rc == 0
    urlopen.assert_not_called()
    out = capsys.readouterr().out
    assert "dev-dashboard" not in out


# ── privacy contract enforced: validate every record before upload ────────────


def test_push_ledger_rejects_smuggled_field_no_network() -> None:
    # A non-schema field (e.g. a prompt) is refused BEFORE any network — the
    # privacy contract is enforced, not just documented.
    bad = (
        '{"timestamp":1.0,"kind":"tool","model_or_tool":"api","cost_usd":0.01,"prompt":"secret"}\n'
    )
    with mock.patch("floe_guard.sync._OPENER.open") as opener:
        with pytest.raises(LedgerSyncError, match="outside the export_log"):
            push_ledger(bad, api_key="floe_abc")
    opener.assert_not_called()


@pytest.mark.parametrize(
    "bad_line",
    [
        '{"kind":"tool","model_or_tool":"api","cost_usd":0.01}',  # missing timestamp
        '{"timestamp":1.0,"kind":"bogus","model_or_tool":"api","cost_usd":0.01}',  # bad kind
        '{"timestamp":1.0,"kind":"tool","model_or_tool":"api","cost_usd":-1}',  # negative cost
        '{"timestamp":1.0,"kind":"tool","model_or_tool":"api","cost_usd":"x"}',  # non-numeric cost
        '{"timestamp":1.0,"kind":"tool","model_or_tool":123,"cost_usd":0.01}',  # non-str model
        "not json at all",  # malformed
    ],
)
def test_push_ledger_rejects_bad_schema_no_network(bad_line: str) -> None:
    with mock.patch("floe_guard.sync._OPENER.open") as opener:
        with pytest.raises(LedgerSyncError):
            push_ledger(bad_line + "\n", api_key="floe_abc")
    opener.assert_not_called()


def test_no_redirect_handler_refuses() -> None:
    # urllib would re-send the key + ledger across a redirect; the handler refuses.
    from floe_guard.sync import _NoRedirect

    with pytest.raises(LedgerSyncError, match="redirect"):
        _NoRedirect().redirect_request(None, None, 302, "Found", {}, "https://evil.example.com/x")


# ── response count validation ─────────────────────────────────────────────────


@pytest.mark.parametrize("bad_synced", [-1, True, "5", 1.9])
def test_invalid_synced_count_raises(bad_synced: object) -> None:
    guard = _guard_with_spend()
    guard.enable_sync(api_key="floe_abc")
    with mock.patch("floe_guard.sync._OPENER.open", return_value=_ok({"synced": bad_synced})):
        with pytest.raises(LedgerSyncError, match="Invalid 'synced'"):
            guard.sync()


def test_synced_absent_raises() -> None:
    # A conformant server ALWAYS includes "synced" on a 2xx; an absent count is a
    # malformed response, not "0 accepted" — reject it so the CLI cannot misread a
    # degenerate {} as "all duplicates" (see push CLI messaging).
    guard = _guard_with_spend()
    guard.enable_sync(api_key="floe_abc")
    with mock.patch("floe_guard.sync._OPENER.open", return_value=_ok({"duplicates": 1})):
        with pytest.raises(LedgerSyncError, match="missing 'synced'"):
            guard.sync()


# ── concurrency: sync uses the snapshotted key, not an env fallback ───────────


def test_sync_uses_snapshotted_key_not_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Even with a different key in the env, sync() sends the key it was enabled
    # with (snapshotted under the lock) — a concurrent disable can't cause an
    # env-fallback send.
    monkeypatch.setenv("FLOE_API_KEY", "floe_env_MUST_NOT_be_used")
    guard = _guard_with_spend()
    guard.enable_sync(api_key="floe_explicit")
    with mock.patch("floe_guard.sync._OPENER.open", return_value=_ok({"synced": 1})) as opener:
        guard.sync()
    assert opener.call_args.args[0].get_header("Authorization") == "Bearer floe_explicit"


def test_no_upload_starts_after_disable_returns() -> None:
    # The disable_sync() ordering guarantee: once it returns, no NEW upload starts.
    # An in-flight sync (started before disable) may finish; a sync after does not.
    import threading

    guard = _guard_with_spend()
    guard.enable_sync(api_key="floe_abc")
    started = threading.Event()
    release = threading.Event()
    calls = 0

    def blocking_open(request, timeout=None):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        started.set()
        release.wait(timeout=5)
        return _ok({"synced": 1})

    with mock.patch("floe_guard.sync._OPENER.open", side_effect=blocking_open):
        first = threading.Thread(target=guard.sync)
        first.start()
        assert started.wait(timeout=5)  # the first sync is now inside open()
        guard.disable_sync()  # returns — no new upload may start after this
        with pytest.raises(RuntimeError, match="not enabled"):
            guard.sync()  # a NEW sync post-disable raises, with no second open()
        assert calls == 1
        release.set()
        first.join(timeout=5)
    assert calls == 1  # still just the one in-flight call
