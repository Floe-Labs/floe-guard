"""``BudgetGuard.from_floe`` sets the local ceiling from hosted Floe headroom.

The one-line upgrade: reads server-side headroom once and enforces it locally.
These tests never hit the network — ``urllib.request.urlopen`` is mocked, so they
exercise the real ``hosted_remaining_usd`` read end to end.
"""

from __future__ import annotations

import io
import json
import urllib.error
from unittest import mock

import pytest

from floe_guard import BudgetGuard
from floe_guard.errors import HostedEnforcementError


def _ok_response(payload: dict[str, object]) -> mock.MagicMock:
    body = json.dumps(payload).encode()
    resp = mock.MagicMock()
    resp.read.return_value = body
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


def _http_error(code: int, payload: dict[str, object] | None = None) -> urllib.error.HTTPError:
    body = json.dumps(payload).encode() if payload is not None else b""
    return urllib.error.HTTPError(
        url="https://credit-api.floelabs.xyz/v1/agents/credit-remaining",
        code=code,
        msg="error",
        hdrs=None,  # type: ignore[arg-type]
        fp=io.BytesIO(body),
    )


# ── ceiling reflects server headroom ──────────────────────────────────────────


def test_ceiling_is_server_headroom() -> None:
    payload = {"headroomToAutoBorrow": "5000000", "sessionSpendRemaining": None}
    with mock.patch("urllib.request.urlopen", return_value=_ok_response(payload)):
        guard = BudgetGuard.from_floe(api_key="floe_abc")
    assert guard.limit_usd == 5.0
    # A working guard: spend accrues and the ceiling holds.
    guard.check()
    assert guard.remaining_usd == 5.0


def test_ceiling_takes_min_of_headroom_and_session() -> None:
    payload = {"headroomToAutoBorrow": "5000000", "sessionSpendRemaining": "2000000"}
    with mock.patch("urllib.request.urlopen", return_value=_ok_response(payload)):
        guard = BudgetGuard.from_floe(api_key="floe_abc")
    assert guard.limit_usd == 2.0


def test_forwards_guard_options() -> None:
    payload = {"headroomToAutoBorrow": "5000000", "sessionSpendRemaining": None}
    with mock.patch("urllib.request.urlopen", return_value=_ok_response(payload)):
        guard = BudgetGuard.from_floe(api_key="floe_abc", near_limit_bps=5000)
    assert guard.near_limit_bps == 5000


# ── zero-telemetry + fail-closed ──────────────────────────────────────────────


def test_no_key_fails_closed_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    # Zero-telemetry invariant: with no key, from_floe raises BEFORE any network
    # call — nothing leaves the process.
    monkeypatch.delenv("FLOE_API_KEY", raising=False)
    with mock.patch("urllib.request.urlopen") as urlopen:
        with pytest.raises(HostedEnforcementError, match="No Floe API key"):
            BudgetGuard.from_floe()
    urlopen.assert_not_called()


@pytest.mark.parametrize("code", [401, 403, 404])
def test_invalid_key_fails_closed(code: int) -> None:
    with mock.patch("urllib.request.urlopen", side_effect=_http_error(code, {"error": "x"})):
        with pytest.raises(HostedEnforcementError, match=str(code)):
            BudgetGuard.from_floe(api_key="floe_bad")


def test_network_error_fails_closed() -> None:
    with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("down")):
        with pytest.raises(HostedEnforcementError, match="Could not reach"):
            BudgetGuard.from_floe(api_key="floe_abc")


# ── opt-in fallback degrades to local enforcement ─────────────────────────────


def test_fallback_used_on_failed_read_with_warning() -> None:
    with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("down")):
        with pytest.warns(UserWarning, match="falling back to a local ceiling"):
            guard = BudgetGuard.from_floe(api_key="floe_abc", fallback_limit_usd=1.50)
    assert guard.limit_usd == 1.50


def test_fallback_ignored_when_read_succeeds() -> None:
    payload = {"headroomToAutoBorrow": "5000000", "sessionSpendRemaining": None}
    with mock.patch("urllib.request.urlopen", return_value=_ok_response(payload)):
        guard = BudgetGuard.from_floe(api_key="floe_abc", fallback_limit_usd=1.50)
    assert guard.limit_usd == 5.0


# ── guard rails ───────────────────────────────────────────────────────────────


def test_limit_usd_in_kwargs_is_rejected() -> None:
    # limit_usd comes from headroom; passing one is a mistake, not a silent override.
    with pytest.raises(TypeError, match="fallback_limit_usd"):
        BudgetGuard.from_floe(api_key="floe_abc", limit_usd=10.0)
