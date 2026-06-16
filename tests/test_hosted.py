"""The hosted hook reads server-side remaining budget from the live endpoint.

These tests never hit the network: ``urllib.request.urlopen`` is mocked.
"""

from __future__ import annotations

import io
import json
import urllib.error
from unittest import mock

import pytest

from floe_guard.errors import HostedEnforcementError
from floe_guard.hosted import (
    hosted_enforcement_available,
    hosted_remaining_usd,
)


def _ok_response(payload: dict[str, object]) -> mock.MagicMock:
    """A context-manager mock that returns ``payload`` as JSON bytes on read()."""
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


# ── availability ────────────────────────────────────────────────────────────


def test_no_key_means_no_hosted_enforcement(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FLOE_API_KEY", raising=False)
    assert hosted_enforcement_available() is False


def test_key_present_signals_hosted_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLOE_API_KEY", "floe_test_key")
    assert hosted_enforcement_available() is True


# ── happy path ──────────────────────────────────────────────────────────────


def test_remaining_takes_min_when_session_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLOE_API_KEY", "floe_test_key")
    payload = {"headroomToAutoBorrow": "5000000", "sessionSpendRemaining": "2000000"}
    with mock.patch("urllib.request.urlopen", return_value=_ok_response(payload)):
        assert hosted_remaining_usd() == 2.0


def test_remaining_uses_headroom_when_session_null(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLOE_API_KEY", "floe_test_key")
    payload = {"headroomToAutoBorrow": "5000000", "sessionSpendRemaining": None}
    with mock.patch("urllib.request.urlopen", return_value=_ok_response(payload)):
        assert hosted_remaining_usd() == 5.0


def test_remaining_uses_headroom_when_session_smaller(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLOE_API_KEY", "floe_test_key")
    payload = {"headroomToAutoBorrow": "1500000", "sessionSpendRemaining": "9000000"}
    with mock.patch("urllib.request.urlopen", return_value=_ok_response(payload)):
        assert hosted_remaining_usd() == 1.5


def test_explicit_api_key_sends_bearer_header() -> None:
    payload = {"headroomToAutoBorrow": "1000000", "sessionSpendRemaining": None}
    with mock.patch(
        "urllib.request.urlopen", return_value=_ok_response(payload)
    ) as urlopen:
        hosted_remaining_usd(api_key="floe_abc")
    request = urlopen.call_args.args[0]
    assert request.get_header("Authorization") == "Bearer floe_abc"
    assert request.full_url.endswith("/v1/agents/credit-remaining")


def test_base_url_override_is_used() -> None:
    payload = {"headroomToAutoBorrow": "1000000", "sessionSpendRemaining": None}
    with mock.patch(
        "urllib.request.urlopen", return_value=_ok_response(payload)
    ) as urlopen:
        hosted_remaining_usd(api_key="floe_abc", base_url="https://staging.example.com/")
    request = urlopen.call_args.args[0]
    assert request.full_url == "https://staging.example.com/v1/agents/credit-remaining"


@pytest.mark.parametrize(
    "bad_base",
    [
        "http://credit-api.floelabs.xyz",  # non-https — would leak the bearer token
        "file:///etc/passwd",
        "ftp://example.com",
        "not-a-url",
        "https://",  # scheme but no host
    ],
)
def test_unsafe_base_url_refuses_to_send_key(bad_base: str) -> None:
    # The agent key is sent as a bearer token; a non-https/malformed base URL must
    # be rejected BEFORE any request is made.
    with mock.patch("urllib.request.urlopen") as urlopen:
        with pytest.raises(HostedEnforcementError, match="https"):
            hosted_remaining_usd(api_key="floe_abc", base_url=bad_base)
    urlopen.assert_not_called()


def test_whitespace_base_url_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FLOE_API_BASE_URL", raising=False)
    payload = {"headroomToAutoBorrow": "1000000", "sessionSpendRemaining": None}
    with mock.patch(
        "urllib.request.urlopen", return_value=_ok_response(payload)
    ) as urlopen:
        hosted_remaining_usd(api_key="floe_abc", base_url="   ")
    request = urlopen.call_args.args[0]
    assert request.full_url == "https://credit-api.floelabs.xyz/v1/agents/credit-remaining"


# ── errors ──────────────────────────────────────────────────────────────────


def test_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FLOE_API_KEY", raising=False)
    with pytest.raises(HostedEnforcementError, match="No Floe API key"):
        hosted_remaining_usd()


@pytest.mark.parametrize(
    ("code", "needle"),
    [
        (401, "401"),
        (403, "403"),
        (404, "404"),
    ],
)
def test_http_status_raises(code: int, needle: str) -> None:
    with mock.patch("urllib.request.urlopen", side_effect=_http_error(code, {"error": "x"})):
        with pytest.raises(HostedEnforcementError, match=needle):
            hosted_remaining_usd(api_key="floe_abc")


def test_404_surfaces_server_error_field() -> None:
    err = _http_error(404, {"error": "no_credit_limit"})
    with mock.patch("urllib.request.urlopen", side_effect=err):
        with pytest.raises(HostedEnforcementError, match="no_credit_limit"):
            hosted_remaining_usd(api_key="floe_abc")


def test_timeout_raises() -> None:
    with mock.patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
        with pytest.raises(HostedEnforcementError, match="Could not reach"):
            hosted_remaining_usd(api_key="floe_abc")


def test_network_error_raises() -> None:
    with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("down")):
        with pytest.raises(HostedEnforcementError, match="Could not reach"):
            hosted_remaining_usd(api_key="floe_abc")


def test_malformed_json_raises() -> None:
    resp = mock.MagicMock()
    resp.read.return_value = b"not json{"
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    with mock.patch("urllib.request.urlopen", return_value=resp):
        with pytest.raises(HostedEnforcementError, match="Malformed JSON"):
            hosted_remaining_usd(api_key="floe_abc")
