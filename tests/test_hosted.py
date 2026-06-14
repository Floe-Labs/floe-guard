"""The hosted upgrade hook is a documented no-op until FLOE_API_KEY is set."""

from __future__ import annotations

import pytest

from floe_guard.hosted import hosted_enforcement_available


def test_no_key_means_no_hosted_enforcement(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FLOE_API_KEY", raising=False)
    assert hosted_enforcement_available() is False


def test_key_present_signals_hosted_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLOE_API_KEY", "floe_test_key")
    assert hosted_enforcement_available() is True
