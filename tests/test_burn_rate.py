"""advisory().burn_rate_usd_per_min — spend ÷ minutes since the guard was created.

The window start (`_created_time`) is set directly and `time.time` is patched only
for the advisory read, so each case asserts the rate from a known spend/elapsed pair.
"""

from __future__ import annotations

import pytest

from floe_guard import BudgetGuard


def test_burn_rate_from_known_spend_and_elapsed(monkeypatch: pytest.MonkeyPatch) -> None:
    guard = BudgetGuard(limit_usd=1.00)
    guard._created_time = 100.0
    guard.record_tool("api", 0.05)  # exact $0.05 spend
    monkeypatch.setattr("floe_guard.guard.time.time", lambda: 130.0)  # +30s = 0.5 min
    assert guard.advisory().burn_rate_usd_per_min == pytest.approx(0.10)  # 0.05 / 0.5


def test_burn_rate_zero_when_nothing_spent(monkeypatch: pytest.MonkeyPatch) -> None:
    guard = BudgetGuard(limit_usd=1.00)
    guard._created_time = 100.0
    monkeypatch.setattr("floe_guard.guard.time.time", lambda: 160.0)  # +60s
    # $0 over real elapsed time is a legitimate 0.0/min, not "unknown".
    assert guard.advisory().burn_rate_usd_per_min == 0.0


def test_burn_rate_none_before_any_time_elapses(monkeypatch: pytest.MonkeyPatch) -> None:
    guard = BudgetGuard(limit_usd=1.00)
    guard._created_time = 100.0
    guard.record_tool("api", 0.05)
    monkeypatch.setattr("floe_guard.guard.time.time", lambda: 100.0)  # no elapsed
    assert guard.advisory().burn_rate_usd_per_min is None
