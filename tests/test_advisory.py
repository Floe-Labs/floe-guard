"""Tests for the context-aware spend advisory (BudgetGuard.advisory)."""
from __future__ import annotations

import pytest

from floe_guard import BudgetAdvisory, BudgetGuard, ManualPrice


def test_fresh_guard_is_far_from_limit() -> None:
    a = BudgetGuard(limit_usd=1.00).advisory()
    assert isinstance(a, BudgetAdvisory)
    assert a.near_limit is False
    assert a.used_bps == 0
    assert a.remaining_usd == 1.00
    assert a.scope == "local"


def test_near_limit_flips_at_default_threshold() -> None:
    g = BudgetGuard(limit_usd=1.00)  # default near_limit_bps = 8000 (80%)
    g.spent_usd = 0.79
    assert g.advisory().near_limit is False
    g.spent_usd = 0.80
    a = g.advisory()
    assert a.near_limit is True
    assert a.used_bps == 8000
    assert a.remaining_usd == pytest.approx(0.20)


def test_custom_near_limit_threshold() -> None:
    g = BudgetGuard(limit_usd=1.00, near_limit_bps=5000)  # 50%
    g.spent_usd = 0.50
    assert g.advisory().near_limit is True


def test_used_bps_clamped_when_over_limit() -> None:
    g = BudgetGuard(limit_usd=1.00)
    g.spent_usd = 1.50  # overshoot
    a = g.advisory()
    assert a.used_bps == 10000  # clamped, not 15000
    assert a.remaining_usd == 0.0  # never negative


def test_zero_limit_reads_fully_used() -> None:
    a = BudgetGuard(limit_usd=0).advisory()
    assert a.used_bps == 10000
    assert a.near_limit is True


def test_used_bps_floors_not_rounds() -> None:
    # 79.999% used: floors to 7999 (not rounded up to 8000), so near_limit does
    # NOT flip before 80% is actually reached. Also keeps Python/JS parity.
    g = BudgetGuard(limit_usd=1.00, near_limit_bps=8000)
    g.spent_usd = 0.79999
    a = g.advisory()
    assert a.used_bps == 7999
    assert a.near_limit is False


def test_expected_cost_unknown_before_first_call() -> None:
    # No call recorded yet: estimate is 0.0 and calls-remaining is unknown (None),
    # not a divide-by-zero and not a misleading 0.
    a = BudgetGuard(limit_usd=1.00).advisory()
    assert a.expected_cost == 0.0
    assert a.est_calls_remaining is None


def test_est_calls_remaining_after_a_call() -> None:
    g = BudgetGuard(limit_usd=1.00)
    # Drive state through the public API (manual price for determinism), not by
    # poking private fields: this also proves record() feeds advisory().
    g.record("m", 1_000, 0, price=ManualPrice(1e-4, 0.0))  # $0.10 spent, $0.90 left
    a = g.advisory()
    assert a.expected_cost == pytest.approx(0.10)
    assert a.est_calls_remaining == 9  # floor(0.90 / 0.10)


def test_est_calls_remaining_uses_costlier_of_llm_and_tool() -> None:
    g = BudgetGuard(limit_usd=1.00)
    g.record("m", 1_000, 0, price=ManualPrice(2e-4, 0.0))  # $0.20 LLM (costlier)
    g.record_tool("db.query", 0.05)                        # $0.05 tool (cheaper)
    a = g.advisory()                                       # spent 0.25, remaining 0.75
    assert a.expected_cost == pytest.approx(0.20)          # max(llm, tool) wins
    assert a.est_calls_remaining == 3  # floor(0.75 / 0.20)


def test_invalid_near_limit_bps_rejected() -> None:
    with pytest.raises(ValueError):
        BudgetGuard(limit_usd=1.00, near_limit_bps=-1)
    with pytest.raises(ValueError):
        BudgetGuard(limit_usd=1.00, near_limit_bps=10001)


def test_non_int_near_limit_bps_rejected() -> None:
    # Floats and bools (bool is an int subclass) are rejected, matching JS.
    with pytest.raises(ValueError):
        BudgetGuard(limit_usd=1.00, near_limit_bps=8000.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        BudgetGuard(limit_usd=1.00, near_limit_bps=True)  # type: ignore[arg-type]
