"""Tests for the context-aware spend advisory (BudgetGuard.advisory)."""
from __future__ import annotations

import pytest

from floe_guard import BudgetAdvisory, BudgetGuard


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


def test_invalid_near_limit_bps_rejected() -> None:
    with pytest.raises(ValueError):
        BudgetGuard(limit_usd=1.00, near_limit_bps=-1)
    with pytest.raises(ValueError):
        BudgetGuard(limit_usd=1.00, near_limit_bps=10001)
