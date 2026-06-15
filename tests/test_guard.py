"""Behavioural tests for the core BudgetGuard kill-switch."""

from __future__ import annotations

import pytest

from floe_guard import (
    BudgetExceeded,
    BudgetGuard,
    ManualPrice,
    UnpriceableModelError,
    UnpriceableModelWarning,
)

MODEL = "gpt-4o"  # 1k in + 1k out = $0.0025 + $0.01 = $0.0125/call


def _run_loop(guard: BudgetGuard, max_calls: int = 1000) -> int:
    """Drive a runaway loop through the guard. Returns the number of LLM calls made."""
    calls_made = 0
    for _ in range(max_calls):
        try:
            guard.check()
        except BudgetExceeded:
            return calls_made
        # The "LLM call" only runs because check() did not block.
        guard.record(MODEL, 1_000, 1_000)
        calls_made += 1
    raise AssertionError("guard never blocked the loop")


def test_guard_hard_stops_before_crossing_call() -> None:
    # $0.05 ceiling, $0.0125/call. Calls 1-4 cost $0.05 total (== ceiling);
    # call 5 would cross, so the guard must block BEFORE it runs.
    guard = BudgetGuard(limit_usd=0.05)
    calls_made = _run_loop(guard)
    assert calls_made == 4
    assert guard.spent_usd <= guard.limit_usd  # never overshot the ceiling


def test_guard_blocks_before_overshoot_when_calls_dont_divide_evenly() -> None:
    # $0.10 ceiling, $0.0125/call -> 8 even calls = $0.10. The predictive check
    # blocks the 9th before it crosses; spend stays at or under the ceiling.
    guard = BudgetGuard(limit_usd=0.10)
    calls_made = _run_loop(guard)
    assert guard.spent_usd <= guard.limit_usd
    # Once blocked, no further calls slip through.
    with pytest.raises(BudgetExceeded):
        guard.check()
    assert calls_made == 8


def test_spend_tally_is_accurate() -> None:
    guard = BudgetGuard(limit_usd=1.00)
    cost1 = guard.record(MODEL, 1_000, 1_000)
    cost2 = guard.record(MODEL, 2_000, 500)
    assert cost1 == pytest.approx(0.0125)
    assert cost2 == pytest.approx(2_000 * 2.5e-6 + 500 * 1e-5)  # 0.005 + 0.005
    assert guard.spent_usd == pytest.approx(cost1 + cost2)


def test_sub_ceiling_run_completes_normally() -> None:
    # A short run that never approaches the ceiling must not raise.
    guard = BudgetGuard(limit_usd=10.00)
    for _ in range(5):
        guard.check()
        guard.record(MODEL, 1_000, 1_000)
    assert guard.spent_usd == pytest.approx(5 * 0.0125)
    guard.check()  # still room — does not raise


def test_unpriceable_model_warns_and_fails_closed_by_default() -> None:
    guard = BudgetGuard(limit_usd=1.00)
    with pytest.warns(UnpriceableModelWarning):
        with pytest.raises(UnpriceableModelError):
            guard.record("totally-made-up-model-x", 100, 100)
    # It did NOT silently accrue $0 and keep going.
    assert guard.spent_usd == 0.0


def test_unpriceable_model_warns_and_skips_when_not_fail_closed() -> None:
    guard = BudgetGuard(limit_usd=1.00, fail_closed=False)
    with pytest.warns(UnpriceableModelWarning):
        cost = guard.record("totally-made-up-model-x", 100, 100)
    assert cost == 0.0  # could not price it, accrued nothing (user opted in)


def test_manual_price_override_makes_model_enforceable() -> None:
    guard = BudgetGuard(
        limit_usd=1.00,
        price_overrides={"my-local-model": ManualPrice(1e-6, 2e-6)},
    )
    cost = guard.record("my-local-model", 1_000, 1_000)
    assert cost == pytest.approx(1_000 * 1e-6 + 1_000 * 2e-6)


def test_per_call_price_override() -> None:
    guard = BudgetGuard(limit_usd=1.00)
    cost = guard.record("ghost-model", 1_000, 0, price=ManualPrice(3e-6, 0.0))
    assert cost == pytest.approx(0.003)


def test_zero_budget_blocks_first_call() -> None:
    guard = BudgetGuard(limit_usd=0.0)
    with pytest.raises(BudgetExceeded):
        guard.check()


def test_negative_limit_rejected() -> None:
    with pytest.raises(ValueError):
        BudgetGuard(limit_usd=-1.0)


def test_on_block_callback_invoked() -> None:
    seen: list[tuple[float, float]] = []
    guard = BudgetGuard(limit_usd=0.0, on_block=lambda s, lim: seen.append((s, lim)))
    with pytest.raises(BudgetExceeded):
        guard.check()
    assert seen == [(0.0, 0.0)]


def test_block_message_printed(capsys: pytest.CaptureFixture[str]) -> None:
    guard = BudgetGuard(limit_usd=0.0)
    with pytest.raises(BudgetExceeded):
        guard.check()
    err = capsys.readouterr().err
    assert "BUDGET EXCEEDED — call blocked" in err


def test_remaining_usd() -> None:
    guard = BudgetGuard(limit_usd=1.00)
    guard.record(MODEL, 1_000, 1_000)
    assert guard.remaining_usd == pytest.approx(1.00 - 0.0125)
