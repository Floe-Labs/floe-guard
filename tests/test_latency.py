"""LatencyBudget — cumulative tool-chain deadline (FLO-624).

Uses an injectable fake monotonic clock so no test ever sleeps.
"""

from __future__ import annotations

import pytest

from floe_guard import DeadlineExceeded, LatencyAdvisory, LatencyBudget


class FakeClock:
    """Monotonic-clock stand-in: seconds, advanced manually."""

    def __init__(self) -> None:
        self.now = 100.0  # arbitrary origin — only deltas matter

    def __call__(self) -> float:
        return self.now

    def advance_ms(self, ms: float) -> None:
        self.now += ms / 1000.0


def make(sla_ms: float = 5000, **kwargs):
    clock = FakeClock()
    budget = LatencyBudget(sla_ms, clock=clock, **kwargs)
    return budget, clock


def test_check_passes_with_headroom_and_blocks_when_projected_over() -> None:
    budget, clock = make(sla_ms=5000)
    clock.advance_ms(3000)
    budget.check(expected_ms=1000)  # 3000 + 1000 <= 5000 — fine

    with pytest.raises(DeadlineExceeded) as exc:
        budget.check(expected_ms=2500)  # 3000 + 2500 > 5000 — shed
    assert exc.value.sla_ms == 5000
    assert exc.value.elapsed_ms == pytest.approx(3000)


def test_check_without_estimate_gates_on_elapsed_only() -> None:
    budget, clock = make(sla_ms=1000)
    clock.advance_ms(999)
    budget.check()  # still inside
    clock.advance_ms(2)
    with pytest.raises(DeadlineExceeded):
        budget.check()


def test_remaining_ms_is_readable_mid_chain_and_floors_at_zero() -> None:
    budget, clock = make(sla_ms=5000)
    clock.advance_ms(1500)
    assert budget.remaining_ms == pytest.approx(3500)
    clock.advance_ms(9000)
    assert budget.remaining_ms == 0.0  # floored, never negative


def test_advisory_is_symmetric_to_budget_advisory() -> None:
    budget, clock = make(sla_ms=5000)  # default near_deadline_bps = 8000
    clock.advance_ms(2500)
    mid = budget.advisory()
    assert isinstance(mid, LatencyAdvisory)
    assert mid.used_bps == 5000
    assert mid.near_deadline is False
    assert mid.remaining_ms == pytest.approx(2500)

    clock.advance_ms(1600)  # 4100/5000 = 82%
    late = budget.advisory()
    assert late.used_bps == 8200
    assert late.near_deadline is True

    clock.advance_ms(9000)  # way past — used_bps caps at 10000
    assert budget.advisory().used_bps == 10000


def test_on_block_fires_before_the_raise() -> None:
    calls: list[tuple[float, float]] = []
    clock = FakeClock()
    budget = LatencyBudget(1000, clock=clock, on_block=lambda e, s: calls.append((e, s)))
    clock.advance_ms(1500)
    with pytest.raises(DeadlineExceeded):
        budget.check()
    assert len(calls) == 1
    assert calls[0][1] == 1000


def test_constructor_and_check_validate_inputs() -> None:
    with pytest.raises(ValueError):
        LatencyBudget(0)
    with pytest.raises(ValueError):
        LatencyBudget(5000, near_deadline_bps=20000)
    budget, _ = make()
    with pytest.raises(ValueError):
        budget.check(expected_ms=-1)
