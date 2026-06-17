"""The ceiling must hold when calls run in parallel (regression for issue #18).

Sequentially, check()/record() stops a runaway loop. But a CrewAI crew fans
calls out in parallel (async tasks, kickoff_for_each_async, hierarchical tool
calls), and check()/record() are not atomic — several callers can read the same
under-limit total, all clear the gate, then all run. reserve()/settle() hold the
estimate in-flight under a lock, so the ceiling holds under fan-out.
"""

from __future__ import annotations

import threading
import time

import pytest

from floe_guard import (
    BudgetExceeded,
    BudgetGuard,
    UnpriceableModelError,
    UnpriceableModelWarning,
)

MODEL = "gpt-4o"  # 1k in + 1k out = $0.0125 / call


def test_reserve_settle_holds_ceiling_under_parallel_calls() -> None:
    guard = BudgetGuard(limit_usd=0.10, on_block=lambda *_: None)
    # Warm one call so the next-call estimate is realistic (~$0.0125).
    guard.record(MODEL, 1_000, 1_000)

    blocked: list[int] = []

    def agent(i: int) -> None:
        try:
            reserved = guard.reserve()
        except BudgetExceeded:
            blocked.append(i)
            return
        time.sleep(0.02)  # API latency — the window the old race exploited
        guard.settle(MODEL, 1_000, 1_000, reserved=reserved)

    threads = [threading.Thread(target=agent, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # The guarantee: 16 concurrent agents never push spend past the ceiling...
    assert guard.spent_usd <= guard.limit_usd + 1e-9
    # ...and the excess was actually stopped, not silently allowed.
    assert blocked
    # No reservations leaked.
    assert guard.remaining_usd >= 0.0


def test_legacy_check_record_path_is_unchanged() -> None:
    # The sequential API behaves exactly as before: $0.05 / $0.0125 -> 4 calls.
    guard = BudgetGuard(limit_usd=0.05, on_block=lambda *_: None)
    calls = 0
    for _ in range(1000):
        try:
            guard.check()
        except BudgetExceeded:
            break
        guard.record(MODEL, 1_000, 1_000)
        calls += 1
    assert calls == 4
    assert guard.spent_usd <= guard.limit_usd


def test_release_frees_inflight_budget() -> None:
    guard = BudgetGuard(limit_usd=0.10, on_block=lambda *_: None)
    guard.record(MODEL, 1_000, 1_000)
    reserved = guard.reserve()
    before = guard.remaining_usd
    guard.release(reserved)  # call failed, give the budget back
    assert guard.remaining_usd >= before


def test_unpriceable_fail_closed_releases_the_reservation() -> None:
    # Regression for the #19 review: settle() on an unpriceable model under
    # fail_closed must release the in-flight reservation before it raises, or
    # _reserved leaks and remaining_usd shrinks permanently until reserve() blocks.
    guard = BudgetGuard(limit_usd=0.10, on_block=lambda *_: None)
    guard.record(MODEL, 1_000, 1_000)
    base = guard.remaining_usd
    reserved = guard.reserve()
    assert guard.remaining_usd < base  # hold is in flight
    with pytest.warns(UnpriceableModelWarning):
        with pytest.raises(UnpriceableModelError):
            guard.settle("totally-made-up-model-x", 100, 100, reserved=reserved)
    assert guard.remaining_usd == base  # released, not leaked
