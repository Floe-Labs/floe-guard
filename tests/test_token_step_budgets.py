"""Aggregate token ceilings and explicit per-step budget scopes (issue #46)."""

from __future__ import annotations

import threading

import pytest

from floe_guard import (
    BudgetExceeded,
    BudgetGuard,
    BudgetReservation,
    ManualPrice,
    TokenBudgetExceeded,
)
from floe_guard.integrations.openai import guarded_completion
from floe_guard.stream import StreamGuard

PRICE = ManualPrice(1e-6, 2e-6)


def quiet_guard(**kwargs) -> BudgetGuard:
    return BudgetGuard(
        limit_usd=10.0,
        on_block=lambda _spent, _limit: None,
        on_token_block=lambda _spent, _limit: None,
        **kwargs,
    )


@pytest.mark.parametrize("bad", [-1, 1.5, True, float("nan"), float("inf")])
def test_token_limit_rejects_invalid_values(bad: object) -> None:
    with pytest.raises(ValueError):
        quiet_guard(token_limit=bad)  # type: ignore[arg-type]


def test_legacy_guard_keeps_numeric_handle_and_behavior() -> None:
    guard = quiet_guard()
    handle = guard.reserve(0.1)
    assert isinstance(handle, float)
    guard.release(handle)
    assert guard.remaining_tokens is None
    assert guard.advisory().token_limit is None


def test_record_accumulates_all_token_buckets() -> None:
    guard = quiet_guard(token_limit=10_000)
    guard.record(
        "manual",
        100,
        20,
        price=PRICE,
        cache_creation_input_tokens=30,
        cache_creation_input_tokens_1h=40,
        cache_read_input_tokens=50,
    )
    assert guard.spent_tokens == 240
    assert guard.remaining_tokens == 9_760


def test_explicit_and_last_call_estimates_block_before_call() -> None:
    guard = quiet_guard(token_limit=100)
    with pytest.raises(TokenBudgetExceeded):
        guard.reserve(0.0, estimated_tokens=101)
    guard.record("manual", 40, 10, price=PRICE)
    assert guard.reserve(0.0).tokens == 50
    with pytest.raises(TokenBudgetExceeded):
        guard.reserve(0.0)


def test_aggregate_token_advisory_flips_overall_near_limit() -> None:
    guard = quiet_guard(token_limit=100, near_limit_bps=8000)
    guard.record("manual", 80, 0, price=PRICE)
    advisory = guard.advisory()
    assert advisory.token_used_bps == 8000
    assert advisory.near_token_limit is True
    assert advisory.near_limit is True
    assert advisory.remaining_tokens == 20


def test_unpriceable_response_still_accrues_known_tokens() -> None:
    guard = quiet_guard(token_limit=100, fail_closed=False)
    with pytest.warns():
        guard.record("unknown-model", 10, 5)
    assert guard.spent_tokens == 15
    assert guard.spent_usd == 0.0


def test_zero_token_limit_blocks_first_llm_but_not_tool() -> None:
    guard = quiet_guard(token_limit=0)
    tool = guard.reserve_tool(0.01)
    guard.settle_tool("cache.lookup", 0.01, reserved=tool)
    with pytest.raises(TokenBudgetExceeded):
        guard.check(estimated_next_tokens=0)


def test_typed_handle_is_exactly_once_and_guard_owned() -> None:
    first = quiet_guard(token_limit=100)
    second = quiet_guard(token_limit=100)
    handle = first.reserve(0.1, estimated_tokens=20)
    assert isinstance(handle, BudgetReservation)
    first.release(handle)
    with pytest.raises(ValueError, match="already"):
        first.release(handle)
    foreign = second.reserve(0.1, estimated_tokens=20)
    with pytest.raises(ValueError, match="different"):
        first.release(foreign)
    second.release(foreign)


def test_concurrent_token_reservations_hold_the_ceiling() -> None:
    guard = quiet_guard(token_limit=100)
    barrier = threading.Barrier(5)
    handles: list[BudgetReservation] = []
    blocked = 0
    lock = threading.Lock()

    def worker() -> None:
        nonlocal blocked
        barrier.wait()
        try:
            handle = guard.reserve(0.0, estimated_tokens=30)
        except TokenBudgetExceeded:
            with lock:
                blocked += 1
            return
        assert isinstance(handle, BudgetReservation)
        with lock:
            handles.append(handle)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(handles) == 3
    assert blocked == 2
    for handle in handles:
        guard.release(handle)
    assert guard.remaining_tokens == 100


def test_step_resets_and_counts_against_parent() -> None:
    guard = quiet_guard(token_limit=1_000)
    with guard.step(max_usd=1.0, max_tokens=100) as step:
        step.record("manual", 40, 10, price=PRICE)
        advisory = step.advisory()
        assert advisory.step_spent_tokens == 50
        assert advisory.step_remaining_tokens == 50
    with guard.step(max_tokens=100) as next_step:
        assert next_step.advisory().step_spent_tokens == 0
        next_step.record("manual", 20, 10, price=PRICE)
    assert guard.spent_tokens == 80


def test_step_tightest_limit_blocks_with_scope() -> None:
    guard = quiet_guard(token_limit=1_000)
    with guard.step(max_usd=0.05, max_tokens=50) as step:
        step.record("manual", 30, 10, price=PRICE)
        with pytest.raises(TokenBudgetExceeded) as caught:
            step.check(0.001, estimated_next_tokens=11)
        assert caught.value.scope == "step"

    with guard.step(max_usd=0.001) as step:
        with pytest.raises(BudgetExceeded) as caught:
            step.check(0.002, estimated_next_tokens=0)
        assert caught.value.scope == "step"


def test_tool_counts_against_step_usd_not_tokens() -> None:
    guard = quiet_guard(token_limit=100)
    with guard.step(max_usd=0.02, max_tokens=0) as step:
        handle = step.reserve_tool(0.01)
        step.settle_tool("search", 0.01, reserved=handle)
        assert step.advisory().step_spent_tokens == 0
        with pytest.raises(BudgetExceeded):
            step.reserve_tool(0.02)


def test_overlapping_steps_keep_state_isolated() -> None:
    guard = quiet_guard(token_limit=1_000)
    first = guard.step(max_tokens=100)
    second = guard.step(max_tokens=200)
    with first as step_a, second as step_b:
        step_a.record("manual", 30, 0, price=PRICE)
        step_b.record("manual", 70, 0, price=PRICE)
        assert step_a.advisory().step_spent_tokens == 30
        assert step_b.advisory().step_spent_tokens == 70
    assert guard.spent_tokens == 100


def test_clean_step_exit_detects_leaked_reservation() -> None:
    guard = quiet_guard(token_limit=100)
    handle: BudgetReservation
    with pytest.raises(RuntimeError, match="active reservation"):
        with guard.step(max_tokens=50) as step:
            reserved = step.reserve(0.0, estimated_tokens=10)
            assert isinstance(reserved, BudgetReservation)
            handle = reserved
    guard.release(handle)


def test_step_exception_is_not_masked_by_leaked_reservation() -> None:
    guard = quiet_guard(token_limit=100)
    handle: BudgetReservation
    with pytest.raises(LookupError, match="provider failed"):
        with guard.step(max_tokens=50) as step:
            reserved = step.reserve(0.0, estimated_tokens=10)
            assert isinstance(reserved, BudgetReservation)
            handle = reserved
            raise LookupError("provider failed")
    guard.release(handle)


def test_step_cannot_open_new_reservations_after_exit() -> None:
    guard = quiet_guard(token_limit=100)
    scoped = guard.step(max_tokens=50)

    with scoped:
        pass

    with pytest.raises(RuntimeError, match="no longer active"):
        scoped.reserve(0.0, estimated_tokens=1)
    with pytest.raises(RuntimeError, match="no longer active"):
        scoped.reserve_tool(0.01)


def test_step_record_tool_rejects_spend_after_exit() -> None:
    guard = quiet_guard(token_limit=100)
    scoped = guard.step(max_usd=0.05)

    with scoped:
        scoped.record_tool("search", 0.01)
    aggregate_spend = guard.spent_usd
    step_spend = scoped.advisory().step_spent_usd

    with pytest.raises(RuntimeError, match="no longer active"):
        scoped.record_tool("search", 0.01)
    assert guard.spent_usd == aggregate_spend
    assert scoped.advisory().step_spent_usd == step_spend


def test_step_rejects_nested_entry_of_the_same_scope() -> None:
    guard = quiet_guard(token_limit=100)
    scoped = guard.step(max_tokens=50)

    with scoped:
        with pytest.raises(RuntimeError, match="cannot be re-entered"):
            with scoped:
                pass
        scoped.check(estimated_next_tokens=1)

    with pytest.raises(RuntimeError, match="cannot be re-entered"):
        with scoped:
            pass


def test_step_advisory_drives_existing_near_limit_signal() -> None:
    guard = quiet_guard(token_limit=1_000, near_limit_bps=8000)
    with guard.step(max_tokens=100) as step:
        step.record("manual", 80, 0, price=PRICE)
        advisory = step.advisory()
        assert advisory.step_used_bps == 8000
        assert advisory.step_near_limit is True
        assert advisory.near_limit is True


def test_stream_hard_stops_on_token_ceiling_and_settles_partial_usage() -> None:
    guard = quiet_guard(token_limit=5)
    handle = guard.reserve(0.0, estimated_tokens=0)
    with pytest.raises(TokenBudgetExceeded):
        with StreamGuard(
            guard,
            "manual",
            prompt_tokens=2,
            reserved=handle,
            price=PRICE,
        ) as stream:
            stream.feed_tokens(3)
            stream.feed_tokens(1)
    assert guard.spent_tokens == 6


def test_scoped_guard_blocks_adapter_before_provider_call() -> None:
    called = False

    class Completions:
        def create(self, **kwargs):
            nonlocal called
            called = True
            return {}

    class Client:
        chat = type("Chat", (), {"completions": Completions()})()

    guard = quiet_guard(token_limit=1_000)
    with guard.step(max_tokens=0) as step:
        with pytest.raises(TokenBudgetExceeded):
            guarded_completion(step, Client(), model="gpt-4o", messages=[])
    assert called is False
