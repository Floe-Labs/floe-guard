"""Aggregate token ceilings and explicit per-step budget scopes (issue #46)."""

from __future__ import annotations

import copy
import threading
from dataclasses import FrozenInstanceError

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
    held = guard.reserve(0.0)
    assert isinstance(held, BudgetReservation)
    assert held.tokens == 50
    with pytest.raises(TokenBudgetExceeded):
        guard.reserve(0.0)
    guard.release(held)


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


def test_typed_handle_is_immutable() -> None:
    guard = quiet_guard(token_limit=100)
    other = quiet_guard(token_limit=100)
    handle = guard.reserve(0.1, estimated_tokens=20)
    assert isinstance(handle, BudgetReservation)
    assert handle.usd == 0.1
    assert handle.tokens == 20

    with pytest.raises(FrozenInstanceError):
        handle.usd = 0.2  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        handle.tokens = 30  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        handle._owner = other  # type: ignore[misc]

    guard.release(handle)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("usd", 0.4),
        ("usd", -0.1),
        ("usd", float("nan")),
        ("tokens", 40),
        ("tokens", -1),
        ("tokens", 1.5),
        ("tokens", float("nan")),
    ],
)
def test_forced_typed_handle_mutation_cannot_change_accounting(field: str, bad: object) -> None:
    guard = quiet_guard(token_limit=100)
    handle = guard.reserve(0.2, estimated_tokens=20)
    assert isinstance(handle, BudgetReservation)
    original = getattr(handle, field)

    object.__setattr__(handle, field, bad)
    with pytest.raises(ValueError, match="modified"):
        guard.release(handle)
    assert guard._reserved == 0.2
    assert guard._reserved_tokens == 20

    object.__setattr__(handle, field, original)
    guard.release(handle)
    assert guard._reserved == 0.0
    assert guard._reserved_tokens == 0


def test_equal_value_type_change_is_still_rejected_as_handle_mutation() -> None:
    guard = quiet_guard(token_limit=100)
    handle = guard.reserve(1.0, estimated_tokens=20)
    assert isinstance(handle, BudgetReservation)

    object.__setattr__(handle, "usd", True)
    with pytest.raises(ValueError, match="modified"):
        guard.release(handle)
    assert guard._reserved == 1.0
    assert guard._reserved_tokens == 20

    object.__setattr__(handle, "usd", 1.0)
    guard.release(handle)


def test_forged_and_copied_typed_handles_are_rejected_without_mutation() -> None:
    guard = quiet_guard(token_limit=100)
    issued = guard.reserve(0.2, estimated_tokens=20)
    assert isinstance(issued, BudgetReservation)
    forged = BudgetReservation(issued.usd, issued.tokens, guard)
    copied = copy.copy(issued)

    for invalid in (forged, copied):
        with pytest.raises(ValueError, match="not issued"):
            guard.release(invalid)
        assert guard._reserved == 0.2
        assert guard._reserved_tokens == 20

    guard.release(issued)


def test_equality_forgery_cannot_alias_an_issued_handle() -> None:
    guard = quiet_guard(token_limit=100)
    issued = guard.reserve(0.2, estimated_tokens=20)
    assert isinstance(issued, BudgetReservation)

    class EqualForgery(BudgetReservation):
        def __hash__(self) -> int:
            return hash(issued)

        def __eq__(self, other: object) -> bool:
            return True

    forged = EqualForgery(issued.usd, issued.tokens, guard)
    with pytest.raises(ValueError, match="not issued"):
        guard.release(forged)
    assert guard._reserved == 0.2
    assert guard._reserved_tokens == 20

    guard.release(issued)


def test_modified_handle_cannot_free_an_unrelated_token_hold() -> None:
    guard = quiet_guard(token_limit=300)
    first = guard.reserve(0.2, estimated_tokens=100)
    second = guard.reserve(0.2, estimated_tokens=100)
    assert isinstance(first, BudgetReservation)
    assert isinstance(second, BudgetReservation)

    object.__setattr__(first, "tokens", 200)
    with pytest.raises(ValueError, match="modified"):
        guard.release(first)
    assert guard._reserved == 0.4
    assert guard._reserved_tokens == 200
    with pytest.raises(TokenBudgetExceeded):
        guard.reserve(0.1, estimated_tokens=101)

    object.__setattr__(first, "tokens", 100)
    guard.release(first)
    assert guard._reserved_tokens == 100
    guard.release(second)
    assert guard._reserved == 0.0
    assert guard._reserved_tokens == 0


def test_concurrent_terminal_operations_consume_typed_handle_once() -> None:
    guard = quiet_guard(token_limit=100)
    target = guard.reserve(0.2, estimated_tokens=20)
    unrelated = guard.reserve(0.3, estimated_tokens=30)
    assert isinstance(target, BudgetReservation)
    assert isinstance(unrelated, BudgetReservation)
    barrier = threading.Barrier(2)
    outcomes: list[str] = []
    lock = threading.Lock()

    def release_target() -> None:
        barrier.wait()
        try:
            guard.release(target)
        except ValueError as exc:
            outcome = str(exc)
        else:
            outcome = "released"
        with lock:
            outcomes.append(outcome)

    threads = [threading.Thread(target=release_target) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert outcomes.count("released") == 1
    assert sum("already" in outcome for outcome in outcomes) == 1
    assert guard._reserved == 0.3
    assert guard._reserved_tokens == 30
    guard.release(unrelated)


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


def test_scoped_guard_rejects_sibling_and_nonzero_numeric_handles() -> None:
    guard = quiet_guard(token_limit=1_000)
    first = guard.step(max_tokens=100)
    second = guard.step(max_tokens=100)
    with first as step_a, second as step_b:
        handle_a = step_a.reserve(0.1, estimated_tokens=20)
        handle_b = step_b.reserve(0.2, estimated_tokens=30)
        assert isinstance(handle_a, BudgetReservation)
        assert isinstance(handle_b, BudgetReservation)

        with pytest.raises(ValueError, match="different step"):
            step_b.settle("manual", 1, 0, reserved=handle_a, price=PRICE)
        with pytest.raises(ValueError, match="different step"):
            step_b.release(handle_a)
        with pytest.raises(ValueError, match="must be zero"):
            step_a.settle("manual", 1, 0, reserved=0.1, price=PRICE)
        assert guard._reserved == pytest.approx(0.3)
        assert guard._reserved_tokens == 50

        step_a.release(handle_a)
        step_b.release(handle_b)


def test_scoped_record_paths_use_registered_zero_handles() -> None:
    guard = quiet_guard(token_limit=100)
    with guard.step(max_usd=1.0, max_tokens=50) as step:
        step.record("manual", 10, 5, price=PRICE)
        step.record_tool("search", 0.25)
        advisory = step.advisory()
        assert advisory.step_spent_tokens == 15
        assert advisory.step_spent_usd == pytest.approx(0.25002)
    assert guard.spent_tokens == 15
    assert guard.spent_usd == pytest.approx(0.25002)


def test_clean_step_exit_detects_leaked_reservation() -> None:
    guard = quiet_guard(token_limit=100)
    handle: BudgetReservation
    with pytest.raises(RuntimeError, match="active reservation"):
        with guard.step(max_tokens=50) as step:
            reserved = step.reserve(0.0, estimated_tokens=10)
            assert isinstance(reserved, BudgetReservation)
            handle = reserved
    step.release(handle)
    assert guard.remaining_tokens == 100


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


def test_step_exit_and_reserve_share_one_lifecycle_boundary() -> None:
    guard = quiet_guard(token_limit=100)
    scoped = guard.step(max_tokens=50)
    scoped.__enter__()
    reached_reserve = threading.Event()
    original_ensure_active = scoped._ensure_active

    def signal_before_reserve() -> None:
        original_ensure_active()
        reached_reserve.set()

    scoped._ensure_active = signal_before_reserve  # type: ignore[method-assign]
    reserve_result: list[BudgetReservation] = []
    reserve_errors: list[BaseException] = []
    exit_errors: list[BaseException] = []

    def reserve() -> None:
        try:
            handle = scoped.reserve(0.0, estimated_tokens=10)
            assert isinstance(handle, BudgetReservation)
            reserve_result.append(handle)
        except BaseException as exc:
            reserve_errors.append(exc)

    def exit_step() -> None:
        try:
            scoped.__exit__(None, None, None)
        except BaseException as exc:
            exit_errors.append(exc)

    reserve_thread = threading.Thread(target=reserve)
    exit_thread = threading.Thread(target=exit_step)
    signalled = False
    guard._lock.acquire()
    try:
        reserve_thread.start()
        signalled = reached_reserve.wait(timeout=1)
        exit_thread.start()
    finally:
        guard._lock.release()
    reserve_thread.join()
    exit_thread.join()
    assert signalled

    if reserve_result:
        assert len(exit_errors) == 1
        assert "active reservation" in str(exit_errors[0])
        guard.release(reserve_result[0])
    else:
        assert len(reserve_errors) == 1
        assert "no longer active" in str(reserve_errors[0])
        assert exit_errors == []
    assert guard.remaining_tokens == 100


def test_step_settlement_rejects_exit_but_release_still_cleans_up() -> None:
    guard = quiet_guard(token_limit=100)
    scoped = guard.step(max_usd=1.0, max_tokens=50)
    with scoped:
        llm_handle = scoped.reserve(0.0, estimated_tokens=0)
        tool_handle = scoped.reserve_tool(0.0)
        assert isinstance(llm_handle, BudgetReservation)
        assert isinstance(tool_handle, BudgetReservation)

    with pytest.raises(RuntimeError, match="no longer active"):
        scoped.settle("manual", 1, 0, reserved=llm_handle, price=PRICE)
    with pytest.raises(RuntimeError, match="no longer active"):
        scoped.settle_tool("search", 0.01, reserved=tool_handle)

    scoped.release(llm_handle)
    scoped.release(tool_handle)
    assert guard.spent_usd == 0.0
    assert guard.spent_tokens == 0
    assert guard.remaining_tokens == 100


def test_step_exposes_only_deliberate_parent_surface() -> None:
    guard = quiet_guard(token_limit=100)
    with guard.step(max_tokens=50) as step:
        assert step.limit_usd == guard.limit_usd
        assert step.token_limit == guard.token_limit
        assert step.remaining_tokens == guard.remaining_tokens
        assert step.estimate_call("manual", 1, price=PRICE) == pytest.approx(1e-6)
        assert step.export_log() == guard.export_log()
        with pytest.raises(AttributeError):
            step.step(max_tokens=1)  # type: ignore[attr-defined]
        with pytest.raises(AttributeError):
            step.unknown_parent_method()  # type: ignore[attr-defined]


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


def test_scoped_stream_rejects_wrong_handles_before_registration() -> None:
    guard = quiet_guard(token_limit=1_000)
    first = guard.step(max_tokens=100)
    second = guard.step(max_tokens=100)
    with first as step_a, second as step_b:
        handle = step_a.reserve(0.0, estimated_tokens=10)
        assert isinstance(handle, BudgetReservation)
        baseline_streams = len(guard._stream_costs)
        with pytest.raises(ValueError, match="different step"):
            StreamGuard(step_b, "manual", reserved=handle, price=PRICE)
        with pytest.raises(ValueError, match="must be zero"):
            StreamGuard(step_b, "manual", reserved=0.1, price=PRICE)
        assert len(guard._stream_costs) == baseline_streams
        assert guard._reserved_tokens == 10
        step_a.release(handle)


def test_scoped_zero_stream_is_attributed_to_step_ceiling() -> None:
    guard = quiet_guard(token_limit=1_000)
    with guard.step(max_tokens=5) as step:
        with pytest.raises(TokenBudgetExceeded) as caught:
            with StreamGuard(step, "manual", price=PRICE) as stream:
                stream.feed_tokens(6)
        assert caught.value.scope == "step"
        assert step.advisory().step_spent_tokens == 6
        assert guard._reserved_tokens == 0
        assert guard._stream_costs == {}


def test_scoped_stream_accepts_same_step_handle_and_settles() -> None:
    guard = quiet_guard(token_limit=1_000)
    with guard.step(max_tokens=100) as step:
        handle = step.reserve(0.0, estimated_tokens=10)
        assert isinstance(handle, BudgetReservation)
        stream = StreamGuard(step, "manual", reserved=handle, price=PRICE)
        stream.feed_tokens(4)
        assert stream.finish() == pytest.approx(8e-6)
        assert step.advisory().step_spent_tokens == 4
        assert guard._reserved_tokens == 0
        assert guard._stream_costs == {}


def test_scoped_stream_counts_same_step_but_not_sibling_streams() -> None:
    guard = quiet_guard(token_limit=1_000)
    first = guard.step(max_tokens=5)
    second = guard.step(max_tokens=5)
    with first as step_a, second as step_b:
        stream_a = StreamGuard(step_a, "manual", price=PRICE)
        stream_b = StreamGuard(step_b, "manual", price=PRICE)
        stream_a.feed_tokens(4)
        stream_b.feed_tokens(4)

        sibling_a = StreamGuard(step_a, "manual", price=PRICE)
        with pytest.raises(TokenBudgetExceeded) as caught:
            sibling_a.feed_tokens(2)
        assert caught.value.scope == "step"

        stream_a.finish()
        stream_b.finish()
        assert guard._stream_costs == {}


def test_scoped_stream_rejects_inactive_step() -> None:
    guard = quiet_guard(token_limit=100)
    scoped = guard.step(max_tokens=50)
    with scoped:
        pass
    with pytest.raises(RuntimeError, match="no longer active"):
        StreamGuard(scoped, "manual", price=PRICE)


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
