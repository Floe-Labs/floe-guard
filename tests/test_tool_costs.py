"""Tool spend as a first-class primitive: reserve_tool/settle_tool/record_tool
share the token ceiling, and tool_costs exposes the per-tool split."""

from __future__ import annotations

import threading
import time

import pytest

from floe_guard import BudgetExceeded, BudgetGuard

MODEL = "gpt-4o"  # 1k in + 1k out = $0.0125/call


# ── validation ──────────────────────────────────────────────────────────────────


def test_reserve_tool_rejects_non_finite_estimates() -> None:
    guard = BudgetGuard(limit_usd=1.00)
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            guard.reserve_tool(bad)


def test_settle_tool_rejects_bad_amounts() -> None:
    guard = BudgetGuard(limit_usd=1.00)
    for bad in (float("nan"), float("inf"), -0.01):
        with pytest.raises(ValueError):
            guard.settle_tool("apollo.people_lookup", bad)
        with pytest.raises(ValueError):
            guard.settle_tool("apollo.people_lookup", 0.01, reserved=bad)
    assert guard.spend_log == []
    assert guard.tool_costs == {}


# ── accrual and attribution ─────────────────────────────────────────────────────


def test_tool_costs_tallies_per_name() -> None:
    guard = BudgetGuard(limit_usd=1.00)
    guard.record_tool("apollo.people_lookup", 0.02)
    guard.record_tool("apollo.people_lookup", 0.02)
    guard.record_tool("exa.search", 0.01)
    assert guard.tool_costs == {
        "apollo.people_lookup": pytest.approx(0.04),
        "exa.search": pytest.approx(0.01),
    }
    assert guard.spent_usd == pytest.approx(0.05)


def test_tokens_and_tools_share_one_ceiling_and_split_is_inspectable() -> None:
    guard = BudgetGuard(limit_usd=1.00)
    guard.record(MODEL, 1_000, 1_000)  # $0.0125 of tokens
    guard.record_tool("apollo.people_lookup", 0.02)
    assert guard.spent_usd == pytest.approx(0.0325)
    assert guard.remaining_usd == pytest.approx(1.00 - 0.0325)
    tool_total = sum(guard.tool_costs.values())
    assert tool_total == pytest.approx(0.02)
    assert guard.spent_usd - tool_total == pytest.approx(0.0125)  # token side


def test_tool_costs_returns_a_snapshot_copy() -> None:
    guard = BudgetGuard(limit_usd=1.00)
    guard.record_tool("exa.search", 0.01)
    snapshot = guard.tool_costs
    snapshot["exa.search"] = 999.0
    assert guard.tool_costs["exa.search"] == pytest.approx(0.01)


# ── pre-call hard-stop (the reserve/settle contract) ────────────────────────────


def test_reserve_tool_blocks_before_the_tool_runs() -> None:
    # The cost is KNOWN pre-call, so the hard-stop is exact: the Apollo call
    # that would cross the cap never happens, and nothing is held afterwards.
    guard = BudgetGuard(limit_usd=0.01, on_block=lambda *_: None)
    with pytest.raises(BudgetExceeded):
        guard.reserve_tool(0.02)
    assert guard.remaining_usd == pytest.approx(0.01)
    assert guard.spent_usd == 0.0


def test_reserve_settle_tool_round_trip() -> None:
    guard = BudgetGuard(limit_usd=1.00)
    handle = guard.reserve_tool(0.02)
    assert handle == pytest.approx(0.02)
    assert guard.remaining_usd == pytest.approx(0.98)  # held while in flight
    cost = guard.settle_tool("apollo.people_lookup", 0.02, reserved=handle, label="prospector")
    assert cost == pytest.approx(0.02)
    assert guard.remaining_usd == pytest.approx(0.98)  # hold swapped for spend
    (event,) = guard.spend_log
    assert event.kind == "tool"
    assert event.model_or_tool == "apollo.people_lookup"
    assert event.reserved == pytest.approx(0.02)
    assert event.label == "prospector"


def test_release_frees_a_tool_reservation_on_failure() -> None:
    guard = BudgetGuard(limit_usd=1.00)
    handle = guard.reserve_tool(0.02)
    guard.release(handle)  # the tool call failed — nothing was spent
    assert guard.remaining_usd == pytest.approx(1.00)
    assert guard.spend_log == []


def test_runaway_tool_loop_dies_at_the_ceiling() -> None:
    # THE user story: a loop hammering a $0.002 paid API. record_tool updates
    # the next-call estimate, so plain check() stops the loop BEFORE the
    # crossing call — the same stop-before contract as tokens.
    guard = BudgetGuard(limit_usd=0.01, on_block=lambda *_: None)
    calls = 0
    with pytest.raises(BudgetExceeded):
        for _ in range(1_000):
            guard.check()
            guard.record_tool("apollo.people_lookup", 0.002)
            calls += 1
    assert calls == 5  # 5 × $0.002 == $0.01 — call 6 was blocked, not refunded
    assert guard.spent_usd <= guard.limit_usd + 1e-9


# ── concurrency (mirrors test_concurrency.py, mixed token + tool spend) ─────────


def test_mixed_token_and_tool_spend_holds_ceiling_under_parallel_calls() -> None:
    guard = BudgetGuard(limit_usd=0.10, on_block=lambda *_: None)
    # Warm one call so the LLM path's next-call estimate is realistic.
    guard.record(MODEL, 1_000, 1_000)

    blocked: list[int] = []

    def llm_agent(i: int) -> None:
        try:
            reserved = guard.reserve()
        except BudgetExceeded:
            blocked.append(i)
            return
        time.sleep(0.02)  # API latency — the window the old race exploited
        guard.settle(MODEL, 1_000, 1_000, reserved=reserved)

    def tool_agent(i: int) -> None:
        try:
            reserved = guard.reserve_tool(0.0125)
        except BudgetExceeded:
            blocked.append(i)
            return
        time.sleep(0.02)
        guard.settle_tool("apollo.people_lookup", 0.0125, reserved=reserved)

    threads = [
        threading.Thread(target=llm_agent if i % 2 else tool_agent, args=(i,))
        for i in range(16)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # The guarantee: mixed token + tool fan-out never pushes spend past the cap...
    assert guard.spent_usd <= guard.limit_usd + 1e-9
    # ...and the excess was actually stopped, not silently allowed.
    assert blocked
    # No reservations leaked, and both kinds of spend landed in the tally.
    assert guard._reserved == pytest.approx(0.0, abs=1e-9)
    assert guard.tool_costs.get("apollo.people_lookup", 0.0) > 0.0
