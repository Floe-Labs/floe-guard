"""LangGraph adapter: the ceiling must hold across a real StateGraph fan-out.

Issue #33's acceptance criteria, as tests:

1. A fan-out with N parallel sub-agents never crosses ``limit_usd`` —
   ``reserve()``/``settle()`` hold under branch concurrency (LangGraph runs the
   branches of a superstep on a thread pool, the exact race of issue #18).
2. Graph state exposes a typed ``BudgetAdvisory``; a router downshifts model
   choice on ``near_limit`` before any ``BudgetExceeded``.
"""

from __future__ import annotations

import operator
import time
from typing import Annotated

import pytest

pytest.importorskip("langgraph")

from langgraph.graph import END, START, StateGraph  # noqa: E402
from typing_extensions import TypedDict  # noqa: E402

from floe_guard import BudgetAdvisory, BudgetExceeded, BudgetGuard  # noqa: E402
from floe_guard.integrations.langgraph import (  # noqa: E402
    AdvisoryChannel,
    guarded_node,
    latest_advisory,
)

MODEL = "gpt-4o"  # 1k in + 1k out = $0.0125 / call
CHEAP = "gpt-4o-mini"  # 1k in + 1k out = ~$0.0008 / call


def _usage(model: str = MODEL) -> dict:
    return {"model": model, "prompt_tokens": 1_000, "completion_tokens": 1_000}


class FanOutState(TypedDict):
    results: Annotated[list, operator.add]
    budget: AdvisoryChannel


class LoopState(TypedDict):
    # Module-level (not defined inside its test): langgraph resolves node/route
    # type hints through the function's module globals, where a test-local
    # class would not exist.
    models_used: Annotated[list, operator.add]
    steps: Annotated[int, operator.add]
    budget: AdvisoryChannel


def _wait_for_settles(guard: BudgetGuard, timeout: float = 2.0) -> None:
    """Wait for in-flight branches to settle/release their holds.

    When a branch raises BudgetExceeded, invoke() re-raises while sibling
    threads may still be inside their simulated API latency; their settle()
    lands moments later. Bounded wait, then the assertions run.
    """
    deadline = time.monotonic() + timeout
    while guard._reserved > 1e-9 and time.monotonic() < deadline:
        time.sleep(0.01)


def test_fan_out_within_budget_settles_exactly() -> None:
    # 6 parallel branches at $0.0125 fit under a $0.10 ceiling: every branch
    # must run, settle its own slice, and leak nothing.
    guard = BudgetGuard(limit_usd=0.10, on_block=lambda *_: None)
    guard.record(MODEL, 1_000, 1_000)  # warm the next-call estimate

    def make_branch(i: int):
        def branch(state: FanOutState) -> dict:
            time.sleep(0.02)  # API latency — the window the old race exploited
            return {"results": [i], "usage": _usage()}

        return branch

    g = StateGraph(FanOutState)
    for i in range(6):
        g.add_node(f"agent_{i}", guarded_node(guard, make_branch(i)))
        g.add_edge(START, f"agent_{i}")
        g.add_edge(f"agent_{i}", END)
    out = g.compile().invoke({"results": []})

    assert sorted(out["results"]) == list(range(6))
    assert guard.spent_usd == pytest.approx(7 * 0.0125)  # warm call + 6 branches
    assert guard._reserved == pytest.approx(0.0, abs=1e-9)
    # The advisory channel carries the fan-out's final utilization.
    assert isinstance(out["budget"], BudgetAdvisory)
    assert out["budget"].used_bps == 8750


def test_fan_out_over_budget_never_crosses_ceiling() -> None:
    # 16 parallel branches want ~$0.21 against a $0.10 ceiling. Some branches
    # must be refused at reserve() time, and the total must hold — the graph
    # port of test_concurrency's 16-agents guarantee.
    guard = BudgetGuard(limit_usd=0.10, on_block=lambda *_: None)
    guard.record(MODEL, 1_000, 1_000)

    def make_branch(i: int):
        def branch(state: FanOutState) -> dict:
            time.sleep(0.02)
            return {"results": [i], "usage": _usage()}

        return branch

    g = StateGraph(FanOutState)
    for i in range(16):
        g.add_node(f"agent_{i}", guarded_node(guard, make_branch(i)))
        g.add_edge(START, f"agent_{i}")
        g.add_edge(f"agent_{i}", END)
    app = g.compile()

    with pytest.raises(BudgetExceeded):
        app.invoke({"results": []})
    _wait_for_settles(guard)

    # The guarantee: parallel branches never push spend past the ceiling...
    assert guard.spent_usd <= guard.limit_usd + 1e-9
    # ...and no reservation leaked from the raced and the refused branches.
    assert guard._reserved == pytest.approx(0.0, abs=1e-9)


def test_router_downshifts_on_near_limit_before_hard_stop() -> None:
    # Acceptance (b): a router reads state["budget"].near_limit and switches to
    # the cheap model, so the run finishes on budget with zero BudgetExceeded.
    guard = BudgetGuard(limit_usd=0.05, near_limit_bps=7000, on_block=lambda *_: None)

    def route(state: LoopState) -> str:
        adv = state.get("budget")
        if state["steps"] >= 12:  # safety valve; the budget should not need it
            return END
        if adv is not None and adv.near_limit and adv.remaining_usd < 0.0008:
            return END  # not even a cheap call left
        return "cheap_step" if adv is not None and adv.near_limit else "full_step"

    def full_step(state: LoopState) -> dict:
        return {"models_used": [MODEL], "steps": 1, "usage": _usage(MODEL)}

    def cheap_step(state: LoopState) -> dict:
        return {"models_used": [CHEAP], "steps": 1, "usage": _usage(CHEAP)}

    g = StateGraph(LoopState)
    g.add_node("full_step", guarded_node(guard, full_step, estimated_cost=0.0125))
    g.add_node("cheap_step", guarded_node(guard, cheap_step, estimated_cost=0.0008))
    g.add_conditional_edges(START, route)
    g.add_conditional_edges("full_step", route)
    g.add_conditional_edges("cheap_step", route)
    out = g.compile().invoke({"models_used": [], "steps": 0}, {"recursion_limit": 100})

    # $0.05 at $0.0125/call trips the 70% advisory after the 3rd full call...
    assert out["models_used"][:3] == [MODEL, MODEL, MODEL]
    # ...then every later call ran on the cheap model, and the ceiling held
    # without a single hard block.
    assert set(out["models_used"][3:]) == {CHEAP}
    assert guard.spent_usd <= guard.limit_usd + 1e-9


def test_node_error_releases_reservation() -> None:
    guard = BudgetGuard(limit_usd=0.10, on_block=lambda *_: None)
    guard.record(MODEL, 1_000, 1_000)

    @guarded_node(guard)
    def broken(state: dict) -> dict:
        raise RuntimeError("upstream API fell over")

    with pytest.raises(RuntimeError):
        broken({})
    assert guard._reserved == pytest.approx(0.0, abs=1e-9)
    assert guard.spent_usd == pytest.approx(0.0125)  # only the warm call


def test_malformed_usage_releases_hold_and_raises() -> None:
    # A malformed usage payload must not leak the reservation: the hold is
    # released before the error propagates, the same fail-safe as settle()'s
    # pricing-error path.
    guard = BudgetGuard(limit_usd=0.10, on_block=lambda *_: None)
    guard.record(MODEL, 1_000, 1_000)

    @guarded_node(guard)
    def broken_usage(state: dict) -> dict:
        return {"usage": {"model": MODEL, "prompt_tokens": "abc", "completion_tokens": 50}}

    with pytest.raises(ValueError):
        broken_usage({})
    assert guard._reserved == pytest.approx(0.0, abs=1e-9)
    assert guard.spent_usd == pytest.approx(0.0125)  # only the warm call


def test_node_without_usage_releases_hold_and_still_reports_advisory() -> None:
    # A node that meters through another floe-guard adapter (or spends nothing)
    # must not be double-counted: the hold is released, not settled.
    guard = BudgetGuard(limit_usd=0.10, on_block=lambda *_: None)
    guard.record(MODEL, 1_000, 1_000)

    @guarded_node(guard)
    def tool_only(state: dict) -> dict:
        return {"results": ["no llm call here"]}

    update = tool_only({})
    assert guard.spent_usd == pytest.approx(0.0125)
    assert guard._reserved == pytest.approx(0.0, abs=1e-9)
    assert isinstance(update["budget"], BudgetAdvisory)


def test_async_node_is_guarded_too() -> None:
    # asyncio.run keeps this self-contained — the dev extra carries no async
    # pytest plugin, and the suite must stay green bare.
    import asyncio

    guard = BudgetGuard(limit_usd=0.10, on_block=lambda *_: None)

    @guarded_node(guard)
    async def async_worker(state: dict) -> dict:
        return {"usage": _usage()}

    update = asyncio.run(async_worker({}))
    assert guard.spent_usd == pytest.approx(0.0125)
    assert isinstance(update["budget"], BudgetAdvisory)


def test_langgraph_example_finishes_on_budget(capsys: pytest.CaptureFixture[str]) -> None:
    # Acceptance (c): the prebuilt budget-aware router example runs end to end
    # with no API key and finishes under its ceiling (same contract as the
    # runaway_loop example test).
    import sys
    from pathlib import Path

    examples = Path(__file__).resolve().parent.parent / "examples"
    sys.path.insert(0, str(examples))
    try:
        import langgraph_budget_aware

        langgraph_budget_aware.main()
    finally:
        sys.path.remove(str(examples))

    out = capsys.readouterr().out
    assert "tapering" in out
    assert "Finished at step" in out
    assert "held under $0.10" in out


def test_latest_advisory_prefers_higher_utilization() -> None:
    early = BudgetAdvisory(
        near_limit=False, used_bps=2500, remaining_usd=0.075, limit_usd=0.10, spent_usd=0.025
    )
    late = BudgetAdvisory(
        near_limit=True, used_bps=8750, remaining_usd=0.0125, limit_usd=0.10, spent_usd=0.0875
    )
    # Branch completion order is nondeterministic — the reducer must keep the
    # fresher (higher-utilization) reading regardless of write order.
    assert latest_advisory(early, late) is late
    assert latest_advisory(late, early) is late
    assert latest_advisory(None, early) is early
    assert latest_advisory(late, None) is late
