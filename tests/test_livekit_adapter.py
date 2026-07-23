"""Tests for the LiveKit Agents integration.

The adapter hooks two surfaces — the agent's ``llm_node`` (reserve) and the
session's ``metrics_collected`` / ``close`` events (settle / release). These
drive them directly through a minimal fake agent + event-emitter session, which
is all ``attach`` touches; a full LiveKit room/runner isn't needed.
"""

from __future__ import annotations

import asyncio
import types

import pytest

# The adapter hard-imports livekit-agents at module load, so skip the whole
# module (rather than erroring collection) when the optional extra is absent.
pytest.importorskip("livekit.agents")

from livekit.agents.metrics import LLMMetrics, STTMetrics, TTSMetrics  # noqa: E402

from floe_guard import BudgetGuard  # noqa: E402
from floe_guard.errors import BudgetExceeded  # noqa: E402
from floe_guard.integrations.livekit import LiveKitBudgetGuard  # noqa: E402


class _FakeSession:
    def __init__(self):
        self._handlers = {}

    def on(self, event, cb):
        self._handlers[event] = cb

    def emit(self, event, arg):
        self._handlers[event](arg)


class _FakeAgent:
    async def llm_node(self, chat_ctx, tools, model_settings):
        for chunk in ("hello", " world"):
            yield chunk


def _attached(guard, **kwargs):
    agent, session = _FakeAgent(), _FakeSession()
    budget = LiveKitBudgetGuard(guard, model="gpt-4o", **kwargs)
    budget.attach(session, agent)
    return budget, agent, session


async def _drive_turn(agent):
    return [chunk async for chunk in agent.llm_node(None, None, None)]


def _llm_metrics(prompt_tokens, completion_tokens, cancelled=False):
    return LLMMetrics(
        label="llm",
        request_id="r",
        timestamp=0.0,
        duration=0.0,
        ttft=0.0,
        cancelled=cancelled,
        completion_tokens=completion_tokens,
        prompt_tokens=prompt_tokens,
        prompt_cached_tokens=0,
        total_tokens=prompt_tokens + completion_tokens,
        tokens_per_second=0.0,
    )


def _event(metric):
    return types.SimpleNamespace(metrics=metric)


@pytest.mark.asyncio
async def test_turn_streams_through_and_settles_on_metrics():
    guard = BudgetGuard(limit_usd=100.00)
    budget, agent, session = _attached(guard)

    chunks = await _drive_turn(agent)
    assert chunks == ["hello", " world"]  # original llm_node output is preserved
    assert budget._pending  # reservation held until usage is reported

    session.emit("metrics_collected", _event(_llm_metrics(100, 50)))

    assert guard.advisory().spent_usd > 0
    assert not budget._pending
    assert budget._reserved == 0.0


@pytest.mark.asyncio
async def test_on_budget_exceeded_callback_used_instead_of_raising():
    guard = BudgetGuard(limit_usd=0.0001)
    called = {}

    async def handle(exc):
        called["exc"] = exc

    budget, agent, session = _attached(guard, on_budget_exceeded=handle)

    await _drive_turn(agent)  # first turn reserves $0 (no prior cost) and passes
    session.emit("metrics_collected", _event(_llm_metrics(1000, 500)))  # now over ceiling

    chunks = await _drive_turn(agent)  # second turn should be blocked
    assert chunks == []  # blocked turn yields nothing
    assert isinstance(called.get("exc"), BudgetExceeded)


@pytest.mark.asyncio
async def test_blocked_turn_raises_without_callback():
    guard = BudgetGuard(limit_usd=0.0001)
    budget, agent, session = _attached(guard)

    await _drive_turn(agent)
    session.emit("metrics_collected", _event(_llm_metrics(1000, 500)))

    with pytest.raises(BudgetExceeded):
        await _drive_turn(agent)


@pytest.mark.asyncio
async def test_cancelled_turn_settles_and_releases_reservation():
    guard = BudgetGuard(limit_usd=100.00)
    budget, agent, session = _attached(guard)

    await _drive_turn(agent)
    session.emit("metrics_collected", _event(_llm_metrics(0, 0, cancelled=True)))

    assert not budget._pending
    assert budget._reserved == 0.0


@pytest.mark.asyncio
async def test_close_releases_dangling_reservation():
    guard = BudgetGuard(limit_usd=100.00)
    budget, agent, session = _attached(guard)

    await _drive_turn(agent)  # reserves, no metrics emitted
    assert budget._pending
    session.emit("close", types.SimpleNamespace())

    assert not budget._pending
    assert guard._reserved == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_failing_turn_releases_its_own_reservation():
    guard = BudgetGuard(limit_usd=100.00)

    class _FailingAgent:
        async def llm_node(self, chat_ctx, tools, model_settings):
            raise RuntimeError("llm blew up")
            yield  # make this an async generator  # noqa: RET503

    agent, session = _FailingAgent(), _FakeSession()
    budget = LiveKitBudgetGuard(guard, model="gpt-4o")
    budget.attach(session, agent)

    with pytest.raises(RuntimeError, match="llm blew up"):
        async for _ in agent.llm_node(None, None, None):
            pass

    assert not budget._pending
    assert budget._reserved == 0.0
    assert guard._reserved == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_failing_turn_does_not_release_later_turns_reservation():
    """If turn A fails after turn B has reserved, A must not free B's hold."""
    guard = BudgetGuard(limit_usd=100.00)
    # Seed a non-zero default estimate so reserve() holds real USD.
    seed_agent, seed_session = _FakeAgent(), _FakeSession()
    seed = LiveKitBudgetGuard(guard, model="gpt-4o")
    seed.attach(seed_session, seed_agent)
    await _drive_turn(seed_agent)
    seed_session.emit("metrics_collected", _event(_llm_metrics(1000, 500)))

    a_ready = asyncio.Event()
    a_may_fail = asyncio.Event()
    calls = {"n": 0}

    class _OverlapAgent:
        async def llm_node(self, chat_ctx, tools, model_settings):
            calls["n"] += 1
            if calls["n"] == 1:
                a_ready.set()
                await a_may_fail.wait()
                raise RuntimeError("turn A interrupted")
            yield "b-ok"

    agent, session = _OverlapAgent(), _FakeSession()
    budget = LiveKitBudgetGuard(guard, model="gpt-4o")
    budget.attach(session, agent)

    async def drive_a():
        with pytest.raises(RuntimeError, match="turn A interrupted"):
            async for _ in agent.llm_node(None, None, None):
                pass

    task_a = asyncio.create_task(drive_a())
    await a_ready.wait()

    # Turn B reserves while A is still in flight (and still owns its cleanup path).
    chunks = [c async for c in agent.llm_node(None, None, None)]
    assert chunks == ["b-ok"]
    assert budget._pending
    b_reserved = budget._reserved
    assert b_reserved > 0
    assert guard._reserved == pytest.approx(b_reserved)

    a_may_fail.set()
    await task_a

    # A's exception cleanup must not have released B's reservation.
    assert budget._pending
    assert budget._reserved == pytest.approx(b_reserved)
    assert guard._reserved == pytest.approx(b_reserved)


@pytest.mark.asyncio
async def test_delayed_metrics_do_not_steal_later_turns_reservation():
    """Turn A's late LLMMetrics must not settle against turn B's hold."""
    guard = BudgetGuard(limit_usd=100.00)
    budget, agent, session = _attached(guard)

    # Seed a non-zero default estimate so holds are real USD.
    await _drive_turn(agent)
    session.emit("metrics_collected", _event(_llm_metrics(1000, 500)))

    # Turn A reserves and finishes streaming; metrics not yet emitted.
    await _drive_turn(agent)
    assert budget._pending
    a_reserved = budget._reserved
    assert a_reserved > 0

    # Turn B starts: early-releases A's guard hold, then reserves its own.
    await _drive_turn(agent)
    assert budget._pending
    b_reserved = budget._reserved
    assert b_reserved > 0
    assert guard._reserved == pytest.approx(b_reserved)

    # A's delayed metrics arrive first — must meter A without consuming B.
    session.emit("metrics_collected", _event(_llm_metrics(100, 50)))
    assert budget._pending
    assert budget._reserved == pytest.approx(b_reserved)
    assert guard._reserved == pytest.approx(b_reserved)

    spent_after_a = guard.advisory().spent_usd

    # B's metrics settle B's own hold.
    session.emit("metrics_collected", _event(_llm_metrics(200, 100)))
    assert not budget._pending
    assert budget._reserved == 0.0
    assert guard._reserved == pytest.approx(0.0)
    assert guard.advisory().spent_usd > spent_after_a


@pytest.mark.asyncio
async def test_stt_and_tts_metered_only_when_priced():
    guard = BudgetGuard(limit_usd=100.00)
    _, _, session = _attached(guard, stt_usd_per_second=0.01, tts_usd_per_1k_chars=0.10)

    session.emit(
        "metrics_collected",
        _event(
            STTMetrics(
                label="stt",
                request_id="r",
                timestamp=0.0,
                duration=0.0,
                audio_duration=10.0,
                streamed=True,
            )
        ),
    )
    session.emit(
        "metrics_collected",
        _event(
            TTSMetrics(
                label="tts",
                request_id="r",
                timestamp=0.0,
                ttfb=0.0,
                duration=0.0,
                audio_duration=0.0,
                cancelled=False,
                characters_count=1000,
                streamed=True,
            )
        ),
    )

    # 10s * $0.01/s + 1000 chars / 1000 * $0.10 = 0.10 + 0.10
    assert guard.advisory().spent_usd == pytest.approx(10 * 0.01 + 0.10)


@pytest.mark.asyncio
async def test_stt_tts_ignored_by_default():
    guard = BudgetGuard(limit_usd=100.00)
    _, _, session = _attached(guard)  # no per-unit prices

    session.emit(
        "metrics_collected",
        _event(
            STTMetrics(
                label="stt",
                request_id="r",
                timestamp=0.0,
                duration=0.0,
                audio_duration=10.0,
                streamed=True,
            )
        ),
    )

    assert guard.advisory().spent_usd == 0.0
