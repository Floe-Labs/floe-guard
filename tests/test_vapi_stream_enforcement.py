"""Vapi streaming enforcement through the public adapter, without provider keys."""

import asyncio
from types import SimpleNamespace

import pytest

from floe_guard import BudgetExceeded, BudgetGuard, ManualPrice, SqliteStore, StreamGuard
from floe_guard.integrations.vapi import VapiBudgetGuard, VapiUsageMissingError


def guard(limit=0.01):
    return BudgetGuard(
        limit, price_overrides={"m": ManualPrice(0.001, 0.001)}, on_block=lambda *_: None
    )


def chunk():
    return {"choices": [{"delta": {"content": "word"}}]}


@pytest.mark.asyncio
@pytest.mark.parametrize("started", [False, True])
@pytest.mark.parametrize(
    "invalid", [(), ("bad",), (ValueError, None, "bad traceback"), (ValueError, None, None, None)]
)
async def test_invalid_throw_preserves_stream_for_retry_and_cleanup(started, invalid):
    g = guard()
    closed = []

    async def source():
        try:
            yield chunk()
            yield chunk()
        finally:
            closed.append(True)

    stream = VapiBudgetGuard(g, model="m").guard_stream(source, estimated_cost=0.003)
    if started:
        await anext(stream)
    # Keep the generator alive: garbage collection must not mask lost ownership.
    native = stream._gen
    try:
        with pytest.raises(TypeError):
            await stream.athrow(*invalid)
        assert stream._gen is native
        assert await anext(stream) == chunk()
        await stream.aclose()
        assert closed == [True]
        assert len(g.spend_log) == 1
        assert g.spent_usd == pytest.approx(0.002 if started else 0.001)
        assert g.remaining_usd == pytest.approx(0.008 if started else 0.009)
        assert not g._stream_costs
    finally:
        await native.aclose()
        await stream.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["next", "close", "send", "throw"])
async def test_overlapping_operation_does_not_lose_cleanup(operation):
    g = guard()
    ready, resume = asyncio.Event(), asyncio.Event()
    closed = []

    async def source():
        try:
            ready.set()
            await resume.wait()
            yield chunk()
        finally:
            closed.append(True)

    stream = VapiBudgetGuard(g, model="m").guard_stream(source, estimated_cost=0.003)
    first = asyncio.create_task(anext(stream))
    await ready.wait()
    try:
        with pytest.raises(RuntimeError, match="already running"):
            if operation == "next":
                await anext(stream)
            elif operation == "close":
                await stream.aclose()
            elif operation == "send":
                await stream.asend(None)
            else:
                await stream.athrow(RuntimeError("cancel"))
    finally:
        resume.set()
        await first
        await stream.aclose()
    assert closed == [True]
    assert len(g.spend_log) == 1
    assert g.spent_usd == pytest.approx(0.001)
    assert g.remaining_usd == pytest.approx(0.009)
    assert not g._stream_costs


@pytest.mark.asyncio
@pytest.mark.parametrize("held", [0, 0.004, 0.008])
@pytest.mark.parametrize("actual", [2, 3])
async def test_final_usage_counts_other_stream_without_double_counting(held, actual):
    g = guard()
    other = StreamGuard(g, "m", reserved=g.reserve(held))
    other.feed_tokens(8)
    closed = []

    async def source():
        try:
            yield {"usage": {"prompt_tokens": 0, "completion_tokens": actual}}
        finally:
            closed.append(True)

    stream = VapiBudgetGuard(g, model="m").guard_stream(source, estimated_cost=0.001)
    try:
        if actual == 3:
            with pytest.raises(BudgetExceeded):
                await anext(stream)
        else:
            assert (await anext(stream))["usage"]["completion_tokens"] == actual
    finally:
        await stream.aclose()
        other.finish()
    assert closed == [True]
    assert len(g.spend_log) == 2
    assert g.spent_usd == pytest.approx((8 + actual) * 0.001)


@pytest.mark.asyncio
async def test_final_usage_preserves_an_unrelated_tool_hold():
    g = guard()
    held = g.reserve_tool(0.008)

    async def source():
        yield {"usage": {"prompt_tokens": 0, "completion_tokens": 3}}

    stream = VapiBudgetGuard(g, model="m").guard_stream(source, estimated_cost=0)
    with pytest.raises(BudgetExceeded):
        await anext(stream)
    assert len(g.spend_log) == 1
    assert g.spent_usd == pytest.approx(0.003)
    assert g.remaining_usd == 0
    g.release(held)
    assert g.remaining_usd == pytest.approx(0.007)


@pytest.mark.asyncio
async def test_crossing_records_partial_spend_and_closes_source():
    g = guard(0.0025)
    closed, seen = [], []

    async def source():
        try:
            for _ in range(10):
                yield chunk()
        finally:
            closed.append(True)

    with pytest.raises(BudgetExceeded):
        async for value in VapiBudgetGuard(g, model="m").guard_stream(source):
            seen.append(value)
    assert len(seen) == 2
    assert closed == [True]
    assert g.spent_usd == pytest.approx(0.003)
    assert len(g.spend_log) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("exit", ["close", "error", "missing", "cancel"])
async def test_partial_exits_preserve_other_reservations(exit):
    g = guard()
    other = g.reserve_tool(0.002)
    waiting = asyncio.Event()
    closed = []

    async def source():
        try:
            yield chunk()
            if exit == "error":
                raise RuntimeError("provider failed")
            if exit == "cancel":
                waiting.set()
                await asyncio.Event().wait()
        finally:
            closed.append(True)

    stream = VapiBudgetGuard(g, model="m").guard_stream(source, estimated_cost=0.003)
    await anext(stream)
    if exit == "close":
        await stream.aclose()
    elif exit == "cancel":
        task = asyncio.create_task(anext(stream))
        await asyncio.wait_for(waiting.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        with pytest.raises(RuntimeError if exit == "error" else VapiUsageMissingError):
            await anext(stream)
    await stream.aclose()
    assert closed == [True]
    assert g.spent_usd == pytest.approx(0.001)
    assert g.remaining_usd == pytest.approx(0.007)
    assert len(g.spend_log) == 1
    g.release(other)


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt", [8, 10])
async def test_final_usage_is_settled_before_yield(prompt):
    g = guard()

    async def source():
        yield chunk()
        yield {"usage": {"prompt_tokens": prompt, "completion_tokens": 1}}

    stream = VapiBudgetGuard(g, model="m").guard_stream(source)
    await anext(stream)
    if prompt == 10:
        with pytest.raises(BudgetExceeded):
            await anext(stream)
    else:
        await anext(stream)
        with pytest.raises(BudgetExceeded):
            g.reserve_tool(0.002)
    assert g.spent_usd == pytest.approx((prompt + 1) * 0.001)
    await stream.aclose()
    assert len(g.spend_log) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("objects", [False, True])
async def test_cached_usage_from_dict_or_sdk_object(objects):
    g = BudgetGuard(1)
    usage = {
        "prompt_tokens": 10000,
        "completion_tokens": 20,
        "prompt_tokens_details": {"cached_tokens": 9000},
    }
    if objects:
        usage["prompt_tokens_details"] = SimpleNamespace(cached_tokens=9000)
        final = SimpleNamespace(usage=SimpleNamespace(**usage))
    else:
        final = {"usage": usage}

    async def source():
        yield final

    stream = VapiBudgetGuard(g, model="gpt-4o").guard_stream(source)
    await anext(stream)
    assert g.spent_usd == pytest.approx(0.01395)
    await stream.aclose()
    assert len(g.spend_log) == 1


@pytest.mark.asyncio
async def test_unstarted_throw_releases_only_its_own_hold():
    g = guard()
    other = g.reserve_tool(0.002)

    async def source():
        yield chunk()

    stream = VapiBudgetGuard(g, model="m").guard_stream(source, estimated_cost=0.003)
    with pytest.raises(RuntimeError):
        await stream.athrow(RuntimeError("cancel"))
    await stream.aclose()
    assert g.remaining_usd == pytest.approx(0.008)
    assert not g._stream_costs
    g.release(other)


def test_persistent_stream_is_rejected_without_leaking_a_hold(tmp_path):
    g = BudgetGuard(1, window="utc-day", store=SqliteStore(tmp_path / "budget.db"))
    other = g.reserve_tool(0.2)
    with pytest.raises(ValueError, match="persistent store"):
        VapiBudgetGuard(g, model="gpt-4o").guard_stream(lambda: None, estimated_cost=0.3)
    assert g.remaining_usd == pytest.approx(0.8)
    g.release(other)


@pytest.mark.asyncio
async def test_source_startup_failure_releases_without_billing_prompt():
    g = guard()

    async def source():
        raise RuntimeError("connect failed")

    stream = VapiBudgetGuard(g, model="m").guard_stream(
        source, estimated_cost=0.003, prompt_tokens=2
    )
    with pytest.raises(RuntimeError, match="connect failed"):
        await anext(stream)
    assert g.remaining_usd == pytest.approx(0.01)
    assert not g.spend_log
    assert not g._stream_costs


@pytest.mark.asyncio
async def test_sdk_object_tool_deltas_and_async_close():
    g = guard()

    class Source:
        closed = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            tool_calls=[
                                SimpleNamespace(
                                    function=SimpleNamespace(name="find", arguments="{}")
                                )
                            ]
                        )
                    )
                ]
            )

        async def close(self):
            self.closed = True

    source = Source()
    stream = VapiBudgetGuard(g, model="m").guard_stream(
        lambda: source, prompt_tokens=2, count_tokens=lambda _: 3
    )
    await anext(stream)
    await stream.aclose()
    assert source.closed
    assert g.spent_usd == pytest.approx(0.005)


@pytest.mark.asyncio
async def test_cleanup_error_does_not_mask_budget_interruption():
    g = guard(0.0005)

    async def source():
        try:
            yield chunk()
        finally:
            raise RuntimeError("close failed")

    with pytest.raises(BudgetExceeded):
        await anext(VapiBudgetGuard(g, model="m").guard_stream(source))
    assert g.spent_usd == pytest.approx(0.001)
    assert not g._stream_costs


@pytest.mark.asyncio
async def test_invalid_first_send_keeps_stream_closeable():
    g = guard()

    async def source():
        yield chunk()

    stream = VapiBudgetGuard(g, model="m").guard_stream(source, estimated_cost=0.003)
    with pytest.raises(TypeError):
        await stream.asend("invalid")
    await stream.aclose()
    assert g.remaining_usd == pytest.approx(0.01)
    assert not g._stream_costs
