"""VapiBudgetGuard — reserve before the model turn, settle on the real OpenAI
usage, release on error/abort, admit the call via the assistant-request gate,
and price the STT/TTS/telephony legs the custom-LLM proxy never sees.

Fabricated OpenAI-shaped completions / SSE chunk streams drive the adapter — no
Vapi SDK, no `openai` runtime, no network, no keys. The adapter is typed
structurally against the OpenAI wire format Vapi speaks.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from floe_guard import BudgetGuard, ManualPrice
from floe_guard.errors import (
    BudgetExceeded,
    UnpriceableModelError,
    UnpriceableModelWarning,
    UnpriceableVoiceError,
)
from floe_guard.integrations.vapi import VapiBudgetGuard, VapiUsageMissingError
from floe_guard.voice_pricing import price_voice_leg

PRICE = ManualPrice(1e-6, 2e-6)


def completion(prompt_tokens: int, completion_tokens: int) -> dict:
    """An OpenAI-format non-streaming completion carrying a usage block."""
    return {
        "id": "chatcmpl-x",
        "choices": [{"message": {"role": "assistant", "content": "hi"}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


async def sse_stream(
    content_chunks: int,
    usage: dict | None = None,
) -> AsyncIterator[dict]:
    """A fake OpenAI SSE stream: content chunks (no usage), then — if `usage` is
    given — a final empty-choices chunk carrying it, exactly as
    `stream_options:{include_usage:true}` produces. Omit `usage` to model a
    stream WITHOUT include_usage."""
    for _ in range(content_chunks):
        yield {"choices": [{"delta": {"content": "tok"}}]}
    if usage is not None:
        yield {"choices": [], "usage": usage}


async def drain_stream(stream) -> int:
    count = 0
    async for _ in stream:
        count += 1
    return count


def _guard(**overrides) -> BudgetGuard:
    return BudgetGuard(limit_usd=1.00, price_overrides={"m": PRICE, **overrides})


async def _prime_estimate(guard: BudgetGuard, budget: VapiBudgetGuard) -> float:
    """Settle one turn so the guard's next-call estimate becomes non-zero."""
    await budget.guard_completion(lambda: completion(1000, 500), model="m")
    return guard.remaining_usd  # 1.0 - 0.002


# ── reserve before the model turn ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_blocks_over_budget_non_streaming_turn_before_upstream() -> None:
    guard = _guard()
    guard.record_tool("prior", 1.0)  # spend the ceiling
    budget = VapiBudgetGuard(guard, model="m")

    ran = False

    async def run():
        nonlocal ran
        ran = True
        return completion(1000, 500)

    with pytest.raises(BudgetExceeded):
        await budget.guard_completion(run, model="m")
    assert ran is False  # upstream never reached


def test_blocks_over_budget_streaming_turn_synchronously() -> None:
    guard = _guard()
    guard.record_tool("prior", 1.0)
    budget = VapiBudgetGuard(guard, model="m")

    opened = False

    def run():
        nonlocal opened
        opened = True
        return sse_stream(3, {"prompt_tokens": 10, "completion_tokens": 5})

    # guard_stream reserves eagerly, so the block raises right here — not lazily
    # on first pull. The handler learns before piping anything to Vapi.
    with pytest.raises(BudgetExceeded):
        budget.guard_stream(run, model="m")
    assert opened is False


@pytest.mark.asyncio
async def test_guard_completion_and_stream_require_a_model() -> None:
    guard = BudgetGuard(limit_usd=1.00)
    budget = VapiBudgetGuard(guard)  # no default model

    with pytest.raises(ValueError, match="no model"):
        await budget.guard_completion(lambda: completion(1, 1), model=None)

    with pytest.raises(ValueError, match="no model"):
        budget.guard_stream(lambda: sse_stream(1), model=None)


# ── settle on real usage ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_settles_priced_cost_from_completion_and_frees_hold() -> None:
    guard = _guard()
    budget = VapiBudgetGuard(guard, model="m")

    out = await budget.guard_completion(lambda: completion(1000, 500), model="m")
    assert out["usage"]["completion_tokens"] == 500  # returned untouched

    # 1000 * 1e-6 + 500 * 2e-6 = 0.002
    assert guard.advisory().spent_usd == pytest.approx(0.002)
    assert guard.remaining_usd == pytest.approx(1.0 - 0.002)  # hold consumed, not leaked
    assert len(guard.spend_log) == 1
    assert guard.spend_log[0].model_or_tool == "m"


@pytest.mark.asyncio
async def test_settles_stream_on_final_chunk_usage_and_passes_chunks() -> None:
    guard = _guard()
    budget = VapiBudgetGuard(guard, model="m")

    stream = budget.guard_stream(
        lambda: sse_stream(4, {"prompt_tokens": 1000, "completion_tokens": 500}), model="m"
    )
    yielded = await drain_stream(stream)

    assert yielded == 5  # 4 content chunks + 1 usage chunk, all forwarded
    assert guard.advisory().spent_usd == pytest.approx(0.002)
    assert guard.remaining_usd == pytest.approx(1.0 - 0.002)


@pytest.mark.asyncio
async def test_per_call_model_wins_over_constructor_default() -> None:
    guard = BudgetGuard(limit_usd=1.00, price_overrides={"req-model": PRICE})
    budget = VapiBudgetGuard(guard)  # no default model

    await budget.guard_completion(lambda: completion(1000, 0), model="req-model")
    assert guard.spend_log[0].model_or_tool == "req-model"


@pytest.mark.asyncio
async def test_completion_may_be_a_sync_source() -> None:
    guard = _guard()
    budget = VapiBudgetGuard(guard, model="m")

    await budget.guard_completion(lambda: completion(100, 50), model="m")
    assert guard.advisory().spent_usd == pytest.approx(100 * 1e-6 + 50 * 2e-6)


# ── release on error / abort ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_releases_hold_when_upstream_throws() -> None:
    guard = _guard()
    budget = VapiBudgetGuard(guard, model="m")
    after_turn1 = await _prime_estimate(guard, budget)

    async def boom():
        raise RuntimeError("upstream 500")

    with pytest.raises(RuntimeError, match="upstream 500"):
        await budget.guard_completion(boom, model="m")
    assert guard.remaining_usd == pytest.approx(after_turn1)  # hold released
    assert guard.advisory().spent_usd == pytest.approx(0.002)  # no new spend


@pytest.mark.asyncio
async def test_releases_hold_when_caller_aborts_stream_early() -> None:
    guard = _guard()
    budget = VapiBudgetGuard(guard, model="m")
    after_turn1 = await _prime_estimate(guard, budget)

    stream = budget.guard_stream(
        lambda: sse_stream(10, {"prompt_tokens": 1000, "completion_tokens": 500}), model="m"
    )
    # Breaking off the async-for and closing the generator unwinds into its
    # finally, which releases the still-open hold. (In Python a bare break
    # finalizes the generator only on a later loop iteration — close it
    # explicitly so the hold clears deterministically.)
    async for _ in stream:
        break
    await stream.aclose()

    assert guard.remaining_usd == pytest.approx(after_turn1)
    assert guard.advisory().spent_usd == pytest.approx(0.002)  # aborted turn metered nothing


# ── missing usage fails loudly ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_missing_usage_completion_fails_loudly_and_releases() -> None:
    guard = _guard()
    budget = VapiBudgetGuard(guard, model="m")
    after_turn1 = await _prime_estimate(guard, budget)

    with pytest.raises(VapiUsageMissingError, match="no token usage"):
        await budget.guard_completion(lambda: {"usage": None}, model="m")
    assert guard.remaining_usd == pytest.approx(after_turn1)  # released, not metered at $0
    assert guard.advisory().spent_usd == pytest.approx(0.002)


@pytest.mark.asyncio
async def test_missing_usage_stream_fails_loudly() -> None:
    guard = _guard()
    budget = VapiBudgetGuard(guard, model="m")

    stream = budget.guard_stream(lambda: sse_stream(3), model="m")  # no usage chunk
    with pytest.raises(VapiUsageMissingError):
        await drain_stream(stream)
    assert guard.advisory().spent_usd == 0.0  # nothing metered
    assert guard.remaining_usd == pytest.approx(1.0)  # hold released


# ── assistant-request admission via gates.vapi ────────────────────────────────


def test_assistant_request_admits_with_assistant_id() -> None:
    guard = BudgetGuard(limit_usd=1.00)
    budget = VapiBudgetGuard(guard)
    assert budget.assistant_request(assistant_id="asst_123") == {"assistantId": "asst_123"}


def test_assistant_request_rejects_exhausted_call_with_spoken_error() -> None:
    guard = BudgetGuard(limit_usd=1.00)
    guard.record_tool("prior", 1.0)
    budget = VapiBudgetGuard(guard)
    assert budget.assistant_request(assistant_id="asst_123", error_message="Out of budget.") == {
        "error": "Out of budget."
    }


# ── voice legs price via the cost map ──────────────────────────────────────────


def test_meters_stt_tts_telephony_from_named_vendors() -> None:
    guard = BudgetGuard(limit_usd=10.00)
    budget = VapiBudgetGuard(
        guard,
        stt_model="deepgram-nova-3",
        tts_model="elevenlabs-flash-v2.5",
        telephony="twilio-us-inbound-local",
    )

    stt = budget.meter_stt(30)  # 30s
    tts = budget.meter_tts(1_200)  # 1.2k chars
    tel = budget.meter_telephony(2.5)  # 2.5 min

    assert stt == pytest.approx(price_voice_leg("stt", 30, model="deepgram-nova-3"))
    assert tts == pytest.approx(price_voice_leg("tts", 1_200, model="elevenlabs-flash-v2.5"))
    assert tel == pytest.approx(price_voice_leg("telephony", 2.5, model="twilio-us-inbound-local"))
    assert guard.tool_costs["vapi-stt"] == pytest.approx(stt)
    assert guard.tool_costs["vapi-tts"] == pytest.approx(tts)
    assert guard.tool_costs["vapi-telephony"] == pytest.approx(tel)


def test_unconfigured_leg_is_unmetered() -> None:
    guard = BudgetGuard(limit_usd=10.00)
    budget = VapiBudgetGuard(guard)  # no voice vendors configured
    assert budget.meter_stt(30) is None
    assert budget.meter_tts(30) is None
    assert budget.meter_telephony(30) is None
    assert guard.advisory().spent_usd == 0.0


@pytest.mark.parametrize(
    "kwarg,mode",
    [("stt_model", "stt"), ("tts_model", "tts"), ("telephony", "telephony")],
)
def test_fails_closed_on_unpriceable_vendor(kwarg: str, mode: str) -> None:
    guard = BudgetGuard(limit_usd=10.00)
    budget = VapiBudgetGuard(guard, **{kwarg: "no-such-vendor"})

    with pytest.raises(UnpriceableVoiceError):
        getattr(budget, f"meter_{mode}")(1)
    assert guard.advisory().spent_usd == 0.0


# ── no double release ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_double_release_when_streaming_settle_throws() -> None:
    guard = BudgetGuard(limit_usd=1.00)  # fail_closed default; "mystery-model" unpriceable
    guard.record_tool("prior", 0.1)  # non-zero next-call estimate → the reserve holds 0.10
    before = guard.remaining_usd  # 0.90
    budget = VapiBudgetGuard(guard, model="mystery-model")

    async def source():
        yield {"usage": {"prompt_tokens": 100, "completion_tokens": 50}}

    stream = budget.guard_stream(source, model="mystery-model")
    # Draining triggers settle("mystery-model", …) → UnpriceableModelError, which
    # releases the reservation ITSELF. The finally must not release it again — a
    # double release would drive `reserved` negative and inflate remaining_usd.
    with pytest.warns(UnpriceableModelWarning), pytest.raises(UnpriceableModelError):
        await drain_stream(stream)
    assert guard.remaining_usd == pytest.approx(before)  # released exactly once


@pytest.mark.asyncio
async def test_malformed_usage_is_treated_as_missing() -> None:
    # A partial or non-numeric usage block is no usage at all (fail-closed).
    guard = _guard()
    budget = VapiBudgetGuard(guard, model="m")
    after_turn1 = await _prime_estimate(guard, budget)

    with pytest.raises(VapiUsageMissingError):
        await budget.guard_completion(lambda: {"usage": {"prompt_tokens": "lots"}})
    assert guard.remaining_usd == pytest.approx(after_turn1)


@pytest.mark.asyncio
async def test_no_double_release_when_completion_settle_throws() -> None:
    # Same invariant as the streaming variant: guard.settle releases the
    # reservation on its own unpriceable-model failure path, and guard_completion
    # has no finally of its own - the hold is released exactly once.
    guard = BudgetGuard(limit_usd=1.00)  # fail_closed default; "mystery-model" unpriceable
    guard.record_tool("prior", 0.1)  # non-zero next-call estimate -> the reserve holds 0.10
    before = guard.remaining_usd  # 0.90
    budget = VapiBudgetGuard(guard, model="mystery-model")

    with pytest.warns(UnpriceableModelWarning), pytest.raises(UnpriceableModelError):
        await budget.guard_completion(lambda: completion(100, 50), model="mystery-model")
    assert guard.remaining_usd == pytest.approx(before)  # released exactly once


@pytest.mark.asyncio
async def test_releases_hold_when_stream_closed_before_consumption() -> None:
    # A generator's finally cannot run before it is started, so closing a never-
    # consumed guarded stream would leak its hold. The returned wrapper closes
    # that gap deterministically on aclose().
    guard = _guard()
    budget = VapiBudgetGuard(guard, model="m")
    after_turn1 = await _prime_estimate(guard, budget)  # next-call estimate 0.002

    stream = budget.guard_stream(lambda: sse_stream(5), model="m")
    assert guard.remaining_usd == pytest.approx(after_turn1 - 0.002)  # held

    await stream.aclose()  # never consumed - released eagerly
    assert guard.remaining_usd == pytest.approx(after_turn1)


@pytest.mark.asyncio
async def test_closing_a_finished_stream_does_not_double_release() -> None:
    guard = _guard()
    budget = VapiBudgetGuard(guard, model="m")

    stream = budget.guard_stream(
        lambda: sse_stream(2, {"prompt_tokens": 1000, "completion_tokens": 500}), model="m"
    )
    await drain_stream(stream)  # consumed to completion -> settled
    assert guard.remaining_usd == pytest.approx(1.0 - 0.002)

    await stream.aclose()  # already finished - must not release the settled hold
    assert guard.remaining_usd == pytest.approx(1.0 - 0.002)


@pytest.mark.asyncio
async def test_no_double_release_when_first_stream_pull_raises() -> None:
    guard = _guard()
    budget = VapiBudgetGuard(guard, model="m")
    after_turn1 = await _prime_estimate(guard, budget)
    before = guard.remaining_usd

    async def source():
        raise RuntimeError("pull boom")
        yield {"choices": []}

    stream = budget.guard_stream(source, model="m")
    assert guard.remaining_usd == pytest.approx(before - 0.002)

    with pytest.raises(RuntimeError, match="pull boom"):
        async for _ in stream:
            pass
    assert guard.remaining_usd == pytest.approx(after_turn1)

    await stream.aclose()
    assert guard.remaining_usd == pytest.approx(after_turn1)


@pytest.mark.asyncio
async def test_no_double_release_when_first_stream_pull_is_usage_less_and_empty() -> None:
    guard = _guard()
    budget = VapiBudgetGuard(guard, model="m")
    after_turn1 = await _prime_estimate(guard, budget)
    before = guard.remaining_usd

    async def source():
        if False:
            yield {}

    stream = budget.guard_stream(source, model="m")
    assert guard.remaining_usd == pytest.approx(before - 0.002)

    with pytest.raises(VapiUsageMissingError):
        await drain_stream(stream)
    assert guard.remaining_usd == pytest.approx(after_turn1)

    await stream.aclose()
    assert guard.remaining_usd == pytest.approx(after_turn1)


@pytest.mark.asyncio
async def test_garbage_collection_releases_hold_for_unstarted_stream() -> None:
    guard = _guard()
    budget = VapiBudgetGuard(guard, model="m")
    after_turn1 = await _prime_estimate(guard, budget)
    before = guard.remaining_usd

    stream = budget.guard_stream(lambda: sse_stream(5), model="m")
    assert guard.remaining_usd == pytest.approx(before - 0.002)

    del stream
    import gc
    gc.collect()

    assert guard.remaining_usd == pytest.approx(after_turn1)


@pytest.mark.asyncio
async def test_malformed_usage_values_rejected() -> None:
    guard = _guard()
    budget = VapiBudgetGuard(guard, model="m")
    await _prime_estimate(guard, budget)

    with pytest.raises(VapiUsageMissingError):
        await budget.guard_completion(
            lambda: {"usage": {"prompt_tokens": -10, "completion_tokens": 5}}
        )

    with pytest.raises(VapiUsageMissingError):
        await budget.guard_completion(
            lambda: {"usage": {"prompt_tokens": 10, "completion_tokens": 5.5}}
        )
