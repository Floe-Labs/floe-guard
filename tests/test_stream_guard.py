"""Tests for request-sized estimates (estimate_call) and mid-stream enforcement
(StreamGuard / guard_stream)."""

from __future__ import annotations

from typing import Any

import pytest

from floe_guard import (
    BudgetExceeded,
    BudgetGuard,
    ManualPrice,
    StreamGuard,
    UnpriceableModelError,
    UnpriceableModelWarning,
    guard_stream,
)
from floe_guard.integrations.langchain import _estimate_start
from floe_guard.integrations.litellm import _estimate_request
from floe_guard.stream import approx_tokens

MODEL = "gpt-4o"  # $2.5e-6/input token, $1e-5/output token


# ── estimate_call ───────────────────────────────────────────────────────────────


def test_estimate_call_prices_the_actual_request() -> None:
    guard = BudgetGuard(limit_usd=1.00)
    est = guard.estimate_call(MODEL, 1_000, 2_000)
    assert est == pytest.approx(1_000 * 2.5e-6 + 2_000 * 1e-5)  # 0.0025 + 0.02


def test_estimate_call_prompt_only_when_no_output_cap() -> None:
    guard = BudgetGuard(limit_usd=1.00)
    assert guard.estimate_call(MODEL, 1_000) == pytest.approx(0.0025)


def test_estimate_call_unpriceable_returns_none() -> None:
    guard = BudgetGuard(limit_usd=1.00)
    assert guard.estimate_call("model-that-does-not-exist", 1_000, 1_000) is None


def test_estimate_call_honors_manual_price() -> None:
    guard = BudgetGuard(limit_usd=1.00)
    price = ManualPrice(input_cost_per_token=1e-6, output_cost_per_token=2e-6)
    est = guard.estimate_call("my-local-model", 1_000, 500, price=price)
    assert est == pytest.approx(1_000 * 1e-6 + 500 * 2e-6)


def test_oversized_first_call_is_blocked_at_its_true_size() -> None:
    # THE acceptance case: a FIRST call (no last-cost baseline) that alone would
    # cross the cap must block pre-call once the reservation is request-sized.
    guard = BudgetGuard(limit_usd=0.01, on_block=lambda s, l: None)
    est = guard.estimate_call(MODEL, 1_000, 100_000)  # ≈ $1.0025 ≫ $0.01
    assert est is not None and est > guard.limit_usd
    with pytest.raises(BudgetExceeded):
        guard.reserve(est)
    # Nothing was held: the budget is untouched for calls that DO fit.
    assert guard.remaining_usd == pytest.approx(guard.limit_usd)
    # The unsized default would have let the same first call through.
    assert guard.reserve() == 0.0


def test_fitting_call_reserves_its_estimated_size() -> None:
    guard = BudgetGuard(limit_usd=1.00)
    est = guard.estimate_call(MODEL, 1_000, 1_000)
    handle = guard.reserve(est)
    assert handle == pytest.approx(0.0125)
    assert guard.remaining_usd == pytest.approx(1.00 - 0.0125)
    guard.release(handle)


# ── adapter estimate wiring ─────────────────────────────────────────────────────


class _FakeLiteLLM:
    """token_counter stub — the only litellm surface _estimate_request touches."""

    def __init__(self, tokens: int = 1_000, raise_on_count: bool = False) -> None:
        self._tokens = tokens
        self._raise = raise_on_count

    def token_counter(self, model: str, messages: Any) -> int:
        if self._raise:
            raise RuntimeError("unknown model")
        return self._tokens


def test_litellm_estimate_prices_model_prompt_and_cap() -> None:
    guard = BudgetGuard(limit_usd=1.00)
    kwargs = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 2_000}
    est = _estimate_request(_FakeLiteLLM(tokens=1_000), guard, kwargs)
    assert est == pytest.approx(0.0025 + 0.02)


def test_litellm_estimate_degrades_to_none() -> None:
    guard = BudgetGuard(limit_usd=1.00)
    assert _estimate_request(_FakeLiteLLM(), guard, {"messages": []}) is None  # no model
    assert _estimate_request(_FakeLiteLLM(), guard, None) is None  # non-dict kwargs
    assert (  # token_counter failure
        _estimate_request(_FakeLiteLLM(raise_on_count=True), guard, {"model": MODEL}) is None
    )
    assert (  # unpriceable model
        _estimate_request(_FakeLiteLLM(), guard, {"model": "model-that-does-not-exist"}) is None
    )


def test_langchain_estimate_uses_serialized_config_and_text_heuristic() -> None:
    guard = BudgetGuard(limit_usd=1.00)
    serialized = {"kwargs": {"model_name": MODEL, "max_tokens": 1_000}}
    est = _estimate_start(guard, serialized, ["x" * 4_000])  # ≈ 1_000 prompt tokens
    assert est == pytest.approx(1_000 * 2.5e-6 + 1_000 * 1e-5)


def test_langchain_estimate_degrades_to_none() -> None:
    guard = BudgetGuard(limit_usd=1.00)
    assert _estimate_start(guard, {}, ["hello"]) is None
    assert _estimate_start(guard, {"kwargs": {}}, ["hello"]) is None
    assert _estimate_start(guard, None, ["hello"]) is None


# ── StreamGuard: mid-stream enforcement ─────────────────────────────────────────

CHUNK = "x" * 40  # 10 tokens by the heuristic → $1e-4 of gpt-4o output


def test_approx_tokens_heuristic() -> None:
    assert approx_tokens("") == 0
    assert approx_tokens("abc") == 1  # non-empty is at least one token
    assert approx_tokens(CHUNK) == 10


def test_runaway_stream_is_cut_off_mid_generation() -> None:
    # $0.01 ceiling ≙ 1_000 gpt-4o output tokens. An endless stream must be
    # aborted mid-flight, not recorded as a big overshoot after the fact.
    guard = BudgetGuard(limit_usd=0.01, on_block=lambda s, l: None)
    sg = StreamGuard(guard, MODEL)
    chunks_fed = 0
    with pytest.raises(BudgetExceeded):
        while True:
            sg.feed_text(CHUNK)
            chunks_fed += 1
    # Cut off at the ceiling (±1 chunk — those tokens had already arrived),
    # instead of running to infinity.
    assert chunks_fed == pytest.approx(100, abs=1)
    assert guard.spent_usd <= guard.limit_usd + 1e-4 + 1e-9
    # The partial spend was settled honestly and hit the ledger.
    (event,) = guard.spend_log
    assert event.completion_tokens == pytest.approx(chunks_fed * 10, abs=10)
    # The guard now blocks everything else, as after any ceiling hit.
    with pytest.raises(BudgetExceeded):
        guard.check()


def test_stream_abort_settles_its_own_reservation() -> None:
    guard = BudgetGuard(limit_usd=0.01, on_block=lambda s, l: None)
    handle = guard.reserve(0.005)
    sg = StreamGuard(guard, MODEL, reserved=handle)
    with pytest.raises(BudgetExceeded):
        while True:
            sg.feed_text(CHUNK)
    # The hold was released by the abort-settle: no phantom reservation left.
    assert guard.remaining_usd == pytest.approx(max(0.0, guard.limit_usd - guard.spent_usd))


def test_finish_reconciles_to_reported_usage() -> None:
    guard = BudgetGuard(limit_usd=1.00)
    sg = StreamGuard(guard, MODEL, prompt_tokens=100)
    sg.feed_text(CHUNK)  # heuristic: 10 tokens
    cost = sg.finish(prompt_tokens=120, completion_tokens=37)  # provider truth
    assert cost == pytest.approx(120 * 2.5e-6 + 37 * 1e-5)
    (event,) = guard.spend_log
    assert (event.prompt_tokens, event.completion_tokens) == (120, 37)


def test_context_manager_settles_partial_spend_on_error() -> None:
    guard = BudgetGuard(limit_usd=1.00)
    handle = guard.reserve(0.01)
    with pytest.raises(RuntimeError, match="network died"):
        with StreamGuard(guard, MODEL, reserved=handle) as sg:
            sg.feed_text(CHUNK)
            raise RuntimeError("network died")
    # The 10 generated tokens were billed by the provider — they are recorded,
    # and the reservation is gone.
    assert guard.spent_usd == pytest.approx(10 * 1e-5)
    assert guard.remaining_usd == pytest.approx(1.00 - guard.spent_usd)


def test_feed_after_settle_is_an_error() -> None:
    guard = BudgetGuard(limit_usd=1.00)
    sg = StreamGuard(guard, MODEL)
    sg.finish(completion_tokens=1)
    with pytest.raises(RuntimeError):
        sg.feed_text(CHUNK)
    with pytest.raises(RuntimeError):
        sg.finish()


def test_stream_guard_rejects_bad_reservation_handles() -> None:
    guard = BudgetGuard(limit_usd=1.00)
    for bad in (float("nan"), float("inf"), -0.01):
        with pytest.raises(ValueError):
            StreamGuard(guard, MODEL, reserved=bad)


def test_unpriceable_stream_fails_closed_before_any_spend() -> None:
    guard = BudgetGuard(limit_usd=1.00)
    handle = guard.reserve(0.01)
    with pytest.warns(UnpriceableModelWarning), pytest.raises(UnpriceableModelError):
        StreamGuard(guard, "model-that-does-not-exist", reserved=handle)
    # The reservation was released — fail-closed must not leak the hold.
    assert guard.remaining_usd == pytest.approx(1.00)


def test_unpriceable_stream_fail_open_passes_through() -> None:
    guard = BudgetGuard(limit_usd=1.00, fail_closed=False)
    sg = StreamGuard(guard, "model-that-does-not-exist")
    sg.feed_text(CHUNK)  # no price to check against — must not raise
    with pytest.warns(UnpriceableModelWarning):
        assert sg.finish() == 0.0
    assert guard.spent_usd == 0.0


# ── guard_stream wrapper ────────────────────────────────────────────────────────


def test_guard_stream_yields_until_the_ceiling_then_raises() -> None:
    guard = BudgetGuard(limit_usd=0.01, on_block=lambda s, l: None)

    def endless() -> Any:
        while True:
            yield CHUNK

    consumed = 0
    with pytest.raises(BudgetExceeded):
        for _ in guard_stream(guard, MODEL, endless()):
            consumed += 1
    # ~100 chunks fit under the $0.01 / 1e-4-per-chunk ceiling; the crossing
    # chunk is metered (its tokens arrived) but never yielded to the consumer.
    assert consumed == pytest.approx(100, abs=1)
    assert len(guard.spend_log) == 1


def test_guard_stream_settles_when_the_consumer_breaks_early() -> None:
    guard = BudgetGuard(limit_usd=1.00)
    stream = guard_stream(guard, MODEL, iter([CHUNK, CHUNK, CHUNK]), label="writer")
    next(stream)
    stream.close()  # consumer abandons the stream mid-way
    (event,) = guard.spend_log
    assert event.completion_tokens == 10  # only the consumed chunk was metered
    assert event.label == "writer"
    assert guard.spent_usd == pytest.approx(10 * 1e-5)


def test_guard_stream_exhaustion_settles_the_accumulated_usage() -> None:
    guard = BudgetGuard(limit_usd=1.00)
    chunks = list(guard_stream(guard, MODEL, iter([CHUNK, CHUNK]), prompt_tokens=50))
    assert chunks == [CHUNK, CHUNK]
    (event,) = guard.spend_log
    assert (event.prompt_tokens, event.completion_tokens) == (50, 20)
    assert guard.spent_usd == pytest.approx(50 * 2.5e-6 + 20 * 1e-5)
