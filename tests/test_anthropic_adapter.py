"""Anthropic adapter tests that need no ``anthropic`` install.

The Anthropic SDK response/client shapes are duck-typed with dataclasses. These
cover the input/output -> prompt/completion usage mapping, the accrual contract,
and the hard-stop (a blocked call never reaches the client) without the extra.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from floe_guard import BudgetExceeded, BudgetGuard, UnpriceableModelError, UnpriceableModelWarning
from floe_guard.integrations.anthropic import (
    _model_from,
    _record_response,
    _settle_model,
    _usage_from,
    guarded_acompletion,
    guarded_completion,
)


@dataclass
class _Usage:
    input_tokens: int
    output_tokens: int


@dataclass
class _Response:
    model: str
    usage: _Usage


class _Messages:
    """Stub of ``client.messages`` that records whether it was called."""

    def __init__(self, response: _Response) -> None:
        self._response = response
        self.called = False

    def create(self, **kwargs: object) -> _Response:
        self.called = True
        return self._response


class _AsyncMessages:
    def __init__(self, response: _Response) -> None:
        self._response = response
        self.called = False

    async def create(self, **kwargs: object) -> _Response:
        self.called = True
        return self._response


class _Client:
    def __init__(self, messages: object) -> None:
        self.messages = messages


_MODEL = "claude-3-7-sonnet-20250219"  # present in the bundled cost map


def test_usage_maps_input_output_to_prompt_completion() -> None:
    # The Anthropic-specific bit: input_tokens -> prompt, output_tokens -> completion, plus cache buckets
    assert _usage_from(_Response(_MODEL, _Usage(5, 7))) == (5, 7, 0, 0)
    assert _usage_from({"usage": {"input_tokens": 5, "output_tokens": 7}}) == (5, 7, 0, 0)


def test_usage_extracts_prompt_cache_tokens() -> None:
    # Cached calls must return their buckets unmodified so the core pricing engine can multiply them.
    usage = {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_creation_input_tokens": 200,
        "cache_read_input_tokens": 1000,
    }
    assert _usage_from({"usage": usage}) == (100, 50, 200, 1000)
    # No cache fields → unchanged (the object path getattr-defaults to 0).
    assert _usage_from(_Response(_MODEL, _Usage(5, 7))) == (5, 7, 0, 0)


def test_model_from_prefers_response_then_kwargs() -> None:
    # The response's served model wins; the kwarg is only a fallback.
    resp = _Response(_MODEL, _Usage(1, 1))
    assert _model_from({"model": "claude-3-haiku-20240307"}, resp) == _MODEL
    assert _model_from({"model": _MODEL}, {"usage": {}}) == _MODEL


def test_settle_model_falls_back_to_priceable_alias() -> None:
    guard = BudgetGuard(limit_usd=10.0)
    # Served snapshot isn't in the bundled cost map, but the requested alias is:
    # settle on the alias so the call isn't fail-closed for a stale map.
    unpriced_served = _Response("claude-3-7-sonnet-20260101", _Usage(1, 1))
    assert _settle_model(guard, {"model": _MODEL}, unpriced_served) == _MODEL
    # Served id IS priceable → it wins (source of truth).
    priced_served = _Response(_MODEL, _Usage(1, 1))
    assert _settle_model(guard, {"model": "claude-3-haiku-20240307"}, priced_served) == _MODEL


def test_record_response_accrues() -> None:
    guard = BudgetGuard(limit_usd=10.0)
    resp = _Response(_MODEL, _Usage(1_000, 1_000))
    _record_response(guard, {}, resp)
    assert guard.spent_usd > 0.0  # priced from the bundled cost map


def test_guarded_completion_records_and_calls_client() -> None:
    guard = BudgetGuard(limit_usd=10.0)
    messages = _Messages(_Response(_MODEL, _Usage(1_000, 1_000)))
    client = _Client(messages)
    resp = guarded_completion(guard, client, model=_MODEL, max_tokens=64, messages=[])
    assert messages.called is True
    assert resp.model == _MODEL
    assert guard.spent_usd > 0.0


def test_guarded_acompletion_records_and_calls_client() -> None:
    guard = BudgetGuard(limit_usd=10.0)
    messages = _AsyncMessages(_Response(_MODEL, _Usage(1_000, 1_000)))
    client = _Client(messages)
    asyncio.run(guarded_acompletion(guard, client, model=_MODEL, max_tokens=64, messages=[]))
    assert messages.called is True
    assert guard.spent_usd > 0.0


def test_hard_stop_blocks_call_before_it_reaches_client() -> None:
    # First, price one call to learn its cost, then set a guard whose ceiling is
    # exactly that cost: the first call spends it, the second must block before
    # the client is reached.
    probe = BudgetGuard(limit_usd=10.0)
    _record_response(probe, {}, _Response(_MODEL, _Usage(1_000, 1_000)))
    one_call = probe.spent_usd

    guard = BudgetGuard(limit_usd=one_call)
    messages = _Messages(_Response(_MODEL, _Usage(1_000, 1_000)))
    client = _Client(messages)
    guarded_completion(
        guard, client, model=_MODEL, max_tokens=64, messages=[]
    )  # spends the ceiling
    messages.called = False

    with pytest.raises(BudgetExceeded):
        guarded_completion(guard, client, model=_MODEL, max_tokens=64, messages=[])
    assert messages.called is False  # blocked before reaching the client


def test_hard_stop_async_blocks_call() -> None:
    probe = BudgetGuard(limit_usd=10.0)
    _record_response(probe, {}, _Response(_MODEL, _Usage(1_000, 1_000)))
    one_call = probe.spent_usd

    guard = BudgetGuard(limit_usd=one_call)
    messages = _AsyncMessages(_Response(_MODEL, _Usage(1_000, 1_000)))
    client = _Client(messages)
    asyncio.run(guarded_acompletion(guard, client, model=_MODEL, max_tokens=64, messages=[]))
    messages.called = False

    with pytest.raises(BudgetExceeded):
        asyncio.run(guarded_acompletion(guard, client, model=_MODEL, max_tokens=64, messages=[]))
    assert messages.called is False


def test_unpriceable_model_fails_closed() -> None:
    guard = BudgetGuard(limit_usd=1.0)  # fail_closed defaults to True
    resp = _Response("totally-made-up-model", _Usage(1_000, 1_000))
    with pytest.warns(UnpriceableModelWarning):
        with pytest.raises(UnpriceableModelError):
            _record_response(guard, {}, resp)
    assert guard.spent_usd == 0.0


def test_streaming_is_rejected_before_the_call() -> None:
    guard = BudgetGuard(limit_usd=10.0)
    messages = _Messages(_Response(_MODEL, _Usage(1, 1)))
    client = _Client(messages)
    with pytest.raises(ValueError, match="stream"):
        guarded_completion(guard, client, model=_MODEL, max_tokens=64, messages=[], stream=True)
    assert messages.called is False
    assert guard.spent_usd == 0.0


def test_usageless_response_releases_the_reservation() -> None:
    guard = BudgetGuard(limit_usd=1.0)
    _record_response(guard, {}, _Response(_MODEL, _Usage(0, 0)), reserved=0.5)
    assert guard.spent_usd == 0.0
    assert guard.remaining_usd == pytest.approx(1.0)
