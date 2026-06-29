"""OpenAI adapter tests that need no ``openai`` install.

The OpenAI SDK response/client shapes are duck-typed with dataclasses, so the
reservation/accrual contract — and the hard-stop (a blocked call never reaches
the client) — is covered even in CI without the optional extra.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from floe_guard import BudgetExceeded, BudgetGuard, UnpriceableModelError, UnpriceableModelWarning
from floe_guard.integrations.openai import (
    _model_from,
    _record_response,
    _usage_from,
    guarded_acompletion,
    guarded_completion,
)


@dataclass
class _Usage:
    prompt_tokens: int
    completion_tokens: int


@dataclass
class _Response:
    model: str
    usage: _Usage


class _Completions:
    """Stub of ``client.chat.completions`` that records whether it was called."""

    def __init__(self, response: _Response) -> None:
        self._response = response
        self.called = False

    def create(self, **kwargs: object) -> _Response:
        self.called = True
        return self._response


class _AsyncCompletions:
    def __init__(self, response: _Response) -> None:
        self._response = response
        self.called = False

    async def create(self, **kwargs: object) -> _Response:
        self.called = True
        return self._response


class _Chat:
    def __init__(self, completions: object) -> None:
        self.completions = completions


class _Client:
    def __init__(self, completions: object) -> None:
        self.chat = _Chat(completions)


def _client(response: _Response) -> _Client:
    return _Client(_Completions(response))


def _async_client(response: _Response) -> _Client:
    return _Client(_AsyncCompletions(response))


def test_usage_from_object_and_dict() -> None:
    assert _usage_from(_Response("gpt-4o", _Usage(5, 7))) == (5, 7)
    assert _usage_from({"usage": {"prompt_tokens": 5, "completion_tokens": 7}}) == (5, 7)


def test_model_from_prefers_response_then_kwargs() -> None:
    # The response's served model wins over the requested alias; the kwarg is
    # only a fallback when the response omits the model.
    resp = _Response("gpt-4o-2024-08-06", _Usage(1, 1))
    assert _model_from({"model": "gpt-4o"}, resp) == "gpt-4o-2024-08-06"
    assert _model_from({"model": "gpt-4o"}, {"usage": {}}) == "gpt-4o"


def test_record_response_accrues() -> None:
    guard = BudgetGuard(limit_usd=1.0)
    resp = _Response("gpt-4o", _Usage(1_000, 1_000))
    _record_response(guard, {}, resp)
    assert guard.spent_usd == pytest.approx(0.0125)


def test_guarded_completion_records_and_calls_client() -> None:
    guard = BudgetGuard(limit_usd=1.0)
    completions = _Completions(_Response("gpt-4o", _Usage(1_000, 1_000)))
    client = _Client(completions)
    resp = guarded_completion(guard, client, model="gpt-4o", messages=[])
    assert completions.called is True
    assert resp.model == "gpt-4o"
    assert guard.spent_usd == pytest.approx(0.0125)


def test_guarded_acompletion_records_and_calls_client() -> None:
    guard = BudgetGuard(limit_usd=1.0)
    completions = _AsyncCompletions(_Response("gpt-4o", _Usage(1_000, 1_000)))
    client = _Client(completions)
    resp = asyncio.run(guarded_acompletion(guard, client, model="gpt-4o", messages=[]))
    assert completions.called is True
    assert resp.model == "gpt-4o"
    assert guard.spent_usd == pytest.approx(0.0125)


def test_hard_stop_blocks_call_before_it_reaches_client() -> None:
    # Spend up to the ceiling, then the next reserve() must raise BEFORE the
    # client is ever called — the request never reaches OpenAI.
    guard = BudgetGuard(limit_usd=0.0125)
    completions = _Completions(_Response("gpt-4o", _Usage(1_000, 1_000)))
    client = _Client(completions)
    guarded_completion(guard, client, model="gpt-4o", messages=[])  # spends exactly the ceiling
    completions.called = False

    with pytest.raises(BudgetExceeded):
        guarded_completion(guard, client, model="gpt-4o", messages=[])
    assert completions.called is False  # blocked before reaching the client


def test_hard_stop_async_blocks_call() -> None:
    guard = BudgetGuard(limit_usd=0.0125)
    completions = _AsyncCompletions(_Response("gpt-4o", _Usage(1_000, 1_000)))
    client = _Client(completions)
    asyncio.run(guarded_acompletion(guard, client, model="gpt-4o", messages=[]))
    completions.called = False

    with pytest.raises(BudgetExceeded):
        asyncio.run(guarded_acompletion(guard, client, model="gpt-4o", messages=[]))
    assert completions.called is False


def test_unpriceable_model_fails_closed() -> None:
    guard = BudgetGuard(limit_usd=1.0)  # fail_closed defaults to True
    resp = _Response("totally-made-up-model", _Usage(1_000, 1_000))
    with pytest.warns(UnpriceableModelWarning):
        with pytest.raises(UnpriceableModelError):
            _record_response(guard, {}, resp)
    assert guard.spent_usd == 0.0


def test_streaming_is_rejected_before_the_call() -> None:
    # A streamed response has no final usage to settle, so it would go unmetered.
    # Reject it up front, before reserving or calling the client.
    guard = BudgetGuard(limit_usd=1.0)
    completions = _Completions(_Response("gpt-4o", _Usage(1, 1)))
    client = _Client(completions)
    with pytest.raises(ValueError, match="stream"):
        guarded_completion(guard, client, model="gpt-4o", messages=[], stream=True)
    assert completions.called is False
    assert guard.spent_usd == 0.0


def test_usageless_response_releases_the_reservation() -> None:
    # A response that reports no usage must free the in-flight reservation, or
    # remaining_usd would shrink permanently for later calls.
    guard = BudgetGuard(limit_usd=1.0)
    _record_response(guard, {}, _Response("gpt-4o", _Usage(0, 0)), reserved=0.5)
    assert guard.spent_usd == 0.0
    assert guard.remaining_usd == pytest.approx(1.0)
