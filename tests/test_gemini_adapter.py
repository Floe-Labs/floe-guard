"""Gemini adapter tests that need no ``google-genai`` install.

The SDK's client/response shapes are duck-typed with dataclasses, so the
reservation/accrual contract — the hard-stop (a blocked call never reaches the
client), the five-bucket token mapping, and the Vertex refusal — is covered even
in CI without the optional extra.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from floe_guard import (
    BudgetExceeded,
    BudgetGuard,
    ManualPrice,
    UnpriceableModelError,
    UnpriceableModelWarning,
)
from floe_guard.integrations.gemini import (
    _settle_model,
    _usage_from,
    guarded_acompletion,
    guarded_completion,
)

MODEL = "gemini-2.5-flash"
PRICE = ManualPrice(1e-6, 2e-6)


@dataclass
class _Usage:
    prompt_token_count: int = 0
    candidates_token_count: int = 0
    cached_content_token_count: int | None = None
    thoughts_token_count: int | None = None
    tool_use_prompt_token_count: int | None = None


@dataclass
class _Response:
    model_version: str = MODEL
    usage_metadata: _Usage | None = field(default_factory=_Usage)


class _Models:
    """Stands in for ``client.models`` / ``client.aio.models``."""

    def __init__(self, response: object = None, raises: BaseException | None = None) -> None:
        self._response = response
        self._raises = raises
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self._raises is not None:
            raise self._raises
        return self._response


class _AsyncModels(_Models):
    async def generate_content(self, **kwargs):  # type: ignore[override]
        return super().generate_content(**kwargs)


class _Aio:
    def __init__(self, models: _Models) -> None:
        self.models = models


class _Client:
    """Stands in for ``google.genai.Client``."""

    def __init__(
        self,
        response: object = None,
        *,
        vertexai: bool = False,
        raises: BaseException | None = None,
        is_async: bool = False,
    ) -> None:
        self.vertexai = vertexai
        models = (_AsyncModels if is_async else _Models)(response, raises)
        self.models = models
        self.aio = _Aio(models)


def _guard(limit: float = 1.0, **kw) -> BudgetGuard:
    kw.setdefault("price_overrides", {MODEL: PRICE})
    return BudgetGuard(limit_usd=limit, **kw)


# ── token bucket mapping ──────────────────────────────────────────────────────


def test_usage_maps_plain_prompt_and_candidates() -> None:
    prompt, completion, cached = _usage_from(
        _Response(usage_metadata=_Usage(prompt_token_count=100, candidates_token_count=40))
    )
    assert (prompt, completion, cached) == (100, 40, 0)


def test_thinking_tokens_count_as_output() -> None:
    # thoughts_token_count is billed as output and is NOT part of
    # candidates_token_count — dropping it would under-meter every thinking model.
    prompt, completion, _ = _usage_from(
        _Response(
            usage_metadata=_Usage(
                prompt_token_count=100, candidates_token_count=40, thoughts_token_count=250
            )
        )
    )
    assert (prompt, completion) == (100, 290)


def test_tool_use_tokens_count_as_input() -> None:
    # tool_use_prompt_token_count is input and is NOT part of prompt_token_count
    # (the SDK documents total = prompt + candidates + tool_use + thoughts).
    prompt, completion, _ = _usage_from(
        _Response(
            usage_metadata=_Usage(
                prompt_token_count=100, candidates_token_count=40, tool_use_prompt_token_count=30
            )
        )
    )
    assert (prompt, completion) == (130, 40)


def test_cached_tokens_are_carved_out_of_the_prompt_not_added() -> None:
    # prompt_token_count INCLUDES cached tokens, so the cached share is subtracted
    # and re-priced at the cheaper cache-read rate. Charging both would bill the
    # cached tokens twice.
    prompt, _, cached = _usage_from(
        _Response(
            usage_metadata=_Usage(
                prompt_token_count=1000,
                candidates_token_count=40,
                cached_content_token_count=800,
            )
        )
    )
    assert (prompt, cached) == (200, 800)


def test_unset_buckets_are_treated_as_zero() -> None:
    # Gemini leaves inapplicable buckets as None rather than 0.
    assert _usage_from(_Response(usage_metadata=_Usage(prompt_token_count=10))) == (10, 0, 0)
    assert _usage_from(_Response(usage_metadata=None)) == (0, 0, 0)


def test_usage_reads_a_plain_dict_response() -> None:
    response = {
        "model_version": MODEL,
        "usage_metadata": {"prompt_token_count": 10, "candidates_token_count": 5},
    }
    assert _usage_from(response) == (10, 5, 0)


# ── reserve / settle contract ─────────────────────────────────────────────────


def test_records_spend_and_returns_response() -> None:
    guard = _guard()
    response = _Response(usage_metadata=_Usage(prompt_token_count=1000, candidates_token_count=500))
    client = _Client(response)

    assert guarded_completion(guard, client, model=MODEL, contents="hi") is response
    # 1000 * 1e-6 + 500 * 2e-6
    assert guard.spent_usd == pytest.approx(0.002)
    assert guard.remaining_usd == pytest.approx(0.998)
    assert len(guard.spend_log) == 1


def test_cached_tokens_are_priced_at_the_cache_read_rate() -> None:
    guard = _guard()
    client = _Client(
        _Response(
            usage_metadata=_Usage(
                prompt_token_count=1000,
                candidates_token_count=0,
                cached_content_token_count=800,
            )
        )
    )
    guarded_completion(guard, client, model=MODEL, contents="hi")
    # 200 fresh input at 1e-6, plus 800 cached at 0.1x — NOT 1000 at full rate.
    assert guard.spent_usd == pytest.approx(200 * 1e-6 + 800 * 1e-6 * 0.10)


def test_blocked_call_never_reaches_the_client() -> None:
    guard = _guard(limit=0.0)
    client = _Client(_Response())

    with pytest.raises(BudgetExceeded):
        guarded_completion(guard, client, model=MODEL, contents="hi")
    assert client.models.calls == []


def test_exception_releases_the_reservation() -> None:
    guard = _guard()
    client = _Client(raises=RuntimeError("boom"))

    with pytest.raises(RuntimeError):
        guarded_completion(guard, client, model=MODEL, contents="hi")
    assert guard.spent_usd == 0.0
    assert guard.remaining_usd == pytest.approx(1.0)  # no leaked hold


def test_usageless_response_releases_the_reservation() -> None:
    guard = _guard()
    client = _Client(_Response(usage_metadata=None))

    guarded_completion(guard, client, model=MODEL, contents="hi")
    assert guard.spent_usd == 0.0
    assert guard.remaining_usd == pytest.approx(1.0)


def test_async_records_spend() -> None:
    guard = _guard()
    client = _Client(
        _Response(usage_metadata=_Usage(prompt_token_count=1000, candidates_token_count=500)),
        is_async=True,
    )

    asyncio.run(guarded_acompletion(guard, client, model=MODEL, contents="hi"))
    assert guard.spent_usd == pytest.approx(0.002)


def test_async_blocked_call_never_reaches_the_client() -> None:
    guard = _guard(limit=0.0)
    client = _Client(_Response(), is_async=True)

    with pytest.raises(BudgetExceeded):
        asyncio.run(guarded_acompletion(guard, client, model=MODEL, contents="hi"))
    assert client.models.calls == []


# ── served vs requested model ─────────────────────────────────────────────────


def test_unpriceable_served_id_falls_back_to_the_priceable_request_alias() -> None:
    guard = _guard()
    response = _Response(model_version="gemini-2.5-flash-brand-new-snapshot")
    assert _settle_model(guard, {"model": MODEL}, response) == MODEL


def test_served_id_wins_when_it_prices() -> None:
    guard = BudgetGuard(limit_usd=1.0, price_overrides={"gemini-2.0-flash": PRICE, MODEL: PRICE})
    response = _Response(model_version="gemini-2.0-flash")
    assert _settle_model(guard, {"model": MODEL}, response) == "gemini-2.0-flash"


# ── Vertex refusal ────────────────────────────────────────────────────────────


def test_vertex_client_fails_closed_before_the_call() -> None:
    # The map prices AI Studio rates; Vertex bills the same ids differently, so a
    # Vertex call without an explicit price would under-meter. Refuse it, and do
    # so BEFORE the request reaches Google or any budget is held.
    guard = BudgetGuard(limit_usd=1.0)
    client = _Client(_Response(), vertexai=True)

    with pytest.warns(UnpriceableModelWarning, match="Vertex"):
        with pytest.raises(UnpriceableModelError):
            guarded_completion(guard, client, model=MODEL, contents="hi")
    assert client.models.calls == []
    assert guard.remaining_usd == pytest.approx(1.0)  # nothing reserved


def test_vertex_client_proceeds_when_the_caller_supplies_a_price() -> None:
    # price_overrides is the documented way to meter Vertex — the caller's rates
    # win, so there is nothing left to refuse.
    guard = _guard()
    client = _Client(
        _Response(usage_metadata=_Usage(prompt_token_count=1000, candidates_token_count=500)),
        vertexai=True,
    )

    guarded_completion(guard, client, model=MODEL, contents="hi")
    assert guard.spent_usd == pytest.approx(0.002)


def test_vertex_client_warns_and_proceeds_when_fail_open() -> None:
    # fail_closed=False is an explicit opt-in to un-enforced spend; the warning
    # still fires so the under-metering is never silent.
    guard = BudgetGuard(limit_usd=1.0, fail_closed=False)
    client = _Client(
        _Response(usage_metadata=_Usage(prompt_token_count=1000, candidates_token_count=500)),
        vertexai=True,
    )

    with pytest.warns(UnpriceableModelWarning, match="Vertex"):
        guarded_completion(guard, client, model=MODEL, contents="hi")
    assert client.models.calls  # the call ran


def test_ai_studio_client_is_not_refused() -> None:
    guard = BudgetGuard(limit_usd=1.0)
    client = _Client(
        _Response(usage_metadata=_Usage(prompt_token_count=1000, candidates_token_count=500)),
        vertexai=False,
    )

    guarded_completion(guard, client, model=MODEL, contents="hi")
    assert guard.spent_usd > 0
