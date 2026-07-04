"""Anthropic Python SDK adapter (optional extra: ``pip install floe-guard[anthropic]``).

:func:`guarded_completion` / :func:`guarded_acompletion` wrap a call to the
Anthropic SDK's ``client.messages.create`` with a pre-call budget reservation
and post-call accrual. This is the **guaranteed** enforcement path: the guard
reserves the call's budget before the request, so a blocked call never reaches
Anthropic.

Like the OpenAI adapter there is no callback variant — wrapping the call site is
the enforcement surface. The contract matches the other adapters: ``reserve()``
before the call, ``settle(model, prompt_tokens, completion_tokens, reserved=...)``
after, ``release()`` on exception. Reserving before the await lets parallel calls
each hold their own slice of the ceiling (issue #18).

The one Anthropic-specific bit: usage is reported as ``usage.input_tokens`` /
``usage.output_tokens``, which we map onto the ``(prompt_tokens, completion_tokens)``
shape the guard settles on.
"""

from __future__ import annotations

from typing import Any

from ..guard import BudgetGuard
from ..pricing import resolve_price

# Anthropic prompt-cache pricing multipliers, relative to the base input rate:
# 5-minute cache writes bill at ~1.25x, cache reads at ~0.1x. See _usage_from.
_CACHE_WRITE_WEIGHT = 1.25
_CACHE_READ_WEIGHT = 0.1


def _model_from(kwargs: dict[str, Any], response: Any) -> str:
    # Prefer the model the response was actually served by, falling back to the
    # request's model= only when the response omits it.
    model = getattr(response, "model", None)
    if model is None and isinstance(response, dict):
        model = response.get("model")
    if not model:
        model = kwargs.get("model")
    return str(model or "")


def _usage_from(response: Any) -> tuple[int, int]:
    """Map an Anthropic message's usage onto (prompt_tokens, completion_tokens).

    Anthropic reports ``usage.input_tokens`` / ``usage.output_tokens``; the guard
    settles on the OpenAI-style ``(prompt_tokens, completion_tokens)`` pair, so
    input maps to prompt and output to completion.

    Prompt caching: ``input_tokens`` counts only the *fresh* prompt tokens —
    cached tokens are reported separately as ``cache_creation_input_tokens``
    (billed ~1.25x base input) and ``cache_read_input_tokens`` (~0.1x). Dropping
    them would under-meter a cached call, and a budget guard must never
    under-count. Since the guard prices with a flat per-model input rate, we fold
    the cache tokens into the prompt bucket at their *effective* weight so the
    metered cost approximates the real spend (an estimate, not exact cache-aware
    pricing — safe-direction).
    """
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return 0, 0
    get = usage.get if isinstance(usage, dict) else lambda k, d=0: getattr(usage, k, d)
    input_tokens = int(get("input_tokens", 0) or 0)
    output_tokens = int(get("output_tokens", 0) or 0)
    cache_write = int(get("cache_creation_input_tokens", 0) or 0)
    cache_read = int(get("cache_read_input_tokens", 0) or 0)
    prompt_tokens = (
        input_tokens
        + round(cache_write * _CACHE_WRITE_WEIGHT)
        + round(cache_read * _CACHE_READ_WEIGHT)
    )
    return prompt_tokens, output_tokens


def _settle_model(guard: BudgetGuard, kwargs: dict[str, Any], response: Any) -> str:
    """Pick the model id to settle against.

    The served model (``response.model``) is the source of truth (issue #23). But
    if that served id can't be priced and the requested alias can, settle on the
    alias instead, so a provider snapshot newer than the bundled cost map doesn't
    fail-closed a call that would otherwise price cleanly. If neither prices, keep
    the served id so the guard still fail-closes on a real id.
    """
    served = _model_from(kwargs, response)
    requested = str(kwargs.get("model") or "")
    if (
        served
        and requested
        and served != requested
        and resolve_price(served, guard.price_overrides) is None
        and resolve_price(requested, guard.price_overrides) is not None
    ):
        return requested
    return served


def _record_response(
    guard: BudgetGuard, kwargs: Any, response: Any, *, reserved: float = 0.0
) -> None:
    if not isinstance(kwargs, dict):
        kwargs = {}
    model = _settle_model(guard, kwargs, response)
    prompt_tokens, completion_tokens = _usage_from(response)
    if prompt_tokens <= 0 and completion_tokens <= 0:
        # No tokens spent — free the reservation.
        guard.release(reserved)
        return
    # There IS spend to account for. Route it through settle() even when the
    # model id is missing, so the guard's policy applies (fail-closed → warn +
    # raise; fail-open → warn + skip) rather than letting a completed call go
    # unmetered and skew the next check().
    guard.settle(model, prompt_tokens, completion_tokens, reserved=reserved)


def _reject_streaming(kwargs: dict[str, Any]) -> None:
    if kwargs.get("stream"):
        raise ValueError(
            "floe-guard's Anthropic adapter does not support stream=True: a streamed "
            "response has no final usage to meter, so the call would go unaccounted. "
            "Use a non-streaming call, or meter the stream yourself with "
            "guard.reserve()/guard.settle()."
        )


def guarded_completion(guard: BudgetGuard, client: Any, **kwargs: Any) -> Any:
    """``client.messages.create`` with a budget reservation and accrual.

    Raises :class:`~floe_guard.BudgetExceeded` before the call if the budget
    would be crossed — the request never reaches Anthropic.

    ``client`` is an ``anthropic.Anthropic`` instance; ``kwargs`` are forwarded
    to ``messages.create`` (e.g. ``model=``, ``max_tokens=``, ``messages=``).

    Streaming is not supported: a streamed response has no final ``usage`` to
    settle from, so ``stream=True`` raises ``ValueError``.
    """
    _reject_streaming(kwargs)
    reserved = guard.reserve()
    try:
        response = client.messages.create(**kwargs)
    except BaseException:
        guard.release(reserved)
        raise
    _record_response(guard, kwargs, response, reserved=reserved)
    return response


async def guarded_acompletion(guard: BudgetGuard, client: Any, **kwargs: Any) -> Any:
    """Async counterpart of :func:`guarded_completion`.

    ``client`` is an ``anthropic.AsyncAnthropic`` instance. Streaming is not
    supported (see :func:`guarded_completion`); ``stream=True`` raises ``ValueError``.
    """
    _reject_streaming(kwargs)
    reserved = guard.reserve()
    try:
        response = await client.messages.create(**kwargs)
    except BaseException:
        guard.release(reserved)
        raise
    _record_response(guard, kwargs, response, reserved=reserved)
    return response


__all__ = [
    "guarded_completion",
    "guarded_acompletion",
]
