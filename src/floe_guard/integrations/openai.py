"""OpenAI Python SDK adapter (optional extra: ``pip install floe-guard[openai]``).

:func:`guarded_completion` / :func:`guarded_acompletion` wrap a call to the
OpenAI SDK's ``client.chat.completions.create`` with a pre-call budget
reservation and post-call accrual. This is the **guaranteed** enforcement path:
the guard reserves the call's budget before the request, so a blocked call never
reaches OpenAI.

The OpenAI SDK has no global callback hook, so unlike the LiteLLM adapter there
is no callback variant — wrapping the call site is the enforcement surface.

The contract matches the LiteLLM adapter: ``reserve()`` before the call,
``settle(model, prompt_tokens, completion_tokens, reserved=...)`` after, and
``release()`` on exception. Reserving before the await means parallel calls each
hold their own slice of the ceiling instead of racing one shared total (issue
#18). Unpriceable models stay fail-closed (the guard's configured policy).
"""

from __future__ import annotations

from typing import Any

from ..guard import BudgetGuard
from ..pricing import resolve_price


def _model_from(kwargs: dict[str, Any], response: Any) -> str:
    # Prefer the model the response was actually served by — OpenAI can return a
    # resolved/canonical id that differs from the requested alias — and fall back
    # to the request's model= only when the response omits it.
    model = getattr(response, "model", None)
    if model is None and isinstance(response, dict):
        model = response.get("model")
    if not model:
        model = kwargs.get("model")
    return str(model or "")


def _usage_from(response: Any) -> tuple[int, int, int]:
    """Map an OpenAI chat completion's usage onto ``(prompt, completion, cache_read)``.

    OpenAI reports ``usage.prompt_tokens`` / ``usage.completion_tokens``, and
    ``usage.prompt_tokens_details.cached_tokens`` for the share of the prompt
    served from its prompt cache. ``prompt_tokens`` *includes* the cached share,
    so it is subtracted out and returned separately to be re-priced at the
    cheaper cache-read rate rather than being charged at the full input rate.
    """
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return 0, 0, 0
    get = usage.get if isinstance(usage, dict) else lambda k, d=0: getattr(usage, k, d)
    prompt = int(get("prompt_tokens", 0) or 0)
    completion = int(get("completion_tokens", 0) or 0)
    cached = _cached_tokens(get("prompt_tokens_details", None))
    # max(0, …): the cached share is documented as part of prompt_tokens, but
    # clamp so a malformed pair can never produce a negative input count.
    return max(0, prompt - cached), completion, cached


def _cached_tokens(details: Any) -> int:
    """``prompt_tokens_details.cached_tokens``, as a dict or object; 0 when absent.

    The SDK leaves the block unset on a response with no cache hit, and reports
    ``cached_tokens=None`` on some providers, so both read as zero.
    """
    if details is None:
        return 0
    get = details.get if isinstance(details, dict) else lambda k, d=0: getattr(details, k, d)
    return max(0, int(get("cached_tokens", 0) or 0))


def _settle_model(guard: BudgetGuard, kwargs: dict[str, Any], response: Any) -> str:
    """Pick the model id to settle against.

    The served model (``response.model``) is the source of truth (issue #23) —
    it reflects OpenAI's canonical/substituted id. But if that served id can't be
    priced and the requested alias can, settle on the alias instead, so a
    provider snapshot newer than the bundled cost map doesn't fail-closed a call
    that would otherwise price cleanly. If neither prices, keep the served id so
    the guard still fail-closes on a real id.
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
    prompt_tokens, completion_tokens, cache_read = _usage_from(response)
    if prompt_tokens <= 0 and completion_tokens <= 0 and cache_read <= 0:
        # No tokens spent (e.g. a usage-less response) — free the reservation.
        guard.release(reserved)
        return
    # There IS spend to account for. Route it through settle() even when the
    # model id is missing, so the guard's policy applies (fail-closed → warn +
    # raise; fail-open → warn + skip) rather than letting a completed call go
    # unmetered and skew the next check().
    guard.settle(
        model,
        prompt_tokens,
        completion_tokens,
        reserved=reserved,
        cache_read_input_tokens=cache_read,
    )


def _reject_streaming(kwargs: dict[str, Any]) -> None:
    if kwargs.get("stream"):
        raise ValueError(
            "floe-guard's OpenAI adapter does not support stream=True: a streamed "
            "response has no final usage to meter, so the call would go unaccounted. "
            "Use a non-streaming call, or meter the stream yourself with "
            "guard.reserve()/guard.settle()."
        )


def guarded_completion(guard: BudgetGuard, client: Any, **kwargs: Any) -> Any:
    """``client.chat.completions.create`` with a budget reservation and accrual.

    Raises :class:`~floe_guard.BudgetExceeded` before the call if the budget
    would be crossed — the request never reaches OpenAI.

    ``client`` is an ``openai.OpenAI`` instance; ``kwargs`` are forwarded to
    ``chat.completions.create`` (e.g. ``model=``, ``messages=``).

    Streaming is not supported: a streamed response has no final ``usage`` to
    settle from, so the call would silently go unmetered. Passing ``stream=True``
    raises ``ValueError``.
    """
    _reject_streaming(kwargs)
    reserved = guard.reserve()
    try:
        response = client.chat.completions.create(**kwargs)
    except BaseException:
        guard.release(reserved)
        raise
    _record_response(guard, kwargs, response, reserved=reserved)
    return response


async def guarded_acompletion(guard: BudgetGuard, client: Any, **kwargs: Any) -> Any:
    """Async counterpart of :func:`guarded_completion`.

    ``client`` is an ``openai.AsyncOpenAI`` instance. Streaming is not supported
    (see :func:`guarded_completion`); ``stream=True`` raises ``ValueError``.
    """
    _reject_streaming(kwargs)
    reserved = guard.reserve()
    try:
        response = await client.chat.completions.create(**kwargs)
    except BaseException:
        guard.release(reserved)
        raise
    _record_response(guard, kwargs, response, reserved=reserved)
    return response


__all__ = [
    "guarded_completion",
    "guarded_acompletion",
]
