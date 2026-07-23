"""Google Gemini adapter (optional extra: ``pip install floe-guard[gemini]``).

:func:`guarded_completion` / :func:`guarded_acompletion` wrap a call to the
``google-genai`` SDK's ``client.models.generate_content`` with a pre-call budget
reservation and post-call accrual. This is the **guaranteed** enforcement path:
the guard reserves the call's budget before the request, so a blocked call never
reaches Google.

Like the OpenAI and Anthropic adapters there is no callback variant — the SDK has
no global hook, so wrapping the call site is the enforcement surface. The contract
matches them exactly: ``reserve()`` before the call, ``settle(model,
prompt_tokens, completion_tokens, reserved=...)`` after, ``release()`` on
exception. Reserving before the await lets parallel calls each hold their own
slice of the ceiling (issue #18).

**Which Google backend you are on matters.** One SDK serves both Google AI Studio
(the Gemini Developer API) and Vertex AI, and the *model id is identical on both*
— but they bill at different rates (``gemini-2.0-flash-001`` costs 50% more on
Vertex). The bundled cost map carries AI Studio prices only, so metering a Vertex
call against it would under-meter, which is the one failure mode this package
exists to prevent. The model id cannot reveal the billing path, but the *client*
can: ``client.vertexai`` is ``True`` for Vertex. This adapter reads it and
refuses (fail-closed, per the guard's policy) unless you have supplied a price
for the model yourself::

    guard = BudgetGuard(limit_usd=1.00, price_overrides={
        "gemini-2.5-flash": ManualPrice(3.0e-7, 2.5e-6),   # your Vertex rates
    })

Read ``client.vertexai`` rather than the constructor argument: BOTH
``Client(vertexai=True, ...)`` and the newer ``Client(enterprise=True, ...)`` set
it, and there is no ``.enterprise`` attribute to check instead.

**Token buckets.** Gemini splits usage across five counters, and the SDK's own
field docs pin down how they compose (``total_token_count`` is documented as
``prompt + candidates + tool_use_prompt + thoughts``):

- ``prompt_token_count`` — input, and it *includes* cached tokens when a cached
  context is used, so the cached share is subtracted out and re-priced at the
  cheaper cache-read rate rather than being charged twice.
- ``tool_use_prompt_token_count`` — results fed back from tool executions. Input,
  and NOT part of ``prompt_token_count``; dropping it would under-meter a
  tool-using agent.
- ``candidates_token_count`` — the generated output.
- ``thoughts_token_count`` — thinking-model reasoning. Billed as output and NOT
  part of ``candidates_token_count``; dropping it would under-meter every
  thinking model.

**Streaming is not wrapped.** ``generate_content_stream`` returns a generator
whose usage only arrives on the final chunk (or never, if the consumer stops
early), so a reserve/settle wrapper around it would hand back an unmetered stream
and leak its reservation. Meter a stream with
:func:`~floe_guard.guard_stream` instead, which prices it chunk-by-chunk and can
hard-stop mid-generation.
"""

from __future__ import annotations

import warnings
from typing import Any

from ..errors import UnpriceableModelError, UnpriceableModelWarning
from ..guard import BudgetGuard
from ..pricing import resolve_price


def _model_from(kwargs: dict[str, Any], response: Any) -> str:
    # Prefer the model the response was actually served by — Gemini reports a
    # resolved id in ``model_version`` that can differ from the requested alias —
    # and fall back to the request's model= only when the response omits it.
    model = getattr(response, "model_version", None)
    if model is None and isinstance(response, dict):
        model = response.get("model_version")
    if not model:
        model = kwargs.get("model")
    return str(model or "")


def _usage_from(response: Any) -> tuple[int, int, int]:
    """Map a Gemini response's usage onto ``(prompt, completion, cache_read)``.

    See the module docstring for why each bucket lands where it does; in short,
    cached tokens are carved out of the prompt count (they are included there and
    bill cheaper), tool-use tokens are added to it (they are not included), and
    thinking tokens are added to the output count (they are not included either).
    """
    usage = getattr(response, "usage_metadata", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage_metadata")
    if usage is None:
        return 0, 0, 0
    # ``or 0`` because Gemini leaves an inapplicable bucket unset (None) rather
    # than reporting a zero.
    get = usage.get if isinstance(usage, dict) else lambda k, d=0: getattr(usage, k, d)
    prompt = int(get("prompt_token_count", 0) or 0)
    cached = int(get("cached_content_token_count", 0) or 0)
    tool_use = int(get("tool_use_prompt_token_count", 0) or 0)
    completion = int(get("candidates_token_count", 0) or 0) + int(
        get("thoughts_token_count", 0) or 0
    )
    # max(0, …): the cached share is documented as part of prompt_token_count, but
    # clamp so a malformed pair can never produce a negative input count that
    # would eat into the tool-use tokens added alongside it.
    return max(0, prompt - cached) + tool_use, completion, cached


def _settle_model(guard: BudgetGuard, kwargs: dict[str, Any], response: Any) -> str:
    """Pick the model id to settle against.

    The served id (``model_version``) is the source of truth, but if it cannot be
    priced and the requested alias can, settle on the alias instead — so a
    provider snapshot newer than the bundled cost map does not fail-close a call
    that would otherwise price cleanly. If neither prices, keep the served id so
    the guard still fail-closes on a real id. (Mirrors the OpenAI adapter.)
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


def _check_vertex_pricing(guard: BudgetGuard, client: Any, model: str) -> None:
    """Refuse a Vertex call the bundled (AI Studio) prices would under-meter.

    Runs BEFORE the reservation, so a refused call never reaches Google and never
    holds budget. A caller-supplied price wins outright — that is the documented
    way to meter Vertex — so an override for this model clears the check. Honours
    ``guard.fail_closed``: warn-and-continue when the caller has explicitly opted
    into un-enforced spend.
    """
    if not getattr(client, "vertexai", False):
        return
    priced = resolve_price(model, guard.price_overrides)
    if priced is not None and priced.source == "override":
        return
    warnings.warn(
        f"Client is configured for Vertex AI, but the bundled cost map prices "
        f"{model!r} at Google AI Studio rates. Vertex bills the same model ids "
        f"differently (up to 50% more), so metering this call against the map "
        f"would under-meter it — and a budget guard that under-meters cannot hold "
        f"a ceiling. Pass your Vertex rates via price_overrides="
        f"{{{model!r}: ManualPrice(input, output)}} to enforce this call.",
        UnpriceableModelWarning,
        stacklevel=3,
    )
    if guard.fail_closed:
        raise UnpriceableModelError(model)


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
    # There IS spend to account for. Route it through settle() even when the model
    # id is missing, so the guard's policy applies (fail-closed → warn + raise;
    # fail-open → warn + skip) rather than letting a completed call go unmetered
    # and skew the next check().
    guard.settle(
        model,
        prompt_tokens,
        completion_tokens,
        reserved=reserved,
        cache_read_input_tokens=cache_read,
    )


def guarded_completion(guard: BudgetGuard, client: Any, **kwargs: Any) -> Any:
    """``client.models.generate_content`` with a budget reservation and accrual.

    Raises :class:`~floe_guard.BudgetExceeded` before the call if the budget would
    be crossed — the request never reaches Google.

    ``client`` is a ``google.genai.Client``; ``kwargs`` are forwarded to
    ``models.generate_content`` (e.g. ``model=``, ``contents=``, ``config=``).

    A Vertex-configured client raises :class:`~floe_guard.UnpriceableModelError`
    unless the model has a ``price_overrides`` entry — see the module docstring.
    Streaming is not supported here; use :func:`~floe_guard.guard_stream`.
    """
    _check_vertex_pricing(guard, client, str(kwargs.get("model") or ""))
    reserved = guard.reserve()
    try:
        response = client.models.generate_content(**kwargs)
    except BaseException:
        guard.release(reserved)
        raise
    _record_response(guard, kwargs, response, reserved=reserved)
    return response


async def guarded_acompletion(guard: BudgetGuard, client: Any, **kwargs: Any) -> Any:
    """Async counterpart of :func:`guarded_completion`.

    Calls ``client.aio.models.generate_content`` — the async face of the same
    ``google.genai.Client``, so pass the client itself, not ``client.aio``.
    """
    _check_vertex_pricing(guard, client, str(kwargs.get("model") or ""))
    reserved = guard.reserve()
    try:
        response = await client.aio.models.generate_content(**kwargs)
    except BaseException:
        guard.release(reserved)
        raise
    _record_response(guard, kwargs, response, reserved=reserved)
    return response


__all__ = [
    "guarded_completion",
    "guarded_acompletion",
]
