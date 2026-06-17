"""LiteLLM adapter (optional extra: ``pip install floe-guard[litellm]``).

Two ways to wire the guard into LiteLLM:

1. :func:`guarded_completion` / :func:`guarded_acompletion` — a thin wrapper
   around ``litellm.completion``. This is the **guaranteed** enforcement path:
   the guard reserves the call's budget before the request, so a blocked call
   never reaches LiteLLM.

2. :class:`BudgetGuardCallback` — a LiteLLM ``CustomLogger`` you register with
   ``litellm.callbacks``. It reserves before the call (``log_pre_api_call``) and
   settles spend after (``log_success_event``). Use this when you cannot wrap the
   call site yourself (e.g. CrewAI, which calls ``litellm.completion`` for you).

Both reserve before the call and settle after, so the ceiling holds even when a
crew fans calls out in parallel (issue #18) — not just on a single sequential
loop. Every priced response routes through the same
:class:`~floe_guard.BudgetGuard`.
"""

from __future__ import annotations

import threading
from typing import Any

from ..guard import BudgetGuard


def _require_litellm() -> Any:
    try:
        import litellm  # noqa: F401
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "The LiteLLM adapter requires litellm. Install with: pip install floe-guard[litellm]"
        ) from e
    return litellm


def _model_from(kwargs: dict[str, Any], response: Any) -> str:
    model = kwargs.get("model")
    if not model:
        # LiteLLM returns either a ModelResponse object or a plain dict; read the
        # model from whichever shape it is so a dict response is still recorded.
        if isinstance(response, dict):
            model = response.get("model")
        else:
            model = getattr(response, "model", None)
    return str(model or "")


def _usage_from(response: Any) -> tuple[int, int]:
    """Pull (prompt_tokens, completion_tokens) from a LiteLLM/OpenAI response."""
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return 0, 0
    get = usage.get if isinstance(usage, dict) else lambda k, d=0: getattr(usage, k, d)
    return int(get("prompt_tokens", 0) or 0), int(get("completion_tokens", 0) or 0)


def _record_response(
    guard: BudgetGuard, kwargs: dict[str, Any], response: Any, *, reserved: float = 0.0
) -> None:
    model = _model_from(kwargs, response)
    prompt_tokens, completion_tokens = _usage_from(response)
    if prompt_tokens <= 0 and completion_tokens <= 0:
        # No tokens were spent (e.g. a usage-less event) — nothing to meter, so
        # free any reservation we were holding for this call.
        guard.release(reserved)
        return
    # There IS spend to account for. Route it through settle() even when the
    # model id is missing, so the guard's configured policy applies (fail-closed
    # → warn + raise; fail-open → warn + skip). Silently skipping here would let
    # a real, completed call go unmetered and skew the next check().
    guard.settle(model, prompt_tokens, completion_tokens, reserved=reserved)


def guarded_completion(guard: BudgetGuard, **kwargs: Any) -> Any:
    """``litellm.completion`` with a pre-call budget reservation and post-call accrual.

    Raises :class:`~floe_guard.BudgetExceeded` before the call if the budget
    would be crossed — the request never reaches LiteLLM.
    """
    litellm = _require_litellm()
    reserved = guard.reserve()
    try:
        response = litellm.completion(**kwargs)
    except BaseException:
        guard.release(reserved)
        raise
    _record_response(guard, kwargs, response, reserved=reserved)
    return response


async def guarded_acompletion(guard: BudgetGuard, **kwargs: Any) -> Any:
    """Async counterpart of :func:`guarded_completion`."""
    litellm = _require_litellm()
    reserved = guard.reserve()
    try:
        response = await litellm.acompletion(**kwargs)
    except BaseException:
        guard.release(reserved)
        raise
    _record_response(guard, kwargs, response, reserved=reserved)
    return response


def budget_guard_callback(guard: BudgetGuard) -> Any:
    """Build a LiteLLM ``CustomLogger`` that enforces ``guard`` on every call.

    Register it globally::

        import litellm
        litellm.callbacks = [budget_guard_callback(guard)]

    ``log_pre_api_call`` reserves the call's budget (raising
    :class:`~floe_guard.BudgetExceeded` to abort), ``log_success_event`` settles
    the response's actual token cost, and ``log_failure_event`` releases the
    reservation. Reservations are keyed per call, so parallel crew calls each
    hold their own slice of the ceiling instead of racing one shared total.
    """
    litellm = _require_litellm()
    from litellm.integrations.custom_logger import CustomLogger

    class BudgetGuardCallback(CustomLogger):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.guard = guard
            self._reservations: dict[Any, float] = {}
            self._rlock = threading.Lock()

        @staticmethod
        def _key(kwargs: Any) -> Any:
            # LiteLLM stamps a stable call id on kwargs for both pre/post events;
            # fall back to the kwargs object identity if it is ever absent.
            call_id = (kwargs or {}).get("litellm_call_id") if isinstance(kwargs, dict) else None
            return call_id if call_id is not None else id(kwargs)

        def _hold(self, kwargs: Any) -> None:
            reserved = self.guard.reserve()  # raises BudgetExceeded -> aborts the call
            with self._rlock:
                self._reservations[self._key(kwargs)] = reserved

        def _pop(self, kwargs: Any) -> float:
            with self._rlock:
                return self._reservations.pop(self._key(kwargs), 0.0)

        def log_pre_api_call(self, model: Any, messages: Any, kwargs: Any) -> None:
            self._hold(kwargs)

        def log_success_event(
            self, kwargs: Any, response_obj: Any, start_time: Any, end_time: Any
        ) -> None:
            _record_response(self.guard, kwargs, response_obj, reserved=self._pop(kwargs))

        def log_failure_event(
            self, kwargs: Any, response_obj: Any, start_time: Any, end_time: Any
        ) -> None:
            self.guard.release(self._pop(kwargs))

        async def async_log_pre_api_call(self, model: Any, messages: Any, kwargs: Any) -> None:
            self._hold(kwargs)

        async def async_log_success_event(
            self, kwargs: Any, response_obj: Any, start_time: Any, end_time: Any
        ) -> None:
            _record_response(self.guard, kwargs, response_obj, reserved=self._pop(kwargs))

        async def async_log_failure_event(
            self, kwargs: Any, response_obj: Any, start_time: Any, end_time: Any
        ) -> None:
            self.guard.release(self._pop(kwargs))

    _ = litellm  # adapter import already validated litellm is present
    return BudgetGuardCallback()


__all__ = [
    "guarded_completion",
    "guarded_acompletion",
    "budget_guard_callback",
]
