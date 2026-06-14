"""LiteLLM adapter (optional extra: ``pip install floe-guard[litellm]``).

Two ways to wire the guard into LiteLLM:

1. :func:`guarded_completion` / :func:`guarded_acompletion` — a thin wrapper
   around ``litellm.completion``. This is the **guaranteed** enforcement path:
   the guard's ``check()`` runs before the call in your own code, so a blocked
   call simply never reaches LiteLLM.

2. :class:`BudgetGuardCallback` — a LiteLLM ``CustomLogger`` you register with
   ``litellm.callbacks``. It checks before the call (``log_pre_api_call``) and
   records spend after (``log_success_event``). Use this when you cannot wrap the
   call site yourself (e.g. CrewAI, which calls ``litellm.completion`` for you).

Both route every priced response through the same :class:`~floe_guard.BudgetGuard`.
"""

from __future__ import annotations

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


def _record_response(guard: BudgetGuard, kwargs: dict[str, Any], response: Any) -> None:
    model = _model_from(kwargs, response)
    prompt_tokens, completion_tokens = _usage_from(response)
    if model:
        guard.record(model, prompt_tokens, completion_tokens)


def guarded_completion(guard: BudgetGuard, **kwargs: Any) -> Any:
    """``litellm.completion`` with a pre-call budget check and post-call accrual.

    Raises :class:`~floe_guard.BudgetExceeded` before the call if the budget
    would be crossed — the request never reaches LiteLLM.
    """
    litellm = _require_litellm()
    guard.check()
    response = litellm.completion(**kwargs)
    _record_response(guard, kwargs, response)
    return response


async def guarded_acompletion(guard: BudgetGuard, **kwargs: Any) -> Any:
    """Async counterpart of :func:`guarded_completion`."""
    litellm = _require_litellm()
    guard.check()
    response = await litellm.acompletion(**kwargs)
    _record_response(guard, kwargs, response)
    return response


def budget_guard_callback(guard: BudgetGuard) -> Any:
    """Build a LiteLLM ``CustomLogger`` that enforces ``guard`` on every call.

    Register it globally::

        import litellm
        litellm.callbacks = [budget_guard_callback(guard)]

    ``log_pre_api_call`` runs ``guard.check()`` (raising
    :class:`~floe_guard.BudgetExceeded` to abort), and ``log_success_event``
    records the response's token cost.
    """
    litellm = _require_litellm()
    from litellm.integrations.custom_logger import CustomLogger

    class BudgetGuardCallback(CustomLogger):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.guard = guard

        def log_pre_api_call(self, model: Any, messages: Any, kwargs: Any) -> None:
            self.guard.check()

        def log_success_event(
            self, kwargs: Any, response_obj: Any, start_time: Any, end_time: Any
        ) -> None:
            _record_response(self.guard, kwargs, response_obj)

        async def async_log_pre_api_call(self, model: Any, messages: Any, kwargs: Any) -> None:
            self.guard.check()

        async def async_log_success_event(
            self, kwargs: Any, response_obj: Any, start_time: Any, end_time: Any
        ) -> None:
            _record_response(self.guard, kwargs, response_obj)

    _ = litellm  # adapter import already validated litellm is present
    return BudgetGuardCallback()


__all__ = [
    "guarded_completion",
    "guarded_acompletion",
    "budget_guard_callback",
]
