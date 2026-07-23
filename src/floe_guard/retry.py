"""Budget-aware retry helpers.

The helpers here are deliberately thin: they do not implement transport
retries, model ranking, or provider-specific pricing. They only decide whether
the next retry should run under the current budget, and let callers supply a
cheaper retry path when the guard is near its limit.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from .errors import BudgetExceeded
from .guard import BudgetAdvisory, BudgetGuard

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPlan(Generic[T]):
    """A concrete retry attempt.

    ``call`` is the operation to run for this retry. ``estimated_cost`` is passed
    to :meth:`BudgetGuard.check` immediately before the retry, so a retry that
    cannot fit the remaining budget is blocked before it runs. Use ``None`` to
    fall back to the guard's normal last-call estimate.
    """

    call: Callable[[], T]
    estimated_cost: float | None = None


RetryPredicate = Callable[[Exception], bool]
DegradeCallback = Callable[[Exception, BudgetAdvisory], RetryPlan[T] | None]
AsyncDegradeCallback = Callable[
    [Exception, BudgetAdvisory],
    RetryPlan[Awaitable[T]] | Awaitable[RetryPlan[Awaitable[T]] | None] | None,
]


def _default_retry_if(exc: Exception) -> bool:
    return not isinstance(exc, BudgetExceeded)


def _validate_max_attempts(max_attempts: int) -> None:
    # Same int-not-bool contract as BudgetGuard.near_limit_bps.
    if (
        isinstance(max_attempts, bool)
        or not isinstance(max_attempts, int)
        or max_attempts < 1
    ):
        raise ValueError(f"max_attempts must be an int >= 1, got {max_attempts!r}")


def with_budget_retry(
    guard: BudgetGuard,
    call: Callable[[], T],
    *,
    estimated_cost: float | None = None,
    max_attempts: int = 2,
    on_degrade: DegradeCallback[T] | None = None,
    retry_if: RetryPredicate | None = None,
) -> T:
    """Run ``call`` with budget-aware retries.

    The first attempt runs unchanged. If it fails with a retryable exception,
    the helper consults the guard:

    * with ample budget, retry the same ``call``;
    * when ``advisory().near_limit`` is true and ``on_degrade`` is supplied, let
      the caller provide a cheaper :class:`RetryPlan`;
    * before every retry, call ``guard.check(plan.estimated_cost)`` so an
      over-budget retry aborts before spending.

    ``max_attempts`` includes the first attempt. The last retryable exception is
    re-raised when attempts are exhausted. Control-flow exceptions
    (``KeyboardInterrupt``, ``SystemExit``, ``CancelledError``) are not caught.
    """

    _validate_max_attempts(max_attempts)
    should_retry = retry_if or _default_retry_if
    plan = RetryPlan(call=call, estimated_cost=estimated_cost)

    for attempt in range(1, max_attempts + 1):
        try:
            return plan.call()
        except Exception as exc:
            if attempt >= max_attempts or not should_retry(exc):
                raise
            plan = _next_plan(guard, exc, plan, on_degrade)

    raise RuntimeError("unreachable")


async def async_with_budget_retry(
    guard: BudgetGuard,
    call: Callable[[], Awaitable[T]],
    *,
    estimated_cost: float | None = None,
    max_attempts: int = 2,
    on_degrade: AsyncDegradeCallback[T] | None = None,
    retry_if: RetryPredicate | None = None,
) -> T:
    """Async variant of :func:`with_budget_retry`."""

    _validate_max_attempts(max_attempts)
    should_retry = retry_if or _default_retry_if
    plan = RetryPlan(call=call, estimated_cost=estimated_cost)

    for attempt in range(1, max_attempts + 1):
        try:
            return await plan.call()
        except Exception as exc:
            if attempt >= max_attempts or not should_retry(exc):
                raise
            plan = await _next_async_plan(guard, exc, plan, on_degrade)

    raise RuntimeError("unreachable")


def _next_plan(
    guard: BudgetGuard,
    exc: Exception,
    current: RetryPlan[T],
    on_degrade: DegradeCallback[T] | None,
) -> RetryPlan[T]:
    advisory = guard.advisory()
    if advisory.near_limit and on_degrade is not None:
        degraded = on_degrade(exc, advisory)
        if degraded is not None:
            guard.check(degraded.estimated_cost)
            return degraded
    guard.check(current.estimated_cost)
    return current


async def _next_async_plan(
    guard: BudgetGuard,
    exc: Exception,
    current: RetryPlan[Awaitable[T]],
    on_degrade: AsyncDegradeCallback[T] | None,
) -> RetryPlan[Awaitable[T]]:
    advisory = guard.advisory()
    if advisory.near_limit and on_degrade is not None:
        degraded = on_degrade(exc, advisory)
        if inspect.isawaitable(degraded):
            degraded = await degraded
        if degraded is not None:
            guard.check(degraded.estimated_cost)
            return degraded
    guard.check(current.estimated_cost)
    return current


__all__ = ["RetryPlan", "with_budget_retry", "async_with_budget_retry"]
