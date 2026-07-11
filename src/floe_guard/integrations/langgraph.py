"""LangGraph adapter (optional extra: ``pip install floe-guard[langgraph]``).

Two pieces (issue #33):

:func:`guarded_node` wraps a graph node with the reserve-before / settle-after
contract the OpenAI and Anthropic adapters use: ``reserve()`` before the node
runs, ``settle(..., reserved=...)`` from the usage it reports, ``release()`` on
error. LangGraph executes the parallel branches of a ``StateGraph`` fan-out on
a thread pool, which is exactly the check-then-record race that atomic
reservations close (issue #18) — N sub-agents each hold their own slice of the
ceiling instead of racing one shared total.

:func:`latest_advisory` (used via :data:`AdvisoryChannel`) is the reducer for a
typed budget channel in the graph state. Every guarded node refreshes it with
:meth:`~floe_guard.BudgetGuard.advisory` after settling, so a router node can
read ``state["budget"].near_limit`` and downshift to a cheaper model *before*
``reserve()`` hard-blocks::

    import operator
    from typing import Annotated
    from typing_extensions import TypedDict

    from floe_guard import BudgetGuard
    from floe_guard.integrations.langgraph import AdvisoryChannel, guarded_node

    class State(TypedDict):
        results: Annotated[list, operator.add]
        budget: AdvisoryChannel

    guard = BudgetGuard(limit_usd=0.10)

    # estimated_cost seeds the very first hold (a fresh guard has no last
    # cost to estimate from); later calls re-estimate from the last settled cost.
    @guarded_node(guard, estimated_cost=0.01)
    def worker(state: State) -> dict:
        response = my_llm_call(state)  # however the node does its work
        return {
            "results": [response["text"]],
            # Report what the call consumed; the wrapper settles it.
            "usage": {
                "model": response["model"],
                "prompt_tokens": response["prompt_tokens"],
                "completion_tokens": response["completion_tokens"],
            },
        }

The wrapped node reports its spend by returning a ``"usage"`` entry (the shape
above — the same one ``examples/budget_aware.py`` uses). If a node meters its
LLM call through another floe-guard adapter instead (e.g. the OpenAI or
Anthropic wrappers), omit ``"usage"``: the reservation still holds the branch's
slice of the ceiling while the call is in flight, and the wrapper releases the
hold afterwards instead of settling it twice.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import wraps
from typing import Annotated, Any

from ..guard import BudgetAdvisory, BudgetGuard


def _require_langgraph() -> None:
    try:
        import langgraph  # noqa: F401
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "The LangGraph adapter requires langgraph. "
            "Install with: pip install floe-guard[langgraph]"
        ) from e


def latest_advisory(
    current: BudgetAdvisory | None, update: BudgetAdvisory | None
) -> BudgetAdvisory | None:
    """Reducer for the budget channel: keep the advisory reflecting the most spend.

    Parallel branches finish in nondeterministic order, so "last write wins"
    does not mean "latest spend wins". ``used_bps`` is monotone in the guard's
    running total, so the higher reading is always the fresher truth.
    """
    if update is None:
        return current
    if current is None:
        return update
    return update if update.used_bps >= current.used_bps else current


# Declare this on your graph state to receive the advisory after each guarded
# node:  ``budget: AdvisoryChannel``
AdvisoryChannel = Annotated[BudgetAdvisory | None, latest_advisory]


def _usage_from_update(update: Any, usage_key: str) -> tuple[str, int, int] | None:
    """Pull (model, prompt_tokens, completion_tokens) from a node's state update.

    Returns ``None`` when the update carries no usage entry at all — distinct
    from an entry reporting zero tokens, which still routes through the guard's
    accounting below.
    """
    if not isinstance(update, dict):
        return None
    usage = update.get(usage_key)
    if usage is None:
        return None
    get = usage.get if isinstance(usage, dict) else lambda k, d=0: getattr(usage, k, d)
    model = str(get("model", "") or "")
    prompt_tokens = int(get("prompt_tokens", 0) or 0)
    completion_tokens = int(get("completion_tokens", 0) or 0)
    return model, prompt_tokens, completion_tokens


def _settle_update(guard: BudgetGuard, update: Any, usage_key: str, *, reserved: float) -> None:
    try:
        usage = _usage_from_update(update, usage_key)
    except (TypeError, ValueError):
        # A malformed usage payload (e.g. prompt_tokens="abc") cannot be priced.
        # Release the in-flight hold before propagating so the reservation
        # doesn't leak and shrink remaining_usd permanently — the same
        # fail-safe as settle()'s pricing-error path.
        guard.release(reserved)
        raise
    if usage is None:
        # No usage reported — the node metered elsewhere (another adapter) or
        # spent nothing. Free the hold; do NOT settle, or the spend would be
        # counted twice.
        guard.release(reserved)
        return
    model, prompt_tokens, completion_tokens = usage
    if prompt_tokens <= 0 and completion_tokens <= 0:
        # A usage entry with no tokens — nothing to meter. Free the hold.
        guard.release(reserved)
        return
    # There IS spend to account for. Route it through settle() even when the
    # model id is missing, so the guard's policy applies (fail-closed → warn +
    # raise; fail-open → warn + skip) rather than letting a completed call go
    # unmetered and skew the next reservation's estimate.
    guard.settle(model, prompt_tokens, completion_tokens, reserved=reserved)


def _inject_advisory(guard: BudgetGuard, update: Any, advisory_key: str | None) -> Any:
    if advisory_key is None:
        return update
    if update is None:
        return {advisory_key: guard.advisory()}
    if isinstance(update, dict):
        # Refresh AFTER settling so the branch's own spend is included. The
        # channel's reducer (latest_advisory) resolves concurrent writes.
        return {**update, advisory_key: guard.advisory()}
    # Command/other update objects pass through untouched — nothing to annotate.
    return update


def guarded_node(
    guard: BudgetGuard,
    node: Callable[..., Any] | None = None,
    *,
    usage_key: str = "usage",
    advisory_key: str | None = "budget",
    estimated_cost: float | None = None,
) -> Callable[..., Any]:
    """Wrap a LangGraph node with a budget reservation, accrual, and advisory.

    Works as a decorator (``@guarded_node(guard)``) or a plain wrapper
    (``guarded_node(guard, fn)``); sync and async nodes are both supported.

    Raises :class:`~floe_guard.BudgetExceeded` before the node runs if the
    reservation would cross the ceiling — the branch never executes. On
    success, the node's ``usage_key`` entry is settled against the reservation
    and the guard's :class:`~floe_guard.BudgetAdvisory` is written to
    ``advisory_key`` in the returned update (declare that key on the graph
    state with :data:`AdvisoryChannel`, or pass ``advisory_key=None`` to skip
    the injection). On error the reservation is released and the exception
    propagates, so a failed branch never leaks its hold.

    ``estimated_cost`` sizes the reservation; it defaults to the guard's last
    recorded call cost (see :meth:`~floe_guard.BudgetGuard.reserve`). On a cold
    guard no call has been recorded yet, so that default is ``0`` — when the
    very first graph step is already a parallel fan-out, pass an explicit
    ``estimated_cost`` per node so each branch holds a realistic slice.
    """
    _require_langgraph()

    def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        if inspect.iscoroutinefunction(fn):

            @wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                reserved = guard.reserve(estimated_cost)
                try:
                    update = await fn(*args, **kwargs)
                except BaseException:
                    guard.release(reserved)
                    raise
                _settle_update(guard, update, usage_key, reserved=reserved)
                return _inject_advisory(guard, update, advisory_key)

            return async_wrapper

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            reserved = guard.reserve(estimated_cost)
            try:
                update = fn(*args, **kwargs)
            except BaseException:
                guard.release(reserved)
                raise
            _settle_update(guard, update, usage_key, reserved=reserved)
            return _inject_advisory(guard, update, advisory_key)

        return wrapper

    if node is None:
        return wrap
    return wrap(node)


__all__ = [
    "AdvisoryChannel",
    "guarded_node",
    "latest_advisory",
]
