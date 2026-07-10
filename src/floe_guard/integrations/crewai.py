"""CrewAI adapter (optional extra: ``pip install floe-guard[crewai]``).

This is the no-account, local descendant of Floe's hosted ``crewai-floe``
integration. The hosted ``FloeLLM`` routes a crew through Floe's metered proxy
and debits a server-side credit line; this version keeps everything in-process
and free: it meters tokens against the bundled cost map and hard-stops the crew
before the next LLM call crosses your ceiling.

CrewAI calls ``litellm.completion`` under the hood for every agent step, so a
single LiteLLM callback meters the budget across the **whole crew** — every
agent, every task — with no per-agent wiring. But LiteLLM can swallow
exceptions raised inside its callbacks (see the litellm adapter's module
docstring), so the callback alone cannot be trusted to *stop* the crew. Use
:func:`budget_guarded_llm`, which also enforces in the call path — where a
raise reliably reaches CrewAI:

    from crewai import Agent, Crew, Task
    from floe_guard import BudgetGuard
    from floe_guard.integrations.crewai import budget_guarded_llm

    guard = BudgetGuard(limit_usd=1.00)
    llm = budget_guarded_llm(guard, "gpt-4o")   # meters + hard-stops
    Crew(agents=[Agent(..., llm=llm)], tasks=[...]).kickoff()
"""

from __future__ import annotations

from typing import Any

from ..guard import BudgetGuard
from .litellm import budget_guard_callback


def _require_crewai() -> Any:
    try:
        import crewai  # noqa: F401
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "The CrewAI adapter requires crewai. Install with: pip install floe-guard[crewai]"
        ) from e
    return crewai


def guard_crew(guard: BudgetGuard) -> Any:
    """Meter ``guard`` across every LLM call any CrewAI agent makes.

    Registers a LiteLLM callback (CrewAI runs on LiteLLM) that reserves budget
    before each call and accrues spend after. Call once before ``kickoff()``.
    Idempotent for the same guard — re-registering will not double-count.

    Returns the registered callback. **Blocking is best-effort on this path**:
    LiteLLM can swallow the callback's enforcement raise (see the litellm
    adapter docstring), in which case the violation is recorded on the returned
    callback's ``tripped`` attribute and logged at ERROR level, but the crew
    keeps running. :func:`budget_guarded_llm` closes that gap by re-raising
    ``tripped`` in the call path, outside LiteLLM — prefer it.
    """
    try:
        import litellm
    except ImportError as e:
        raise ImportError(
            "guard_crew requires litellm (CrewAI runs on LiteLLM). "
            "Install: pip install floe-guard[crewai]"
        ) from e

    existing = list(getattr(litellm, "callbacks", None) or [])
    # Reuse a previously-installed floe-guard callback bound to this same guard:
    # it keeps its in-flight reservations and tripped state, and any LLM wrapper
    # already holding a reference to it stays live.
    for cb in existing:
        if getattr(cb, "guard", None) is guard:
            return cb
    callback = budget_guard_callback(guard)
    existing.append(callback)
    litellm.callbacks = existing
    return callback


def budget_guarded_llm(guard: BudgetGuard, model: str, **kwargs: Any) -> Any:
    """Return a ``crewai.LLM`` for ``model`` with ``guard`` enforced crew-wide.

    Metering runs through the same LiteLLM callback as :func:`guard_crew`;
    enforcement additionally runs in the LLM's ``call`` / ``acall`` paths,
    where a raise reliably propagates to CrewAI even when LiteLLM swallows
    callback exceptions: before each call the LLM re-raises any violation the
    callback recorded (``tripped``) and runs ``guard.check()``. A crew whose
    spend crosses the ceiling — or that hits an unpriceable model with
    ``fail_closed=True`` — stops at the next call instead of running unmetered.

    A recorded violation latches: it persists for the life of the callback
    (which :func:`guard_crew` reuses per guard) even if you later add a price
    override or raise ``limit_usd``. After remediating, call
    ``callback.reset()`` (the callback is on ``llm`` via :func:`guard_crew`'s
    return value) or build a fresh ``BudgetGuard``.
    """
    _require_crewai()
    from crewai import LLM

    callback = guard_crew(guard)

    def _enforce() -> None:
        tripped = callback.tripped
        if tripped is not None:
            # Drop the accumulated traceback: re-raising the same latched
            # instance on every subsequent call would grow it unboundedly.
            raise tripped.with_traceback(None)
        guard.check()

    class BudgetGuardedLLM(LLM):  # type: ignore[misc]
        """``crewai.LLM`` that enforces the budget outside LiteLLM's callbacks."""

        def call(self, *args: Any, **call_kwargs: Any) -> Any:
            _enforce()
            return super().call(*args, **call_kwargs)

        async def acall(self, *args: Any, **call_kwargs: Any) -> Any:
            # CrewAI 1.x async crews await acall(); without this override they
            # would bypass enforcement entirely. Harmless on 0.x (never invoked).
            _enforce()
            return await super().acall(*args, **call_kwargs)

    if "__new__" in vars(LLM):
        # CrewAI >= 1.0: LLM.__new__ is a factory that can return a native
        # provider-SDK instance, silently discarding this subclass AND
        # bypassing LiteLLM (so the metering callback never fires). Force the
        # LiteLLM route, which makes __new__ honor the subclass.
        return BudgetGuardedLLM(model=model, is_litellm=True, **kwargs)
    return BudgetGuardedLLM(model, **kwargs)


__all__ = ["guard_crew", "budget_guarded_llm"]
