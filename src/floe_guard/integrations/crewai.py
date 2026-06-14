"""CrewAI adapter (optional extra: ``pip install floe-guard[crewai]``).

This is the no-account, local descendant of Floe's hosted ``crewai-floe``
integration. The hosted ``FloeLLM`` routes a crew through Floe's metered proxy
and debits a server-side credit line; this version keeps everything in-process
and free: it meters tokens against the bundled cost map and hard-stops the crew
before the next LLM call crosses your ceiling.

CrewAI calls ``litellm.completion`` under the hood for every agent step, so a
single LiteLLM callback enforces the budget across the **whole crew** — every
agent, every task — with no per-agent wiring.

    from crewai import Agent, Crew, Task
    from floe_guard import BudgetGuard
    from floe_guard.integrations.crewai import guard_crew

    guard = BudgetGuard(limit_usd=1.00)
    guard_crew(guard)          # 1 line — enforces across the whole crew
    Crew(agents=[...], tasks=[...]).kickoff()
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


def guard_crew(guard: BudgetGuard) -> None:
    """Enforce ``guard`` across every LLM call any CrewAI agent makes.

    Registers a LiteLLM callback (CrewAI runs on LiteLLM) that checks the budget
    before each call and accrues spend after. Call once before ``kickoff()``.
    Idempotent for the same guard — re-registering will not double-count.
    """
    import litellm

    callback = budget_guard_callback(guard)
    existing = list(getattr(litellm, "callbacks", None) or [])
    # Drop any previously-installed floe-guard callback bound to this same guard
    # so repeated calls don't stack duplicate accrual.
    existing = [cb for cb in existing if getattr(cb, "guard", None) is not guard]
    existing.append(callback)
    litellm.callbacks = existing


def budget_guarded_llm(guard: BudgetGuard, model: str, **kwargs: Any) -> Any:
    """Return a ``crewai.LLM`` for ``model`` with ``guard`` enforced crew-wide.

    Convenience over :func:`guard_crew`: builds the LLM you pass to your agents
    and registers the budget callback in one step.
    """
    _require_crewai()
    from crewai import LLM

    guard_crew(guard)
    return LLM(model, **kwargs)


__all__ = ["guard_crew", "budget_guarded_llm"]
