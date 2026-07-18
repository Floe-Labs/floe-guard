"""floe-guard — a local, framework-agnostic budget guardrail for AI agents.

Hard-stops an agent before its next LLM call when it would cross a spend ceiling.
Zero account, no network, runs in-process. Hosted Floe is the un-bypassable,
cross-vendor upgrade path (see the README).

    from floe_guard import BudgetGuard

    guard = BudgetGuard(limit_usd=5.00)
    guard.check()                       # before each LLM call (may raise)
    guard.record("gpt-4o", 1200, 350)   # after each response
"""

from __future__ import annotations

from .errors import (
    BudgetExceeded,
    DeadlineExceeded,
    FloeGuardError,
    HostedEnforcementError,
    UnpriceableModelError,
    UnpriceableModelWarning,
)
from .guard import BudgetAdvisory, BudgetGuard, SpendEvent
from .hosted import hosted_enforcement_available, hosted_remaining_usd
from .latency import LatencyAdvisory, LatencyBudget
from .pricing import ManualPrice, PricedModel, price_tokens, resolve_price
from .stream import StreamGuard, guard_stream

__version__ = "0.5.0"  # keep in lockstep with pyproject.toml

__all__ = [
    "BudgetGuard",
    "BudgetAdvisory",
    "SpendEvent",
    "LatencyBudget",
    "LatencyAdvisory",
    "StreamGuard",
    "guard_stream",
    "BudgetExceeded",
    "DeadlineExceeded",
    "FloeGuardError",
    "HostedEnforcementError",
    "UnpriceableModelError",
    "UnpriceableModelWarning",
    "ManualPrice",
    "PricedModel",
    "price_tokens",
    "resolve_price",
    "hosted_enforcement_available",
    "hosted_remaining_usd",
]
