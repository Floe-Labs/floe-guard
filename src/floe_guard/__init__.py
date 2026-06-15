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
    FloeGuardError,
    UnpriceableModelError,
    UnpriceableModelWarning,
)
from .guard import BudgetGuard
from .pricing import ManualPrice, PricedModel, price_tokens, resolve_price

__version__ = "0.1.0"

__all__ = [
    "BudgetGuard",
    "BudgetExceeded",
    "FloeGuardError",
    "UnpriceableModelError",
    "UnpriceableModelWarning",
    "ManualPrice",
    "PricedModel",
    "price_tokens",
    "resolve_price",
]
