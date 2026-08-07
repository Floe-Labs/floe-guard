"""floe-guard — a local, framework-agnostic budget guardrail for AI agents.

Hard-stops an agent before its next LLM or paid tool call when it would cross a
spend ceiling — tokens and tool costs share one local ceiling.
Zero account and no network; enforcement runs locally, with optional SQLite
state shared across processes. Hosted Floe is the un-bypassable, cross-vendor
upgrade path (see the README).

    from floe_guard import BudgetGuard

    guard = BudgetGuard(limit_usd=5.00)
    guard.check()                       # before each LLM call (may raise)
    guard.record("gpt-4o", 1200, 350)   # after each response
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version

from .errors import (
    BudgetExceeded,
    DeadlineExceeded,
    FloeGuardError,
    HostedEnforcementError,
    TokenBudgetExceeded,
    UnpriceableModelError,
    UnpriceableModelWarning,
)
from .guard import (
    BudgetAdvisory,
    BudgetGuard,
    BudgetReservation,
    ReservationHandle,
    SpendEvent,
)
from .hosted import hosted_enforcement_available, hosted_remaining_usd
from .latency import LatencyAdvisory, LatencyBudget
from .pricing import ManualPrice, PricedModel, price_tokens, resolve_price
from .retry import RetryPlan, async_with_budget_retry, with_budget_retry
from .store import SqliteStore, StateStore
from .stream import StreamGuard, guard_stream

# Single-sourced from the installed package metadata, which pyproject.toml
# already fills in. The hand-maintained literal that used to live here drifted:
# the published 0.11.0 wheel shipped __version__ == "0.10.0".
try:
    __version__ = _package_version("floe-guard")
except PackageNotFoundError:  # source tree with no install
    __version__ = "0.0.0.dev0"

__all__ = [
    "BudgetGuard",
    "BudgetAdvisory",
    "BudgetReservation",
    "ReservationHandle",
    "SpendEvent",
    "StateStore",
    "SqliteStore",
    "LatencyBudget",
    "LatencyAdvisory",
    "StreamGuard",
    "guard_stream",
    "RetryPlan",
    "with_budget_retry",
    "async_with_budget_retry",
    "BudgetExceeded",
    "TokenBudgetExceeded",
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
