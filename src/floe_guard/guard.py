"""The local, in-process budget guard.

``BudgetGuard`` is a kill-switch that lives in the LLM call path. The contract:

1. Call :meth:`check` BEFORE every LLM call. If the *next* call would cross the
   ceiling, it raises :class:`BudgetExceeded` and the call never runs.
2. Call :meth:`record` AFTER every response, with the token usage. It prices the
   tokens offline and accrues the USD into a running total.

Why a call-path wrapper and not an event listener: a passive event-bus listener
is notified *after* the fact and cannot halt the run. To actually stop spend, the
guard has to sit in front of the next call. That is the whole point.

This is **estimate-based**: it prices tokens from a vendored cost map, it does
not reconcile against a wallet. Hosted Floe is the un-bypassable, cross-vendor
upgrade (see the README).
"""

from __future__ import annotations

import sys
import warnings
from collections.abc import Callable
from dataclasses import dataclass

from .errors import (
    BudgetExceeded,
    UnpriceableModelError,
    UnpriceableModelWarning,
)
from .pricing import ManualPrice, price_tokens, resolve_price

# Tolerance for float rounding in the running spend total (well below $0.000001).
_EPS = 1e-12


@dataclass(frozen=True)
class BudgetAdvisory:
    """A context-aware spend signal for the single local budget.

    Mirrors the core fields of hosted Floe's ``X-Floe-Budget-Advisory`` header, so
    agent logic that reads it (taper as you approach the cap, stop at it) ports
    unchanged to the hosted path. Hosted adds what a local, single-budget guard
    cannot know: which of several caps is tightest (``scope`` across
    ``credit_line | session | task | api | vendor``), cross-vendor reasoning,
    server-truth balances, and rolling-window reset timing.

    This is a **soft** signal — the model may ignore it. The hard-stop
    (:meth:`BudgetGuard.check`) is what actually enforces the ceiling; the
    advisory is upside (let the agent finish on budget rather than be cut off).
    """

    near_limit: bool
    used_bps: int  # utilization in basis points, 0..10000 (8500 = 85%)
    remaining_usd: float
    limit_usd: float
    spent_usd: float
    scope: str = "local"  # hosted reports the tightest cap across all scopes


class BudgetGuard:
    """Hard-stop an agent before its next LLM call crosses a USD ceiling.

    Args:
        limit_usd: the spend ceiling, in USD. ``0`` blocks the very first call.
        price_overrides: per-model manual prices for models the bundled cost map
            cannot price (e.g. a brand-new or self-hosted model).
        fail_closed: when ``True`` (default), recording an unpriceable model
            without a manual price warns loudly AND raises
            :class:`UnpriceableModelError` — the guard refuses to keep going when
            it cannot measure spend. When ``False``, it warns and skips accrual
            (you have explicitly opted into un-enforced spend for that model).
        on_block: optional callback invoked with ``(spent_usd, limit_usd)`` right
            before :class:`BudgetExceeded` is raised. Defaults to printing the
            ``BUDGET EXCEEDED — call blocked`` banner to stderr.
        near_limit_bps: utilization (basis points, 0..10000) at which
            :meth:`advisory` flags ``near_limit`` so an agent can taper before the
            hard-stop. Defaults to ``8000`` (80%).
    """

    def __init__(
        self,
        limit_usd: float,
        *,
        price_overrides: dict[str, ManualPrice] | None = None,
        fail_closed: bool = True,
        on_block: Callable[[float, float], None] | None = None,
        near_limit_bps: int = 8000,
    ) -> None:
        if limit_usd < 0:
            raise ValueError(f"limit_usd must not be negative, got {limit_usd!r}")
        # Require a real int (bool is an int subclass in Python — exclude it) in
        # 0..10000, matching the TS Number.isInteger check for cross-language parity.
        if (
            isinstance(near_limit_bps, bool)
            or not isinstance(near_limit_bps, int)
            or not 0 <= near_limit_bps <= 10000
        ):
            raise ValueError(f"near_limit_bps must be an int in 0..10000, got {near_limit_bps!r}")
        self.limit_usd = float(limit_usd)
        self.price_overrides = price_overrides
        self.fail_closed = fail_closed
        self._on_block = on_block or _default_on_block
        self.near_limit_bps = near_limit_bps
        self.spent_usd = 0.0
        # Cost of the most recent priced call, used to predict the next one so we
        # can block BEFORE the crossing call runs (not one call too late).
        self._last_cost = 0.0

    # ── enforcement ───────────────────────────────────────────────────────────

    def check(self, estimated_next_cost: float | None = None) -> None:
        """Raise :class:`BudgetExceeded` if the next call would cross the ceiling.

        Call this immediately before each LLM request. The "next call" is
        estimated from the last recorded call's cost (override with
        ``estimated_next_cost``); the first call is always allowed unless the
        ceiling is already met. A belt-and-suspenders check on the running total
        catches an overshoot if the estimate was too low.
        """
        estimate = self._last_cost if estimated_next_cost is None else max(0.0, estimated_next_cost)
        projected = self.spent_usd + estimate
        # Compare with an epsilon so float rounding in the running total doesn't
        # block a call early or let one slip past the ceiling.
        if self.spent_usd > self.limit_usd - _EPS or projected > self.limit_usd + _EPS:
            self._on_block(self.spent_usd, self.limit_usd)
            raise BudgetExceeded(self.spent_usd, self.limit_usd)

    def record(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        *,
        price: ManualPrice | None = None,
    ) -> float:
        """Price one response's tokens offline and add the cost to the total.

        Returns the USD cost of this call. If the model is unpriceable and no
        ``price`` is given, behaviour depends on ``fail_closed`` (see the class
        docstring): warn + raise (default), or warn + skip accrual.
        """
        overrides = self.price_overrides
        if price is not None:
            overrides = {**(overrides or {}), model: price}

        priced = resolve_price(model, overrides)
        if priced is None:
            warnings.warn(
                f"Cannot price model {model!r}: not in the bundled cost map and no "
                f"manual price given. The budget guard cannot enforce a ceiling on "
                f"spend it cannot measure — pass price=ManualPrice(...) or set it in "
                f"price_overrides.",
                UnpriceableModelWarning,
                stacklevel=2,
            )
            if self.fail_closed:
                raise UnpriceableModelError(model)
            return 0.0

        cost = price_tokens(priced, prompt_tokens, completion_tokens)
        self.spent_usd += cost
        # Clamp a sub-epsilon float overshoot back to the limit so the running
        # total never reports as having crossed the ceiling by a rounding artifact.
        if 0.0 < self.spent_usd - self.limit_usd < _EPS:
            self.spent_usd = self.limit_usd
        self._last_cost = cost
        return cost

    @property
    def remaining_usd(self) -> float:
        """USD left before the ceiling (never negative)."""
        return max(0.0, self.limit_usd - self.spent_usd)

    def advisory(self) -> BudgetAdvisory:
        """Context-aware spend advisory for this budget — see :class:`BudgetAdvisory`.

        ``near_limit`` flips once utilization reaches ``near_limit_bps`` (default
        80%), so an agent can taper *before* the hard-stop. Advisory only: read it
        to adapt; :meth:`check` is what enforces the ceiling.
        """
        if self.limit_usd <= 0.0:
            used_bps = 10000
        else:
            # Floor (not round) so used_bps never over-reports utilization and
            # near_limit flips exactly when the threshold is reached, not a hair
            # early. The tiny epsilon absorbs float noise (0.7*10000 = 6999.9999…),
            # and floor matches JS Math.floor exactly — round() would diverge
            # (Python banker's rounding vs JS ties-up).
            used_bps = max(0, min(10000, int(self.spent_usd / self.limit_usd * 10000 + 1e-9)))
        return BudgetAdvisory(
            near_limit=used_bps >= self.near_limit_bps,
            used_bps=used_bps,
            remaining_usd=max(0.0, self.limit_usd - self.spent_usd),
            limit_usd=self.limit_usd,
            spent_usd=self.spent_usd,
        )


def _default_on_block(spent_usd: float, limit_usd: float) -> None:
    print(
        "BUDGET EXCEEDED — call blocked\n"
        f"  spent so far: ${spent_usd:.6f}  |  ceiling: ${limit_usd:.6f}\n"
        "  The next call would cross your budget; floe-guard stopped your agent "
        "before it ran.",
        file=sys.stderr,
    )
