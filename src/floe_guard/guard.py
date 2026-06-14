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

from .errors import (
    BudgetExceeded,
    UnpriceableModelError,
    UnpriceableModelWarning,
)
from .pricing import ManualPrice, price_tokens, resolve_price


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
    """

    def __init__(
        self,
        limit_usd: float,
        *,
        price_overrides: dict[str, ManualPrice] | None = None,
        fail_closed: bool = True,
        on_block: Callable[[float, float], None] | None = None,
    ) -> None:
        if limit_usd < 0:
            raise ValueError(f"limit_usd must not be negative, got {limit_usd!r}")
        self.limit_usd = float(limit_usd)
        self.price_overrides = price_overrides
        self.fail_closed = fail_closed
        self._on_block = on_block or _default_on_block
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
        if self.spent_usd >= self.limit_usd or projected > self.limit_usd:
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
        self._last_cost = cost
        return cost

    @property
    def remaining_usd(self) -> float:
        """USD left before the ceiling (never negative)."""
        return max(0.0, self.limit_usd - self.spent_usd)


def _default_on_block(spent_usd: float, limit_usd: float) -> None:
    print(
        "BUDGET EXCEEDED — call blocked\n"
        f"  spent so far: ${spent_usd:.6f}  |  ceiling: ${limit_usd:.6f}\n"
        "  The next call would cross your budget; floe-guard stopped your agent "
        "before it ran.",
        file=sys.stderr,
    )
