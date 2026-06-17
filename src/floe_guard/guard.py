"""The local, in-process budget guard.

``BudgetGuard`` is a kill-switch that lives in the LLM call path. The contract:

1. Call :meth:`check` BEFORE every LLM call. If the *next* call would cross the
   ceiling, it raises :class:`BudgetExceeded` and the call never runs.
2. Call :meth:`record` AFTER every response, with the token usage. It prices the
   tokens offline and accrues the USD into a running total.

Why a call-path wrapper and not an event listener: a passive event-bus listener
is notified *after* the fact and cannot halt the run. To actually stop spend, the
guard has to sit in front of the next call. That is the whole point.

**Concurrency.** ``check()`` then ``record()`` is two non-atomic steps. When calls
run in parallel — the default for a CrewAI crew (async tasks,
``kickoff_for_each_async``, hierarchical tool calls) — several can read the same
under-limit total, all clear ``check()``, then all run, and the ceiling is blown
(see issue #18). :meth:`reserve` / :meth:`settle` close that gap: ``reserve``
atomically checks the ceiling *and* holds the estimated cost in-flight, so N
parallel callers can't all clear a stale total. The framework adapters use it;
the sequential ``check`` / ``record`` API is unchanged.

This is **estimate-based**: it prices tokens from a vendored cost map, it does
not reconcile against a wallet. Hosted Floe is the un-bypassable, cross-vendor
upgrade (see the README).
"""

from __future__ import annotations

import sys
import threading
import warnings
from collections.abc import Callable

from .errors import (
    BudgetExceeded,
    UnpriceableModelError,
    UnpriceableModelWarning,
)
from .pricing import ManualPrice, price_tokens, resolve_price

# Tolerance for float rounding in the running spend total (well below $0.000001).
_EPS = 1e-12


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

    Thread-safe: the running total and in-flight reservations are guarded by a
    lock, so the guard can back a parallel crew (use :meth:`reserve` /
    :meth:`settle`).
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
        # USD held for calls that are in flight (reserved but not yet settled).
        # Counted against the ceiling so concurrent callers can't overshoot.
        self._reserved = 0.0
        self._lock = threading.Lock()

    # ── enforcement ───────────────────────────────────────────────────────────

    def check(self, estimated_next_cost: float | None = None) -> None:
        """Raise :class:`BudgetExceeded` if the next call would cross the ceiling.

        Call this immediately before each LLM request. The "next call" is
        estimated from the last recorded call's cost (override with
        ``estimated_next_cost``); the first call is always allowed unless the
        ceiling is already met. A belt-and-suspenders check on the running total
        catches an overshoot if the estimate was too low. In-flight reservations
        count toward the total, so this stays correct alongside :meth:`reserve`.

        Note: ``check`` is a non-binding peek. For parallel calls, use
        :meth:`reserve` / :meth:`settle`, which hold the estimate atomically.
        """
        if self._would_cross(estimated_next_cost):
            self._block()

    def reserve(self, estimated_cost: float | None = None) -> float:
        """Atomically check the ceiling AND hold the estimated cost in-flight.

        This is the concurrency-safe enforcement path. Each parallel caller
        reserves before its call, so N callers can't all clear the same stale
        total and overshoot. Raises :class:`BudgetExceeded` (without reserving)
        if the reservation would cross the ceiling.

        Returns a reservation handle (the USD amount held) to pass to
        :meth:`settle` after the response, or to :meth:`release` if the call
        fails. ``estimated_cost`` defaults to the last call's cost.
        """
        with self._lock:
            estimate = self._last_cost if estimated_cost is None else max(0.0, estimated_cost)
            committed = self.spent_usd + self._reserved
            if committed > self.limit_usd - _EPS or committed + estimate > self.limit_usd + _EPS:
                spent, limit = self.spent_usd, self.limit_usd
            else:
                self._reserved += estimate
                return estimate
        # Blocked — notify and raise outside the lock.
        self._on_block(spent, limit)
        raise BudgetExceeded(spent, limit)

    def settle(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        *,
        reserved: float = 0.0,
        price: ManualPrice | None = None,
    ) -> float:
        """Release a reservation and record the actual cost. Concurrency-safe.

        ``record`` is ``settle`` with no reservation. Returns the USD cost of
        this call. Unpriceable-model handling matches :meth:`record` (warn +
        raise when ``fail_closed``, else warn + skip), and any held reservation
        is released even on the skip path.
        """
        priced = self._resolve(model, price)
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
            # Opted into un-metered spend for this model: free the hold, accrue $0.
            self.release(reserved)
            return 0.0

        cost = price_tokens(priced, prompt_tokens, completion_tokens)
        with self._lock:
            if reserved:
                self._reserved = max(0.0, self._reserved - reserved)
            self.spent_usd += cost
            # Clamp a sub-epsilon float overshoot back to the limit so the running
            # total never reports as having crossed the ceiling by a rounding artifact.
            if 0.0 < self.spent_usd - self.limit_usd < _EPS:
                self.spent_usd = self.limit_usd
            self._last_cost = cost
        return cost

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
        return self.settle(model, prompt_tokens, completion_tokens, reserved=0.0, price=price)

    def release(self, reserved: float) -> None:
        """Drop an in-flight reservation without recording spend (e.g. the call
        failed before producing usage). Safe to call with ``0``."""
        if not reserved:
            return
        with self._lock:
            self._reserved = max(0.0, self._reserved - reserved)

    @property
    def remaining_usd(self) -> float:
        """USD left before the ceiling, net of in-flight reservations (never negative)."""
        with self._lock:
            return max(0.0, self.limit_usd - self.spent_usd - self._reserved)

    # ── internals ──────────────────────────────────────────────────────────────

    def _would_cross(self, estimated_next_cost: float | None) -> bool:
        with self._lock:
            estimate = self._last_cost if estimated_next_cost is None else max(0.0, estimated_next_cost)
            committed = self.spent_usd + self._reserved
            return committed > self.limit_usd - _EPS or committed + estimate > self.limit_usd + _EPS

    def _block(self) -> None:
        self._on_block(self.spent_usd, self.limit_usd)
        raise BudgetExceeded(self.spent_usd, self.limit_usd)

    def _resolve(self, model: str, price: ManualPrice | None):
        overrides = self.price_overrides
        if price is not None:
            overrides = {**(overrides or {}), model: price}
        return resolve_price(model, overrides)


def _default_on_block(spent_usd: float, limit_usd: float) -> None:
    print(
        "BUDGET EXCEEDED — call blocked\n"
        f"  spent so far: ${spent_usd:.6f}  |  ceiling: ${limit_usd:.6f}\n"
        "  The next call would cross your budget; floe-guard stopped your agent "
        "before it ran.",
        file=sys.stderr,
    )
