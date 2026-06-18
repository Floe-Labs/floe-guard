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

import math
import sys
import threading
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
        near_limit_bps: int = 8000,
    ) -> None:
        if not math.isfinite(limit_usd) or limit_usd < 0:
            # NaN/inf would make every check() comparison evaluate False and
            # silently disable the guard — reject them (matches the JS
            # Number.isFinite contract).
            raise ValueError(f"limit_usd must be a finite, non-negative number, got {limit_usd!r}")
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
            # Release any held reservation on BOTH paths. Fail-closed must not
            # leak the in-flight hold, or _reserved grows permanently and
            # remaining_usd shrinks until reserve() starts blocking everything.
            self.release(reserved)
            if self.fail_closed:
                raise UnpriceableModelError(model)
            return 0.0

        try:
            cost = price_tokens(priced, prompt_tokens, completion_tokens)
        except Exception:
            # price_tokens can raise (e.g. non-finite token counts). Release the
            # in-flight hold before propagating so _reserved doesn't leak and
            # shrink remaining_usd permanently — same fail-safe as the unpriceable
            # path above.
            self.release(reserved)
            raise
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

    # ── internals ──────────────────────────────────────────────────────────────

    def _would_cross(self, estimated_next_cost: float | None) -> bool:
        with self._lock:
            estimate = (
                self._last_cost if estimated_next_cost is None else max(0.0, estimated_next_cost)
            )
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
