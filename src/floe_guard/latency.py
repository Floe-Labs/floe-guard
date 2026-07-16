"""LatencyBudget — a cumulative tool-chain deadline, sibling to BudgetGuard.

BudgetGuard stops an agent before its next call crosses a USD ceiling;
LatencyBudget stops it before the next call would blow an end-user SLA::

    from floe_guard import LatencyBudget

    deadline = LatencyBudget(sla_ms=5000)
    ...
    deadline.check(expected_ms=800)   # raises DeadlineExceeded when projected over
    if deadline.advisory().near_deadline:
        use_faster_model()            # taper BEFORE the wall, like near_limit
    router.pick(max_latency_ms=deadline.remaining_ms)

Design notes (mirroring BudgetGuard's ergonomics):

- **Monotonic clock.** Elapsed time comes from ``time.monotonic()``, never wall
  time — NTP steps and DST can't corrupt the budget.
- **Cooperative, not preemptive.** The guard provides the deadline *signal*;
  killing a stalled in-flight call is the framework's job (asyncio
  cancellation, an ``AbortSignal``, a thread timeout). ``check()`` prevents the
  NEXT call from starting; it cannot interrupt one that already did.
- **Advisory symmetry.** :meth:`advisory` returns ``near_deadline`` /
  ``used_bps`` / ``remaining_ms``, the latency twin of BudgetGuard's
  ``near_limit`` / ``used_bps`` / ``remaining_usd`` — taper logic written
  against one shape ports to the other.
- **In-process scope.** One instance covers one request/run in one process;
  distributed/server-side latency tracking is explicitly out of scope.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from .errors import DeadlineExceeded


@dataclass(frozen=True)
class LatencyAdvisory:
    """A context-aware deadline signal — the latency twin of BudgetAdvisory.

    Soft by design: the model may ignore it. The hard-stop
    (:meth:`LatencyBudget.check`) is what actually sheds the next call.
    """

    near_deadline: bool
    used_bps: int  # SLA consumed, basis points 0..10000 (8500 = 85%)
    remaining_ms: float
    sla_ms: float
    elapsed_ms: float


class LatencyBudget:
    """Track cumulative elapsed time across an agentic chain against an SLA.

    Args:
        sla_ms: the end-to-end deadline in milliseconds. Must be > 0.
        near_deadline_bps: utilization (basis points, 0..10000) at which
            :meth:`advisory` flags ``near_deadline`` so the agent can downshift
            to a faster path before the wall. Default 8000 (80%), matching
            BudgetGuard's ``near_limit_bps``.
        on_block: optional callback invoked with ``(elapsed_ms, sla_ms)`` right
            before :class:`DeadlineExceeded` is raised.
        clock: seconds-returning monotonic clock, injectable for tests.
            Defaults to :func:`time.monotonic`.

    The budget starts counting at construction — build it when the request
    (and its SLA) starts.
    """

    def __init__(
        self,
        sla_ms: float,
        *,
        near_deadline_bps: int = 8000,
        on_block: Callable[[float, float], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not (sla_ms > 0):
            raise ValueError("sla_ms must be > 0")
        if not (isinstance(near_deadline_bps, int) and 0 <= near_deadline_bps <= 10000):
            raise ValueError("near_deadline_bps must be an integer 0..10000")
        self.sla_ms = float(sla_ms)
        self.near_deadline_bps = near_deadline_bps
        self._on_block = on_block
        self._clock = clock
        self._started_at = clock()

    @property
    def elapsed_ms(self) -> float:
        """Milliseconds since construction (monotonic)."""
        return (self._clock() - self._started_at) * 1000.0

    @property
    def remaining_ms(self) -> float:
        """Milliseconds left before the SLA, floored at 0.

        This is the readable signal a router uses to pick a faster fallback
        or truncate work mid-chain.
        """
        return max(0.0, self.sla_ms - self.elapsed_ms)

    def check(self, expected_ms: float = 0.0) -> None:
        """Raise :class:`DeadlineExceeded` when the projected elapsed time
        (now + ``expected_ms`` for the upcoming call) would blow the SLA.

        Call it immediately before each tool/model call. ``expected_ms`` is
        the caller's estimate for the next call — pass 0 to only gate on time
        already spent.
        """
        if expected_ms < 0:
            raise ValueError("expected_ms must be >= 0")
        elapsed = self.elapsed_ms
        if elapsed + expected_ms > self.sla_ms:
            if self._on_block is not None:
                self._on_block(elapsed, self.sla_ms)
            raise DeadlineExceeded(elapsed, self.sla_ms)

    def advisory(self) -> LatencyAdvisory:
        """The soft near-deadline signal — symmetric to BudgetGuard.advisory()."""
        elapsed = self.elapsed_ms
        # round (not floor): float clock arithmetic can land a hair under an
        # exact boundary (4099.999…ms of 5000 must read 82%, not 81.99%).
        used_bps = min(10000, round(elapsed * 10000 / self.sla_ms)) if elapsed > 0 else 0
        return LatencyAdvisory(
            near_deadline=used_bps >= self.near_deadline_bps,
            used_bps=used_bps,
            remaining_ms=max(0.0, self.sla_ms - elapsed),
            sla_ms=self.sla_ms,
            elapsed_ms=elapsed,
        )
