"""Exceptions and warnings for floe-guard.

Everything derives from :class:`FloeGuardError` (the package-root base) so callers
can catch the whole family with a single ``except FloeGuardError``.
"""

from __future__ import annotations

import math


def _round_half_up(ms: float) -> int:
    """Shared millisecond rounding for cross-language message parity.

    Python's ``:.0f`` rounds half-to-even while JS ``toFixed(0)`` rounds
    half-up, so tie values would break the byte-for-byte message contract.
    Both packages format deadline messages through floor(x + 0.5) instead —
    identical semantics in both runtimes (see ``roundHalfUp`` in
    ``js/src/errors.ts``).
    """
    return math.floor(ms + 0.5)


class FloeGuardError(Exception):
    """Base class for every error raised by floe-guard."""


class BudgetExceeded(FloeGuardError):
    """Raised before an LLM call that would cross the configured spend ceiling.

    The guard raises this *instead of* letting the next call run, so a runaway
    loop stops here rather than burning more money.
    """

    def __init__(self, spent_usd: float, limit_usd: float) -> None:
        self.spent_usd = spent_usd
        self.limit_usd = limit_usd
        super().__init__(
            f"BUDGET EXCEEDED — call blocked (spent ${spent_usd:.6f} of ${limit_usd:.6f} ceiling)"
        )


class UnpriceableModelError(FloeGuardError):
    """Raised when a model cannot be priced and the guard is fail-closed.

    We refuse rather than silently accrue $0 — "we cannot cap what we cannot
    price". Pass a manual price (``price_overrides`` or ``record(..., price=...)``)
    to make the model enforceable.
    """

    def __init__(self, model: str) -> None:
        self.model = model
        super().__init__(
            f"Cannot price model {model!r}: not in the bundled cost map and no "
            f"manual price was given. The guard cannot enforce a budget on spend "
            f"it cannot measure. Pass a price override to enable enforcement."
        )


class HostedEnforcementError(FloeGuardError):
    """Raised when a read against the hosted Floe budget endpoint fails.

    Covers a missing API key, a non-200 response (401 bad/missing key, 403 agent
    closed/suspended, 404 agent not provisioned), a network/timeout failure, or a
    malformed response body. The message states plainly what went wrong — this
    client only *reads* server-side remaining budget; it does not enforce.
    """


class UnpriceableModelWarning(UserWarning):
    """Warned (loudly) whenever an unpriceable model is seen.

    Always emitted regardless of ``fail_closed``. In fail-closed mode an
    :class:`UnpriceableModelError` is additionally raised; in fail-open mode the
    warning is emitted and the call's spend is skipped.
    """


class DeadlineExceeded(FloeGuardError):
    """Raised before a call whose projected duration would blow the SLA.

    The latency twin of :class:`BudgetExceeded`: :meth:`LatencyBudget.check`
    raises this *instead of* letting the next tool/model call start, so the
    chain sheds work or falls back to a faster path rather than violating the
    end-user SLA. This is a cooperative signal — killing an already-running
    stalled task is the framework's job (asyncio cancellation / AbortSignal),
    not the guard's.
    """

    def __init__(self, elapsed_ms: float, sla_ms: float) -> None:
        self.elapsed_ms = elapsed_ms
        self.sla_ms = sla_ms
        super().__init__(
            f"DEADLINE EXCEEDED — call blocked (elapsed {_round_half_up(elapsed_ms)}ms "
            f"of {_round_half_up(sla_ms)}ms SLA)"
        )
