"""Exceptions and warnings for floe-guard.

Everything derives from :class:`FloeGuardError` (the package-root base) so callers
can catch the whole family with a single ``except FloeGuardError``.
"""

from __future__ import annotations


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


class UnpriceableModelWarning(UserWarning):
    """Warned (loudly) whenever an unpriceable model is seen.

    Always emitted regardless of ``fail_closed``. In fail-closed mode an
    :class:`UnpriceableModelError` is additionally raised; in fail-open mode the
    warning is emitted and the call's spend is skipped.
    """
