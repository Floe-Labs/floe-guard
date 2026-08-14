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

import json
import math
import sys
import threading
import time
import warnings
from collections import deque
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, NoReturn

from .errors import (
    BudgetExceeded,
    HostedEnforcementError,
    TokenBudgetExceeded,
    UnpriceableModelError,
    UnpriceableModelWarning,
)
from .pricing import ManualPrice, price_tokens, resolve_price
from .store import StateStore

# Tolerance for float rounding in the running spend total (well below $0.000001).
_EPS = 1e-12


@dataclass(frozen=True)
class BudgetReservation:
    """A two-dimensional in-flight hold (USD + tokens) returned by
    :meth:`BudgetGuard.reserve` when a token ceiling or an active
    :meth:`BudgetGuard.step` is involved.

    A dumb value handle — it carries the amounts held, nothing else. Pass it
    straight back to :meth:`BudgetGuard.settle` / :meth:`BudgetGuard.release`.
    When neither tokens nor a step are involved, ``reserve`` returns a plain
    ``float`` instead (byte-for-byte the old behaviour), so ``ReservationHandle``
    is the union of the two.
    """

    usd: float
    tokens: int

    def __post_init__(self) -> None:
        # Public, re-exported value object: validate at construction so a
        # hand-rolled BudgetReservation(usd=nan, tokens=-5) can't slip past the
        # raw-float guard and corrupt _reserved / _reserved_tokens downstream
        # (settle/release trust these fields). Same contracts as reserve()'s
        # inputs and the JS twin's reservedUsdOf validation.
        if not math.isfinite(self.usd) or self.usd < 0:
            raise ValueError(f"BudgetReservation.usd must be finite and >= 0, got {self.usd!r}")
        if isinstance(self.tokens, bool) or not isinstance(self.tokens, int) or self.tokens < 0:
            raise ValueError(
                f"BudgetReservation.tokens must be a non-negative int, got {self.tokens!r}"
            )


# A plain float when USD-only (backward compatible); a BudgetReservation when a
# token ceiling or an active step is in play.
ReservationHandle = float | BudgetReservation


class _PersistentReservation(float):
    """Numeric reservation handle carrying its authoritative store window.

    Returned by :meth:`BudgetGuard.reserve` only when a persistent ``store`` is
    configured (USD-only path). Pass it through unchanged: it is float-compatible,
    but arithmetic or normalization such as ``float()``, ``abs()``, ``round()``,
    or ``max()`` can return a plain float and discard its issuing-window
    provenance — which its terminal settle/release needs to hit the right window.
    """

    _window_id: str

    def __new__(cls, amount_usd: float, window_id: str) -> _PersistentReservation:
        handle = super().__new__(cls, amount_usd)
        handle._window_id = window_id
        return handle

    def __reduce__(self) -> tuple[object, tuple[float, str]]:
        return _PersistentReservation, (float(self), self._window_id)


# eq=False: identity, not value, equality. The step stack is popped by identity
# (``is`` / ``list.remove`` on the exact object), so two nested steps with the
# same caps and no spend yet must not compare equal or ``remove`` could drop the
# wrong one. Matches the JS twin, which splices by ``lastIndexOf`` (reference).
@dataclass(eq=False)
class _StepState:
    """One scope on the step stack (see :meth:`BudgetGuard.step`). Mutable: the
    guard accrues into ``spent_*`` / ``reserved_*`` as calls settle within the
    step. ``max_usd`` / ``max_tokens`` are ``None`` for an uncapped dimension."""

    max_usd: float | None
    max_tokens: int | None
    spent_usd: float = 0.0
    spent_tokens: int = 0
    reserved_usd: float = 0.0
    reserved_tokens: int = 0


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
    # The guard's own next-call estimate (the costlier of the last LLM and last
    # tool call — same value the default reservation uses). 0.0 until the first
    # call is recorded, so a planner can't divide by a cold estimate.
    expected_cost: float = 0.0
    # How many more calls the remaining budget buys at expected_cost:
    # floor(remaining_usd / expected_cost). None when expected_cost is 0.0
    # (no call recorded yet) — unknown, not zero.
    est_calls_remaining: int | None = None
    # Aggregate token utilization in basis points (0..10000), mirroring used_bps
    # for the token ceiling. None when no token_limit is set (nothing to signal).
    token_used_bps: int | None = None
    # Tokens left before the aggregate token ceiling (never negative). None when
    # no token_limit is set.
    remaining_tokens: int | None = None
    # Per-step headroom for the innermost active step, present (non-None) ONLY
    # while a step() is active AND that dimension is capped. Lets a router read
    # how much room the current step has before its own cap, not just the global
    # one. None when no step is active or the dimension is uncapped.
    step_remaining_usd: float | None = None
    step_remaining_tokens: int | None = None
    # Spend rate in USD per minute over this guard's lifetime (spent_usd ÷ minutes
    # since the guard was created) — the number voice teams watch. None only when
    # no wall-clock time has elapsed yet. Make one guard per call/turn for a
    # per-call burn rate; a long-lived guard reports the session average.
    burn_rate_usd_per_min: float | None = None


@dataclass(frozen=True)
class SpendEvent:
    """One priced spend event in the guard's per-call ledger.

    Every :meth:`BudgetGuard.record` / :meth:`BudgetGuard.settle` /
    :meth:`BudgetGuard.record_tool` / :meth:`BudgetGuard.settle_tool`
    that accrues spend appends exactly one event,
    so ``sum(e.cost_usd for e in guard.spend_log)`` equals ``guard.spent_usd``
    (unless a ``max_log_events`` ring buffer has evicted old events).
    The schema is identical in the TS package (``SpendEvent`` in ``js/src/guard.ts``)
    and :meth:`BudgetGuard.export_log` serialises it with the same snake_case keys
    in both languages, so every agent emits the same shape regardless of stack.
    """

    timestamp: float  # Unix epoch seconds (UTC)
    kind: Literal["llm", "tool"]
    model_or_tool: str
    prompt_tokens: int | None  # None for tool events
    completion_tokens: int | None  # None for tool events
    cost_usd: float
    label: str | None = None  # caller-supplied tag (agent/task name)
    reserved: float | None = None  # the reservation settled by this call, if any

    def to_dict(self) -> dict[str, object]:
        """The stable wire shape used by :meth:`BudgetGuard.export_log`.

        Key order is fixed and the optional fields (``label``, ``reserved``) are
        omitted when absent — not emitted as null — matching the TS package's
        ``exportLog()`` field-for-field. (The *schema* is the contract, not the
        bytes: the two runtimes may render the same float differently, e.g.
        Python ``2.5e-06`` vs JS ``0.0000025``.)
        """
        out: dict[str, object] = {
            "timestamp": self.timestamp,
            "kind": self.kind,
            "model_or_tool": self.model_or_tool,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": self.cost_usd,
        }
        if self.label is not None:
            out["label"] = self.label
        if self.reserved is not None:
            out["reserved"] = self.reserved
        return out


class BudgetGuard:
    """Hard-stop an agent before its next LLM or tool call crosses a USD ceiling.

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
            ``BUDGET EXCEEDED — call blocked`` banner to stderr. Fires on **USD**
            ceiling crossings only; a token-ceiling block
            (:class:`TokenBudgetExceeded`, aggregate or step) is raised without
            invoking it, because the callback is dollar-shaped (spent/limit USD).
        near_limit_bps: utilization (basis points, 0..10000) at which
            :meth:`advisory` flags ``near_limit`` so an agent can taper before the
            hard-stop. Defaults to ``8000`` (80%).
        max_log_events: optional cap on the per-call spend ledger
            (:attr:`spend_log`). When set, the ledger is a ring buffer keeping the
            most recent N events so a long-running agent's memory stays bounded;
            the running totals are unaffected. ``None`` (default) keeps every event.
        token_limit: optional aggregate token ceiling (prompt + completion + cache
            buckets) enforced alongside ``limit_usd``. ``check`` / ``reserve`` take
            ``estimated_tokens`` to pre-emptively block; a cross raises
            :class:`TokenBudgetExceeded` (``scope="aggregate"``). ``None`` (default)
            disables the token dimension entirely.
        window: optional persistence window. ``"utc-day"`` shares one ceiling from
            midnight to midnight UTC across every process using the same ``store``;
            requires ``store``. ``None`` (default) is the in-memory guard.
        store: persistent :class:`~floe_guard.store.StateStore` (e.g.
            :class:`~floe_guard.store.SqliteStore`) used with ``window``. When set,
            ``spent_usd`` / reservations are the store's authoritative window
            snapshot, so a fresh process continues where the last left off and the
            cap resets at the UTC-day boundary. Persistence is **USD-only**: it
            cannot be combined with ``token_limit``, :meth:`step`, or a per-call
            ``estimated_tokens`` (each raises). ``window`` and ``store`` are
            all-or-nothing. Note ``spend_log`` / ``tool_costs`` remain
            process-local (not persisted, not window-reset).

    Thread-safe: the running total and in-flight reservations are guarded by a
    lock, so the guard can back a parallel crew (use :meth:`reserve` /
    :meth:`settle`). A configured ``store`` extends that hard-stop across
    processes (SQLite transactions), where the in-process lock cannot reach.
    """

    def __init__(
        self,
        limit_usd: float,
        *,
        price_overrides: dict[str, ManualPrice] | None = None,
        fail_closed: bool = True,
        on_block: Callable[[float, float], None] | None = None,
        near_limit_bps: int = 8000,
        max_log_events: int | None = None,
        token_limit: int | None = None,
        window: Literal["utc-day"] | None = None,
        store: StateStore | None = None,
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
        # Same int-not-bool contract as near_limit_bps (parity with TS
        # Number.isInteger). None disables the token ceiling entirely.
        if token_limit is not None and (
            isinstance(token_limit, bool) or not isinstance(token_limit, int) or token_limit < 0
        ):
            raise ValueError(f"token_limit must be None or a non-negative int, got {token_limit!r}")
        # Persistence config: window and store are all-or-nothing, and persistence
        # is USD-only for now — it cannot compose with the token dimension (the
        # store tracks USD, and a persistent handle can't also carry a token hold).
        if (window is None) != (store is None):
            raise ValueError("window and store must be configured together")
        if window not in (None, "utc-day"):
            raise ValueError(f"window must be 'utc-day' or None, got {window!r}")
        if store is not None and token_limit is not None:
            raise ValueError(
                "token_limit is not supported with a persistent store (persistence is USD-only)"
            )
        self._window = window
        self._store = store
        self._active_window_id: str | None = None
        self.limit_usd = float(limit_usd)
        # Reject a non-ManualPrice override at construction (clear TypeError at the
        # call site) rather than three frames deep in pricing on the first call.
        for _key, _override in (price_overrides or {}).items():
            _require_manual_price(_override, f"price_overrides[{_key!r}]")
        self.token_limit = token_limit
        # Aggregate tokens accrued (prompt + completion + cache buckets), and
        # tokens held in-flight — the token twin of spent_usd / _reserved.
        self.spent_tokens = 0
        self._reserved_tokens = 0
        # Step stack: innermost step is last. Empty when no step() is active.
        self._steps: list[_StepState] = []
        self.price_overrides = price_overrides
        self.fail_closed = fail_closed
        self._on_block = on_block or _default_on_block
        self.near_limit_bps = near_limit_bps
        # Wall-clock start of this guard's spend window, for the $/min burn rate
        # (advisory().burn_rate_usd_per_min). One guard per call/turn ⇒ per-call rate.
        self._created_time = time.time()
        self.spent_usd = 0.0
        # Costs of the most recent priced LLM call and tool call, tracked
        # SEPARATELY: the default next-call prediction is the max of the two, so
        # a cheap tool call can't shrink the estimate right before an expensive
        # LLM call (or vice versa) — conservative beats one-call-too-late.
        self._last_llm_cost = 0.0
        self._last_tool_cost = 0.0
        # USD held for calls that are in flight (reserved but not yet settled).
        # Counted against the ceiling so concurrent callers can't overshoot.
        self._reserved = 0.0
        # Same int-not-bool contract as near_limit_bps (parity with TS Number.isInteger).
        if max_log_events is not None and (
            isinstance(max_log_events, bool)
            or not isinstance(max_log_events, int)
            or max_log_events < 0
        ):
            raise ValueError(
                f"max_log_events must be None or a non-negative int, got {max_log_events!r}"
            )
        # Per-call ledger; deque(maxlen=None) is unbounded, otherwise a ring buffer.
        self._spend_log: deque[SpendEvent] = deque(maxlen=max_log_events)
        # Active streams' (accrued_usd, reserved_usd), keyed by registry token —
        # see _stream_register(). Lets parallel streams count each other's
        # in-flight accrual against the ceiling before anything settles.
        self._stream_costs: dict[object, tuple[float, float]] = {}
        # Per-tool running totals (settle_tool/record_tool) — the tool side of
        # the one shared ceiling, exposed via the tool_costs property.
        self._tool_costs: dict[str, float] = {}
        self._lock = threading.Lock()
        # Adopt the store's authoritative snapshot for today's window at startup,
        # so a fresh process (cron/serverless) continues where the last left off.
        if self._store is not None:
            with self._lock:
                self._refresh_persistent_state_locked()

    # ── hosted upgrade ────────────────────────────────────────────────────────

    @classmethod
    def from_floe(
        cls,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        fallback_limit_usd: float | None = None,
        timeout: float = 10.0,
        **guard_kwargs: object,
    ) -> BudgetGuard:
        """Build a guard whose local ceiling is set from your hosted Floe headroom.

        The one-line upgrade from a hand-picked ``limit_usd`` to a ceiling that
        tracks server-side headroom: reads
        :func:`~floe_guard.hosted.hosted_remaining_usd` once (the **minimum** of
        auto-borrow headroom and session spend remaining) and constructs a
        :class:`BudgetGuard` with that number as ``limit_usd``. Everything else —
        ``check`` / ``record`` / ``reserve`` / ``advisory`` — is unchanged.

        Budget, not balance. The hosted read is a *headroom / budget* signal, not
        an account balance, and this guard still enforces **locally**. Hosted Floe
        remains the source of truth for the un-bypassable, cross-vendor cap; the
        local ceiling is a fresh per-process budget bounded by what the server
        currently allows.

        Refresh policy: headroom is read **once, at construction**. It moves as you
        borrow/repay, so to re-read later either call ``from_floe`` again or set
        ``guard.limit_usd = hosted_remaining_usd(api_key)`` on a schedule you own.

        Zero-telemetry invariant: no network call happens unless a key is supplied
        (via ``api_key=`` or ``FLOE_API_KEY``). With no key the hosted read raises
        before touching the network.

        Fail-closed: a failed hosted read (missing/invalid key, 401/403/404,
        network/timeout) raises
        :class:`~floe_guard.errors.HostedEnforcementError` — the guard is *not*
        silently built with an unknown or unbounded ceiling. Pass
        ``fallback_limit_usd`` to instead degrade to a known local ceiling (with a
        loud warning), so a transient blip falls back to local enforcement rather
        than crashing the agent.

        Args:
            api_key: Floe agent key (``floe_<hex>``). Defaults to ``FLOE_API_KEY``.
            base_url: API base. Defaults to ``FLOE_API_BASE_URL``, else production.
            fallback_limit_usd: local ceiling to use if the hosted read fails.
                ``None`` (default) re-raises the error (fail closed).
            timeout: socket timeout for the hosted read, in seconds.
            **guard_kwargs: any other :class:`BudgetGuard` option (``fail_closed``,
                ``on_block``, ``near_limit_bps``, ``price_overrides``,
                ``token_limit``, ``window`` / ``store``, …). ``limit_usd`` is
                derived from headroom and must not be passed here.

        Raises:
            HostedEnforcementError: the hosted read failed and no
                ``fallback_limit_usd`` was given.
            TypeError: ``limit_usd`` was passed in ``guard_kwargs``.
        """
        if "limit_usd" in guard_kwargs:
            raise TypeError(
                "from_floe derives limit_usd from hosted headroom; pass "
                "fallback_limit_usd= for the offline fallback instead."
            )
        # Local import: hosted.py is a leaf module, but importing it at the top of
        # guard.py would couple the core guard to the network client for no gain.
        from .hosted import hosted_remaining_usd

        try:
            ceiling = hosted_remaining_usd(api_key, base_url=base_url, timeout=timeout)
        except HostedEnforcementError:
            if fallback_limit_usd is None:
                raise
            fallback = float(fallback_limit_usd)
            warnings.warn(
                "floe-guard: could not read hosted Floe headroom; falling back to "
                f"a local ceiling of ${fallback:.2f}. Enforcement is local-only "
                "until the hosted read succeeds.",
                stacklevel=2,
            )
            ceiling = fallback
        return cls(limit_usd=ceiling, **guard_kwargs)  # type: ignore[arg-type]

    # ── enforcement ───────────────────────────────────────────────────────────

    def check(
        self,
        estimated_next_cost: float | None = None,
        *,
        estimated_tokens: int = 0,
    ) -> None:
        """Raise if the next call would cross any active ceiling.

        Call this immediately before each LLM request. The "next call" USD cost
        is estimated conservatively as the costlier of the last LLM call and the
        last tool call (override with ``estimated_next_cost``); the first call
        is always allowed unless a ceiling is already met. A belt-and-suspenders
        check on the running total catches an overshoot if the estimate was too
        low. In-flight reservations count toward the total, so this stays
        correct alongside :meth:`reserve`.

        Pass ``estimated_tokens`` to also pre-emptively block on the aggregate
        ``token_limit`` or the active step's token cap (raises
        :class:`TokenBudgetExceeded`). A USD block raises :class:`BudgetExceeded`.

        Note: ``check`` is a non-binding peek. For parallel calls, use
        :meth:`reserve` / :meth:`settle`, which hold the estimate atomically.
        """
        self._validate_estimate(estimated_next_cost)
        self._validate_token_estimate(estimated_tokens)
        self._reject_tokens_with_store(estimated_tokens)
        with self._lock:
            # Refresh from the store so this peek sees other processes' spend. It's
            # still a non-binding peek; reserve()/settle() are the cross-process
            # hard guarantee (atomic in the store).
            if self._store is not None:
                self._refresh_persistent_state_locked()
            estimate = (
                self._default_estimate_locked()
                if estimated_next_cost is None
                else max(0.0, estimated_next_cost)
            )
            blocked = self._blocking_cross_locked(estimate, max(0, estimated_tokens))
        if blocked is not None:
            self._raise_block(blocked)

    def estimate_call(
        self,
        model: str,
        prompt_tokens: int,
        max_completion_tokens: int = 0,
        *,
        price: ManualPrice | None = None,
    ) -> float | None:
        """Price the ACTUAL incoming request, for a request-sized reserve()/check().

        :meth:`check` and :meth:`reserve` default to predicting the next call
        from the LAST call's cost — which is blind on the first call and wrong
        for a call much larger than the previous one. Feed this the request you
        are about to send (its real prompt size and output cap) and pass the
        result straight through::

            est = guard.estimate_call("gpt-4o", prompt_tokens, max_completion_tokens=1024)
            handle = guard.reserve(est)   # blocks NOW if this call alone would cross

        The estimate is worst-case on output (the model may stop well short of
        ``max_completion_tokens``); the hold is corrected to actual cost at
        :meth:`settle`. Returns ``None`` when the model is unpriceable — and
        ``reserve(None)`` / ``check(None)`` fall back to the last-cost
        prediction, so the wiring degrades gracefully instead of failing.
        """
        priced = self._resolve(model, price)
        if priced is None:
            return None
        return price_tokens(priced, prompt_tokens, max_completion_tokens)

    def reserve(
        self,
        estimated_cost: float | None = None,
        *,
        estimated_tokens: int = 0,
    ) -> ReservationHandle:
        """Atomically check every active ceiling AND hold the estimate in-flight.

        This is the concurrency-safe enforcement path. Each parallel caller
        reserves before its call, so N callers can't all clear the same stale
        total and overshoot. Raises :class:`BudgetExceeded` (USD) or
        :class:`TokenBudgetExceeded` (tokens) — without reserving — if the
        reservation would cross the aggregate ceiling or the active step's cap.

        Returns a reservation handle to pass to :meth:`settle` after the
        response, or to :meth:`release` if the call fails. **For backward
        compatibility the handle is a plain ``float`` (the USD held) when no
        tokens and no active :meth:`step` are involved** — old callers are
        unchanged. Otherwise it is a :class:`BudgetReservation` carrying both
        dimensions. ``estimated_cost`` defaults to the costlier of the last LLM
        call and the last tool call.
        """
        self._validate_estimate(estimated_cost)
        self._validate_token_estimate(estimated_tokens)
        self._reject_tokens_with_store(estimated_tokens)
        tokens = max(0, estimated_tokens)
        with self._lock:
            estimate = (
                self._default_estimate_locked()
                if estimated_cost is None
                else max(0.0, estimated_cost)
            )
            if self._store is not None:
                # Cross-process path: the store atomically checks the ceiling AND
                # holds the estimate in one transaction, so overlapping processes
                # can't both clear a stale total. It returns the fresh snapshot.
                window_id = self._current_window_id_locked()
                accepted, spent, reserved = self._store.reserve(
                    window_id, self.limit_usd, estimate
                )
                self.spent_usd, self._reserved = spent, reserved
                if accepted:
                    return _PersistentReservation(estimate, window_id)
                blocked: tuple[str, str, float, float] | None = (
                    "usd",
                    "aggregate",
                    self.spent_usd,
                    self.limit_usd,
                )
            else:
                blocked = self._blocking_cross_locked(estimate, tokens)
                if blocked is None:
                    self._reserved += estimate
                    self._reserved_tokens += tokens
                    step = self._steps[-1] if self._steps else None
                    if step is not None:
                        step.reserved_usd += estimate
                        step.reserved_tokens += tokens
                    # Plain float only when nothing token- or step-shaped is in play,
                    # so pre-existing USD-only callers get byte-for-byte the old handle.
                    if tokens == 0 and step is None:
                        return estimate
                    return BudgetReservation(usd=estimate, tokens=tokens)
        # Blocked — notify and raise outside the lock.
        self._raise_block(blocked)

    def settle(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        *,
        reserved: ReservationHandle = 0.0,
        price: ManualPrice | None = None,
        cache_creation_input_tokens: int = 0,
        cache_creation_input_tokens_1h: int = 0,
        cache_read_input_tokens: int = 0,
        label: str | None = None,
    ) -> float:
        """Release a reservation and record the actual cost. Concurrency-safe.

        ``record`` is ``settle`` with no reservation. Returns the USD cost of
        this call. Unpriceable-model handling matches :meth:`record` (warn +
        raise when ``fail_closed``, else warn + skip), and any held reservation
        is released even on the skip path. A priced call appends one
        :class:`SpendEvent` to :attr:`spend_log` (``label`` tags it, e.g. with an
        agent or task name); the warn-and-skip path accrues nothing and logs
        nothing, so the ledger stays in lockstep with ``spent_usd``.

        ``reserved`` accepts a plain float (USD-only, backward compatible) or a
        :class:`BudgetReservation` from a token/step-aware :meth:`reserve`; the
        token hold is drained the same way. Tokens accrue free from the counts
        this method already receives.
        """
        # A bad reserved handle would corrupt _reserved and break the ceiling for
        # OTHER in-flight calls (negative → phantom hold; inf → clears all holds).
        # _reserved_usd_of validates the handle (raw float, or a BudgetReservation's fields).
        reserved_usd = self._reserved_usd_of(reserved)
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
            cost = price_tokens(
                priced,
                prompt_tokens,
                completion_tokens,
                cache_creation_input_tokens=cache_creation_input_tokens,
                cache_creation_input_tokens_1h=cache_creation_input_tokens_1h,
                cache_read_input_tokens=cache_read_input_tokens,
            )
        except Exception:
            # price_tokens can raise (e.g. non-finite token counts). Release the
            # in-flight hold before propagating so _reserved doesn't leak and
            # shrink remaining_usd permanently — same fail-safe as the unpriceable
            # path above.
            self.release(reserved)
            raise
        # Tokens accrued match what USD bills: prompt + completion + every cache
        # bucket price_tokens charges (negative counts clamp to 0, as in pricing).
        accrued_tokens = (
            max(0, prompt_tokens)
            + max(0, completion_tokens)
            + max(0, cache_creation_input_tokens)
            + max(0, cache_creation_input_tokens_1h)
            + max(0, cache_read_input_tokens)
        )
        with self._lock:
            if self._store is not None:
                # The store is authoritative for USD: consume the hold and add the
                # actual cost atomically, then adopt the returned snapshot. Don't
                # `+= cost` — that would double-count what the store already applied.
                self._apply_persistent_terminal_locked(
                    reserved,
                    lambda window_id: self._store.settle(
                        window_id, self.limit_usd, reserved_usd, cost
                    ),
                )
            else:
                if reserved:
                    self._consume_reservation_locked(reserved)
                self.spent_usd += cost
                self._accrue_step_locked(cost, accrued_tokens)
                # Clamp a sub-epsilon float overshoot back to the limit so the running
                # total never reports as having crossed the ceiling by a rounding artifact.
                if 0.0 < self.spent_usd - self.limit_usd < _EPS:
                    self.spent_usd = self.limit_usd
            # Process-local either way (tokens aren't persisted; the store is USD-only).
            self.spent_tokens += accrued_tokens
            self._last_llm_cost = cost
            self._spend_log.append(
                SpendEvent(
                    timestamp=time.time(),
                    kind="llm",
                    model_or_tool=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cost_usd=cost,
                    label=label,
                    # 0.0 means "no reservation" (the plain record() path) — omit
                    # rather than log a meaningless zero. Log the USD amount so the
                    # ledger schema stays float-shaped even for a BudgetReservation.
                    reserved=reserved_usd if reserved_usd else None,
                )
            )
        return cost

    def record(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        *,
        price: ManualPrice | None = None,
        cache_creation_input_tokens: int = 0,
        cache_creation_input_tokens_1h: int = 0,
        cache_read_input_tokens: int = 0,
        label: str | None = None,
    ) -> float:
        """Price one response's tokens offline and add the cost to the total.

        Returns the USD cost of this call. If the model is unpriceable and no
        ``price`` is given, behaviour depends on ``fail_closed`` (see the class
        docstring): warn + raise (default), or warn + skip accrual.
        """
        return self.settle(
            model,
            prompt_tokens,
            completion_tokens,
            reserved=0.0,
            price=price,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_creation_input_tokens_1h=cache_creation_input_tokens_1h,
            cache_read_input_tokens=cache_read_input_tokens,
            label=label,
        )

    def reserve_tool(self, estimated_cost: float) -> ReservationHandle:
        """Atomically check the ceiling AND hold a tool call's cost in-flight.

        The tool-spend counterpart of :meth:`reserve` — and STRONGER than the
        LLM path, because a paid tool's price is usually known exactly before
        the call, so the pre-call hard-stop is precise rather than estimated::

            handle = guard.reserve_tool(0.02)      # raises BEFORE Apollo runs
            result = apollo.people_lookup(...)
            guard.settle_tool("apollo.people_lookup", 0.02, reserved=handle)

        Raises :class:`BudgetExceeded` (without reserving) if the call would
        cross the ceiling. The estimate is required — tools have no last-cost
        prediction worth falling back to. Pass the returned handle to
        :meth:`settle_tool`, or :meth:`release` if the call fails.
        """
        if estimated_cost is None:
            # reserve(None) would silently fall back to the last-cost prediction
            # (0 on a fresh guard) — an unguarded tool call. A missing price must
            # fail loudly, e.g. guard.reserve_tool(price_table.get(tool)).
            raise ValueError("reserve_tool requires an estimated cost, got None")
        if not math.isfinite(estimated_cost) or estimated_cost < 0:
            # reserve() clamps a negative estimate to 0 (lenient LLM contract) —
            # for a tool that would reserve nothing: the same unguarded call.
            raise ValueError(
                f"estimated_cost must be a finite, non-negative number, got {estimated_cost!r}"
            )
        return self.reserve(estimated_cost)

    def settle_tool(
        self,
        tool: str,
        cost_usd: float,
        *,
        reserved: ReservationHandle = 0.0,
        label: str | None = None,
    ) -> float:
        """Release a reservation and record a tool call's actual cost.

        Concurrency-safe; ``record_tool`` is ``settle_tool`` with no
        reservation. The caller supplies the cost — tools have no token usage
        to price. Accrues into the same ``spent_usd`` ceiling as tokens,
        tallies the per-tool total (:attr:`tool_costs`), updates the tool side
        of the next-call estimate (tracked separately from the LLM side; the
        default prediction is the max of the two, so a tool-hammering loop's
        plain :meth:`check` stops BEFORE the crossing call without a cheap tool
        shrinking the LLM prediction), and appends a ``kind="tool"`` :class:`SpendEvent` to
        :attr:`spend_log`. Returns ``cost_usd``.
        """
        if not math.isfinite(cost_usd) or cost_usd < 0:
            raise ValueError(f"cost_usd must be a finite, non-negative number, got {cost_usd!r}")
        # A bad reserved handle would corrupt _reserved and break the ceiling for
        # OTHER in-flight calls — same contract as settle(). _reserved_usd_of
        # validates the handle (raw float, or a BudgetReservation's fields).
        reserved_usd = self._reserved_usd_of(reserved)
        # int (and bool) are valid inputs; coerce so the logged event and the
        # return value are always float, like every other cost in the guard.
        cost_usd = float(cost_usd)
        with self._lock:
            if self._store is not None:
                # Store is authoritative for USD (see settle()): consume + accrue
                # atomically and adopt the snapshot rather than `+= cost_usd`.
                self._apply_persistent_terminal_locked(
                    reserved,
                    lambda window_id: self._store.settle(
                        window_id, self.limit_usd, reserved_usd, cost_usd
                    ),
                )
            else:
                if reserved:
                    self._consume_reservation_locked(reserved)
                self.spent_usd += cost_usd
                # A paid tool spends the step's USD too (no tokens to price).
                self._accrue_step_locked(cost_usd, 0)
                # Same sub-epsilon clamp as settle(): never report a rounding-artifact
                # crossing of the ceiling.
                if 0.0 < self.spent_usd - self.limit_usd < _EPS:
                    self.spent_usd = self.limit_usd
            self._last_tool_cost = cost_usd
            self._tool_costs[tool] = self._tool_costs.get(tool, 0.0) + cost_usd
            self._spend_log.append(
                SpendEvent(
                    timestamp=time.time(),
                    kind="tool",
                    model_or_tool=tool,
                    prompt_tokens=None,
                    completion_tokens=None,
                    cost_usd=cost_usd,
                    label=label,
                    reserved=reserved_usd if reserved_usd else None,
                )
            )
        return cost_usd

    def record_tool(self, tool: str, cost_usd: float, *, label: str | None = None) -> float:
        """Accrue a non-LLM cost (a paid tool/API call) against the same ceiling.

        Post-hoc accrual for costs only known after the call (metered APIs);
        when the price is known up front, :meth:`reserve_tool` /
        :meth:`settle_tool` give the stronger pre-call hard-stop. See
        :meth:`settle_tool` for the full contract. Returns ``cost_usd``.
        """
        return self.settle_tool(tool, cost_usd, reserved=0.0, label=label)

    def release(self, reserved: ReservationHandle) -> None:
        """Drop an in-flight reservation without recording spend (e.g. the call
        failed before producing usage). Safe to call with ``0``. Accepts a plain
        float or a :class:`BudgetReservation` (drains its token hold too)."""
        # Validate before the zero-check so a NaN handle raises instead of being
        # silently dropped (which would leak the hold). A bad handle here corrupts
        # _reserved for other in-flight calls. _reserved_usd_of validates a raw float.
        reserved_usd = self._reserved_usd_of(reserved)
        if not reserved:
            return
        with self._lock:
            if self._store is not None:
                self._apply_persistent_terminal_locked(
                    reserved,
                    lambda window_id: self._store.release(
                        window_id, self.limit_usd, reserved_usd
                    ),
                )
            else:
                self._consume_reservation_locked(reserved)

    @property
    def remaining_usd(self) -> float:
        """USD left before the ceiling, net of in-flight reservations (never negative)."""
        with self._lock:
            if self._store is not None:
                self._refresh_persistent_state_locked()
            return max(0.0, self.limit_usd - self.spent_usd - self._reserved)

    @property
    def tool_costs(self) -> dict[str, float]:
        """Per-tool running USD totals, keyed by the name given to
        :meth:`settle_tool` / :meth:`record_tool` — e.g.
        ``{"apollo.people_lookup": 0.42, "exa.search": 0.11}``. Makes the
        token/tool split of the one shared ceiling inspectable: in the in-memory
        guard, ``spent_usd - sum(tool_costs.values())`` is the token side.
        Returns a snapshot copy.

        With a persistent ``store``, ``spent_usd`` is the current UTC-day window's
        authoritative total while ``tool_costs`` is this process's own cumulative
        tally (not windowed, not shared), so that identity does not hold across a
        day rollover or a second process — use it for per-process attribution, not
        as a cross-window invariant."""
        with self._lock:
            return dict(self._tool_costs)

    @property
    def spend_log(self) -> list[SpendEvent]:
        """The per-call spend ledger, oldest first — one :class:`SpendEvent` per
        priced :meth:`record` / :meth:`settle` / :meth:`record_tool` /
        :meth:`settle_tool`.

        Returns a snapshot copy: safe to iterate while other threads keep
        recording, and mutating it cannot corrupt the ledger.
        """
        with self._lock:
            return list(self._spend_log)

    def export_log(self) -> str:
        """The spend ledger as JSONL — one event per line, newline-terminated.

        The schema is stable and language-independent (snake_case keys, fixed
        order; optional fields omitted when absent), identical to the TS
        package's ``exportLog()``, so heterogeneous agents produce logs you can
        concatenate and analyse as one stream. Empty ledger yields ``""``.
        """
        # Compact separators and raw (non-escaped) unicode match JS
        # JSON.stringify's layout; float rendering may still differ between the
        # runtimes (2.5e-06 vs 0.0000025) — the schema, not the bytes, is the
        # cross-language contract.
        return "".join(
            f"{json.dumps(event.to_dict(), separators=(',', ':'), ensure_ascii=False)}\n"
            for event in self.spend_log
        )

    @contextmanager
    def step(
        self, *, max_usd: float | None = None, max_tokens: int | None = None
    ) -> Iterator[BudgetGuard]:
        """Scope a per-step USD and/or token cap — the **per-call budget** primitive.

        Voice budgets are per-*call*, not per-day: wrap one call (or one turn) in a
        ``step`` to cap that call independently of the guard's aggregate ceiling.
        The step's headroom is ``advisory().step_remaining_usd``, and
        ``advisory().est_calls_remaining`` reports how many more calls the budget
        buys at the current per-call estimate::

            with guard.step(max_usd=0.05) as call:    # this call's own $0.05 cap
                call.check(); call.record("gpt-4o", ...)

        Push a step onto the guard's stack on enter, pop it on exit; the
        enforcement/accrual path honours the innermost active step *on top of*
        the aggregate ceilings. A call that would cross the step's ``max_usd`` /
        ``max_tokens`` is hard-blocked (``BudgetExceeded`` / ``TokenBudgetExceeded``
        with ``scope="step"``) even if the aggregate budget has room::

            with guard.step(max_tokens=5_000) as g:   # g IS guard
                g.check(estimated_tokens=...)          # may raise scope="step"
                g.record("gpt-4o", 1200, 350)

        Yields the SAME guard, so no adapter needs to know about steps — pass the
        guard through as usual. Steps nest (an inner step is checked first); the
        innermost is the one that owns each call. **Not for concurrent parallel
        steps on one guard** — that's a per-step identity registry, out of scope
        for issue #46. Use one guard per parallel branch instead.
        """
        if self._store is not None:
            # Steps are an in-process, token-aware construct; persistence is USD-only
            # and cross-process. Combining them needs a unified handle (not yet).
            raise ValueError(
                "step() is not supported with a persistent store (persistence is USD-only)"
            )
        if max_usd is not None and (not math.isfinite(max_usd) or max_usd < 0):
            raise ValueError(
                f"max_usd must be None or a finite, non-negative number, got {max_usd!r}"
            )
        if max_tokens is not None and (
            isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 0
        ):
            raise ValueError(f"max_tokens must be None or a non-negative int, got {max_tokens!r}")
        state = _StepState(
            max_usd=float(max_usd) if max_usd is not None else None, max_tokens=max_tokens
        )
        with self._lock:
            self._steps.append(state)
        try:
            yield self
        finally:
            with self._lock:
                # Pop our own state. Identity match (not just pop()) so an
                # out-of-order exit can't silently drop someone else's step.
                if self._steps and self._steps[-1] is state:
                    self._steps.pop()
                else:
                    self._steps.remove(state)

    def advisory(self) -> BudgetAdvisory:
        """Context-aware spend advisory for this budget — see :class:`BudgetAdvisory`.

        ``near_limit`` flips once utilization reaches ``near_limit_bps`` (default
        80%), so an agent can taper *before* the hard-stop. Advisory only: read it
        to adapt; :meth:`check` is what enforces the ceiling.
        """
        # Snapshot every shared field under the lock — running totals, last-call
        # costs, and the innermost step's mutable fields — so a concurrent
        # settle()/step() can't change them mid-read, and a step() exit can't pop
        # the last step between the truthiness check and the index (IndexError).
        # The arithmetic below runs on the snapshot, lock-free.
        with self._lock:
            # Adopt the store's authoritative window snapshot before snapshotting,
            # so the advisory reflects other processes' spend (USD-only path).
            if self._store is not None:
                self._refresh_persistent_state_locked()
            limit_usd = self.limit_usd
            spent_usd = self.spent_usd
            last_llm = self._last_llm_cost
            last_tool = self._last_tool_cost
            token_limit = self.token_limit
            spent_tokens = self.spent_tokens
            near_limit_bps = self.near_limit_bps
            step = self._steps[-1] if self._steps else None
            step_active = step is not None
            step_max_usd = step.max_usd if step is not None else None
            step_spent_usd = step.spent_usd if step is not None else 0.0
            step_max_tokens = step.max_tokens if step is not None else None
            step_spent_tokens = step.spent_tokens if step is not None else 0

        if limit_usd <= 0.0:
            used_bps = 10000
        else:
            # Floor (not round) so used_bps never over-reports utilization and
            # near_limit flips exactly when the threshold is reached, not a hair
            # early. The tiny epsilon absorbs float noise (0.7*10000 = 6999.9999…),
            # and floor matches JS Math.floor exactly — round() would diverge
            # (Python banker's rounding vs JS ties-up).
            used_bps = max(0, min(10000, int(spent_usd / limit_usd * 10000 + 1e-9)))
        remaining = max(0.0, limit_usd - spent_usd)
        # The estimate is the costlier of the last LLM and tool call (the guard's
        # own default reservation), 0.0 before any call.
        expected_cost = max(last_llm, last_tool)
        # +1e-9 absorbs float noise so e.g. 0.6/0.2 floors to 3 not 2 (same
        # epsilon rationale as used_bps above, and keeps JS Math.floor parity).
        est_calls_remaining = int(remaining / expected_cost + 1e-9) if expected_cost > 0.0 else None
        # Burn rate: spend ÷ minutes elapsed since the guard was created. None until
        # any wall-clock time has passed (avoids a divide-by-zero at t0); $0 spent
        # over real elapsed time is a legitimate 0.0/min, not None.
        elapsed_min = (time.time() - self._created_time) / 60.0
        burn_rate_usd_per_min = spent_usd / elapsed_min if elapsed_min > 0.0 else None
        # Aggregate token utilization, mirroring used_bps. None (no signal) when
        # the token dimension is unset — an aggregate-only USD guard is unchanged.
        token_used_bps: int | None = None
        remaining_tokens: int | None = None
        near_token = False
        if token_limit is not None:
            if token_limit <= 0:
                token_used_bps = 10000
            else:
                token_used_bps = max(0, min(10000, int(spent_tokens / token_limit * 10000 + 1e-9)))
            remaining_tokens = max(0, token_limit - spent_tokens)
            near_token = token_used_bps >= near_limit_bps
        # Innermost active step's headroom + nearness (per dimension, only when
        # that dimension is capped). None keeps the fields absent for old readers.
        step_remaining_usd: float | None = None
        step_remaining_tokens: int | None = None
        near_step = False
        if step_active:
            if step_max_usd is not None:
                step_remaining_usd = max(0.0, step_max_usd - step_spent_usd)
                if step_max_usd <= 0.0:
                    near_step = True
                else:
                    s_bps = int(step_spent_usd / step_max_usd * 10000 + 1e-9)
                    near_step = near_step or s_bps >= near_limit_bps
            if step_max_tokens is not None:
                step_remaining_tokens = max(0, step_max_tokens - step_spent_tokens)
                if step_max_tokens <= 0:
                    near_step = True
                else:
                    s_tbps = int(step_spent_tokens / step_max_tokens * 10000 + 1e-9)
                    near_step = near_step or s_tbps >= near_limit_bps
        return BudgetAdvisory(
            # near_limit now also flips when a token ceiling or the active step is
            # near its cap, so a router can downshift before ANY hard-stop.
            near_limit=used_bps >= near_limit_bps or near_token or near_step,
            used_bps=used_bps,
            # Settled budget: limit minus accrued spend, deliberately NOT net of
            # in-flight reservations. This differs from the remaining_usd property
            # (which subtracts _reserved): the advisory is a soft utilization signal
            # about money already spent, while the property reports what a new call
            # can still claim.
            remaining_usd=remaining,
            limit_usd=limit_usd,
            spent_usd=spent_usd,
            expected_cost=expected_cost,
            est_calls_remaining=est_calls_remaining,
            token_used_bps=token_used_bps,
            remaining_tokens=remaining_tokens,
            step_remaining_usd=step_remaining_usd,
            step_remaining_tokens=step_remaining_tokens,
            burn_rate_usd_per_min=burn_rate_usd_per_min,
        )

    # ── internals ──────────────────────────────────────────────────────────────

    def _default_estimate_locked(self) -> float:
        """The default next-call prediction when the caller supplies no
        estimate. Caller must hold ``self._lock``. Conservative: the costlier
        of the last LLM call and the last tool call — a mixed loop predicts the
        pricier kind, which at worst blocks one call early (fail-closed) rather
        than letting a crossing call through because the LAST event happened to
        be cheap.
        """
        return max(self._last_llm_cost, self._last_tool_cost)

    def _consume_reservation_locked(self, reserved: ReservationHandle) -> None:
        """Subtract a settled/released hold from the in-flight tallies. Caller
        must hold ``self._lock``. A handle larger than EVERYTHING currently
        held cannot have come from a matching :meth:`reserve` — raising beats
        silently clamping, which would free OTHER callers' holds and fail the
        ceiling open. The epsilon absorbs float dust from accumulating and
        draining many holds; per-caller over-release (a handle within the
        total but larger than the caller's own hold) is undetectable without
        per-handle tracking and remains the caller's responsibility.

        Drains both dimensions the SAME way — a ``BudgetReservation`` carries a
        token hold too, and (like the USD float) it also unwinds the innermost
        active step. A plain ``float`` handle drains only USD (the backward-compat
        path). We do NOT track which step a handle came from: a handle is drained
        against whatever step is innermost now, matching the sequential-loop
        contract (concurrent parallel steps on one guard are out of scope, #46).
        """
        usd = reserved.usd if isinstance(reserved, BudgetReservation) else reserved
        tokens = reserved.tokens if isinstance(reserved, BudgetReservation) else 0
        if usd > self._reserved + _EPS:
            raise ValueError(
                f"reserved handle ({usd!r}) exceeds total in-flight reservations "
                f"({self._reserved!r}) — a handle must come from a matching reserve()"
            )
        if tokens > self._reserved_tokens:
            raise ValueError(
                f"reserved token handle ({tokens!r}) exceeds total in-flight token "
                f"reservations ({self._reserved_tokens!r}) — a handle must come from a "
                f"matching reserve()"
            )
        self._reserved = max(0.0, self._reserved - usd)
        self._reserved_tokens = max(0, self._reserved_tokens - tokens)
        step = self._steps[-1] if self._steps else None
        if step is not None:
            step.reserved_usd = max(0.0, step.reserved_usd - usd)
            step.reserved_tokens = max(0, step.reserved_tokens - tokens)

    def _validate_estimate(self, estimated: float | None) -> None:
        # NaN/inf would poison the ceiling comparisons and fail-open (or poison
        # _reserved) — reject a non-finite caller-supplied estimate up front,
        # matching the constructor's math.isfinite guard and the TS Number.isFinite.
        if estimated is not None and not math.isfinite(estimated):
            raise ValueError(f"estimated cost must be a finite number, got {estimated!r}")

    @staticmethod
    def _validate_token_estimate(estimated_tokens: int) -> None:
        # Reject a float (incl. NaN/inf) or bool token estimate BEFORE reserve()
        # mutates _reserved_tokens / the step's hold. Otherwise the mutation lands
        # and BudgetReservation.__post_init__ rejects the handle only afterwards —
        # leaking the hold — and a NaN cast into the integer token comparisons
        # would fail-open the ceiling. A negative int is clamped by max(0, ...),
        # matching the lenient USD estimate. Mirrors the TS Number.isInteger guard.
        if isinstance(estimated_tokens, bool) or not isinstance(estimated_tokens, int):
            raise ValueError(f"estimated_tokens must be an int, got {estimated_tokens!r}")

    def _reject_tokens_with_store(self, estimated_tokens: int) -> None:
        # Persistence is USD-only. token_limit is already forbidden at construction;
        # this also rejects a per-call token estimate so the store path never has to
        # return a token-shaped handle it can't round-trip to the store.
        if self._store is not None and estimated_tokens:
            raise ValueError(
                "estimated_tokens is not supported with a persistent store "
                "(persistence is USD-only)"
            )

    # ── persistent-window internals ─────────────────────────────────────────────

    def _current_window_id_locked(self) -> str:
        """Today's store key. On a UTC-day rollover, reset the process-local
        next-call estimate (a new day starts fresh); the spend_log / tool_costs
        ledgers are deliberately NOT cleared here — they are process-local history,
        and a read path must never destroy them as a side effect."""
        window_id = _utc_day_window_id()
        if window_id != self._active_window_id:
            self._active_window_id = window_id
            self._last_llm_cost = 0.0
            self._last_tool_cost = 0.0
        return window_id

    def _refresh_persistent_state_locked(self) -> None:
        """Adopt the authoritative store snapshot for the current window."""
        if self._store is None:
            return
        self.spent_usd, self._reserved = self._store.load(
            self._current_window_id_locked(), self.limit_usd
        )

    def _reservation_window_id_locked(self, reserved: ReservationHandle) -> str:
        """A persistent handle settles/releases against its ISSUING window, so a
        hold taken just before midnight is consumed from the right day even after
        the guard has rolled to a new window."""
        if isinstance(reserved, _PersistentReservation):
            return reserved._window_id
        return self._current_window_id_locked()

    def _apply_persistent_terminal_locked(
        self,
        reserved: ReservationHandle,
        operation: Callable[[str], tuple[float, float]],
    ) -> None:
        """Run a store terminal op (settle/release) against the reservation's
        issuing window, then adopt the returned snapshot ONLY when that window is
        the active one — a cross-midnight settlement must not overwrite today's
        live totals with yesterday's. The single home for this ordered protocol
        (load-before-consume) so the three call sites can't drift."""
        window_id = self._reservation_window_id_locked(reserved)
        updates_active_window = self._prepare_persistent_terminal_locked(window_id)
        snapshot = operation(window_id)
        if updates_active_window:
            self.spent_usd, self._reserved = snapshot

    def _prepare_persistent_terminal_locked(self, window_id: str) -> bool:
        """True when ``window_id`` is the active window (so its snapshot should be
        adopted). When it differs (a handle from a prior window), load the current
        window first so a stale terminal op can't leave the live totals wrong."""
        current_window_id = self._current_window_id_locked()
        if window_id == current_window_id:
            return True
        if self._store is None:  # pragma: no cover - persistent callers only
            raise RuntimeError("persistent terminal operation requires a configured store")
        self.spent_usd, self._reserved = self._store.load(current_window_id, self.limit_usd)
        return False

    @staticmethod
    def _reserved_usd_of(reserved: ReservationHandle) -> float:
        """The USD amount of a handle, validated. A ``BudgetReservation`` self-
        validates at construction (:meth:`BudgetReservation.__post_init__`), so its
        ``usd`` is already finite/non-negative; a raw float gets the same
        finite/non-negative guard here so a bad hand-rolled handle can't corrupt
        the in-flight tally."""
        if isinstance(reserved, BudgetReservation):
            return reserved.usd
        if not math.isfinite(reserved) or reserved < 0:
            raise ValueError(f"reserved must be a finite, non-negative number, got {reserved!r}")
        return reserved

    def _accrue_step_locked(self, cost: float, tokens: int) -> None:
        """Accrue a settled call into the innermost active step. Caller must
        hold ``self._lock``. No-op when no step is active — the reservation was
        already drained by _consume_reservation_locked, this adds the *settled*
        amount. Sequential-loop contract: the innermost step is the one that
        owns the call (concurrent parallel steps on one guard are out of scope)."""
        step = self._steps[-1] if self._steps else None
        if step is not None:
            step.spent_usd += cost
            step.spent_tokens += tokens

    def _stream_register(self, reserved: float) -> object:
        """Register an active stream (see :class:`~floe_guard.stream.StreamGuard`)
        and return its registry key. Active streams' accrued-but-unsettled costs
        count against the ceiling for each OTHER stream, so parallel unreserved
        streams share the budget instead of each spending the full ceiling."""
        key = object()
        with self._lock:
            self._stream_costs[key] = (0.0, max(0.0, reserved))
        return key

    def _stream_unregister(self, key: object) -> None:
        """Drop a stream's registry entry once its cost is settled (settle()
        moves the accrual into ``spent_usd``, so keeping it would double-count)."""
        with self._lock:
            self._stream_costs.pop(key, None)

    def _stream_would_cross(self, key: object, cumulative_call_cost: float) -> bool:
        """Atomically record stream ``key``'s cumulative cost so far and answer:
        would it cross the ceiling? Counted against the limit: settled spend,
        other calls' reservations (this stream's own hold is excluded — its real
        accrued cost replaces the estimate), and each OTHER active stream's
        accrual beyond its own reservation (the reservation part is already
        inside ``_reserved``). Used by :class:`~floe_guard.stream.StreamGuard`.
        """
        with self._lock:
            own_reserved = self._stream_costs.get(key, (0.0, 0.0))[1]
            self._stream_costs[key] = (cumulative_call_cost, own_reserved)
            other_overage = sum(
                max(0.0, accrued - held)
                for k, (accrued, held) in self._stream_costs.items()
                if k is not key
            )
            others = self.spent_usd + max(0.0, self._reserved - own_reserved) + other_overage
            return others + cumulative_call_cost > self.limit_usd + _EPS

    def _blocking_cross_locked(
        self, estimate_usd: float, estimate_tokens: int
    ) -> tuple[str, str, float, float] | None:
        """The ONE choke point, now across two dimensions and two scopes. Caller
        must hold ``self._lock``. Returns the first ceiling that blocks as
        ``(dimension, scope, spent, limit)`` — dimension ``"usd" | "tokens"``,
        scope ``"aggregate" | "step"`` — or ``None`` if the call fits everywhere.
        The ``spent`` / ``limit`` pair is the crossed ceiling's own figures (USD
        for a USD block, token counts for a token block), captured HERE under the
        lock so :meth:`_raise_block` (which runs after the lock is released) builds
        the error and fires ``on_block`` with a snapshot — never re-reading shared
        state, so a concurrent write or step() exit can't make the reported values
        inconsistent or raise an ``IndexError``.

        Order (aggregate before step) is deliberate: the guard-wide ceiling is
        the hard money/token limit, so it's reported first when both would block.
        The token ceilings are only consulted when ``token_limit`` / the step's
        ``max_tokens`` is set — an unset dimension never blocks.
        """
        # Aggregate USD — same comparison the original _would_cross used. The
        # message/callback report the accrued total (spent_usd), as _block did.
        committed = self.spent_usd + self._reserved
        if committed > self.limit_usd - _EPS or committed + estimate_usd > self.limit_usd + _EPS:
            return ("usd", "aggregate", self.spent_usd, self.limit_usd)
        # Aggregate tokens (integers — no epsilon needed).
        if self.token_limit is not None:
            committed_t = self.spent_tokens + self._reserved_tokens
            if committed_t >= self.token_limit or committed_t + estimate_tokens > self.token_limit:
                return ("tokens", "aggregate", committed_t, self.token_limit)
        # Innermost active step's caps (an outer step can only be crossed by
        # first crossing the inner one, so checking the innermost is sufficient).
        step = self._steps[-1] if self._steps else None
        if step is not None:
            if step.max_usd is not None:
                s_committed = step.spent_usd + step.reserved_usd
                if (
                    s_committed > step.max_usd - _EPS
                    or s_committed + estimate_usd > step.max_usd + _EPS
                ):
                    # USD message stays aggregate-shaped (matches the original
                    # _raise_block, which reported spent_usd/limit_usd for a step
                    # USD block too).
                    return ("usd", "step", self.spent_usd, self.limit_usd)
            if step.max_tokens is not None:
                s_committed_t = step.spent_tokens + step.reserved_tokens
                if (
                    s_committed_t >= step.max_tokens
                    or s_committed_t + estimate_tokens > step.max_tokens
                ):
                    return ("tokens", "step", s_committed_t, step.max_tokens)
        return None

    def _block(self) -> NoReturn:
        """Notify + raise a USD :class:`BudgetExceeded`. The aggregate-USD path
        used by :class:`~floe_guard.stream.StreamGuard`; :meth:`_raise_block` is
        the dimension-aware generalisation used by check/reserve."""
        self._on_block(self.spent_usd, self.limit_usd)
        raise BudgetExceeded(self.spent_usd, self.limit_usd)

    def _raise_block(self, blocked: tuple[str, str, float, float]) -> NoReturn:
        """Notify + raise the right error for a (dimension, scope) block. Called
        OUTSIDE the lock (like the original _block). ``spent`` / ``limit`` were
        snapshotted under the lock by :meth:`_blocking_cross_locked`, so this path
        reads no shared state — the reported figures stay consistent with the
        blocking decision and can't race a concurrent write or step() exit."""
        dimension, scope, spent, limit = blocked
        if dimension == "usd":
            self._on_block(spent, limit)
            raise BudgetExceeded(spent, limit)
        # Token block — counts came in with `blocked`; on_block is dollar-shaped
        # so a token block skips it.
        raise TokenBudgetExceeded(int(spent), int(limit), scope)

    def _resolve(self, model: str, price: ManualPrice | None):
        overrides = self.price_overrides
        if price is not None:
            _require_manual_price(price, "price")
            overrides = {**(overrides or {}), model: price}
        return resolve_price(model, overrides)


def _require_manual_price(value: object, where: str) -> None:
    """Fail at the call site, not three frames into pricing.

    A cost_map.json entry is a plain dict with the right key names, so passing
    one straight through is the obvious move and used to surface as
    ``AttributeError: 'dict' object has no attribute 'input_cost_per_token'``
    from inside resolve_price.
    """
    if isinstance(value, ManualPrice):
        return
    hint = ""
    if isinstance(value, Mapping):
        try:
            hint = (
                f" Build one from it: ManualPrice("
                f"input_cost_per_token={value['input_cost_per_token']!r}, "
                f"output_cost_per_token={value['output_cost_per_token']!r})"
            )
        except KeyError:
            hint = (
                " A mapping needs both 'input_cost_per_token' and "
                "'output_cost_per_token' to become a ManualPrice."
            )
    raise TypeError(f"{where} must be a ManualPrice, got {type(value).__name__}.{hint}")


def _default_on_block(spent_usd: float, limit_usd: float) -> None:
    print(
        "BUDGET EXCEEDED — call blocked\n"
        f"  spent so far: ${spent_usd:.6f}  |  ceiling: ${limit_usd:.6f}\n"
        "  The next call would cross your budget; floe-guard stopped your agent "
        "before it ran.",
        file=sys.stderr,
    )


def _utc_day_window_id() -> str:
    """The current UTC calendar date, used as the persistent window key."""
    return datetime.now(timezone.utc).date().isoformat()
