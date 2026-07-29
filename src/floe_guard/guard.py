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
import weakref
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from .errors import (
    BudgetExceeded,
    TokenBudgetExceeded,
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
    # The guard's own next-call estimate (the costlier of the last LLM and last
    # tool call — same value the default reservation uses). 0.0 until the first
    # call is recorded, so a planner can't divide by a cold estimate.
    expected_cost: float = 0.0
    # How many more calls the remaining budget buys at expected_cost:
    # floor(remaining_usd / expected_cost). None when expected_cost is 0.0
    # (no call recorded yet) — unknown, not zero.
    est_calls_remaining: int | None = None
    token_limit: int | None = None
    spent_tokens: int = 0
    remaining_tokens: int | None = None
    token_used_bps: int | None = None
    near_token_limit: bool = False
    step_limit_usd: float | None = None
    step_spent_usd: float | None = None
    step_remaining_usd: float | None = None
    step_token_limit: int | None = None
    step_spent_tokens: int | None = None
    step_remaining_tokens: int | None = None
    step_used_bps: int | None = None
    step_near_limit: bool = False


@dataclass(eq=False)
class _StepState:
    limit_usd: float | None
    token_limit: int | None
    spent_usd: float = 0.0
    spent_tokens: int = 0
    reserved_usd: float = 0.0
    reserved_tokens: int = 0
    active: bool = True


@dataclass(frozen=True, eq=False)
class BudgetReservation:
    """Opaque immutable reservation returned by token/step guards.

    Read ``usd`` and ``tokens`` for diagnostics, but do not construct, copy, or
    modify handles. Only the issuing guard's private registry is authoritative.
    """

    usd: float
    tokens: int
    _owner: BudgetGuard


@dataclass(frozen=True)
class _ReservationState:
    issued: weakref.ReferenceType[BudgetReservation]
    usd: float
    tokens: int
    step: _StepState | None


ReservationHandle = float | BudgetReservation


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
        token_limit: optional aggregate LLM-token ceiling. Tool calls consume
            USD only. ``0`` blocks the first LLM call.
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
        on_token_block: token counterpart invoked with
            ``(spent_tokens, token_limit)`` before
            :class:`~floe_guard.TokenBudgetExceeded`.
        near_limit_bps: utilization (basis points, 0..10000) at which
            :meth:`advisory` flags ``near_limit`` so an agent can taper before the
            hard-stop. Defaults to ``8000`` (80%).
        max_log_events: optional cap on the per-call spend ledger
            (:attr:`spend_log`). When set, the ledger is a ring buffer keeping the
            most recent N events so a long-running agent's memory stays bounded;
            the running totals are unaffected. ``None`` (default) keeps every event.

    Thread-safe: the running total and in-flight reservations are guarded by a
    lock, so the guard can back a parallel crew (use :meth:`reserve` /
    :meth:`settle`).
    """

    def __init__(
        self,
        limit_usd: float,
        *,
        token_limit: int | None = None,
        price_overrides: dict[str, ManualPrice] | None = None,
        fail_closed: bool = True,
        on_block: Callable[[float, float], None] | None = None,
        on_token_block: Callable[[int, int], None] | None = None,
        near_limit_bps: int = 8000,
        max_log_events: int | None = None,
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
        _validate_optional_tokens("token_limit", token_limit)
        self.token_limit = token_limit
        self.price_overrides = price_overrides
        self.fail_closed = fail_closed
        self._on_block = on_block or _default_on_block
        self._on_token_block = on_token_block or _default_on_token_block
        self.near_limit_bps = near_limit_bps
        self.spent_usd = 0.0
        self.spent_tokens = 0
        # Costs of the most recent priced LLM call and tool call, tracked
        # SEPARATELY: the default next-call prediction is the max of the two, so
        # a cheap tool call can't shrink the estimate right before an expensive
        # LLM call (or vice versa) — conservative beats one-call-too-late.
        self._last_llm_cost = 0.0
        self._last_tool_cost = 0.0
        self._last_llm_tokens = 0
        # USD held for calls that are in flight (reserved but not yet settled).
        # Counted against the ceiling so concurrent callers can't overshoot.
        self._reserved = 0.0
        self._reserved_tokens = 0
        # Typed handles are immutable caller-visible identities. Accounting and
        # lifecycle state live here so forged, copied, or modified handles can
        # never release another caller's hold.
        self._reservations: weakref.WeakKeyDictionary[BudgetReservation, _ReservationState] = (
            weakref.WeakKeyDictionary()
        )
        self._consumed_reservations: weakref.WeakSet[BudgetReservation] = weakref.WeakSet()
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
        # see _stream_prepare(). Lets parallel streams count each other's
        # in-flight accrual against the ceiling before anything settles.
        self._stream_costs: dict[object, tuple[float, int, float, int, _StepState | None]] = {}
        # Per-tool running totals (settle_tool/record_tool) — the tool side of
        # the one shared ceiling, exposed via the tool_costs property.
        self._tool_costs: dict[str, float] = {}
        self._lock = threading.Lock()

    # ── enforcement ───────────────────────────────────────────────────────────

    def check(
        self,
        estimated_next_cost: float | None = None,
        *,
        estimated_next_tokens: int | None = None,
    ) -> None:
        """Raise :class:`BudgetExceeded` if the next call would cross the ceiling.

        Call this immediately before each LLM request. The "next call" is
        estimated conservatively as the costlier of the last LLM call and the
        last tool call (override with ``estimated_next_cost``); the first call
        is always allowed unless the ceiling is already met. A belt-and-suspenders
        check on the running total catches an overshoot if the estimate was too
        low. In-flight reservations count toward the total, so this stays
        correct alongside :meth:`reserve`.

        Note: ``check`` is a non-binding peek. For parallel calls, use
        :meth:`reserve` / :meth:`settle`, which hold the estimate atomically.
        """
        self._check(estimated_next_cost, estimated_next_tokens, step=None)

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
        estimated_tokens: int | None = None,
    ) -> ReservationHandle:
        """Atomically check the ceiling AND hold the estimated cost in-flight.

        This is the concurrency-safe enforcement path. Each parallel caller
        reserves before its call, so N callers can't all clear the same stale
        total and overshoot. Raises :class:`BudgetExceeded` (without reserving)
        if the reservation would cross the ceiling.

        Returns a reservation handle (the USD amount held) to pass to
        :meth:`settle` after the response, or to :meth:`release` if the call
        fails. ``estimated_cost`` defaults to the costlier of the last LLM call
        and the last tool call.
        """
        return self._reserve(estimated_cost, estimated_tokens, step=None)

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
        """
        reserved_usd, reserved_tokens, step = self._reservation_parts(reserved)
        try:
            total_tokens = sum(
                max(0, int(value))
                for value in (
                    prompt_tokens,
                    completion_tokens,
                    cache_creation_input_tokens,
                    cache_creation_input_tokens_1h,
                    cache_read_input_tokens,
                )
            )
        except (TypeError, ValueError, OverflowError):
            self.release(reserved)
            raise
        priced = self._resolve(model, price)
        if priced is None:
            with self._lock:
                self._consume_handle_locked(reserved, reserved_usd, reserved_tokens, step)
                self._accrue_tokens_locked(total_tokens, step)
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
        with self._lock:
            self._consume_handle_locked(reserved, reserved_usd, reserved_tokens, step)
            self.spent_usd += cost
            self._accrue_tokens_locked(total_tokens, step)
            # Clamp a sub-epsilon float overshoot back to the limit so the running
            # total never reports as having crossed the ceiling by a rounding artifact.
            if 0.0 < self.spent_usd - self.limit_usd < _EPS:
                self.spent_usd = self.limit_usd
            self._last_llm_cost = cost
            if step is not None:
                step.spent_usd += cost
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
                    # rather than log a meaningless zero.
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
        return self._reserve(estimated_cost, 0, step=None, enforce_tokens=False)

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
        reserved_usd, reserved_tokens, step = self._reservation_parts(reserved)
        # int (and bool) are valid inputs; coerce so the logged event and the
        # return value are always float, like every other cost in the guard.
        cost_usd = float(cost_usd)
        with self._lock:
            self._consume_handle_locked(reserved, reserved_usd, reserved_tokens, step)
            self.spent_usd += cost_usd
            if step is not None:
                step.spent_usd += cost_usd
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
        failed before producing usage). Safe to call with ``0``."""
        # Validate before the zero-check so a NaN handle raises instead of being
        # silently dropped (which would leak the hold). A bad handle here corrupts
        # _reserved for other in-flight calls.
        reserved_usd, reserved_tokens, step = self._reservation_parts(reserved)
        if not isinstance(reserved, BudgetReservation) and not reserved_usd:
            return
        with self._lock:
            self._consume_handle_locked(reserved, reserved_usd, reserved_tokens, step)

    @property
    def remaining_usd(self) -> float:
        """USD left before the ceiling, net of in-flight reservations (never negative)."""
        with self._lock:
            return max(0.0, self.limit_usd - self.spent_usd - self._reserved)

    @property
    def remaining_tokens(self) -> int | None:
        """Tokens left before the aggregate ceiling, net of reservations."""
        if self.token_limit is None:
            return None
        with self._lock:
            return max(0, self.token_limit - self.spent_tokens - self._reserved_tokens)

    @property
    def tool_costs(self) -> dict[str, float]:
        """Per-tool running USD totals, keyed by the name given to
        :meth:`settle_tool` / :meth:`record_tool` — e.g.
        ``{"apollo.people_lookup": 0.42, "exa.search": 0.11}``. Makes the
        token/tool split of the one shared ceiling inspectable
        (``spent_usd - sum(tool_costs.values())`` is the token side).
        Returns a snapshot copy."""
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

    def advisory(self) -> BudgetAdvisory:
        """Return aggregate USD/token headroom and the overall taper signal."""
        return self._advisory(step=None)

    def step(
        self, *, max_usd: float | None = None, max_tokens: int | None = None
    ) -> StepBudgetGuard:
        """Create an explicit scoped guard with fresh per-step ceilings."""
        return StepBudgetGuard(self, max_usd=max_usd, max_tokens=max_tokens)

    def _advisory(self, step: _StepState | None) -> BudgetAdvisory:
        used_bps = _used_bps(self.spent_usd, self.limit_usd)
        remaining = max(0.0, self.limit_usd - self.spent_usd)
        token_used_bps = (
            _used_bps(self.spent_tokens, self.token_limit) if self.token_limit is not None else None
        )
        remaining_tokens = (
            max(0, self.token_limit - self.spent_tokens) if self.token_limit is not None else None
        )
        near_token_limit = token_used_bps is not None and token_used_bps >= self.near_limit_bps
        step_usd_bps = (
            _used_bps(step.spent_usd, step.limit_usd)
            if step is not None and step.limit_usd is not None
            else None
        )
        step_token_bps = (
            _used_bps(step.spent_tokens, step.token_limit)
            if step is not None and step.token_limit is not None
            else None
        )
        step_used_bps = (
            max(v for v in (step_usd_bps, step_token_bps) if v is not None)
            if step is not None
            else None
        )
        step_near_limit = step_used_bps is not None and step_used_bps >= self.near_limit_bps
        expected_cost = max(self._last_llm_cost, self._last_tool_cost)
        est_calls_remaining = int(remaining / expected_cost + 1e-9) if expected_cost > 0.0 else None
        return BudgetAdvisory(
            near_limit=(used_bps >= self.near_limit_bps or near_token_limit or step_near_limit),
            used_bps=used_bps,
            remaining_usd=remaining,
            limit_usd=self.limit_usd,
            spent_usd=self.spent_usd,
            expected_cost=expected_cost,
            est_calls_remaining=est_calls_remaining,
            token_limit=self.token_limit,
            spent_tokens=self.spent_tokens,
            remaining_tokens=remaining_tokens,
            token_used_bps=token_used_bps,
            near_token_limit=near_token_limit,
            step_limit_usd=step.limit_usd if step is not None else None,
            step_spent_usd=step.spent_usd if step is not None else None,
            step_remaining_usd=(
                max(0.0, step.limit_usd - step.spent_usd)
                if step is not None and step.limit_usd is not None
                else None
            ),
            step_token_limit=step.token_limit if step is not None else None,
            step_spent_tokens=step.spent_tokens if step is not None else None,
            step_remaining_tokens=(
                max(0, step.token_limit - step.spent_tokens)
                if step is not None and step.token_limit is not None
                else None
            ),
            step_used_bps=step_used_bps,
            step_near_limit=step_near_limit,
        )

    # ── internals ──────────────────────────────────────────────────────────────

    def _check(
        self,
        estimated_cost: float | None,
        estimated_tokens: int | None,
        *,
        step: _StepState | None,
    ) -> None:
        self._validate_estimate(estimated_cost)
        _validate_optional_tokens("estimated tokens", estimated_tokens)
        with self._lock:
            cost = (
                self._default_estimate_locked()
                if estimated_cost is None
                else max(0.0, estimated_cost)
            )
            tokens = self._last_llm_tokens if estimated_tokens is None else estimated_tokens
            blocked = self._blocking_limit_locked(cost, tokens, step)
        if blocked is not None:
            self._raise_block(blocked)

    def _reserve(
        self,
        estimated_cost: float | None,
        estimated_tokens: int | None,
        *,
        step: _StepState | None,
        enforce_tokens: bool = True,
    ) -> ReservationHandle:
        self._validate_estimate(estimated_cost)
        _validate_optional_tokens("estimated tokens", estimated_tokens)
        with self._lock:
            if step is not None and not step.active:
                raise RuntimeError("step budget is no longer active")
            cost = (
                self._default_estimate_locked()
                if estimated_cost is None
                else max(0.0, estimated_cost)
            )
            tokens = self._last_llm_tokens if estimated_tokens is None else estimated_tokens
            blocked = self._blocking_limit_locked(cost, tokens, step, enforce_tokens=enforce_tokens)
            if blocked is None:
                self._reserved += cost
                self._reserved_tokens += tokens
                if step is not None:
                    step.reserved_usd += cost
                    step.reserved_tokens += tokens
                if self.token_limit is None and step is None:
                    return cost
                return self._make_reservation_locked(cost, tokens, step)
        self._raise_block(blocked)
        raise AssertionError("unreachable")

    def _blocking_limit_locked(
        self,
        cost: float,
        tokens: int,
        step: _StepState | None,
        *,
        enforce_tokens: bool = True,
    ) -> tuple[str, str, float | int, float | int] | None:
        committed_usd = self.spent_usd + self._reserved
        if committed_usd > self.limit_usd - _EPS or committed_usd + cost > self.limit_usd + _EPS:
            return ("usd", "aggregate", self.spent_usd, self.limit_usd)
        committed_tokens = self.spent_tokens + self._reserved_tokens
        if (
            enforce_tokens
            and self.token_limit is not None
            and (
                committed_tokens >= self.token_limit or committed_tokens + tokens > self.token_limit
            )
        ):
            return ("tokens", "aggregate", self.spent_tokens, self.token_limit)
        if step is not None:
            committed_step_usd = step.spent_usd + step.reserved_usd
            if step.limit_usd is not None and (
                committed_step_usd > step.limit_usd - _EPS
                or committed_step_usd + cost > step.limit_usd + _EPS
            ):
                return ("usd", "step", step.spent_usd, step.limit_usd)
            committed_step_tokens = step.spent_tokens + step.reserved_tokens
            if (
                enforce_tokens
                and step.token_limit is not None
                and (
                    committed_step_tokens >= step.token_limit
                    or committed_step_tokens + tokens > step.token_limit
                )
            ):
                return ("tokens", "step", step.spent_tokens, step.token_limit)
        return None

    def _raise_block(self, blocked: tuple[str, str, float | int, float | int]) -> None:
        metric, scope, spent, limit = blocked
        if metric == "usd":
            self._on_block(float(spent), float(limit))
            raise BudgetExceeded(float(spent), float(limit), scope=scope)
        self._on_token_block(int(spent), int(limit))
        raise TokenBudgetExceeded(int(spent), int(limit), scope=scope)

    def _reservation_parts(
        self, reserved: ReservationHandle
    ) -> tuple[float, int, _StepState | None]:
        if isinstance(reserved, BudgetReservation):
            with self._lock:
                state = self._typed_reservation_state_locked(reserved)
                return state.usd, state.tokens, state.step
        if not math.isfinite(reserved) or reserved < 0:
            raise ValueError(f"reserved must be a finite, non-negative number, got {reserved!r}")
        return float(reserved), 0, None

    def _make_reservation_locked(
        self, usd: float, tokens: int, step: _StepState | None
    ) -> BudgetReservation:
        """Create and register a typed handle. Caller must hold ``self._lock``."""
        handle = BudgetReservation(usd, tokens, self)
        self._reservations[handle] = _ReservationState(weakref.ref(handle), usd, tokens, step)
        return handle

    def _make_reservation(
        self, usd: float, tokens: int, step: _StepState | None
    ) -> BudgetReservation:
        """Create a registered zero/default handle for a scoped post-hoc record."""
        with self._lock:
            return self._make_reservation_locked(usd, tokens, step)

    def _typed_reservation_state_locked(self, handle: BudgetReservation) -> _ReservationState:
        """Validate a typed handle without mutating accounting state."""
        if handle._owner is not self:
            raise ValueError("reservation belongs to a different BudgetGuard")
        if handle in self._consumed_reservations:
            raise ValueError("reservation has already been settled or released")
        state = self._reservations.get(handle)
        if state is None or state.issued() is not handle:
            raise ValueError("reservation handle was not issued by this BudgetGuard")
        if (
            not math.isfinite(handle.usd)
            or handle.usd < 0
            or isinstance(handle.tokens, bool)
            or not isinstance(handle.tokens, int)
            or handle.tokens < 0
            or type(handle.usd) is not type(state.usd)
            or type(handle.tokens) is not type(state.tokens)
            or handle.usd != state.usd
            or handle.tokens != state.tokens
        ):
            raise ValueError("reservation handle has been modified")
        return state

    def _validate_scoped_locked(
        self,
        reserved: ReservationHandle,
        step: _StepState,
        *,
        require_active: bool,
    ) -> BudgetReservation | None:
        """Validate a scoped handle. Caller must hold ``self._lock``.

        Return the genuine typed handle, or ``None`` for a legal numeric zero
        that the caller may normalize into a registered zero-valued handle.
        """
        if require_active and not step.active:
            raise RuntimeError("step budget is no longer active")
        if isinstance(reserved, BudgetReservation):
            state = self._typed_reservation_state_locked(reserved)
            if state.step is not step:
                raise ValueError("reservation belongs to a different step budget")
            return reserved
        if not math.isfinite(reserved) or reserved < 0:
            raise ValueError(f"reserved must be a finite, non-negative number, got {reserved!r}")
        if reserved != 0:
            raise ValueError("scoped numeric reservation handles must be zero")
        return None

    def _scoped_reservation(
        self,
        reserved: ReservationHandle,
        step: _StepState,
        *,
        require_active: bool,
    ) -> BudgetReservation:
        """Return a genuine handle for ``step``; numeric scoped handles must be zero."""
        with self._lock:
            normalized = self._validate_scoped_locked(
                reserved,
                step,
                require_active=require_active,
            )
            if normalized is not None:
                return normalized
            return self._make_reservation_locked(0.0, 0, step)

    def _validate_scoped_reservation(self, reserved: ReservationHandle, step: _StepState) -> None:
        """Validate a scoped handle without creating a zero-valued replacement."""
        with self._lock:
            self._validate_scoped_locked(reserved, step, require_active=True)

    def _consume_handle_locked(
        self,
        handle: ReservationHandle,
        reserved_usd: float,
        reserved_tokens: int,
        step: _StepState | None,
    ) -> None:
        if isinstance(handle, BudgetReservation):
            state = self._typed_reservation_state_locked(handle)
            if (
                reserved_usd != state.usd
                or reserved_tokens != state.tokens
                or step is not state.step
            ):
                raise ValueError("reservation handle state changed during settlement")
        elif not reserved_usd and not reserved_tokens:
            return
        if reserved_usd > self._reserved + _EPS:
            raise ValueError("reserved USD handle exceeds total in-flight reservations")
        if reserved_tokens > self._reserved_tokens:
            raise ValueError("reserved token handle exceeds total in-flight reservations")
        if step is not None:
            if reserved_usd > step.reserved_usd + _EPS or reserved_tokens > step.reserved_tokens:
                raise ValueError("reservation exceeds its step's in-flight totals")
        self._consume_reservation_locked(reserved_usd)
        self._reserved_tokens -= reserved_tokens
        if step is not None:
            step.reserved_usd = max(0.0, step.reserved_usd - reserved_usd)
            step.reserved_tokens -= reserved_tokens
        if isinstance(handle, BudgetReservation):
            del self._reservations[handle]
            self._consumed_reservations.add(handle)

    def _accrue_tokens_locked(self, total_tokens: int, step: _StepState | None) -> None:
        self.spent_tokens += total_tokens
        self._last_llm_tokens = total_tokens
        if step is not None:
            step.spent_tokens += total_tokens

    def _default_estimate_locked(self) -> float:
        """The default next-call prediction when the caller supplies no
        estimate. Caller must hold ``self._lock``. Conservative: the costlier
        of the last LLM call and the last tool call — a mixed loop predicts the
        pricier kind, which at worst blocks one call early (fail-closed) rather
        than letting a crossing call through because the LAST event happened to
        be cheap.
        """
        return max(self._last_llm_cost, self._last_tool_cost)

    def _consume_reservation_locked(self, reserved: float) -> None:
        """Subtract a settled/released hold from the in-flight tally. Caller
        must hold ``self._lock``. A handle larger than EVERYTHING currently
        held cannot have come from a matching :meth:`reserve` — raising beats
        silently clamping, which would free OTHER callers' holds and fail the
        ceiling open. The epsilon absorbs float dust from accumulating and
        draining many holds; per-caller over-release (a handle within the
        total but larger than the caller's own hold) is undetectable without
        per-handle tracking and remains the caller's responsibility.
        """
        if reserved > self._reserved + _EPS:
            raise ValueError(
                f"reserved handle ({reserved!r}) exceeds total in-flight reservations "
                f"({self._reserved!r}) — a handle must come from a matching reserve()"
            )
        self._reserved = max(0.0, self._reserved - reserved)

    def _validate_estimate(self, estimated: float | None) -> None:
        # NaN/inf would poison the ceiling comparisons and fail-open (or poison
        # _reserved) — reject a non-finite caller-supplied estimate up front,
        # matching the constructor's math.isfinite guard and the TS Number.isFinite.
        if estimated is not None and not math.isfinite(estimated):
            raise ValueError(f"estimated cost must be a finite number, got {estimated!r}")

    def _stream_register_locked(
        self,
        reserved_usd: float,
        reserved_tokens: int,
        step: _StepState | None,
    ) -> object:
        """Register a stream after its handle is validated. Caller holds ``_lock``."""
        key = object()
        self._stream_costs[key] = (
            0.0,
            0,
            reserved_usd,
            reserved_tokens,
            step,
        )
        return key

    def _stream_validate_reservation(self, reserved: ReservationHandle) -> None:
        """Reject malformed handles before stream construction does any setup."""
        self._reservation_parts(reserved)

    def _stream_prepare(self, reserved: ReservationHandle) -> tuple[ReservationHandle, object]:
        """Register an active stream (see :class:`~floe_guard.stream.StreamGuard`)
        and return its registry key. Active streams' accrued-but-unsettled costs
        count against the ceiling for each OTHER stream, so parallel unreserved
        streams share the budget instead of each spending the full ceiling."""
        reserved_usd, reserved_tokens, step = self._reservation_parts(reserved)
        with self._lock:
            key = self._stream_register_locked(reserved_usd, reserved_tokens, step)
        return reserved, key

    def _stream_prepare_scoped(
        self, reserved: ReservationHandle, step: _StepState
    ) -> tuple[BudgetReservation, object]:
        """Atomically validate scope liveness, normalize its handle, and register."""
        with self._lock:
            normalized = self._validate_scoped_locked(reserved, step, require_active=True)
            if normalized is None:
                normalized = self._make_reservation_locked(0.0, 0, step)
            state = self._typed_reservation_state_locked(normalized)
            key = self._stream_register_locked(state.usd, state.tokens, state.step)
        return normalized, key

    def _stream_unregister(self, key: object) -> None:
        """Drop a stream's registry entry once its cost is settled (settle()
        moves the accrual into ``spent_usd``, so keeping it would double-count)."""
        with self._lock:
            self._stream_costs.pop(key, None)

    def _stream_would_cross(
        self, key: object, cumulative_call_cost: float, cumulative_tokens: int
    ) -> tuple[str, str, float | int, float | int] | None:
        """Atomically record stream ``key``'s cumulative cost so far and answer:
        would it cross the ceiling? Counted against the limit: settled spend,
        other calls' reservations (this stream's own hold is excluded — its real
        accrued cost replaces the estimate), and each OTHER active stream's
        accrual beyond its own reservation (the reservation part is already
        inside ``_reserved``). Used by :class:`~floe_guard.stream.StreamGuard`.
        """
        with self._lock:
            _, _, own_usd, own_tokens, step = self._stream_costs.get(key, (0.0, 0, 0.0, 0, None))
            self._stream_costs[key] = (
                cumulative_call_cost,
                cumulative_tokens,
                own_usd,
                own_tokens,
                step,
            )
            other_usd_overage = 0.0
            other_token_overage = 0
            same_step_usd_overage = 0.0
            same_step_token_overage = 0
            for (
                other_key,
                (
                    accrued_usd,
                    accrued_tokens,
                    held_usd,
                    held_tokens,
                    other_step,
                ),
            ) in self._stream_costs.items():
                if other_key is key:
                    continue
                usd_overage = max(0.0, accrued_usd - held_usd)
                token_overage = max(0, accrued_tokens - held_tokens)
                other_usd_overage += usd_overage
                other_token_overage += token_overage
                if step is not None and other_step is step:
                    same_step_usd_overage += usd_overage
                    same_step_token_overage += token_overage
            aggregate_usd = self.spent_usd + max(0.0, self._reserved - own_usd) + other_usd_overage
            if aggregate_usd + cumulative_call_cost > self.limit_usd + _EPS:
                return ("usd", "aggregate", self.spent_usd, self.limit_usd)
            aggregate_tokens = (
                self.spent_tokens + max(0, self._reserved_tokens - own_tokens) + other_token_overage
            )
            if (
                self.token_limit is not None
                and aggregate_tokens + cumulative_tokens > self.token_limit
            ):
                return ("tokens", "aggregate", self.spent_tokens, self.token_limit)
            if step is not None:
                step_usd = (
                    step.spent_usd + max(0.0, step.reserved_usd - own_usd) + same_step_usd_overage
                )
                if (
                    step.limit_usd is not None
                    and step_usd + cumulative_call_cost > step.limit_usd + _EPS
                ):
                    return ("usd", "step", step.spent_usd, step.limit_usd)
                step_tokens = (
                    step.spent_tokens
                    + max(0, step.reserved_tokens - own_tokens)
                    + same_step_token_overage
                )
                if (
                    step.token_limit is not None
                    and step_tokens + cumulative_tokens > step.token_limit
                ):
                    return ("tokens", "step", step.spent_tokens, step.token_limit)
            return None

    def _would_cross(self, estimated_next_cost: float | None) -> bool:
        with self._lock:
            estimate = (
                self._default_estimate_locked()
                if estimated_next_cost is None
                else max(0.0, estimated_next_cost)
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


class StepBudgetGuard:
    """A scoped guard that enforces fresh per-step limits plus its parent limits."""

    def __init__(
        self,
        parent: BudgetGuard,
        *,
        max_usd: float | None,
        max_tokens: int | None,
    ) -> None:
        if max_usd is None and max_tokens is None:
            raise ValueError("step requires max_usd, max_tokens, or both")
        if max_usd is not None and (not math.isfinite(max_usd) or max_usd < 0):
            raise ValueError(f"max_usd must be a finite, non-negative number, got {max_usd!r}")
        _validate_optional_tokens("max_tokens", max_tokens)
        self._parent = parent
        self._state = _StepState(
            limit_usd=float(max_usd) if max_usd is not None else None,
            token_limit=max_tokens,
        )
        self._entered = False

    def __enter__(self) -> StepBudgetGuard:
        if self._entered or not self._state.active:
            raise RuntimeError("step budget cannot be re-entered")
        self._entered = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        with self._parent._lock:
            self._state.active = False
            leaked = self._state.reserved_usd > _EPS or self._state.reserved_tokens > 0
        if exc_type is None and leaked:
            raise RuntimeError(
                "step exited with an active reservation; settle or release every handle"
            )

    @property
    def limit_usd(self) -> float:
        return self._parent.limit_usd

    @property
    def token_limit(self) -> int | None:
        return self._parent.token_limit

    @property
    def spent_usd(self) -> float:
        return self._parent.spent_usd

    @property
    def spent_tokens(self) -> int:
        return self._parent.spent_tokens

    @property
    def remaining_usd(self) -> float:
        return self._parent.remaining_usd

    @property
    def remaining_tokens(self) -> int | None:
        return self._parent.remaining_tokens

    @property
    def price_overrides(self) -> dict[str, ManualPrice] | None:
        return self._parent.price_overrides

    @property
    def fail_closed(self) -> bool:
        return self._parent.fail_closed

    @property
    def near_limit_bps(self) -> int:
        return self._parent.near_limit_bps

    @property
    def tool_costs(self) -> dict[str, float]:
        return self._parent.tool_costs

    @property
    def spend_log(self) -> list[SpendEvent]:
        return self._parent.spend_log

    def estimate_call(
        self,
        model: str,
        prompt_tokens: int,
        max_completion_tokens: int = 0,
        *,
        price: ManualPrice | None = None,
    ) -> float | None:
        return self._parent.estimate_call(
            model,
            prompt_tokens,
            max_completion_tokens,
            price=price,
        )

    def export_log(self) -> str:
        return self._parent.export_log()

    def check(
        self,
        estimated_next_cost: float | None = None,
        *,
        estimated_next_tokens: int | None = None,
    ) -> None:
        self._ensure_active()
        self._parent._check(estimated_next_cost, estimated_next_tokens, step=self._state)

    def reserve(
        self,
        estimated_cost: float | None = None,
        *,
        estimated_tokens: int | None = None,
    ) -> ReservationHandle:
        self._ensure_active()
        return self._parent._reserve(estimated_cost, estimated_tokens, step=self._state)

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
        self._ensure_active()
        reserved = self._parent._scoped_reservation(
            reserved,
            self._state,
            require_active=True,
        )
        return self._parent.settle(
            model,
            prompt_tokens,
            completion_tokens,
            reserved=reserved,
            price=price,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_creation_input_tokens_1h=cache_creation_input_tokens_1h,
            cache_read_input_tokens=cache_read_input_tokens,
            label=label,
        )

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
        self._ensure_active()
        handle = self._parent._make_reservation(0.0, 0, self._state)
        return self.settle(
            model,
            prompt_tokens,
            completion_tokens,
            reserved=handle,
            price=price,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_creation_input_tokens_1h=cache_creation_input_tokens_1h,
            cache_read_input_tokens=cache_read_input_tokens,
            label=label,
        )

    def reserve_tool(self, estimated_cost: float) -> ReservationHandle:
        self._ensure_active()
        return self._parent._reserve(estimated_cost, 0, step=self._state, enforce_tokens=False)

    def settle_tool(
        self,
        tool: str,
        cost_usd: float,
        *,
        reserved: ReservationHandle = 0.0,
        label: str | None = None,
    ) -> float:
        self._ensure_active()
        reserved = self._parent._scoped_reservation(
            reserved,
            self._state,
            require_active=True,
        )
        return self._parent.settle_tool(tool, cost_usd, reserved=reserved, label=label)

    def record_tool(self, tool: str, cost_usd: float, *, label: str | None = None) -> float:
        self._ensure_active()
        handle = self._parent._make_reservation(0.0, 0, self._state)
        return self.settle_tool(tool, cost_usd, reserved=handle, label=label)

    def release(self, reserved: ReservationHandle) -> None:
        reserved = self._parent._scoped_reservation(
            reserved,
            self._state,
            require_active=False,
        )
        self._parent.release(reserved)

    def advisory(self) -> BudgetAdvisory:
        return self._parent._advisory(self._state)

    def _stream_validate_reservation(self, reserved: ReservationHandle) -> None:
        self._parent._validate_scoped_reservation(reserved, self._state)

    def _stream_prepare(self, reserved: ReservationHandle) -> tuple[BudgetReservation, object]:
        return self._parent._stream_prepare_scoped(reserved, self._state)

    def _stream_would_cross(
        self, key: object, cumulative_call_cost: float, cumulative_tokens: int
    ) -> tuple[str, str, float | int, float | int] | None:
        return self._parent._stream_would_cross(key, cumulative_call_cost, cumulative_tokens)

    def _stream_unregister(self, key: object) -> None:
        self._parent._stream_unregister(key)

    def _resolve(self, model: str, price: ManualPrice | None):
        return self._parent._resolve(model, price)

    def _raise_block(self, blocked: tuple[str, str, float | int, float | int]) -> None:
        self._parent._raise_block(blocked)

    def _ensure_active(self) -> None:
        if not self._state.active:
            raise RuntimeError("step budget is no longer active")


def _validate_optional_tokens(name: str, value: int | None) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
        raise ValueError(f"{name} must be None or a non-negative int, got {value!r}")


def _used_bps(used: float | int, limit: float | int) -> int:
    if limit <= 0:
        return 10000
    return max(0, min(10000, int(used / limit * 10000 + 1e-9)))


def _default_on_block(spent_usd: float, limit_usd: float) -> None:
    print(
        "BUDGET EXCEEDED — call blocked\n"
        f"  spent so far: ${spent_usd:.6f}  |  ceiling: ${limit_usd:.6f}\n"
        "  The next call would cross your budget; floe-guard stopped your agent "
        "before it ran.",
        file=sys.stderr,
    )


def _default_on_token_block(spent_tokens: int, limit_tokens: int) -> None:
    print(
        "TOKEN BUDGET EXCEEDED — call blocked\n"
        f"  tokens so far: {spent_tokens}  |  ceiling: {limit_tokens}\n"
        "  The next call would cross your token budget; floe-guard stopped your "
        "agent before it ran.",
        file=sys.stderr,
    )
