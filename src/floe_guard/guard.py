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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

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
            ``BUDGET EXCEEDED — call blocked`` banner to stderr.
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
        price_overrides: dict[str, ManualPrice] | None = None,
        fail_closed: bool = True,
        on_block: Callable[[float, float], None] | None = None,
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
        self.price_overrides = price_overrides
        self.fail_closed = fail_closed
        self._on_block = on_block or _default_on_block
        self.near_limit_bps = near_limit_bps
        self.spent_usd = 0.0
        # Cost of the most recent priced call — LLM or tool — used to predict the
        # next one so we can block BEFORE the crossing call runs (not one too late).
        self._last_cost = 0.0
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
        self._validate_estimate(estimated_next_cost)
        if self._would_cross(estimated_next_cost):
            self._block()

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
        self._validate_estimate(estimated_cost)
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
        # A bad reserved handle would corrupt _reserved and break the ceiling for
        # OTHER in-flight calls (negative → phantom hold; inf → clears all holds).
        if not math.isfinite(reserved) or reserved < 0:
            raise ValueError(f"reserved must be a finite, non-negative number, got {reserved!r}")
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
        with self._lock:
            if reserved:
                self._reserved = max(0.0, self._reserved - reserved)
            self.spent_usd += cost
            # Clamp a sub-epsilon float overshoot back to the limit so the running
            # total never reports as having crossed the ceiling by a rounding artifact.
            if 0.0 < self.spent_usd - self.limit_usd < _EPS:
                self.spent_usd = self.limit_usd
            self._last_cost = cost
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
                    reserved=reserved if reserved else None,
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

    def reserve_tool(self, estimated_cost: float) -> float:
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
        reserved: float = 0.0,
        label: str | None = None,
    ) -> float:
        """Release a reservation and record a tool call's actual cost.

        Concurrency-safe; ``record_tool`` is ``settle_tool`` with no
        reservation. The caller supplies the cost — tools have no token usage
        to price. Accrues into the same ``spent_usd`` ceiling as tokens,
        tallies the per-tool total (:attr:`tool_costs`), updates the next-call
        estimate (so a tool-hammering loop's plain :meth:`check` predicts one
        tool call ahead and stops BEFORE the crossing call — the same contract
        as tokens), and appends a ``kind="tool"`` :class:`SpendEvent` to
        :attr:`spend_log`. Returns ``cost_usd``.
        """
        if not math.isfinite(cost_usd) or cost_usd < 0:
            raise ValueError(f"cost_usd must be a finite, non-negative number, got {cost_usd!r}")
        # A bad reserved handle would corrupt _reserved and break the ceiling for
        # OTHER in-flight calls — same contract as settle().
        if not math.isfinite(reserved) or reserved < 0:
            raise ValueError(f"reserved must be a finite, non-negative number, got {reserved!r}")
        # int (and bool) are valid inputs; coerce so the logged event and the
        # return value are always float, like every other cost in the guard.
        cost_usd = float(cost_usd)
        with self._lock:
            if reserved:
                self._reserved = max(0.0, self._reserved - reserved)
            self.spent_usd += cost_usd
            # Same sub-epsilon clamp as settle(): never report a rounding-artifact
            # crossing of the ceiling.
            if 0.0 < self.spent_usd - self.limit_usd < _EPS:
                self.spent_usd = self.limit_usd
            self._last_cost = cost_usd
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
                    reserved=reserved if reserved else None,
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

    def release(self, reserved: float) -> None:
        """Drop an in-flight reservation without recording spend (e.g. the call
        failed before producing usage). Safe to call with ``0``."""
        # Validate before the zero-check so a NaN handle raises instead of being
        # silently dropped (which would leak the hold). A bad handle here corrupts
        # _reserved for other in-flight calls.
        if not math.isfinite(reserved) or reserved < 0:
            raise ValueError(f"reserved must be a finite, non-negative number, got {reserved!r}")
        if not reserved:
            return
        with self._lock:
            self._reserved = max(0.0, self._reserved - reserved)

    @property
    def remaining_usd(self) -> float:
        """USD left before the ceiling, net of in-flight reservations (never negative)."""
        with self._lock:
            return max(0.0, self.limit_usd - self.spent_usd - self._reserved)

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
            # Settled budget: limit minus accrued spend, deliberately NOT net of
            # in-flight reservations. This differs from the remaining_usd property
            # (which subtracts _reserved): the advisory is a soft utilization signal
            # about money already spent, while the property reports what a new call
            # can still claim.
            remaining_usd=max(0.0, self.limit_usd - self.spent_usd),
            limit_usd=self.limit_usd,
            spent_usd=self.spent_usd,
        )

    # ── internals ──────────────────────────────────────────────────────────────

    def _validate_estimate(self, estimated: float | None) -> None:
        # NaN/inf would poison the ceiling comparisons and fail-open (or poison
        # _reserved) — reject a non-finite caller-supplied estimate up front,
        # matching the constructor's math.isfinite guard and the TS Number.isFinite.
        if estimated is not None and not math.isfinite(estimated):
            raise ValueError(f"estimated cost must be a finite number, got {estimated!r}")

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
