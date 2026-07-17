/**
 * The local, in-process budget guard.
 *
 * `BudgetGuard` is a kill-switch that lives in the LLM call path. The contract:
 *
 * 1. Call {@link BudgetGuard.check} BEFORE every LLM call. If the *next* call
 *    would cross the ceiling, it throws {@link BudgetExceeded} and the call never
 *    runs.
 * 2. Call {@link BudgetGuard.record} AFTER every response, with the token usage.
 *    It prices the tokens offline and accrues the USD into a running total.
 *
 * **Concurrency.** `check()` then `record()` is a check-then-act with an `await`
 * in between. Fire several model calls at once (e.g. `Promise.all`) and they all
 * `check()` against the same under-limit total before any `record()` lands, so
 * the ceiling is blown (see issue #18). {@link BudgetGuard.reserve} /
 * {@link BudgetGuard.settle} close that gap: `reserve()` holds the estimated cost
 * in flight (synchronously, before the await), so parallel callers each take
 * their own slice of the ceiling. JS is single-threaded, so an in-flight counter
 * is enough — no lock needed. The middleware uses it; `check`/`record` are
 * unchanged.
 *
 * This is a faithful port of `src/floe_guard/guard.py` — same prediction logic,
 * same epsilon handling, same fail-closed default.
 */

import { BudgetExceeded, UnpriceableModelError } from "./errors.js";
import {
  type ManualPrice,
  priceTokens,
  resolvePrice,
} from "./pricing.js";

/** Tolerance for float rounding in the running spend total (well below $0.000001). */
const EPS = 1e-12;

/**
 * One priced spend event in the guard's per-call ledger.
 *
 * Every {@link BudgetGuard.record} / {@link BudgetGuard.settle} /
 * {@link BudgetGuard.recordTool} / {@link BudgetGuard.settleTool} that accrues
 * spend appends exactly one event, so
 * the ledger's costs sum to `spentUsd` (unless a `maxLogEvents` ring buffer has
 * evicted old events). The schema is identical in the Python
 * package (`SpendEvent` in `src/floe_guard/guard.py`) and
 * {@link BudgetGuard.exportLog} serialises it with the same snake_case keys in
 * both languages, so every agent emits the same shape regardless of stack.
 */
export interface SpendEvent {
  /** Unix epoch seconds (UTC). */
  readonly timestamp: number;
  readonly kind: "llm" | "tool";
  readonly modelOrTool: string;
  /** `null` for tool events. */
  readonly promptTokens: number | null;
  /** `null` for tool events. */
  readonly completionTokens: number | null;
  readonly costUsd: number;
  /** Caller-supplied tag (agent/task name). */
  readonly label?: string;
  /** The reservation settled by this call, if any. */
  readonly reserved?: number;
}

export interface BudgetGuardOptions {
  /** Per-model manual prices for models the bundled cost map cannot price. */
  priceOverrides?: Record<string, ManualPrice>;
  /**
   * When `true` (default), recording an unpriceable model without a manual price
   * warns loudly AND throws {@link UnpriceableModelError}. When `false`, it warns
   * and skips accrual (you have opted into un-enforced spend for that model).
   */
  failClosed?: boolean;
  /**
   * Optional callback invoked with `(spentUsd, limitUsd)` right before
   * {@link BudgetExceeded} is thrown. Defaults to printing the
   * `BUDGET EXCEEDED — call blocked` banner to stderr.
   */
  onBlock?: (spentUsd: number, limitUsd: number) => void;
  /**
   * Utilization (basis points, 0..10000) at which {@link BudgetGuard.advisory}
   * flags `nearLimit` so an agent can taper before the hard-stop. Default 8000.
   */
  nearLimitBps?: number;
  /**
   * Optional cap on the per-call spend ledger ({@link BudgetGuard.spendLog}).
   * When set, the ledger is a ring buffer keeping the most recent N events so a
   * long-running agent's memory stays bounded; the running totals are
   * unaffected. Default: keep every event.
   */
  maxLogEvents?: number;
}

/**
 * A context-aware spend signal for the single local budget.
 *
 * Mirrors the core fields of hosted Floe's `X-Floe-Budget-Advisory` header, so
 * agent logic that reads it (taper as you approach the cap, stop at it) ports
 * unchanged to the hosted path. Hosted adds what a local, single-budget guard
 * cannot know: which of several caps is tightest (`scope` across
 * `credit_line | session | task | api | vendor`), cross-vendor reasoning,
 * server-truth balances, and rolling-window reset timing.
 *
 * This is a **soft** signal — the model may ignore it. The hard-stop
 * ({@link BudgetGuard.check}) is what enforces the ceiling; the advisory is
 * upside (let the agent finish on budget rather than be cut off).
 */
export interface BudgetAdvisory {
  nearLimit: boolean;
  /** Utilization in basis points, 0..10000 (8500 = 85%). */
  usedBps: number;
  remainingUsd: number;
  limitUsd: number;
  spentUsd: number;
  /** Hosted reports the tightest cap across all scopes; local is always "local". */
  scope: "local";
}

export class BudgetGuard {
  readonly limitUsd: number;
  spentUsd = 0;
  priceOverrides?: Record<string, ManualPrice>;
  failClosed: boolean;
  nearLimitBps: number;

  private readonly onBlock: (spentUsd: number, limitUsd: number) => void;
  /** Cost of the most recent priced call, used to predict the next one. */
  private lastCost = 0;
  /** USD held for in-flight calls (reserved, not yet settled). Counts toward the ceiling. */
  private reserved = 0;
  /** Per-call ledger, oldest first; a ring buffer when maxLogEvents is set. */
  private readonly spendEvents: SpendEvent[] = [];
  private readonly maxLogEvents?: number;
  /**
   * Per-tool running totals (settleTool/recordTool) — the tool side of the one
   * shared ceiling, exposed via the toolCosts getter. null-prototype: tool
   * names are caller-supplied strings, so a "__proto__" name is stored as
   * plain data instead of mutating the object's prototype.
   */
  private readonly toolCostTotals: Record<string, number> = Object.create(null);

  /**
   * @param limitUsd the spend ceiling, in USD. `0` blocks the very first call.
   */
  constructor(limitUsd: number, options: BudgetGuardOptions = {}) {
    if (!Number.isFinite(limitUsd) || limitUsd < 0) {
      // NaN/Infinity would make every check() comparison fail-open, silently
      // disabling the guard — reject them up front.
      throw new RangeError(
        `limitUsd must be a finite, non-negative number, got ${limitUsd}`,
      );
    }
    // `=== undefined` (not `??`) so an explicit null is rejected by validation
    // rather than silently defaulting — matches Python, which rejects None.
    const nearLimitBps = options.nearLimitBps === undefined ? 8000 : options.nearLimitBps;
    if (!Number.isInteger(nearLimitBps) || nearLimitBps < 0 || nearLimitBps > 10000) {
      throw new RangeError(
        `nearLimitBps must be an integer in 0..10000, got ${nearLimitBps}`,
      );
    }
    if (
      options.maxLogEvents !== undefined &&
      (!Number.isInteger(options.maxLogEvents) || options.maxLogEvents < 0)
    ) {
      throw new RangeError(
        `maxLogEvents must be a non-negative integer, got ${options.maxLogEvents}`,
      );
    }
    this.limitUsd = limitUsd;
    this.maxLogEvents = options.maxLogEvents;
    this.priceOverrides = options.priceOverrides;
    this.failClosed = options.failClosed ?? true;
    this.onBlock = options.onBlock ?? defaultOnBlock;
    this.nearLimitBps = nearLimitBps;
  }

  /**
   * Throw {@link BudgetExceeded} if the next call would cross the ceiling.
   *
   * Call this immediately before each LLM request. The "next call" is estimated
   * from the last recorded call's cost (override with `estimatedNextCost`); the
   * first call is always allowed unless the ceiling is already met. In-flight
   * reservations count toward the total, so this stays correct alongside
   * {@link BudgetGuard.reserve}.
   *
   * Note: `check` is a non-binding peek. For parallel calls, use `reserve()` /
   * `settle()`, which hold the estimate across the await.
   */
  check(estimatedNextCost?: number): void {
    const rawEstimate =
      estimatedNextCost === undefined ? this.lastCost : estimatedNextCost;
    if (!Number.isFinite(rawEstimate)) {
      // NaN/Infinity would poison the comparisons and fail-open — reject it
      // (parity with the constructor's Number.isFinite guard).
      throw new RangeError(
        `estimatedNextCost must be a finite number, got ${rawEstimate}`,
      );
    }
    const estimate = Math.max(0, rawEstimate);
    const committed = this.spentUsd + this.reserved;
    if (committed > this.limitUsd - EPS || committed + estimate > this.limitUsd + EPS) {
      this.onBlock(this.spentUsd, this.limitUsd);
      throw new BudgetExceeded(this.spentUsd, this.limitUsd);
    }
  }

  /**
   * Atomically check the ceiling AND hold the estimated cost in flight.
   *
   * The concurrency-safe enforcement path: call before the request and hold the
   * returned reservation across the await, so parallel callers can't all clear
   * the same stale total. Throws {@link BudgetExceeded} (without reserving) if
   * the reservation would cross the ceiling. Returns the reservation handle to
   * pass to {@link BudgetGuard.settle} (or {@link BudgetGuard.release} on error).
   * `estimatedCost` defaults to the last call's cost.
   */
  reserve(estimatedCost?: number): number {
    const rawEstimate = estimatedCost === undefined ? this.lastCost : estimatedCost;
    if (!Number.isFinite(rawEstimate)) {
      // NaN would poison this.reserved and fail-open the ceiling — reject it.
      throw new RangeError(
        `estimatedCost must be a finite number, got ${rawEstimate}`,
      );
    }
    const estimate = Math.max(0, rawEstimate);
    const committed = this.spentUsd + this.reserved;
    if (committed > this.limitUsd - EPS || committed + estimate > this.limitUsd + EPS) {
      this.onBlock(this.spentUsd, this.limitUsd);
      throw new BudgetExceeded(this.spentUsd, this.limitUsd);
    }
    this.reserved += estimate;
    return estimate;
  }

  /**
   * Release a reservation and record the actual cost. `record` is `settle` with
   * no reservation. Returns the USD cost of this call; unpriceable-model handling
   * matches {@link BudgetGuard.record}, and any held reservation is released even
   * on the warn-and-skip path. A priced call appends one {@link SpendEvent} to
   * {@link BudgetGuard.spendLog} (`label` tags it, e.g. with an agent/task name);
   * the warn-and-skip path accrues nothing and logs nothing, so the ledger stays
   * in lockstep with `spentUsd`.
   */
  settle(
    model: string,
    promptTokens: number,
    completionTokens: number,
    options: { reserved?: number; price?: ManualPrice; label?: string } = {},
  ): number {
    const reserved = options.reserved ?? 0;
    // A bad reserved handle would corrupt this.reserved and break the ceiling for
    // OTHER in-flight calls (negative → phantom hold; Infinity → clears all holds).
    if (!Number.isFinite(reserved) || reserved < 0) {
      throw new RangeError(`reserved must be a finite, non-negative number, got ${reserved}`);
    }
    let overrides = this.priceOverrides;
    if (options.price !== undefined) {
      overrides = { ...(overrides ?? {}), [model]: options.price };
    }

    const priced = resolvePrice(model, overrides);
    if (priced === null) {
      console.warn(
        `Cannot price model '${model}': not in the bundled cost map and no ` +
          `manual price given. The budget guard cannot enforce a ceiling on ` +
          `spend it cannot measure — pass { price } or set it in priceOverrides.`,
      );
      // Release any held reservation on BOTH paths. Fail-closed must not leak
      // the in-flight hold, or reserved grows permanently and remainingUsd
      // shrinks until reserve() starts blocking everything.
      this.release(reserved);
      if (this.failClosed) {
        throw new UnpriceableModelError(model);
      }
      return 0;
    }

    let cost: number;
    try {
      cost = priceTokens(priced, promptTokens, completionTokens);
    } catch (err) {
      // priceTokens can throw (e.g. non-finite costs). Release the in-flight
      // hold before re-throwing so `reserved` doesn't leak and shrink
      // remainingUsd permanently — same fail-safe as the unpriceable path above.
      this.release(reserved);
      throw err;
    }
    if (reserved) {
      this.reserved = Math.max(0, this.reserved - reserved);
    }
    this.spentUsd += cost;
    // Clamp a sub-epsilon float overshoot back to the limit so the running total
    // never reports as having crossed the ceiling by a rounding artifact.
    if (this.spentUsd - this.limitUsd > 0 && this.spentUsd - this.limitUsd < EPS) {
      this.spentUsd = this.limitUsd;
    }
    this.lastCost = cost;
    this.appendEvent({
      timestamp: Date.now() / 1000,
      kind: "llm",
      modelOrTool: model,
      promptTokens,
      completionTokens,
      costUsd: cost,
      ...(options.label !== undefined ? { label: options.label } : {}),
      // 0 means "no reservation" (the plain record() path) — omit rather than
      // log a meaningless zero.
      ...(reserved ? { reserved } : {}),
    });
    return cost;
  }

  /**
   * Price one response's tokens offline and add the cost to the total.
   *
   * Returns the USD cost of this call. If the model is unpriceable and no `price`
   * is given, behaviour depends on `failClosed`: warn + throw (default), or
   * warn + skip accrual.
   */
  record(
    model: string,
    promptTokens: number,
    completionTokens: number,
    options: { price?: ManualPrice; label?: string } = {},
  ): number {
    return this.settle(model, promptTokens, completionTokens, {
      reserved: 0,
      price: options.price,
      label: options.label,
    });
  }

  /**
   * Atomically check the ceiling AND hold a tool call's cost in flight.
   *
   * The tool-spend counterpart of {@link BudgetGuard.reserve} — and STRONGER
   * than the LLM path, because a paid tool's price is usually known exactly
   * before the call, so the pre-call hard-stop is precise rather than
   * estimated:
   *
   *     const handle = guard.reserveTool(0.02);   // throws BEFORE Apollo runs
   *     const result = await apollo.peopleLookup(...);
   *     guard.settleTool("apollo.people_lookup", 0.02, { reserved: handle });
   *
   * Throws {@link BudgetExceeded} (without reserving) if the call would cross
   * the ceiling. The estimate is required — tools have no last-cost prediction
   * worth falling back to. Pass the returned handle to
   * {@link BudgetGuard.settleTool}, or {@link BudgetGuard.release} on failure.
   */
  reserveTool(estimatedCost: number): number {
    if (estimatedCost === undefined) {
      // reserve(undefined) would silently fall back to the last-cost prediction
      // (0 on a fresh guard) — an unguarded tool call. A missing price must
      // fail loudly, e.g. guard.reserveTool(priceTable[tool]).
      throw new RangeError("reserveTool requires an estimated cost, got undefined");
    }
    return this.reserve(estimatedCost);
  }

  /**
   * Release a reservation and record a tool call's actual cost.
   *
   * `recordTool` is `settleTool` with no reservation. The caller supplies the
   * cost — tools have no token usage to price. Accrues into the same
   * `spentUsd` ceiling as tokens, tallies the per-tool total
   * ({@link BudgetGuard.toolCosts}), updates the next-call estimate (so a
   * tool-hammering loop's plain `check()` predicts one tool call ahead and
   * stops BEFORE the crossing call — the same contract as tokens), and appends
   * a `kind: "tool"` {@link SpendEvent} to {@link BudgetGuard.spendLog}.
   * Returns `costUsd`.
   */
  settleTool(
    tool: string,
    costUsd: number,
    options: { reserved?: number; label?: string } = {},
  ): number {
    if (!Number.isFinite(costUsd) || costUsd < 0) {
      throw new RangeError(`costUsd must be a finite, non-negative number, got ${costUsd}`);
    }
    const reserved = options.reserved ?? 0;
    // A bad reserved handle would corrupt the in-flight tally and break the
    // ceiling for OTHER calls — same contract as settle().
    if (!Number.isFinite(reserved) || reserved < 0) {
      throw new RangeError(`reserved must be a finite, non-negative number, got ${reserved}`);
    }
    if (reserved) {
      this.reserved = Math.max(0, this.reserved - reserved);
    }
    this.spentUsd += costUsd;
    // Same sub-epsilon clamp as settle(): never report a rounding-artifact
    // crossing of the ceiling.
    if (this.spentUsd - this.limitUsd > 0 && this.spentUsd - this.limitUsd < EPS) {
      this.spentUsd = this.limitUsd;
    }
    this.lastCost = costUsd;
    this.toolCostTotals[tool] = (this.toolCostTotals[tool] ?? 0) + costUsd;
    this.appendEvent({
      timestamp: Date.now() / 1000,
      kind: "tool",
      modelOrTool: tool,
      promptTokens: null,
      completionTokens: null,
      costUsd,
      ...(options.label !== undefined ? { label: options.label } : {}),
      ...(reserved ? { reserved } : {}),
    });
    return costUsd;
  }

  /**
   * Accrue a non-LLM cost (a paid tool/API call) against the same ceiling.
   *
   * Post-hoc accrual for costs only known after the call (metered APIs); when
   * the price is known up front, {@link BudgetGuard.reserveTool} /
   * {@link BudgetGuard.settleTool} give the stronger pre-call hard-stop. See
   * `settleTool` for the full contract. Returns `costUsd`.
   */
  recordTool(tool: string, costUsd: number, options: { label?: string } = {}): number {
    return this.settleTool(tool, costUsd, { reserved: 0, label: options.label });
  }

  /**
   * Drop an in-flight reservation without recording spend (e.g. the call failed
   * before producing usage). Safe to call with `0`.
   */
  release(reserved: number): void {
    // Validate before the zero-check so a NaN handle throws instead of being
    // silently dropped (a leak); a bad handle corrupts the in-flight tally.
    if (!Number.isFinite(reserved) || reserved < 0) {
      throw new RangeError(`reserved must be a finite, non-negative number, got ${reserved}`);
    }
    if (!reserved) return;
    this.reserved = Math.max(0, this.reserved - reserved);
  }

  /** USD left before the ceiling, net of in-flight reservations (never negative). */
  get remainingUsd(): number {
    return Math.max(0, this.limitUsd - this.spentUsd - this.reserved);
  }

  /**
   * Per-tool running USD totals, keyed by the name given to `settleTool()` /
   * `recordTool()` — e.g. `{"apollo.people_lookup": 0.42, "exa.search": 0.11}`.
   * Makes the token/tool split of the one shared ceiling inspectable
   * (`spentUsd - sum of toolCosts` is the token side). Returns a snapshot copy.
   */
  get toolCosts(): Record<string, number> {
    return { ...this.toolCostTotals };
  }

  /**
   * The per-call spend ledger, oldest first — one {@link SpendEvent} per priced
   * `record()` / `settle()` / `recordTool()` / `settleTool()`. Returns a
   * snapshot copy: mutating it cannot corrupt the ledger.
   */
  get spendLog(): SpendEvent[] {
    return [...this.spendEvents];
  }

  /**
   * The spend ledger as JSONL — one event per line, newline-terminated.
   *
   * The schema is stable and language-independent (snake_case keys, fixed order;
   * optional fields omitted when absent), identical to the Python package's
   * `export_log()`, so heterogeneous agents produce logs you can concatenate and
   * analyse as one stream. (The *schema* is the contract, not the bytes: the two
   * runtimes may render the same float differently, e.g. JS `0.0000025` vs
   * Python `2.5e-06`.) Empty ledger yields `""`.
   */
  exportLog(): string {
    return this.spendEvents
      .map((e) => {
        // snake_case wire shape, fixed key order — the cross-language schema.
        const row: Record<string, unknown> = {
          timestamp: e.timestamp,
          kind: e.kind,
          model_or_tool: e.modelOrTool,
          prompt_tokens: e.promptTokens,
          completion_tokens: e.completionTokens,
          cost_usd: e.costUsd,
        };
        if (e.label !== undefined) row.label = e.label;
        if (e.reserved !== undefined) row.reserved = e.reserved;
        return `${JSON.stringify(row)}\n`;
      })
      .join("");
  }

  private appendEvent(event: SpendEvent): void {
    // Frozen for parity with Python's frozen dataclass: spendLog copies the
    // array but shares the event objects, so an unfrozen event would let a
    // consumer silently rewrite logged history.
    this.spendEvents.push(Object.freeze(event));
    if (this.maxLogEvents !== undefined && this.spendEvents.length > this.maxLogEvents) {
      // Ring buffer: drop the oldest overflow (at most one per append).
      this.spendEvents.splice(0, this.spendEvents.length - this.maxLogEvents);
    }
  }

  /**
   * Context-aware spend advisory for this budget — see {@link BudgetAdvisory}.
   *
   * `nearLimit` flips once utilization reaches `nearLimitBps` (default 80%), so an
   * agent can taper *before* the hard-stop. Advisory only: read it to adapt;
   * {@link BudgetGuard.check} is what enforces the ceiling.
   */
  advisory(): BudgetAdvisory {
    // Floor (not round) so usedBps never over-reports utilization and nearLimit
    // flips exactly when the threshold is reached; the epsilon absorbs float noise
    // and Math.floor matches Python's int() exactly (round() would diverge).
    const usedBps =
      this.limitUsd <= 0
        ? 10000
        : Math.max(0, Math.min(10000, Math.floor((this.spentUsd / this.limitUsd) * 10000 + 1e-9)));
    return {
      nearLimit: usedBps >= this.nearLimitBps,
      usedBps,
      // Settled budget: limit minus accrued spend, deliberately NOT net of
      // in-flight reservations. Unlike the remainingUsd getter (which subtracts
      // `reserved`), the advisory is a soft utilization signal about money already
      // spent, while the getter reports what a new call can still claim.
      remainingUsd: Math.max(0, this.limitUsd - this.spentUsd),
      limitUsd: this.limitUsd,
      spentUsd: this.spentUsd,
      scope: "local",
    };
  }
}

function defaultOnBlock(spentUsd: number, limitUsd: number): void {
  console.error(
    "BUDGET EXCEEDED — call blocked\n" +
      `  spent so far: $${spentUsd.toFixed(6)}  |  ceiling: $${limitUsd.toFixed(6)}\n` +
      "  The next call would cross your budget; floe-guard stopped your agent " +
      "before it ran.",
  );
}
