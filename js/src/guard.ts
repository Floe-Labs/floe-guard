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

import {
  BudgetExceeded,
  TokenBudgetExceeded,
  UnpriceableModelError,
} from "./errors.js";
import {
  type ManualPrice,
  priceTokens,
  resolvePrice,
} from "./pricing.js";

/** Tolerance for float rounding in the running spend total (well below $0.000001). */
const EPS = 1e-12;

interface StepState {
  limitUsd: number | null;
  tokenLimit: number | null;
  spentUsd: number;
  spentTokens: number;
  reservedUsd: number;
  reservedTokens: number;
  active: boolean;
}

interface ReservationParts {
  usd: number;
  tokens: number;
  step: StepState | null;
}

interface ReservationState extends ReservationParts {
  owner: object;
  active: boolean;
}

type BlockingLimit = {
  metric: "usd" | "tokens";
  scope: "aggregate" | "step";
  spent: number;
  limit: number;
};

declare const reservationBrand: unique symbol;

/**
 * An opaque, immutable reservation issued by {@link BudgetGuard.reserve}.
 *
 * Pass the original object to `settle()` or `release()` exactly once. Do not
 * construct, clone, or copy reservation handles: the guard validates object
 * identity against private authoritative state.
 */
export interface BudgetReservation {
  readonly usd: number;
  readonly tokens: number;
  /** Nominal brand: reservation handles can only be issued by this module. */
  readonly [reservationBrand]: never;
}

export type ReservationHandle = number | BudgetReservation;

/**
 * Authoritative per-handle state. The public object is only an immutable view;
 * accounting and lifecycle decisions never trust caller-visible properties.
 */
const reservationStates = new WeakMap<BudgetReservation, ReservationState>();

function issueReservation(
  owner: object,
  usd: number,
  tokens: number,
  step: StepState | null,
): BudgetReservation {
  const handle = Object.freeze({
    usd,
    tokens,
  }) as BudgetReservation;
  reservationStates.set(handle, {
    owner,
    usd,
    tokens,
    step,
    active: true,
  });
  return handle;
}

function normalizeStepHandle(
  owner: object,
  handle: ReservationHandle,
  step: StepState,
): BudgetReservation {
  if (typeof handle === "number") {
    if (!Number.isFinite(handle) || handle < 0) {
      throw new RangeError(
        `reserved must be a finite, non-negative number, got ${handle}`,
      );
    }
    if (handle !== 0) {
      throw new RangeError(
        "scoped budget APIs require a reservation issued by this step",
      );
    }
    return issueReservation(owner, 0, 0, step);
  }
  const state = reservationStates.get(handle);
  if (state === undefined) {
    throw new RangeError(
      "invalid reservation handle; use the original object returned by reserve()",
    );
  }
  if (state.owner !== owner) {
    throw new RangeError("reservation belongs to a different BudgetGuard");
  }
  if (state.step !== step) {
    throw new RangeError("reservation belongs to a different budget step");
  }
  return handle;
}

export interface StepBudgetOptions {
  maxUsd?: number;
  maxTokens?: number;
}

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

export interface BudgetGuardOptions<
  TokenLimit extends number | undefined = number | undefined,
> {
  /** Optional aggregate token ceiling. */
  tokenLimit?: TokenLimit;
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
  /** Callback invoked immediately before a token ceiling blocks a call. */
  onTokenBlock?: (spentTokens: number, limitTokens: number) => void;
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
  /**
   * The guard's own next-call estimate (the costlier of the last LLM and last
   * tool call — the same value the default reservation uses). 0 until the first
   * call is recorded, so a planner can't divide by a cold estimate.
   *
   * Optional so adding it stays a non-breaking, additive change for any code
   * that constructs a `BudgetAdvisory` literal; `advisory()` always sets it.
   */
  expectedCost?: number;
  /**
   * How many more calls the remaining budget buys at expectedCost:
   * floor(remainingUsd / expectedCost). null when expectedCost is 0 (no call
   * recorded yet) — unknown, not zero. Optional for the same additive reason as
   * expectedCost; `advisory()` always sets it.
   */
  estCallsRemaining?: number | null;
  /** Token and step fields are always set by advisory(); optional for additive compatibility. */
  tokenLimit?: number | null;
  spentTokens?: number;
  remainingTokens?: number | null;
  tokenUsedBps?: number | null;
  nearTokenLimit?: boolean;
  stepLimitUsd?: number | null;
  stepSpentUsd?: number | null;
  stepRemainingUsd?: number | null;
  stepTokenLimit?: number | null;
  stepSpentTokens?: number | null;
  stepRemainingTokens?: number | null;
  stepUsedBps?: number | null;
  stepNearLimit?: boolean;
}

export class BudgetGuard<
  TokenLimit extends number | undefined = undefined,
> {
  readonly limitUsd: number;
  readonly tokenLimit: number | null;
  spentUsd = 0;
  spentTokens = 0;
  priceOverrides?: Record<string, ManualPrice>;
  failClosed: boolean;
  nearLimitBps: number;

  private readonly onBlock: (spentUsd: number, limitUsd: number) => void;
  private readonly onTokenBlock: (spentTokens: number, limitTokens: number) => void;
  /**
   * Costs of the most recent priced LLM call and tool call, tracked
   * SEPARATELY: the default next-call prediction is the max of the two, so a
   * cheap tool call can't shrink the estimate right before an expensive LLM
   * call (or vice versa) — conservative beats one-call-too-late.
   */
  private lastLlmCost = 0;
  private lastToolCost = 0;
  private lastLlmTokens = 0;
  /** USD held for in-flight calls (reserved, not yet settled). Counts toward the ceiling. */
  private reserved = 0;
  private reservedTokens = 0;
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
  constructor(
    limitUsd: number,
    options: BudgetGuardOptions<TokenLimit> = {} as BudgetGuardOptions<TokenLimit>,
  ) {
    if (!Number.isFinite(limitUsd) || limitUsd < 0) {
      // NaN/Infinity would make every check() comparison fail-open, silently
      // disabling the guard — reject them up front.
      throw new RangeError(
        `limitUsd must be a finite, non-negative number, got ${limitUsd}`,
      );
    }
    validateOptionalTokens("tokenLimit", options.tokenLimit);
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
    this.tokenLimit = options.tokenLimit ?? null;
    this.maxLogEvents = options.maxLogEvents;
    this.priceOverrides = options.priceOverrides;
    this.failClosed = options.failClosed ?? true;
    this.onBlock = options.onBlock ?? defaultOnBlock;
    this.onTokenBlock = options.onTokenBlock ?? defaultOnTokenBlock;
    this.nearLimitBps = nearLimitBps;
  }

  /**
   * Throw {@link BudgetExceeded} if the next call would cross the ceiling.
   *
   * Call this immediately before each LLM request. The "next call" is estimated
   * conservatively as the costlier of the last LLM call and the last tool call
   * (override with `estimatedNextCost`); the
   * first call is always allowed unless the ceiling is already met. In-flight
   * reservations count toward the total, so this stays correct alongside
   * {@link BudgetGuard.reserve}.
   *
   * Note: `check` is a non-binding peek. For parallel calls, use `reserve()` /
   * `settle()`, which hold the estimate across the await.
   */
  check(estimatedNextCost?: number, estimatedNextTokens?: number): void {
    this.checkForStep(estimatedNextCost, estimatedNextTokens, null);
  }

  /**
   * Atomically check the ceiling AND hold the estimated cost in flight.
   *
   * The concurrency-safe enforcement path: call before the request and hold the
   * returned reservation across the await, so parallel callers can't all clear
   * the same stale total. Throws {@link BudgetExceeded} (without reserving) if
   * the reservation would cross the ceiling. Returns the reservation handle to
   * pass to {@link BudgetGuard.settle} (or {@link BudgetGuard.release} on error).
   * `estimatedCost` defaults to the costlier of the last LLM call and the last
   * tool call.
   */
  reserve(
    estimatedCost?: number,
  ): TokenLimit extends number ? BudgetReservation : number;
  reserve(
    estimatedCost: number | undefined,
    estimatedTokens: number,
  ): TokenLimit extends number ? BudgetReservation : number;
  reserve(estimatedCost?: number, estimatedTokens?: number): ReservationHandle {
    return this.reserveForStep(estimatedCost, estimatedTokens, null);
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
    options: { reserved?: ReservationHandle; price?: ManualPrice; label?: string } = {},
  ): number {
    const reserved = options.reserved ?? 0;
    const parts = this.reservationParts(reserved);
    let totalTokens: number;
    try {
      totalTokens = actualTokenCount(promptTokens, completionTokens);
    } catch (error) {
      this.release(reserved);
      throw error;
    }
    let overrides = this.priceOverrides;
    if (options.price !== undefined) {
      overrides = { ...(overrides ?? {}), [model]: options.price };
    }

    const priced = resolvePrice(model, overrides);
    if (priced === null) {
      this.consumeHandle(reserved, parts);
      this.accrueTokens(totalTokens, parts.step);
      console.warn(
        `Cannot price model '${model}': not in the bundled cost map and no ` +
          `manual price given. The budget guard cannot enforce a ceiling on ` +
          `spend it cannot measure — pass { price } or set it in priceOverrides.`,
      );
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
    this.consumeHandle(reserved, parts);
    this.spentUsd += cost;
    this.accrueTokens(totalTokens, parts.step);
    // Clamp a sub-epsilon float overshoot back to the limit so the running total
    // never reports as having crossed the ceiling by a rounding artifact.
    if (this.spentUsd - this.limitUsd > 0 && this.spentUsd - this.limitUsd < EPS) {
      this.spentUsd = this.limitUsd;
    }
    this.lastLlmCost = cost;
    if (parts.step !== null) {
      parts.step.spentUsd += cost;
    }
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
      ...(parts.usd ? { reserved: parts.usd } : {}),
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
  reserveTool(
    estimatedCost: number,
  ): TokenLimit extends number ? BudgetReservation : number {
    if (estimatedCost === undefined) {
      // reserve(undefined) would silently fall back to the last-cost prediction
      // (0 on a fresh guard) — an unguarded tool call. A missing price must
      // fail loudly, e.g. guard.reserveTool(priceTable[tool]).
      throw new RangeError("reserveTool requires an estimated cost, got undefined");
    }
    if (!Number.isFinite(estimatedCost) || estimatedCost < 0) {
      // reserve() clamps a negative estimate to 0 (lenient LLM contract) — for
      // a tool that would reserve nothing: the same unguarded call.
      throw new RangeError(
        `estimatedCost must be a finite, non-negative number, got ${estimatedCost}`,
      );
    }
    return this.reserveForStep(
      estimatedCost,
      0,
      null,
      false,
    ) as TokenLimit extends number ? BudgetReservation : number;
  }

  /**
   * Release a reservation and record a tool call's actual cost.
   *
   * `recordTool` is `settleTool` with no reservation. The caller supplies the
   * cost — tools have no token usage to price. Accrues into the same
   * `spentUsd` ceiling as tokens, tallies the per-tool total
   * ({@link BudgetGuard.toolCosts}), updates the tool side of the next-call
   * estimate (tracked separately from the LLM side; the default prediction is
   * the max of the two, so a tool-hammering loop's plain `check()` stops
   * BEFORE the crossing call without a cheap tool shrinking the LLM
   * prediction), and appends
   * a `kind: "tool"` {@link SpendEvent} to {@link BudgetGuard.spendLog}.
   * Returns `costUsd`.
   */
  settleTool(
    tool: string,
    costUsd: number,
    options: { reserved?: ReservationHandle; label?: string } = {},
  ): number {
    if (!Number.isFinite(costUsd) || costUsd < 0) {
      throw new RangeError(`costUsd must be a finite, non-negative number, got ${costUsd}`);
    }
    const reserved = options.reserved ?? 0;
    const parts = this.reservationParts(reserved);
    this.consumeHandle(reserved, parts);
    this.spentUsd += costUsd;
    if (parts.step !== null) parts.step.spentUsd += costUsd;
    // Same sub-epsilon clamp as settle(): never report a rounding-artifact
    // crossing of the ceiling.
    if (this.spentUsd - this.limitUsd > 0 && this.spentUsd - this.limitUsd < EPS) {
      this.spentUsd = this.limitUsd;
    }
    this.lastToolCost = costUsd;
    this.toolCostTotals[tool] = (this.toolCostTotals[tool] ?? 0) + costUsd;
    this.appendEvent({
      timestamp: Date.now() / 1000,
      kind: "tool",
      modelOrTool: tool,
      promptTokens: null,
      completionTokens: null,
      costUsd,
      ...(options.label !== undefined ? { label: options.label } : {}),
      ...(parts.usd ? { reserved: parts.usd } : {}),
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
  release(reserved: ReservationHandle): void {
    // Validate before the zero-check so a NaN handle throws instead of being
    // silently dropped (a leak); a bad handle corrupts the in-flight tally.
    const parts = this.reservationParts(reserved);
    this.consumeHandle(reserved, parts);
  }

  /** USD left before the ceiling, net of in-flight reservations (never negative). */
  get remainingUsd(): number {
    return Math.max(0, this.limitUsd - this.spentUsd - this.reserved);
  }

  /** Tokens left before the aggregate ceiling, net of in-flight reservations. */
  get remainingTokens(): number | null {
    return this.tokenLimit === null
      ? null
      : Math.max(0, this.tokenLimit - this.spentTokens - this.reservedTokens);
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

  /**
   * Run work through a fresh explicit per-step guard. The callback may be sync
   * or async; overlapping steps remain isolated while sharing aggregate totals.
   */
  step<T>(
    options: StepBudgetOptions,
    callback: (step: StepBudgetGuard) => T,
  ): T {
    if (options.maxUsd === undefined && options.maxTokens === undefined) {
      throw new RangeError("step requires maxUsd, maxTokens, or both");
    }
    if (
      options.maxUsd !== undefined &&
      (!Number.isFinite(options.maxUsd) || options.maxUsd < 0)
    ) {
      throw new RangeError(
        `maxUsd must be a finite, non-negative number, got ${options.maxUsd}`,
      );
    }
    validateOptionalTokens("maxTokens", options.maxTokens);
    const state: StepState = {
      limitUsd: options.maxUsd ?? null,
      tokenLimit: options.maxTokens ?? null,
      spentUsd: 0,
      spentTokens: 0,
      reservedUsd: 0,
      reservedTokens: 0,
      active: true,
    };
    const scoped = new StepBudgetGuard(this, state);
    let result: T;
    try {
      result = callback(scoped);
    } catch (error) {
      state.active = false;
      throw error;
    }
    let promiseLike: boolean;
    try {
      promiseLike = isPromiseLike(result);
    } catch (error) {
      state.active = false;
      throw error;
    }
    if (promiseLike) {
      return Promise.resolve(result).then(
        (value) => {
          this.closeStep(state);
          return value;
        },
        (error) => {
          state.active = false;
          throw error;
        },
      ) as T;
    }
    this.closeStep(state);
    return result;
  }

  checkForStep(
    estimatedCost: number | undefined,
    estimatedTokens: number | undefined,
    step: StepState | null,
  ): void {
    const cost = this.normalizedCostEstimate(estimatedCost);
    const tokens = this.normalizedTokenEstimate(estimatedTokens);
    const blocked = this.blockingLimit(cost, tokens, step);
    if (blocked !== null) this.raiseBlock(blocked);
  }

  reserveForStep(
    estimatedCost: number | undefined,
    estimatedTokens: number | undefined,
    step: StepState | null,
    enforceTokens = true,
  ): ReservationHandle {
    if (step !== null && !step.active) {
      throw new Error("step budget is no longer active");
    }
    const cost = this.normalizedCostEstimate(estimatedCost);
    const tokens = this.normalizedTokenEstimate(estimatedTokens);
    const reservedTokens =
      this.tokenLimit !== null || step !== null ? tokens : 0;
    const blocked = this.blockingLimit(
      cost,
      reservedTokens,
      step,
      enforceTokens,
    );
    if (blocked !== null) this.raiseBlock(blocked);
    this.reserved += cost;
    this.reservedTokens += reservedTokens;
    if (step !== null) {
      step.reservedUsd += cost;
      step.reservedTokens += reservedTokens;
    }
    if (this.tokenLimit === null && step === null) return cost;
    return issueReservation(this, cost, reservedTokens, step);
  }

  private normalizedCostEstimate(estimate: number | undefined): number {
    const raw = estimate === undefined ? this.defaultEstimate() : estimate;
    if (!Number.isFinite(raw)) {
      throw new RangeError(`estimated cost must be a finite number, got ${raw}`);
    }
    return Math.max(0, raw);
  }

  private normalizedTokenEstimate(estimate: number | undefined): number {
    validateOptionalTokens("estimated tokens", estimate);
    return estimate === undefined ? this.lastLlmTokens : estimate;
  }

  private blockingLimit(
    cost: number,
    tokens: number,
    step: StepState | null,
    enforceTokens = true,
  ): BlockingLimit | null {
    const committedUsd = this.spentUsd + this.reserved;
    if (
      committedUsd > this.limitUsd - EPS ||
      committedUsd + cost > this.limitUsd + EPS
    ) {
      return {
        metric: "usd",
        scope: "aggregate",
        spent: this.spentUsd,
        limit: this.limitUsd,
      };
    }
    const committedTokens = this.spentTokens + this.reservedTokens;
    if (
      enforceTokens &&
      this.tokenLimit !== null &&
      (committedTokens >= this.tokenLimit ||
        committedTokens + tokens > this.tokenLimit)
    ) {
      return {
        metric: "tokens",
        scope: "aggregate",
        spent: this.spentTokens,
        limit: this.tokenLimit,
      };
    }
    if (step !== null) {
      const stepUsd = step.spentUsd + step.reservedUsd;
      if (
        step.limitUsd !== null &&
        (stepUsd > step.limitUsd - EPS || stepUsd + cost > step.limitUsd + EPS)
      ) {
        return {
          metric: "usd",
          scope: "step",
          spent: step.spentUsd,
          limit: step.limitUsd,
        };
      }
      const stepTokens = step.spentTokens + step.reservedTokens;
      if (
        enforceTokens &&
        step.tokenLimit !== null &&
        (stepTokens >= step.tokenLimit ||
          stepTokens + tokens > step.tokenLimit)
      ) {
        return {
          metric: "tokens",
          scope: "step",
          spent: step.spentTokens,
          limit: step.tokenLimit,
        };
      }
    }
    return null;
  }

  private raiseBlock(blocked: BlockingLimit): never {
    if (blocked.metric === "usd") {
      this.onBlock(blocked.spent, blocked.limit);
      throw new BudgetExceeded(blocked.spent, blocked.limit, blocked.scope);
    }
    this.onTokenBlock(blocked.spent, blocked.limit);
    throw new TokenBudgetExceeded(
      blocked.spent,
      blocked.limit,
      blocked.scope,
    );
  }

  private reservationParts(handle: ReservationHandle): ReservationParts {
    if (typeof handle === "number") {
      if (!Number.isFinite(handle) || handle < 0) {
        throw new RangeError(
          `reserved must be a finite, non-negative number, got ${handle}`,
        );
      }
      return { usd: handle, tokens: 0, step: null };
    }
    const state = reservationStates.get(handle);
    if (state === undefined) {
      throw new RangeError(
        "invalid reservation handle; use the original object returned by reserve()",
      );
    }
    if (state.owner !== this) {
      throw new RangeError("reservation belongs to a different BudgetGuard");
    }
    if (!state.active) {
      throw new RangeError("reservation has already been settled or released");
    }
    if (
      !Number.isFinite(handle.usd) ||
      handle.usd < 0 ||
      !Number.isInteger(handle.tokens) ||
      handle.tokens < 0 ||
      handle.usd !== state.usd ||
      handle.tokens !== state.tokens
    ) {
      throw new RangeError("reservation contains invalid USD or token values");
    }
    return { usd: state.usd, tokens: state.tokens, step: state.step };
  }

  private consumeHandle(
    handle: ReservationHandle,
    parts: ReservationParts,
  ): void {
    if (
      parts.usd > this.reserved + EPS ||
      parts.tokens > this.reservedTokens
    ) {
      throw new RangeError("reservation exceeds total in-flight reservations");
    }
    if (
      parts.step !== null &&
      (parts.usd > parts.step.reservedUsd + EPS ||
        parts.tokens > parts.step.reservedTokens)
    ) {
      throw new RangeError("reservation exceeds its step's in-flight totals");
    }
    this.consumeReservation(parts.usd);
    this.reservedTokens -= parts.tokens;
    if (parts.step !== null) {
      parts.step.reservedUsd = Math.max(0, parts.step.reservedUsd - parts.usd);
      parts.step.reservedTokens -= parts.tokens;
    }
    if (typeof handle !== "number") {
      const state = reservationStates.get(handle);
      // reservationParts() validated issuance, ownership, and active state.
      if (state === undefined || state.owner !== this || !state.active) {
        throw new RangeError("invalid or already consumed reservation handle");
      }
      state.active = false;
    }
  }

  private accrueTokens(tokens: number, step: StepState | null): void {
    this.spentTokens += tokens;
    this.lastLlmTokens = tokens;
    if (step !== null) step.spentTokens += tokens;
  }

  private closeStep(step: StepState): void {
    step.active = false;
    if (step.reservedUsd > EPS || step.reservedTokens > 0) {
      throw new Error(
        "step exited with an active reservation; settle or release every handle",
      );
    }
  }

  /**
   * The default next-call prediction when the caller supplies no estimate.
   * Conservative: the costlier of the last LLM call and the last tool call — a
   * mixed loop predicts the pricier kind, which at worst blocks one call early
   * (fail-closed) rather than letting a crossing call through because the LAST
   * event happened to be cheap.
   */
  private defaultEstimate(): number {
    return Math.max(this.lastLlmCost, this.lastToolCost);
  }

  /**
   * Subtract a settled/released hold from the in-flight tally. A handle larger
   * than EVERYTHING currently held cannot have come from a matching
   * `reserve()` — throwing beats silently clamping, which would free OTHER
   * callers' holds and fail the ceiling open. The epsilon absorbs float dust
   * from accumulating and draining many holds; per-caller over-release (a
   * handle within the total but larger than the caller's own hold) is
   * undetectable without per-handle tracking and remains the caller's
   * responsibility.
   */
  private consumeReservation(reserved: number): void {
    if (reserved > this.reserved + EPS) {
      throw new RangeError(
        `reserved handle (${reserved}) exceeds total in-flight reservations ` +
          `(${this.reserved}) — a handle must come from a matching reserve()`,
      );
    }
    this.reserved = Math.max(0, this.reserved - reserved);
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
    return this.advisoryForStep(null);
  }

  advisoryForStep(step: StepState | null): BudgetAdvisory {
    // Floor (not round) so usedBps never over-reports utilization and nearLimit
    // flips exactly when the threshold is reached; the epsilon absorbs float noise
    // and Math.floor matches Python's int() exactly (round() would diverge).
    const usedBps =
      this.limitUsd <= 0
        ? 10000
        : Math.max(0, Math.min(10000, Math.floor((this.spentUsd / this.limitUsd) * 10000 + 1e-9)));
    const remainingUsd = Math.max(0, this.limitUsd - this.spentUsd);
    const tokenUsedBps =
      this.tokenLimit === null
        ? null
        : usedBasisPoints(this.spentTokens, this.tokenLimit);
    const remainingTokens =
      this.tokenLimit === null
        ? null
        : Math.max(0, this.tokenLimit - this.spentTokens);
    const nearTokenLimit =
      tokenUsedBps !== null && tokenUsedBps >= this.nearLimitBps;
    const stepUsdBps =
      step?.limitUsd != null
        ? usedBasisPoints(step.spentUsd, step.limitUsd)
        : null;
    const stepTokenBps =
      step?.tokenLimit != null
        ? usedBasisPoints(step.spentTokens, step.tokenLimit)
        : null;
    const stepUsedBps =
      step === null
        ? null
        : Math.max(...[stepUsdBps, stepTokenBps].filter((v): v is number => v !== null));
    const stepNearLimit =
      stepUsedBps !== null && stepUsedBps >= this.nearLimitBps;
    // The costlier of the last LLM and tool call — the guard's own default
    // reservation estimate; 0 before any call is recorded.
    const expectedCost = Math.max(this.lastLlmCost, this.lastToolCost);
    // +1e-9 absorbs float noise so e.g. 0.6/0.2 floors to 3 not 2 (same epsilon
    // rationale as usedBps above, and keeps Python int() parity).
    const estCallsRemaining =
      expectedCost > 0 ? Math.floor(remainingUsd / expectedCost + 1e-9) : null;
    return {
      nearLimit:
        usedBps >= this.nearLimitBps || nearTokenLimit || stepNearLimit,
      usedBps,
      // Settled budget: limit minus accrued spend, deliberately NOT net of
      // in-flight reservations. Unlike the remainingUsd getter (which subtracts
      // `reserved`), the advisory is a soft utilization signal about money already
      // spent, while the getter reports what a new call can still claim.
      remainingUsd,
      limitUsd: this.limitUsd,
      spentUsd: this.spentUsd,
      scope: "local",
      expectedCost,
      estCallsRemaining,
      tokenLimit: this.tokenLimit,
      spentTokens: this.spentTokens,
      remainingTokens,
      tokenUsedBps,
      nearTokenLimit,
      stepLimitUsd: step?.limitUsd ?? null,
      stepSpentUsd: step?.spentUsd ?? null,
      stepRemainingUsd:
        step?.limitUsd != null
          ? Math.max(0, step.limitUsd - step.spentUsd)
          : null,
      stepTokenLimit: step?.tokenLimit ?? null,
      stepSpentTokens: step?.spentTokens ?? null,
      stepRemainingTokens:
        step?.tokenLimit != null
          ? Math.max(0, step.tokenLimit - step.spentTokens)
          : null,
      stepUsedBps,
      stepNearLimit,
    };
  }
}

/** Scoped view over a parent guard with independent per-step ceilings. */
export class StepBudgetGuard {
  constructor(
    private readonly parent: BudgetGuard<number | undefined>,
    private readonly state: StepState,
  ) {}

  get limitUsd(): number { return this.parent.limitUsd; }
  get tokenLimit(): number | null { return this.parent.tokenLimit; }
  get spentUsd(): number { return this.parent.spentUsd; }
  get spentTokens(): number { return this.parent.spentTokens; }
  get remainingUsd(): number { return this.parent.remainingUsd; }
  get remainingTokens(): number | null { return this.parent.remainingTokens; }
  get priceOverrides(): Record<string, ManualPrice> | undefined {
    return this.parent.priceOverrides;
  }
  get failClosed(): boolean { return this.parent.failClosed; }

  check(estimatedCost?: number, estimatedTokens?: number): void {
    this.ensureActive();
    this.parent.checkForStep(estimatedCost, estimatedTokens, this.state);
  }

  reserve(estimatedCost?: number, estimatedTokens?: number): ReservationHandle {
    this.ensureActive();
    return this.parent.reserveForStep(estimatedCost, estimatedTokens, this.state);
  }

  settle(
    model: string,
    promptTokens: number,
    completionTokens: number,
    options: { reserved?: ReservationHandle; price?: ManualPrice; label?: string } = {},
  ): number {
    this.ensureActive();
    const reserved = normalizeStepHandle(
      this.parent,
      options.reserved ?? 0,
      this.state,
    );
    return this.parent.settle(model, promptTokens, completionTokens, {
      ...options,
      reserved,
    });
  }

  record(
    model: string,
    promptTokens: number,
    completionTokens: number,
    options: { price?: ManualPrice; label?: string } = {},
  ): number {
    return this.settle(model, promptTokens, completionTokens, options);
  }

  reserveTool(estimatedCost: number): ReservationHandle {
    this.ensureActive();
    return this.parent.reserveForStep(estimatedCost, 0, this.state, false);
  }

  settleTool(
    tool: string,
    costUsd: number,
    options: { reserved?: ReservationHandle; label?: string } = {},
  ): number {
    this.ensureActive();
    const reserved = normalizeStepHandle(
      this.parent,
      options.reserved ?? 0,
      this.state,
    );
    return this.parent.settleTool(tool, costUsd, { ...options, reserved });
  }

  recordTool(tool: string, costUsd: number, options: { label?: string } = {}): number {
    return this.settleTool(tool, costUsd, options);
  }

  release(reserved: ReservationHandle): void {
    this.parent.release(normalizeStepHandle(this.parent, reserved, this.state));
  }

  advisory(): BudgetAdvisory {
    return this.parent.advisoryForStep(this.state);
  }

  private ensureActive(): void {
    if (!this.state.active) throw new Error("step budget is no longer active");
  }
}

function usedBasisPoints(used: number, limit: number): number {
  if (limit <= 0) return 10000;
  return Math.max(0, Math.min(10000, Math.floor((used / limit) * 10000 + 1e-9)));
}

function validateOptionalTokens(name: string, value: number | undefined): void {
  if (
    value !== undefined &&
    (!Number.isInteger(value) || !Number.isFinite(value) || value < 0)
  ) {
    throw new RangeError(`${name} must be a non-negative integer, got ${value}`);
  }
}

function actualTokenCount(promptTokens: number, completionTokens: number): number {
  if (!Number.isFinite(promptTokens) || !Number.isFinite(completionTokens)) {
    throw new RangeError("token counts must be finite numbers");
  }
  return Math.trunc(Math.max(0, promptTokens)) + Math.trunc(Math.max(0, completionTokens));
}

function defaultOnBlock(spentUsd: number, limitUsd: number): void {
  console.error(
    "BUDGET EXCEEDED — call blocked\n" +
      `  spent so far: $${spentUsd.toFixed(6)}  |  ceiling: $${limitUsd.toFixed(6)}\n` +
      "  The next call would cross your budget; floe-guard stopped your agent " +
      "before it ran.",
  );
}

function defaultOnTokenBlock(spentTokens: number, limitTokens: number): void {
  console.error(
    "TOKEN BUDGET EXCEEDED — call blocked\n" +
      `  tokens so far: ${spentTokens}  |  ceiling: ${limitTokens}\n` +
      "  The next call would cross your token budget; floe-guard stopped your " +
      "agent before it ran.",
  );
}

function isPromiseLike(value: unknown): value is PromiseLike<unknown> {
  if (
    value === null ||
    (typeof value !== "object" && typeof value !== "function")
  ) {
    return false;
  }
  return typeof (value as { then?: unknown }).then === "function";
}
