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
    this.limitUsd = limitUsd;
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
    const estimate =
      estimatedNextCost === undefined
        ? this.lastCost
        : Math.max(0, estimatedNextCost);
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
    const estimate =
      estimatedCost === undefined ? this.lastCost : Math.max(0, estimatedCost);
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
   * on the warn-and-skip path.
   */
  settle(
    model: string,
    promptTokens: number,
    completionTokens: number,
    options: { reserved?: number; price?: ManualPrice } = {},
  ): number {
    const reserved = options.reserved ?? 0;
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
    options: { price?: ManualPrice } = {},
  ): number {
    return this.settle(model, promptTokens, completionTokens, {
      reserved: 0,
      price: options.price,
    });
  }

  /**
   * Drop an in-flight reservation without recording spend (e.g. the call failed
   * before producing usage). Safe to call with `0`.
   */
  release(reserved: number): void {
    if (!reserved) return;
    this.reserved = Math.max(0, this.reserved - reserved);
  }

  /** USD left before the ceiling, net of in-flight reservations (never negative). */
  get remainingUsd(): number {
    return Math.max(0, this.limitUsd - this.spentUsd - this.reserved);
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
