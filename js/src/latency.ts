/**
 * LatencyBudget — a cumulative tool-chain deadline, sibling to BudgetGuard.
 *
 * BudgetGuard stops an agent before its next call crosses a USD ceiling;
 * LatencyBudget stops it before the next call would blow an end-user SLA:
 *
 * ```ts
 * const deadline = new LatencyBudget(5000);
 * ...
 * deadline.check(800);                       // throws DeadlineExceeded when projected over
 * if (deadline.advisory().nearDeadline) useFasterModel();
 * router.pick({ maxLatencyMs: deadline.remainingMs });
 * ```
 *
 * Design notes (mirroring `src/floe_guard/latency.py`):
 * - **Monotonic clock** — `performance.now()`, never wall time.
 * - **Cooperative, not preemptive** — the guard provides the deadline signal;
 *   aborting a stalled in-flight call is the framework's job (AbortSignal).
 *   `check()` prevents the NEXT call from starting.
 * - **Advisory symmetry** — `nearDeadline` / `usedBps` / `remainingMs` are the
 *   latency twin of BudgetGuard's `nearLimit` / `usedBps` / `remainingUsd`.
 * - **In-process scope** — one instance per request/run; distributed latency
 *   tracking is out of scope.
 */

import { DeadlineExceeded } from "./errors.js";

/** A context-aware deadline signal — the latency twin of {@link BudgetAdvisory}.
 *  Soft by design; the hard-stop is {@link LatencyBudget.check}. */
export interface LatencyAdvisory {
  nearDeadline: boolean;
  /** SLA consumed, basis points 0..10000 (8500 = 85%). */
  usedBps: number;
  remainingMs: number;
  slaMs: number;
  elapsedMs: number;
}

export interface LatencyBudgetOptions {
  /**
   * Utilization (basis points, 0..10000) at which {@link LatencyBudget.advisory}
   * flags `nearDeadline` so an agent can downshift to a faster path before the
   * wall. Default 8000 (80%), matching BudgetGuard's `nearLimitBps`.
   */
  nearDeadlineBps?: number;
  /** Invoked with `(elapsedMs, slaMs)` right before {@link DeadlineExceeded} is thrown. */
  onBlock?: (elapsedMs: number, slaMs: number) => void;
  /** Milliseconds-returning monotonic clock, injectable for tests. Defaults to `performance.now`. */
  clock?: () => number;
}

export class LatencyBudget {
  readonly slaMs: number;
  readonly nearDeadlineBps: number;
  private readonly onBlock?: (elapsedMs: number, slaMs: number) => void;
  private readonly clock: () => number;
  private readonly startedAt: number;

  /** The budget starts counting at construction — build it when the request
   *  (and its SLA) starts. */
  constructor(slaMs: number, options: LatencyBudgetOptions = {}) {
    if (!(Number.isFinite(slaMs) && slaMs > 0)) {
      throw new RangeError("slaMs must be > 0");
    }
    const nearDeadlineBps = options.nearDeadlineBps ?? 8000;
    if (!Number.isInteger(nearDeadlineBps) || nearDeadlineBps < 0 || nearDeadlineBps > 10000) {
      throw new RangeError("nearDeadlineBps must be an integer 0..10000");
    }
    this.slaMs = slaMs;
    this.nearDeadlineBps = nearDeadlineBps;
    this.onBlock = options.onBlock;
    this.clock = options.clock ?? (() => performance.now());
    this.startedAt = this.clock();
  }

  /** Milliseconds since construction (monotonic). */
  get elapsedMs(): number {
    return this.clock() - this.startedAt;
  }

  /** Milliseconds left before the SLA, floored at 0 — the readable signal a
   *  router uses to pick a faster fallback or truncate work mid-chain. */
  get remainingMs(): number {
    return Math.max(0, this.slaMs - this.elapsedMs);
  }

  /**
   * Throw {@link DeadlineExceeded} when the projected elapsed time (now +
   * `expectedMs` for the upcoming call) would blow the SLA. Call it
   * immediately before each tool/model call; pass 0 to only gate on time
   * already spent.
   */
  check(expectedMs = 0): void {
    if (!(Number.isFinite(expectedMs) && expectedMs >= 0)) {
      throw new RangeError("expectedMs must be >= 0");
    }
    const elapsed = this.elapsedMs;
    if (elapsed + expectedMs > this.slaMs) {
      this.onBlock?.(elapsed, this.slaMs);
      throw new DeadlineExceeded(elapsed, this.slaMs);
    }
  }

  /** The soft near-deadline signal — symmetric to `BudgetGuard.advisory()`. */
  advisory(): LatencyAdvisory {
    const elapsed = this.elapsedMs;
    // round (not floor) — matches the Python twin; float clock arithmetic can
    // land a hair under an exact boundary.
    const usedBps = elapsed > 0 ? Math.min(10000, Math.round((elapsed * 10000) / this.slaMs)) : 0;
    return {
      nearDeadline: usedBps >= this.nearDeadlineBps,
      usedBps,
      remainingMs: Math.max(0, this.slaMs - elapsed),
      slaMs: this.slaMs,
      elapsedMs: elapsed,
    };
  }
}
