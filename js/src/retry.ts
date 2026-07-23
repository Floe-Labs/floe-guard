import { BudgetExceeded } from "./errors.js";
import { type BudgetAdvisory, BudgetGuard } from "./guard.js";

export interface RetryPlan<T> {
  /** Operation to run for this retry attempt. */
  call: () => T | Promise<T>;
  /**
   * Estimated retry cost passed to `guard.check()` before the retry runs.
   * Leave undefined to use the guard's default last-call estimate.
   */
  estimatedCost?: number;
}

export interface BudgetRetryOptions<T> {
  /** Estimated cost for retrying the original call. */
  estimatedCost?: number;
  /** Total attempts, including the first call. Default: 2. */
  maxAttempts?: number;
  /** Choose a cheaper retry plan when `guard.advisory().nearLimit` is true. */
  onDegrade?: (
    error: unknown,
    advisory: BudgetAdvisory,
  ) => RetryPlan<T> | Promise<RetryPlan<T> | undefined> | undefined;
  /** Decide whether an error is retryable. Defaults to all non-budget errors. */
  retryIf?: (error: unknown) => boolean;
}

function defaultRetryIf(error: unknown): boolean {
  return !(error instanceof BudgetExceeded);
}

/**
 * Run `call` with budget-aware retries.
 *
 * The first attempt runs unchanged. If it fails with a retryable error, the
 * helper retries as-is with ample budget, asks `onDegrade` for a cheaper plan
 * when near the limit, and always calls `guard.check(estimatedCost)` before a
 * retry so an over-budget retry is blocked before it runs.
 */
export async function withBudgetRetry<T>(
  guard: BudgetGuard,
  call: () => T | Promise<T>,
  options: BudgetRetryOptions<T> = {},
): Promise<T> {
  const maxAttempts = options.maxAttempts ?? 2;
  if (!Number.isInteger(maxAttempts) || maxAttempts < 1) {
    throw new RangeError(`maxAttempts must be an integer >= 1, got ${maxAttempts}`);
  }
  const retryIf = options.retryIf ?? defaultRetryIf;
  let plan: RetryPlan<T> = { call, estimatedCost: options.estimatedCost };

  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    try {
      return await plan.call();
    } catch (error) {
      if (attempt >= maxAttempts || !retryIf(error)) {
        throw error;
      }
      plan = await nextPlan(guard, error, plan, options.onDegrade);
    }
  }

  throw new Error("unreachable");
}

async function nextPlan<T>(
  guard: BudgetGuard,
  error: unknown,
  current: RetryPlan<T>,
  onDegrade: BudgetRetryOptions<T>["onDegrade"],
): Promise<RetryPlan<T>> {
  const advisory = guard.advisory();
  if (advisory.nearLimit && onDegrade !== undefined) {
    const degraded = await onDegrade(error, advisory);
    if (degraded !== undefined) {
      guard.check(degraded.estimatedCost);
      return degraded;
    }
  }
  guard.check(current.estimatedCost);
  return current;
}
