/**
 * Exceptions for floe-guard.
 *
 * Everything derives from {@link FloeGuardError} (the package-root base) so callers
 * can catch the whole family with a single `catch (e) { if (e instanceof FloeGuardError) ... }`.
 *
 * Mirrors `src/floe_guard/errors.py` in the Python package — message formats are
 * kept byte-for-byte identical so both adapters read the same.
 */

/** Base class for every error raised by floe-guard. */
export class FloeGuardError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "FloeGuardError";
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/**
 * Thrown before an LLM call that would cross the configured spend ceiling.
 *
 * The guard throws this *instead of* letting the next call run, so a runaway loop
 * stops here rather than burning more money.
 */
export class BudgetExceeded extends FloeGuardError {
  readonly spentUsd: number;
  readonly limitUsd: number;

  constructor(spentUsd: number, limitUsd: number) {
    super(
      `BUDGET EXCEEDED — call blocked (spent $${spentUsd.toFixed(6)} of $${limitUsd.toFixed(6)} ceiling)`,
    );
    this.name = "BudgetExceeded";
    this.spentUsd = spentUsd;
    this.limitUsd = limitUsd;
  }
}

/**
 * Thrown when a model cannot be priced and the guard is fail-closed.
 *
 * We refuse rather than silently accrue $0 — "we cannot cap what we cannot price".
 * Pass a manual price (`priceOverrides` or `record(..., { price })`) to make the
 * model enforceable.
 */
export class UnpriceableModelError extends FloeGuardError {
  readonly model: string;

  constructor(model: string) {
    super(
      `Cannot price model '${model}': not in the bundled cost map and no ` +
        `manual price was given. The guard cannot enforce a budget on spend ` +
        `it cannot measure. Pass a price override to enable enforcement.`,
    );
    this.name = "UnpriceableModelError";
    this.model = model;
  }
}
