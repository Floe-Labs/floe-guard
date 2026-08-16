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

  constructor(spentUsd: number, limitUsd: number, message?: string) {
    // Subclasses (the token twin) pass their own message through so they don't
    // have to reassign `this.message` after super().
    super(
      message ??
        `BUDGET EXCEEDED — call blocked (spent $${spentUsd.toFixed(6)} of $${limitUsd.toFixed(6)} ceiling)`,
    );
    this.name = "BudgetExceeded";
    this.spentUsd = spentUsd;
    this.limitUsd = limitUsd;
  }
}

/**
 * Thrown before a call that would cross a token ceiling (aggregate or step).
 *
 * The token twin of {@link BudgetExceeded} — it extends it so the same retry
 * logic that treats a USD block as terminal (`!(error instanceof
 * BudgetExceeded)`) treats a token block as terminal too, with no extra wiring.
 * `spentUsd` / `limitUsd` are inherited but not meaningful here; the token
 * fields are the payload.
 *
 * `scope` is `"aggregate"` (the guard-wide `tokenLimit`) or `"step"` (the
 * innermost active {@link BudgetGuard.step} cap).
 */
export class TokenBudgetExceeded extends BudgetExceeded {
  readonly spentTokens: number;
  readonly limitTokens: number;
  readonly scope: "aggregate" | "step";

  constructor(spentTokens: number, limitTokens: number, scope: "aggregate" | "step") {
    // Pass the token-shaped message through super (BudgetExceeded's default is
    // dollar-shaped); 0/0 for the inherited spentUsd/limitUsd — a token block
    // spent no tracked USD.
    super(
      0,
      0,
      `TOKEN BUDGET EXCEEDED — call blocked (${scope}: ${spentTokens} of ` +
        `${limitTokens} token ceiling)`,
    );
    this.name = "TokenBudgetExceeded";
    this.spentTokens = spentTokens;
    this.limitTokens = limitTokens;
    this.scope = scope;
  }
}

/**
 * Thrown when an opt-in ledger sync to Reconcile Mode fails.
 *
 * Covers a missing API key, a non-2xx response (401 bad/missing key, 403 agent
 * closed/suspended, or a read-only key), a network/timeout failure, or a
 * malformed response. Sync is off by default and only runs when the caller
 * explicitly opts in (`pushLedger` / the `floe-guard push` CLI) — this error
 * never fires for a guard that hasn't opted in, because such a guard never sends.
 *
 * Mirrors `LedgerSyncError` in `src/floe_guard/errors.py`.
 */
export class LedgerSyncError extends FloeGuardError {
  constructor(message: string) {
    super(message);
    this.name = "LedgerSyncError";
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

/**
 * Thrown when a voice leg (STT/TTS/telephony) cannot be priced and the guard is
 * fail-closed.
 *
 * The voice twin of {@link UnpriceableModelError}: we refuse rather than silently
 * accrue $0 — "we cannot cap what we cannot price". It fires when an adapter is
 * asked to meter a leg for a vendor that is absent from the bundled voice cost map
 * (or whose entry has the wrong unit/mode for the leg) and no per-unit override was
 * given. Pass a per-unit rate (`stt_usd_per_second` / `tts_usd_per_1k_chars` /
 * `telephony_usd_per_minute`) to make the leg enforceable.
 *
 * Mirrors `UnpriceableVoiceError` in `src/floe_guard/errors.py`.
 */
export class UnpriceableVoiceError extends FloeGuardError {
  readonly vendor: string | null;
  readonly mode: string;

  constructor(vendor: string | null, mode: string) {
    // Match Python's `{vendor!r}`: a string is quoted, None renders as `None`.
    const shown = vendor === null ? "None" : `'${vendor}'`;
    super(
      `Cannot price ${mode} vendor ${shown}: not in the bundled voice cost ` +
        `map (or its entry has the wrong unit for a ${mode} leg) and no per-unit ` +
        `override was given. The guard cannot enforce a budget on spend it ` +
        `cannot measure. Pass a per-unit rate to enable enforcement.`,
    );
    this.name = "UnpriceableVoiceError";
    this.vendor = vendor;
    this.mode = mode;
  }
}

/**
 * Thrown before a call whose projected duration would blow the SLA.
 *
 * The latency twin of {@link BudgetExceeded}: `LatencyBudget.check()` throws this
 * *instead of* letting the next tool/model call start, so the chain sheds work or
 * falls back to a faster path rather than violating the end-user SLA. Cooperative —
 * killing an already-running stalled call is the framework's job (AbortSignal),
 * not the guard's.
 *
 * Mirrors `DeadlineExceeded` in `src/floe_guard/errors.py` — message format is
 * kept byte-for-byte identical so both adapters read the same.
 */
/** Shared millisecond rounding for cross-language message parity: `toFixed(0)`
 *  rounds half-up while Python's `:.0f` rounds half-to-even, so tie values
 *  would break the byte-for-byte contract. Both packages use floor(x + 0.5)
 *  (see `_round_half_up` in `src/floe_guard/errors.py`). */
function roundHalfUp(ms: number): number {
  return Math.floor(ms + 0.5);
}

export class DeadlineExceeded extends FloeGuardError {
  readonly elapsedMs: number;
  readonly slaMs: number;

  constructor(elapsedMs: number, slaMs: number) {
    super(
      `DEADLINE EXCEEDED — call blocked (elapsed ${roundHalfUp(elapsedMs)}ms of ${roundHalfUp(slaMs)}ms SLA)`,
    );
    this.name = "DeadlineExceeded";
    this.elapsedMs = elapsedMs;
    this.slaMs = slaMs;
  }
}
