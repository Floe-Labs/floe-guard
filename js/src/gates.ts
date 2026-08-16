/**
 * Pre-call admission gates for voice orchestrators.
 *
 * These decide **at the call boundary** whether to *admit* or *reject* an incoming
 * call, from the guard's remaining budget. They return the exact JSON shape each
 * provider's inbound webhook expects, so a free/local user can reject a call on
 * budget exhaustion with the **same contract** the hosted gateway serves — the paid
 * upgrade is a URL swap, not a rewrite.
 *
 * Scope — strictly pre-call. A gate answers one question: *given the budget left,
 * do we let this call start?* It does **not** intervene mid-call. Once a call is
 * admitted it runs to completion; nothing here cuts a call off partway.
 *
 * Budget, not balance: admission reads the guard's local *remaining budget*, a
 * ceiling signal — not an account balance. US-only telephony, v1.
 *
 * A faithful port of `src/floe_guard/gates.py`.
 */

import type { BudgetGuard } from "./guard.js";

/**
 * True when there isn't budget left to admit one more call.
 *
 * A **non-binding preflight read**: it checks the guard's `remainingUsd` but does
 * *not* reserve anything, so under concurrent inbound calls it can admit more than
 * the budget strictly covers (two calls may see the same headroom and both pass).
 * It is coarse admission control — the binding, atomic hard-stop is the in-call
 * guard (`check` / `reserve` while the call runs). This mirrors the hosted pre-dial
 * gate, which is likewise check-only.
 *
 * With the default `estimatedCallUsd = 0` a call is rejected only once the budget
 * is fully spent (`remainingUsd` at or below zero). Pass an estimate (e.g.
 * `$/min × expected minutes`) to reject earlier, when the remaining budget can't
 * cover it; an estimate that *exactly* equals the remaining budget still admits
 * (inclusive ceiling, matching `BudgetGuard.reserve`). Budget signal, not a balance.
 *
 * @throws RangeError when `estimatedCallUsd` is negative or non-finite — a bad
 *   estimate must not silently admit an exhausted guard (NaN/negative comparisons
 *   are always false, the same trap `BudgetGuard` rejects for `limitUsd`).
 */
export function budgetExhausted(
  guard: BudgetGuard,
  options: { estimatedCallUsd?: number } = {},
): boolean {
  // Default ONLY undefined — a `null` estimate is malformed input and must reach
  // validation (below) rather than being silently coerced to 0 (which would admit
  // the call with no intended headroom). Mirrors Python raising on a None estimate.
  const estimatedCallUsd = options.estimatedCallUsd === undefined ? 0 : options.estimatedCallUsd;
  if (!Number.isFinite(estimatedCallUsd) || estimatedCallUsd < 0) {
    throw new RangeError(
      `estimatedCallUsd must be a finite, non-negative number, got ${estimatedCallUsd}`,
    );
  }
  const remainingUsd = guard.remainingUsd;
  return remainingUsd <= 0 || remainingUsd < estimatedCallUsd;
}

/**
 * Generic pre-call admission decision: `true` = admit, `false` = reject.
 *
 * The provider-agnostic gate for Pipecat, custom telephony, or any inbound hook
 * without a first-class helper below. Wire the `false` case into your provider's
 * reject path.
 *
 * TODO(bland): Bland's *Send Call* metadata field name is an OPEN VERIFICATION
 * ITEM (pending confirmation) — this package does not invent it. Use `preCall` to
 * get the admit/reject decision and map `false` onto Bland's Pathway Webhook node
 * once the field name is confirmed.
 */
export function preCall(guard: BudgetGuard, options: { estimatedCallUsd?: number } = {}): boolean {
  return !budgetExhausted(guard, options);
}

/**
 * Response for Retell's inbound-call webhook — reject or admit the call.
 *
 * On budget exhaustion returns `{ call_inbound: { reject: true } }`; otherwise
 * `{ call_inbound: {...} }` carrying any `admit` overrides you pass
 * (`dynamic_variables`, `metadata`, `override_agent_id`, …).
 *
 * Retell contract (verified against docs.retellai.com/features/inbound-call-webhook):
 * **only the boolean `true` rejects** — any other value is ignored. The webhook
 * fires for inbound **phone and SMS** calls (not dial-to-sip); its behaviour for
 * web calls is not documented, so treat this as phone/SMS-only. 10s timeout, up to
 * 3 retries.
 */
export function retell(
  guard: BudgetGuard,
  options: { estimatedCallUsd?: number; admit?: Record<string, unknown> } = {},
): { call_inbound: Record<string, unknown> } {
  if (budgetExhausted(guard, { estimatedCallUsd: options.estimatedCallUsd })) {
    return { call_inbound: { reject: true } };
  }
  // The gate's budget decision — not the caller's overrides — controls admission.
  // Strip `reject` from admit overrides so an errant `admit: { reject: true }` on
  // an available-budget call can't flip the response into Retell's reject shape.
  const safeAdmit = { ...(options.admit ?? {}) };
  delete safeAdmit.reject;
  return { call_inbound: safeAdmit };
}

/**
 * Response for Vapi's `assistant-request` webhook — reject or admit the call.
 *
 * On budget exhaustion returns `{ error: errorMessage }` (Vapi speaks it, then ends
 * the call). Otherwise admits with `{ assistantId }` if given (`assistantId` takes
 * precedence), else `{ assistant }`.
 *
 * Vapi has **no** `reject: true` boolean — rejection *is* returning an error, and
 * admission *is* returning an assistant. Respond within **~7.5 s** or the call may
 * fail (verified against docs.vapi.ai/server-url/spam-call-rejection).
 *
 * @throws RangeError when admitted (budget available) but neither `assistant` nor
 *   `assistantId` was provided — there'd be nothing to hand Vapi.
 */
export function vapi(
  guard: BudgetGuard,
  options: {
    assistant?: Record<string, unknown>;
    assistantId?: string;
    errorMessage?: string;
    estimatedCallUsd?: number;
  } = {},
): Record<string, unknown> {
  const {
    assistant,
    assistantId,
    errorMessage = "Sorry, this agent is out of budget right now.",
    estimatedCallUsd,
  } = options;
  if (budgetExhausted(guard, { estimatedCallUsd })) {
    return { error: errorMessage };
  }
  if (assistantId !== undefined && assistantId !== null) {
    return { assistantId };
  }
  if (assistant !== undefined && assistant !== null) {
    return { assistant };
  }
  throw new RangeError(
    "vapi() admitted the call but has nothing to return: pass assistant or " +
      "assistantId for the admit path.",
  );
}
