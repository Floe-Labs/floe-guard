/**
 * Pre-call admission gates return the exact provider inbound-webhook shapes.
 *
 * Reject on budget exhaustion, admit otherwise — the same contract a free/local
 * user serves as the hosted gateway. Pre-call only; no mid-call behaviour here.
 *
 * Mirrors `tests/test_gates.py`.
 */

import { describe, expect, it } from "vitest";

import { BudgetGuard, gates } from "../src/index.js";

function spent(limit: number, spend: number): BudgetGuard {
  const guard = new BudgetGuard(limit);
  if (spend > 0) guard.recordTool("api", spend);
  return guard;
}

describe("gates — generic decision", () => {
  it("budgetExhausted and preCall", () => {
    expect(gates.budgetExhausted(spent(1.0, 1.0))).toBe(true);
    expect(gates.budgetExhausted(spent(1.0, 0.5))).toBe(false);
    expect(gates.preCall(spent(1.0, 0.5))).toBe(true);
    expect(gates.preCall(spent(1.0, 1.0))).toBe(false);
  });

  it("estimatedCallUsd reserves headroom", () => {
    // $0.30 left, but a call estimated at $0.50 → reject at admission.
    const guard = spent(1.0, 0.7);
    expect(gates.preCall(guard, { estimatedCallUsd: 0.5 })).toBe(false);
    expect(gates.preCall(guard, { estimatedCallUsd: 0.2 })).toBe(true);
  });

  it("an estimate exactly equal to remaining admits (inclusive ceiling)", () => {
    const guard = spent(1.0, 0.5); // $0.50 remaining
    expect(gates.budgetExhausted(guard, { estimatedCallUsd: 0.5 })).toBe(false);
    expect(gates.budgetExhausted(guard, { estimatedCallUsd: 0.5001 })).toBe(true);
  });

  it("fully spent is exhausted regardless of estimate", () => {
    const guard = spent(1.0, 1.0); // $0 remaining
    expect(gates.budgetExhausted(guard)).toBe(true);
    expect(gates.budgetExhausted(guard, { estimatedCallUsd: 0.0 })).toBe(true);
  });

  it("a negative/NaN/Infinity estimate throws instead of admitting", () => {
    // A bad estimate must not silently admit an exhausted guard.
    const guard = spent(1.0, 1.0); // exhausted
    for (const bad of [-1.0, -0.01, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(() => gates.budgetExhausted(guard, { estimatedCallUsd: bad })).toThrow(RangeError);
    }
  });
});

describe("gates — Retell", () => {
  it("rejects on exhaustion", () => {
    expect(gates.retell(spent(1.0, 1.0))).toEqual({ call_inbound: { reject: true } });
  });

  it("admits with overrides and no reject key", () => {
    const resp = gates.retell(spent(1.0, 0.1), { admit: { dynamic_variables: { name: "Ada" } } });
    expect(resp).toEqual({ call_inbound: { dynamic_variables: { name: "Ada" } } });
    expect("reject" in resp.call_inbound).toBe(false);
  });

  it("reject is the boolean true, not a string", () => {
    const reject = gates.retell(spent(1.0, 1.0)).call_inbound.reject;
    expect(reject).toBe(true);
  });
});

describe("gates — Vapi", () => {
  it("rejects with the error shape", () => {
    const resp = gates.vapi(spent(1.0, 1.0), { assistantId: "asst_1", errorMessage: "No budget." });
    expect(resp).toEqual({ error: "No budget." });
  });

  it("admits with assistantId (takes precedence over assistant)", () => {
    const resp = gates.vapi(spent(1.0, 0.1), {
      assistantId: "asst_1",
      assistant: { model: { provider: "openai", model: "gpt-4o" } },
    });
    expect(resp).toEqual({ assistantId: "asst_1" });
  });

  it("admits with an inline assistant", () => {
    const assistant = { model: { provider: "openai", model: "gpt-4o" } };
    expect(gates.vapi(spent(1.0, 0.1), { assistant })).toEqual({ assistant });
  });

  it("admitting without a target throws", () => {
    expect(() => gates.vapi(spent(1.0, 0.1))).toThrow(RangeError);
  });
});
