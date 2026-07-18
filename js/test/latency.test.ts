/**
 * LatencyBudget — cumulative tool-chain deadline (FLO-624).
 *
 * Mirrors tests/test_latency.py; an injectable fake clock keeps every test
 * sleep-free.
 */

import { describe, it, expect } from "vitest";
import { LatencyBudget, DeadlineExceeded } from "../src/index.js";

function make(slaMs = 5000, options: Record<string, unknown> = {}) {
  let now = 100_000; // arbitrary origin — only deltas matter
  const clock = () => now;
  const advanceMs = (ms: number) => {
    now += ms;
  };
  const budget = new LatencyBudget(slaMs, { clock, ...options });
  return { budget, advanceMs };
}

describe("LatencyBudget", () => {
  it("check passes with headroom and blocks when projected over", () => {
    const { budget, advanceMs } = make(5000);
    advanceMs(3000);
    expect(() => budget.check(1000)).not.toThrow(); // 3000 + 1000 <= 5000

    expect(() => budget.check(2500)).toThrow(DeadlineExceeded); // projected over
    try {
      budget.check(2500);
    } catch (e) {
      expect((e as DeadlineExceeded).slaMs).toBe(5000);
      expect((e as DeadlineExceeded).elapsedMs).toBeCloseTo(3000);
    }
  });

  it("check without an estimate gates on elapsed only", () => {
    const { budget, advanceMs } = make(1000);
    advanceMs(999);
    expect(() => budget.check()).not.toThrow();
    advanceMs(2);
    expect(() => budget.check()).toThrow(DeadlineExceeded);
  });

  it("remainingMs is readable mid-chain and floors at zero", () => {
    const { budget, advanceMs } = make(5000);
    advanceMs(1500);
    expect(budget.remainingMs).toBeCloseTo(3500);
    advanceMs(9000);
    expect(budget.remainingMs).toBe(0);
  });

  it("advisory is symmetric to BudgetGuard's (nearDeadline at the 80% default)", () => {
    const { budget, advanceMs } = make(5000);
    advanceMs(2500);
    const mid = budget.advisory();
    expect(mid.usedBps).toBe(5000);
    expect(mid.nearDeadline).toBe(false);
    expect(mid.remainingMs).toBeCloseTo(2500);

    advanceMs(1600); // 4100/5000 = 82%
    const late = budget.advisory();
    expect(late.usedBps).toBe(8200);
    expect(late.nearDeadline).toBe(true);

    advanceMs(9000);
    expect(budget.advisory().usedBps).toBe(10000); // capped
  });

  it("onBlock fires before the throw", () => {
    const calls: Array<[number, number]> = [];
    const { budget, advanceMs } = make(1000, { onBlock: (e: number, s: number) => calls.push([e, s]) });
    advanceMs(1500);
    expect(() => budget.check()).toThrow(DeadlineExceeded);
    expect(calls).toHaveLength(1);
    expect(calls[0]![1]).toBe(1000);
  });

  it("validates constructor and check inputs", () => {
    expect(() => new LatencyBudget(0)).toThrow(RangeError);
    expect(() => new LatencyBudget(5000, { nearDeadlineBps: 20000 })).toThrow(RangeError);
    const { budget } = make();
    expect(() => budget.check(-1)).toThrow(RangeError);
  });

  it("rejects non-finite inputs (inf would disable the deadline; NaN slips comparisons)", () => {
    expect(() => new LatencyBudget(Infinity)).toThrow(RangeError);
    expect(() => new LatencyBudget(NaN)).toThrow(RangeError);
    const { budget } = make();
    expect(() => budget.check(NaN)).toThrow(RangeError);
    expect(() => budget.check(Infinity)).toThrow(RangeError);
  });

  it("deadline message uses the shared half-up rounding (byte-parity with Python)", () => {
    expect(new DeadlineExceeded(0.5, 5000.5).message).toBe(
      "DEADLINE EXCEEDED — call blocked (elapsed 1ms of 5001ms SLA)",
    );
    expect(new DeadlineExceeded(-0.5, 1000).message).toBe(
      "DEADLINE EXCEEDED — call blocked (elapsed 0ms of 1000ms SLA)",
    );
  });
});
