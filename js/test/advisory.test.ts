import { describe, expect, it } from "vitest";

import { BudgetGuard } from "../src/index.js";

describe("BudgetGuard.advisory", () => {
  it("a fresh guard is far from the limit", () => {
    const a = new BudgetGuard(1.0).advisory();
    expect(a.nearLimit).toBe(false);
    expect(a.usedBps).toBe(0);
    expect(a.remainingUsd).toBe(1.0);
    expect(a.scope).toBe("local");
  });

  it("nearLimit flips at the default 80% threshold", () => {
    const g = new BudgetGuard(1.0);
    g.spentUsd = 0.79;
    expect(g.advisory().nearLimit).toBe(false);
    g.spentUsd = 0.8;
    const a = g.advisory();
    expect(a.nearLimit).toBe(true);
    expect(a.usedBps).toBe(8000);
    expect(a.remainingUsd).toBeCloseTo(0.2, 9);
  });

  it("honors a custom nearLimitBps", () => {
    const g = new BudgetGuard(1.0, { nearLimitBps: 5000 });
    g.spentUsd = 0.5;
    expect(g.advisory().nearLimit).toBe(true);
  });

  it("clamps usedBps when over the limit and never reports negative remaining", () => {
    const g = new BudgetGuard(1.0);
    g.spentUsd = 1.5;
    const a = g.advisory();
    expect(a.usedBps).toBe(10000);
    expect(a.remainingUsd).toBe(0);
  });

  it("a zero limit reads as fully used", () => {
    const a = new BudgetGuard(0).advisory();
    expect(a.usedBps).toBe(10000);
    expect(a.nearLimit).toBe(true);
  });

  it("floors usedBps rather than rounding (no early nearLimit, Python parity)", () => {
    const g = new BudgetGuard(1.0, { nearLimitBps: 8000 });
    g.spentUsd = 0.79999; // 79.999%
    const a = g.advisory();
    expect(a.usedBps).toBe(7999); // floored, not rounded up to 8000
    expect(a.nearLimit).toBe(false); // 80% not actually reached yet
  });

  it("expectedCost is 0 and estCallsRemaining is null before any call", () => {
    // No call recorded: no estimate yet, so calls-remaining is unknown (null),
    // never a divide-by-zero or a misleading 0.
    const a = new BudgetGuard(1.0).advisory();
    expect(a.expectedCost).toBe(0);
    expect(a.estCallsRemaining).toBeNull();
  });

  it("estCallsRemaining is floor(remaining / expectedCost) after a call", () => {
    const g = new BudgetGuard(1.0);
    g.recordTool("apollo.people_lookup", 0.1); // spent 0.10, remaining 0.90
    const a = g.advisory();
    expect(a.expectedCost).toBeCloseTo(0.1, 9);
    expect(a.estCallsRemaining).toBe(9); // floor(0.90 / 0.10)
  });

  it("expectedCost is the costlier of the last LLM and tool call", () => {
    // Parity with the Python suite: a cheap tool after an expensive LLM call
    // must not shrink the estimate — the max wins (conservative).
    const g = new BudgetGuard(1.0);
    g.record("gpt-4o", 1_000, 1_000); // $0.0125 LLM
    g.recordTool("exa.search", 0.001); // cheaper tool, spent = $0.0135
    const a = g.advisory();
    expect(a.expectedCost).toBeCloseTo(0.0125, 9); // LLM side, not the cheaper tool
    // floor((1.0 - 0.0135) / 0.0125) = floor(78.92) = 78
    expect(a.estCallsRemaining).toBe(78);
  });

  it("rejects an out-of-range or non-integer nearLimitBps", () => {
    expect(() => new BudgetGuard(1.0, { nearLimitBps: -1 })).toThrow(RangeError);
    expect(() => new BudgetGuard(1.0, { nearLimitBps: 10001 })).toThrow(RangeError);
    expect(() => new BudgetGuard(1.0, { nearLimitBps: 8000.5 })).toThrow(RangeError);
  });

  it("rejects an explicit null nearLimitBps instead of silently defaulting", () => {
    // undefined → default 8000; null is an explicit bad value → reject (matches
    // Python rejecting None).
    expect(new BudgetGuard(1.0, { nearLimitBps: undefined }).nearLimitBps).toBe(8000);
    expect(
      () => new BudgetGuard(1.0, { nearLimitBps: null as unknown as number }),
    ).toThrow(RangeError);
  });
});
