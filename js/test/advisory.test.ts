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

  it("rejects an out-of-range nearLimitBps", () => {
    expect(() => new BudgetGuard(1.0, { nearLimitBps: -1 })).toThrow(RangeError);
    expect(() => new BudgetGuard(1.0, { nearLimitBps: 10001 })).toThrow(RangeError);
  });
});
