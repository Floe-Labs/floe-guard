/**
 * advisory().burnRateUsdPerMin — spend ÷ minutes since the guard was created.
 *
 * The window start (`createdAtMs`) is set directly and `Date.now` is stubbed for
 * the advisory read, so each case asserts the rate from a known spend/elapsed pair.
 *
 * Mirrors `tests/test_burn_rate.py`.
 */

import { afterEach, describe, expect, it, vi } from "vitest";

import { BudgetGuard } from "../src/index.js";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("advisory().burnRateUsdPerMin", () => {
  it("computes from a known spend and elapsed pair", () => {
    const guard = new BudgetGuard(1.0);
    guard.createdAtMs = 100_000;
    guard.recordTool("api", 0.05); // exact $0.05 spend
    vi.spyOn(Date, "now").mockReturnValue(130_000); // +30_000ms = 0.5 min
    expect(guard.advisory().burnRateUsdPerMin).toBeCloseTo(0.1, 9); // 0.05 / 0.5
  });

  it("is 0 when nothing has been spent over real elapsed time", () => {
    const guard = new BudgetGuard(1.0);
    guard.createdAtMs = 100_000;
    vi.spyOn(Date, "now").mockReturnValue(160_000); // +60_000ms = 1 min
    // $0 over real elapsed time is a legitimate 0/min, not "unknown".
    expect(guard.advisory().burnRateUsdPerMin).toBe(0);
  });

  it("is null before any wall-clock time elapses", () => {
    const guard = new BudgetGuard(1.0);
    guard.createdAtMs = 100_000;
    guard.recordTool("api", 0.05);
    vi.spyOn(Date, "now").mockReturnValue(100_000); // no elapsed
    expect(guard.advisory().burnRateUsdPerMin).toBeNull();
  });
});
