import { describe, expect, it } from "vitest";
import { BudgetExceeded, BudgetGuard } from "../src/index.js";

const MODEL = "gpt-4o"; // $2.5e-6/input token, $1e-5/output token

describe("BudgetGuard.estimateCall", () => {
  it("prices the actual request", () => {
    const guard = new BudgetGuard(1.00);
    const est = guard.estimateCall(MODEL, 1_000, 2_000);
    expect(est).toBeCloseTo(1_000 * 2.5e-6 + 2_000 * 1e-5, 12); // 0.0025 + 0.02
  });

  it("prices prompt only when no output cap", () => {
    const guard = new BudgetGuard(1.00);
    const est = guard.estimateCall(MODEL, 1_000);
    expect(est).toBeCloseTo(0.0025, 12);
  });

  it("returns undefined when model is unpriceable", () => {
    const guard = new BudgetGuard(1.00);
    const est = guard.estimateCall("model-that-does-not-exist", 1_000, 1_000);
    expect(est).toBeUndefined();
  });

  it("honors manual price", () => {
    const guard = new BudgetGuard(1.00);
    const price = { inputCostPerToken: 1e-6, outputCostPerToken: 2e-6 };
    const est = guard.estimateCall("my-local-model", 1_000, 500, { price });
    expect(est).toBeCloseTo(1_000 * 1e-6 + 500 * 2e-6, 12);
  });

  it("blocks oversized first call at its true size", () => {
    // THE acceptance case: a FIRST call (no last-cost baseline) that alone would
    // cross the cap must block pre-call once the reservation is request-sized.
    const guard = new BudgetGuard(0.01, { onBlock: () => {} });
    const est = guard.estimateCall(MODEL, 1_000, 100_000); // ≈ $1.0025 ≫ $0.01
    expect(est).toBeDefined();
    expect(est!).toBeGreaterThan(guard.limitUsd);
    expect(() => guard.reserve(est)).toThrow(BudgetExceeded);
    // Nothing was held: the budget is untouched for calls that DO fit.
    expect(guard.remainingUsd).toBeCloseTo(guard.limitUsd, 12);
    // The unsized default would have let the same first call through.
    expect(guard.reserve()).toBe(0.0);
  });

  it("reserves its estimated size for fitting calls", () => {
    const guard = new BudgetGuard(1.00);
    const est = guard.estimateCall(MODEL, 1_000, 1_000);
    const handle = guard.reserve(est);
    expect(handle).toBeCloseTo(0.0125, 12);
    expect(guard.remainingUsd).toBeCloseTo(1.00 - 0.0125, 12);
    guard.release(handle);
  });
});
