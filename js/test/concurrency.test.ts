/**
 * The ceiling must hold when calls run in parallel (regression for issue #18).
 *
 * `check()`/`record()` is a check-then-act with an await in between. Fire several
 * generations at once and they all `check()` against the same under-limit total
 * before any `record()` lands. `reserve()`/`settle()` hold the estimate in flight
 * (synchronously, before the await), so parallel callers each take their slice.
 */

import { describe, expect, it, vi } from "vitest";

import { BudgetExceeded, BudgetGuard, UnpriceableModelError, budgetGuardMiddleware } from "../src/index.js";

function fakeModel(modelId: string) {
  return { modelId } as never;
}
const fakeParams = {} as never;
const tick = () => new Promise((resolve) => setTimeout(resolve, 5));

describe("BudgetGuard — concurrency (issue #18)", () => {
  it("reserve()/settle() holds the ceiling under parallel calls", async () => {
    const guard = new BudgetGuard(0.1, { onBlock: () => {} });
    guard.record("gpt-4o", 1000, 1000); // warm: realistic next-call estimate

    let blocked = 0;
    const agent = async () => {
      let reserved: number;
      try {
        reserved = guard.reserve();
      } catch (err) {
        if (err instanceof BudgetExceeded) {
          blocked++;
          return;
        }
        throw err;
      }
      await tick(); // API latency — the window the old race exploited
      guard.settle("gpt-4o", 1000, 1000, { reserved });
    };

    await Promise.all(Array.from({ length: 16 }, agent));

    expect(guard.spentUsd).toBeLessThanOrEqual(0.1 + 1e-9); // ceiling held
    expect(blocked).toBeGreaterThan(0); // excess was actually stopped
    // no leaked reservation: the in-flight tally is back to zero (remainingUsd is
    // clamped >= 0, so asserting only that would always pass — a tautology).
    expect((guard as unknown as { reserved: number }).reserved).toBeCloseTo(0, 9);
  });

  it("wrapGenerate honours the ceiling across Promise.all", async () => {
    const guard = new BudgetGuard(0.1, { onBlock: () => {} });
    const mw = budgetGuardMiddleware(guard);
    const doGenerate = vi.fn(async () => {
      await tick();
      return { usage: { promptTokens: 1000, completionTokens: 1000 } };
    });
    const launch = () =>
      mw.wrapGenerate!({
        doGenerate: doGenerate as never,
        doStream: vi.fn() as never,
        params: fakeParams,
        model: fakeModel("gpt-4o"),
      });

    await launch(); // warm one call so the estimate is realistic

    const results = await Promise.allSettled(Array.from({ length: 16 }, launch));
    const rejected = results.filter(
      (r) => r.status === "rejected" && r.reason instanceof BudgetExceeded,
    ).length;

    expect(rejected).toBeGreaterThan(0); // parallel fan-out is gated, not raced
    expect(guard.spentUsd).toBeLessThanOrEqual(0.1 + 1e-9); // ceiling held
  });

  it("release() returns in-flight budget when a call fails", () => {
    const guard = new BudgetGuard(0.1, { onBlock: () => {} });
    guard.record("gpt-4o", 1000, 1000);
    const reserved = guard.reserve();
    const before = guard.remainingUsd;
    guard.release(reserved);
    expect(guard.remainingUsd).toBeGreaterThanOrEqual(before);
  });

  it("releases the reservation when settle() hits an unpriceable model (fail-closed)", () => {
    // Regression for the #19 review: fail-closed must not leak the in-flight hold.
    const guard = new BudgetGuard(0.1, { onBlock: () => {} });
    guard.record("gpt-4o", 1000, 1000);
    const base = guard.remainingUsd;
    const reserved = guard.reserve();
    expect(guard.remainingUsd).toBeLessThan(base);
    expect(() => guard.settle("totally-made-up-model-x", 100, 100, { reserved })).toThrow(
      UnpriceableModelError,
    );
    expect(guard.remainingUsd).toBe(base); // released, not leaked
  });
});
