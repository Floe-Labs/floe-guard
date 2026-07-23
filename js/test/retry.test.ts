import { describe, expect, it } from "vitest";

import { BudgetExceeded, BudgetGuard, type RetryPlan, withBudgetRetry } from "../src/index.js";

class RetryableError extends Error {}

describe("withBudgetRetry", () => {
  it("retries the same call when budget is ample", async () => {
    const guard = new BudgetGuard(1.0);
    let primaryCalls = 0;

    const result = await withBudgetRetry(
      guard,
      () => {
        primaryCalls += 1;
        if (primaryCalls === 1) throw new RetryableError("temporary failure");
        return "primary-ok";
      },
      { estimatedCost: 0.05, maxAttempts: 2 },
    );

    expect(result).toBe("primary-ok");
    expect(primaryCalls).toBe(2);
  });

  it("uses a degraded retry plan when near the limit", async () => {
    const guard = new BudgetGuard(1.0, { nearLimitBps: 8000 });
    guard.recordTool("seed", 0.85);
    let primaryCalls = 0;
    let cheapCalls = 0;

    const result = await withBudgetRetry(
      guard,
      () => {
        primaryCalls += 1;
        throw new RetryableError("temporary failure");
      },
      {
        estimatedCost: 0.2,
        maxAttempts: 2,
        onDegrade: (error): RetryPlan<string> => {
          expect(error).toBeInstanceOf(RetryableError);
          return {
            estimatedCost: 0.01,
            call: () => {
              cheapCalls += 1;
              return "cheap-ok";
            },
          };
        },
      },
    );

    expect(result).toBe("cheap-ok");
    expect(primaryCalls).toBe(1);
    expect(cheapCalls).toBe(1);
  });

  it("aborts before a retry whose estimate would cross the budget", async () => {
    const guard = new BudgetGuard(1.0, { onBlock: () => undefined });
    guard.recordTool("seed", 0.95);
    let primaryCalls = 0;

    await expect(
      withBudgetRetry(
        guard,
        () => {
          primaryCalls += 1;
          throw new RetryableError("temporary failure");
        },
        { estimatedCost: 0.1, maxAttempts: 2 },
      ),
    ).rejects.toBeInstanceOf(BudgetExceeded);
    expect(primaryCalls).toBe(1);
  });

  it("does not retry non-retryable failures", async () => {
    const guard = new BudgetGuard(1.0);
    let primaryCalls = 0;

    await expect(
      withBudgetRetry(
        guard,
        () => {
          primaryCalls += 1;
          throw new TypeError("bad request");
        },
        {
          estimatedCost: 0.01,
          retryIf: (error) => !(error instanceof TypeError),
        },
      ),
    ).rejects.toThrow("bad request");
    expect(primaryCalls).toBe(1);
  });

  it("rejects invalid maxAttempts", async () => {
    await expect(
      withBudgetRetry(new BudgetGuard(1.0), () => "ok", { maxAttempts: 0 }),
    ).rejects.toBeInstanceOf(RangeError);
  });
});
