import { describe, expect, it } from "vitest";

import {
  BudgetExceeded,
  BudgetGuard,
  type RetryPlan,
  withBudgetRetry,
} from "../src/index.js";
import {
  DeadlineExceeded,
  FloeGuardError,
  UnpriceableModelError,
  UnpriceableVoiceError,
} from "../src/errors.js";

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

// ── FloeGuardError family is non-retryable ────────────────────────────────────

describe("FloeGuardError family is non-retryable", () => {
  it("does not retry UnpriceableModelError (call count == 1)", async () => {
    const guard = new BudgetGuard(1.0);
    let calls = 0;
    await expect(
      withBudgetRetry(
        guard,
        () => {
          calls += 1;
          throw new UnpriceableModelError("mystery-model");
        },
        { estimatedCost: 0.01, maxAttempts: 3 },
      ),
    ).rejects.toBeInstanceOf(UnpriceableModelError);
    expect(calls).toBe(1);
  });

  it("still retries ordinary (non-FloeGuard) errors", async () => {
    const guard = new BudgetGuard(1.0);
    let calls = 0;
    const result = await withBudgetRetry(
      guard,
      () => {
        calls += 1;
        if (calls < 3) throw new Error("transient");
        return "ok";
      },
      { estimatedCost: 0.01, maxAttempts: 3 },
    );
    expect(result).toBe("ok");
    expect(calls).toBe(3);
  });

  it("does not retry BudgetExceeded (FloeGuardError subclass)", async () => {
    const guard = new BudgetGuard(1.0, { onBlock: () => undefined });
    guard.recordTool("seed", 0.99);
    let calls = 0;
    await expect(
      withBudgetRetry(
        guard,
        () => {
          calls += 1;
          throw new RetryableError("fail");
        },
        { estimatedCost: 0.02, maxAttempts: 3 },
      ),
    ).rejects.toBeInstanceOf(BudgetExceeded);
    expect(calls).toBe(1);
  });

  it.each([
    ["UnpriceableVoiceError", new UnpriceableVoiceError("elevenlabs", "tts")],
    ["DeadlineExceeded", new DeadlineExceeded(500, 300)],
  ] as [string, FloeGuardError][])(
    "does not retry %s",
    async (_name, exc) => {
      const guard = new BudgetGuard(1.0);
      let calls = 0;
      await expect(
        withBudgetRetry(
          guard,
          () => {
            calls += 1;
            throw exc;
          },
          { estimatedCost: 0.01, maxAttempts: 3 },
        ),
      ).rejects.toBeInstanceOf(FloeGuardError);
      expect(calls).toBe(1);
    },
  );
});
