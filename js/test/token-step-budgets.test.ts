import { describe, expect, it } from "vitest";

import {
  BudgetExceeded,
  type BudgetAdvisory,
  BudgetGuard,
  type BudgetGuardOptions,
  type StepBudgetGuard,
  TokenBudgetExceeded,
  budgetGuardMiddleware,
  type BudgetReservation,
} from "../src/index.js";

const quiet = (tokenLimit?: number) =>
  new BudgetGuard(10, {
    tokenLimit,
    onBlock: () => {},
    onTokenBlock: () => {},
  });

describe("aggregate token ceilings", () => {
  it("keeps aggregate-only advisory and options literals source-compatible", () => {
    const legacyAdvisory: BudgetAdvisory = {
      nearLimit: false,
      usedBps: 0,
      remainingUsd: 10,
      limitUsd: 10,
      spentUsd: 0,
      scope: "local",
    };
    const options: BudgetGuardOptions = { tokenLimit: 100 };

    expect(legacyAdvisory.nearLimit).toBe(false);
    expect(new BudgetGuard(10, options).tokenLimit).toBe(100);
  });

  it.each([-1, 1.5, NaN, Infinity])("rejects invalid tokenLimit %s", (bad) => {
    expect(() => quiet(bad)).toThrow(RangeError);
  });

  it("preserves numeric handles when new limits are absent", () => {
    const guard = quiet();
    const handle = guard.reserve(0.1);
    expect(typeof handle).toBe("number");
    guard.release(handle);
    expect(guard.remainingTokens).toBeNull();
  });

  it("infers a typed handle for an explicitly token-enabled guard", () => {
    const guard = new BudgetGuard(10, { tokenLimit: 100 });
    const handle: BudgetReservation = guard.reserve(0);
    expect(handle.tokens).toBe(0);
    guard.release(handle);
  });

  it("accumulates tokens and blocks explicit and fallback estimates", () => {
    const guard = quiet(100);
    expect(() => guard.reserve(0, 101)).toThrow(TokenBudgetExceeded);
    guard.record("manual", 40, 10, {
      price: { inputCostPerToken: 1e-6, outputCostPerToken: 2e-6 },
    });
    expect(guard.spentTokens).toBe(50);
    const handle = guard.reserve(0);
    expect(typeof handle).toBe("object");
    expect((handle as unknown as BudgetReservation).tokens).toBe(50);
    expect(() => guard.reserve(0)).toThrow(TokenBudgetExceeded);
    guard.release(handle);
  });

  it("extends nearLimit with aggregate token utilization", () => {
    const guard = quiet(100);
    guard.record("manual", 80, 0, {
      price: { inputCostPerToken: 1e-6, outputCostPerToken: 2e-6 },
    });
    expect(guard.advisory()).toMatchObject({
      tokenUsedBps: 8000,
      nearTokenLimit: true,
      nearLimit: true,
      remainingTokens: 20,
    });
  });

  it("accrues known tokens when unpriceable spend is fail-open", () => {
    const guard = new BudgetGuard(10, {
      tokenLimit: 100,
      failClosed: false,
      onBlock: () => {},
      onTokenBlock: () => {},
    });
    guard.record("unknown-model", 10, 5);
    expect(guard.spentTokens).toBe(15);
    expect(guard.spentUsd).toBe(0);
  });

  it("enforces typed handle ownership and exactly-once disposal", () => {
    const first = quiet(100);
    const second = quiet(100);
    const handle = first.reserve(0.1, 20);
    first.release(handle);
    expect(() => first.release(handle)).toThrow(/already/);
    const foreign = second.reserve(0.1, 20);
    expect(() => first.release(foreign)).toThrow(/different/);
    second.release(foreign);
  });

  it("returns frozen opaque handles with immutable caller-visible values", () => {
    const guard = quiet(100);
    const handle = guard.reserve(0.25, 20) as BudgetReservation;

    expect(handle).toMatchObject({ usd: 0.25, tokens: 20 });
    expect(Object.isFrozen(handle)).toBe(true);
    expect("_owner" in handle).toBe(false);
    expect("_step" in handle).toBe(false);
    expect("_active" in handle).toBe(false);
    expect(Reflect.set(handle as unknown as object, "usd", 9)).toBe(false);
    expect(Reflect.set(handle as unknown as object, "tokens", 99)).toBe(false);
    expect(() =>
      Object.defineProperty(handle, "tokens", { value: Number.NaN }),
    ).toThrow(TypeError);
    expect(handle).toMatchObject({ usd: 0.25, tokens: 20 });

    guard.release(handle);
    expect(guard.remainingUsd).toBe(10);
    expect(guard.remainingTokens).toBe(100);
  });

  it.each([
    { usd: 0.5, tokens: 10, description: "finite oversized" },
    { usd: -1, tokens: 10, description: "negative USD" },
    { usd: Number.NaN, tokens: 10, description: "NaN USD" },
    { usd: 0.1, tokens: -1, description: "negative tokens" },
    { usd: 0.1, tokens: 1.5, description: "non-integer tokens" },
    { usd: 0.1, tokens: Number.NaN, description: "NaN tokens" },
  ])(
    "rejects a forged $description handle without changing accounting",
    ({ usd, tokens }) => {
      const guard = quiet(100);
      const legitimate = guard.reserve(0.2, 20);
      const forged = { usd, tokens } as unknown as BudgetReservation;
      const before = guard as unknown as {
        reserved: number;
        reservedTokens: number;
      };

      expect(() => guard.release(forged)).toThrow(/invalid reservation handle/);
      expect(before.reserved).toBe(0.2);
      expect(before.reservedTokens).toBe(20);
      expect(guard.remainingUsd).toBeCloseTo(9.8);
      expect(guard.remainingTokens).toBe(80);

      guard.release(legitimate);
      expect(before.reserved).toBe(0);
      expect(before.reservedTokens).toBe(0);
    },
  );

  it("rejects copied and cloned handles without consuming the original", () => {
    const guard = quiet(100);
    const handle = guard.reserve(0.2, 20) as BudgetReservation;
    const spreadClone = { ...handle } as unknown as BudgetReservation;
    const assignedClone = Object.assign({}, handle) as unknown as BudgetReservation;
    const sameOwnerForgery = {
      usd: 0.2,
      tokens: 20,
      _owner: guard,
    } as unknown as BudgetReservation;

    expect(() => guard.release(spreadClone)).toThrow(/original object/);
    expect(() => guard.release(assignedClone)).toThrow(/original object/);
    expect(() => guard.release(sameOwnerForgery)).toThrow(/original object/);
    expect(guard.remainingTokens).toBe(80);
    guard.release(handle);
    expect(guard.remainingTokens).toBe(100);
  });

  it("cannot free concurrent holds with a forged larger handle", () => {
    const guard = new BudgetGuard(10, {
      tokenLimit: 400,
      onBlock: () => {},
      onTokenBlock: () => {},
    });
    const first = guard.reserve(0, 100);
    const second = guard.reserve(0, 100);
    const forged = { usd: 0, tokens: 300 } as unknown as BudgetReservation;

    expect(() => guard.release(forged)).toThrow(/invalid reservation handle/);
    expect(guard.remainingTokens).toBe(200);
    expect(() => guard.reserve(0, 300)).toThrow(TokenBudgetExceeded);

    guard.release(first);
    expect(guard.remainingTokens).toBe(300);
    guard.release(second);
    expect(guard.remainingTokens).toBe(400);
  });

  it("allows exactly one concurrent terminal operation per handle", async () => {
    const guard = quiet(100);
    const contested = guard.reserve(0.2, 20);
    const unrelated = guard.reserve(0.3, 30);

    const results = await Promise.allSettled([
      Promise.resolve().then(() => guard.release(contested)),
      Promise.resolve().then(() => guard.release(contested)),
    ]);
    expect(results.filter((result) => result.status === "fulfilled")).toHaveLength(1);
    expect(results.filter((result) => result.status === "rejected")).toHaveLength(1);
    expect(guard.remainingUsd).toBeCloseTo(9.7);
    expect(guard.remainingTokens).toBe(70);

    guard.release(unrelated);
    expect(guard.remainingUsd).toBe(10);
    expect(guard.remainingTokens).toBe(100);
  });

  it("holds tokens synchronously across Promise fan-out", async () => {
    const guard = quiet(100);
    const launches = Array.from({ length: 5 }, async () => {
      const handle = guard.reserve(0, 30);
      await Promise.resolve();
      return handle;
    });
    const results = await Promise.allSettled(launches);
    expect(results.filter((r) => r.status === "fulfilled")).toHaveLength(3);
    expect(results.filter((r) => r.status === "rejected")).toHaveLength(2);
    for (const result of results) {
      if (result.status === "fulfilled") guard.release(result.value);
    }
    expect(guard.remainingTokens).toBe(100);
  });
});

describe("per-step budgets", () => {
  it("resets each step while retaining aggregate totals", async () => {
    const guard = quiet(1_000);
    await guard.step({ maxTokens: 100 }, async (step) => {
      step.record("manual", 40, 10, {
        price: { inputCostPerToken: 1e-6, outputCostPerToken: 2e-6 },
      });
      expect(step.advisory().stepSpentTokens).toBe(50);
    });
    guard.step({ maxTokens: 100 }, (step) => {
      expect(step.advisory().stepSpentTokens).toBe(0);
      step.record("manual", 20, 10, {
        price: { inputCostPerToken: 1e-6, outputCostPerToken: 2e-6 },
      });
    });
    expect(guard.spentTokens).toBe(80);
  });

  it("reports step scope for the tightest token and USD limits", () => {
    const guard = quiet(1_000);
    guard.step({ maxTokens: 50 }, (step) => {
      step.record("manual", 40, 0, {
        price: { inputCostPerToken: 1e-6, outputCostPerToken: 2e-6 },
      });
      try {
        step.check(0, 11);
        throw new Error("expected token block");
      } catch (error) {
        expect(error).toBeInstanceOf(TokenBudgetExceeded);
        expect((error as TokenBudgetExceeded).scope).toBe("step");
      }
    });
    guard.step({ maxUsd: 0.001 }, (step) => {
      try {
        step.check(0.002, 0);
        throw new Error("expected USD block");
      } catch (error) {
        expect(error).toBeInstanceOf(BudgetExceeded);
        expect((error as BudgetExceeded).scope).toBe("step");
      }
    });
  });

  it("counts tools against step USD but not tokens", () => {
    const guard = quiet(100);
    guard.step({ maxUsd: 0.02, maxTokens: 0 }, (step) => {
      const handle = step.reserveTool(0.01);
      step.settleTool("search", 0.01, { reserved: handle });
      expect(step.advisory().stepSpentTokens).toBe(0);
      expect(() => step.reserveTool(0.02)).toThrow(BudgetExceeded);
    });
  });

  it("keeps overlapping async step state isolated", async () => {
    const guard = quiet(1_000);
    const [first, second] = await Promise.all([
      guard.step({ maxTokens: 100 }, async (step) => {
        await Promise.resolve();
        step.record("manual", 30, 0, {
          price: { inputCostPerToken: 1e-6, outputCostPerToken: 2e-6 },
        });
        return step.advisory().stepSpentTokens;
      }),
      guard.step({ maxTokens: 200 }, async (step) => {
        step.record("manual", 70, 0, {
          price: { inputCostPerToken: 1e-6, outputCostPerToken: 2e-6 },
        });
        return step.advisory().stepSpentTokens;
      }),
    ]);
    expect([first, second]).toEqual([30, 70]);
    expect(guard.spentTokens).toBe(100);
  });

  it("keeps a step active until a custom thenable settles", async () => {
    const guard = quiet(1_000);

    const result = await guard.step(
      { maxTokens: 100 },
      (step) => {
        const thenable: PromiseLike<string> = {
          then(onfulfilled, onrejected) {
            const pending = new Promise<string>((resolve) => {
              queueMicrotask(() => {
                step.record("manual", 30, 0, {
                  price: { inputCostPerToken: 1e-6, outputCostPerToken: 2e-6 },
                });
                resolve("done");
              });
            });
            return pending.then(onfulfilled, onrejected);
          },
        };
        return thenable;
      },
    );

    expect(result).toBe("done");
    expect(guard.spentTokens).toBe(30);
  });

  it("closes the step when inspecting a thenable throws", () => {
    const guard = quiet(100);
    let scoped: StepBudgetGuard | undefined;
    const failure = new URIError("then getter failed");

    expect(() =>
      guard.step({ maxTokens: 50 }, (step) => {
        scoped = step;
        return Object.defineProperty({}, "then", {
          get() {
            throw failure;
          },
        }) as PromiseLike<never>;
      }),
    ).toThrow(failure);

    expect(() => scoped!.reserve(0, 1)).toThrow(/no longer active/);
    expect(() =>
      scoped!.record("manual", 1, 0, {
        price: { inputCostPerToken: 1e-6, outputCostPerToken: 2e-6 },
      }),
    ).toThrow(/no longer active/);
    expect(() => scoped!.reserveTool(0.01)).toThrow(/no longer active/);
  });

  it("detects a leaked reservation on clean exit", () => {
    const guard = quiet(100);
    let scoped: StepBudgetGuard | undefined;
    let handle: BudgetReservation | undefined;
    expect(() =>
      guard.step({ maxTokens: 50 }, (step) => {
        scoped = step;
        handle = step.reserve(0, 10) as BudgetReservation;
      }),
    ).toThrow(/active reservation/);
    scoped!.release(handle!);
    expect(guard.remainingTokens).toBe(100);
  });

  it("rejects cloned and sibling-step handles without consuming their holds", () => {
    const guard = quiet(200);

    guard.step({ maxTokens: 100 }, (firstStep) => {
      const handle = firstStep.reserve(0.1, 20) as BudgetReservation;
      const clone = { ...handle } as unknown as BudgetReservation;

      expect(() => firstStep.release(clone)).toThrow(/original object/);
      guard.step({ maxTokens: 100 }, (secondStep) => {
        expect(() => secondStep.release(handle)).toThrow(/different budget step/);
        expect(() => secondStep.release(0.1)).toThrow(/issued by this step/);
        secondStep.release(0);
      });

      expect(guard.remainingTokens).toBe(180);
      firstStep.release(handle);
      expect(guard.remainingTokens).toBe(200);
    });
  });

  it("registers scoped zero handles for record and recordTool", () => {
    const guard = quiet(100);

    guard.step({ maxUsd: 1, maxTokens: 100 }, (step) => {
      step.record("manual", 10, 5, {
        price: { inputCostPerToken: 0.01, outputCostPerToken: 0.02 },
      });
      step.recordTool("search", 0.25);

      expect(step.advisory()).toMatchObject({
        stepSpentUsd: 0.45,
        stepSpentTokens: 15,
      });
    });
    expect(guard.spentUsd).toBeCloseTo(0.45);
    expect(guard.spentTokens).toBe(15);
  });

  it("does not mask a callback failure with leak detection", async () => {
    const guard = quiet(100);
    let handle: BudgetReservation | undefined;
    await expect(
      guard.step({ maxTokens: 50 }, async (step) => {
        handle = step.reserve(0, 10) as BudgetReservation;
        throw new URIError("provider failed");
      }),
    ).rejects.toThrow("provider failed");
    guard.release(handle!);
  });

  it("does not open new reservations after a step closes", () => {
    const guard = quiet(100);
    let scoped: StepBudgetGuard | undefined;

    guard.step({ maxTokens: 50 }, (step) => {
      scoped = step;
    });

    expect(() => scoped!.reserve(0, 1)).toThrow(/no longer active/);
    expect(() => scoped!.reserveTool(0.01)).toThrow(/no longer active/);
  });

  it("extends nearLimit with step utilization", () => {
    const guard = quiet(1_000);
    guard.step({ maxTokens: 100 }, (step) => {
      step.record("manual", 80, 0, {
        price: { inputCostPerToken: 1e-6, outputCostPerToken: 2e-6 },
      });
      expect(step.advisory()).toMatchObject({
        stepUsedBps: 8000,
        stepNearLimit: true,
        nearLimit: true,
      });
    });
  });

  it("blocks scoped middleware before the provider call", async () => {
    const guard = quiet(1_000);
    let called = false;
    await guard.step({ maxTokens: 0 }, async (step) => {
      const middleware = budgetGuardMiddleware(step);
      await expect(
        middleware.wrapGenerate({
          model: { modelId: "gpt-4o" },
          doGenerate: async () => {
            called = true;
            return { usage: { promptTokens: 1, completionTokens: 1 } };
          },
        }),
      ).rejects.toBeInstanceOf(TokenBudgetExceeded);
    });
    expect(called).toBe(false);
  });
});
