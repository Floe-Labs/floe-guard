import { describe, expect, it, vi } from "vitest";

import {
  BudgetExceeded,
  BudgetGuard,
  budgetGuardMiddleware,
  pricing,
} from "../src/index.js";

// The middleware only reads `model.modelId`, so a minimal stub is enough.
function fakeModel(modelId: string) {
  return { modelId } as never;
}

// The middleware never inspects `params`.
const fakeParams = {} as never;

function readableOf(parts: unknown[]): ReadableStream {
  return new ReadableStream({
    start(controller) {
      for (const part of parts) controller.enqueue(part);
      controller.close();
    },
  });
}

async function drain(stream: ReadableStream): Promise<void> {
  const reader = stream.getReader();
  // eslint-disable-next-line no-constant-condition
  while (true) {
    const { done } = await reader.read();
    if (done) break;
  }
}

describe("budgetGuardMiddleware — wrapGenerate", () => {
  it("passes an under-budget call through and records priced spend", async () => {
    const guard = new BudgetGuard(1.0);
    const mw = budgetGuardMiddleware(guard);

    const doGenerate = vi.fn(async () => ({
      usage: { promptTokens: 1000, completionTokens: 1000 },
    }));

    const result = await mw.wrapGenerate!({
      doGenerate: doGenerate as never,
      doStream: vi.fn() as never,
      params: fakeParams,
      model: fakeModel("gpt-4o"),
    });

    expect(doGenerate).toHaveBeenCalledOnce();
    expect((result as { usage: { promptTokens: number } }).usage.promptTokens).toBe(1000);

    const priced = pricing.resolvePrice("gpt-4o")!;
    const expected = pricing.priceTokens(priced, 1000, 1000);
    expect(expected).toBeGreaterThan(0);
    expect(guard.spentUsd).toBeCloseTo(expected, 12);
  });

  it("throws BudgetExceeded BEFORE doGenerate when the next call would cross", async () => {
    const guard = new BudgetGuard(0.0); // zero ceiling blocks the very first call
    const mw = budgetGuardMiddleware(guard);

    const doGenerate = vi.fn(async () => ({
      usage: { promptTokens: 0, completionTokens: 0 },
    }));

    await expect(
      mw.wrapGenerate!({
        doGenerate: doGenerate as never,
        doStream: vi.fn() as never,
        params: fakeParams,
        model: fakeModel("gpt-4o"),
      }),
    ).rejects.toBeInstanceOf(BudgetExceeded);

    expect(doGenerate).not.toHaveBeenCalled();
  });
});

describe("budgetGuardMiddleware — wrapGenerate (ai@5 usage shape)", () => {
  it("records spend from inputTokens/outputTokens", async () => {
    const guard = new BudgetGuard(1.0);
    const mw = budgetGuardMiddleware(guard);

    const doGenerate = vi.fn(async () => ({
      usage: { inputTokens: 1000, outputTokens: 1000, totalTokens: 2000 },
    }));

    await mw.wrapGenerate!({
      doGenerate: doGenerate as never,
      doStream: vi.fn() as never,
      params: fakeParams,
      model: fakeModel("gpt-4o"),
    });

    const priced = pricing.resolvePrice("gpt-4o")!;
    expect(guard.spentUsd).toBeCloseTo(pricing.priceTokens(priced, 1000, 1000), 12);
  });

  it("rejects a result with no usable token counts instead of metering $0", async () => {
    const guard = new BudgetGuard(1.0);
    const mw = budgetGuardMiddleware(guard);

    const doGenerate = vi.fn(async () => ({
      usage: { totalTokens: 2000 }, // ai@5 allows undefined input/output tokens
    }));

    await expect(
      mw.wrapGenerate!({
        doGenerate: doGenerate as never,
        doStream: vi.fn() as never,
        params: fakeParams,
        model: fakeModel("gpt-4o"),
      }),
    ).rejects.toThrow(/no token usage/);

    // The reservation must be released — nothing spent, nothing held in flight.
    expect(guard.spentUsd).toBe(0);
    expect(guard.remainingUsd).toBe(1.0);
  });
});

describe("budgetGuardMiddleware — wrapStream", () => {
  it("records usage from the stream finish part", async () => {
    const guard = new BudgetGuard(1.0);
    const mw = budgetGuardMiddleware(guard);

    const doStream = vi.fn(async () => ({
      stream: readableOf([
        { type: "text-delta", textDelta: "hello" },
        {
          type: "finish",
          finishReason: "stop",
          usage: { promptTokens: 1000, completionTokens: 1000 },
        },
      ]),
      rawCall: { rawPrompt: null, rawSettings: {} },
    }));

    const result = await mw.wrapStream!({
      doGenerate: vi.fn() as never,
      doStream: doStream as never,
      params: fakeParams,
      model: fakeModel("gpt-4o"),
    });

    expect(doStream).toHaveBeenCalledOnce();
    await drain((result as { stream: ReadableStream }).stream);

    const priced = pricing.resolvePrice("gpt-4o")!;
    expect(guard.spentUsd).toBeCloseTo(pricing.priceTokens(priced, 1000, 1000), 12);
  });

  it("throws BudgetExceeded BEFORE doStream when the budget is exhausted", async () => {
    const guard = new BudgetGuard(0.0);
    const mw = budgetGuardMiddleware(guard);

    const doStream = vi.fn();

    await expect(
      mw.wrapStream!({
        doGenerate: vi.fn() as never,
        doStream: doStream as never,
        params: fakeParams,
        model: fakeModel("gpt-4o"),
      }),
    ).rejects.toBeInstanceOf(BudgetExceeded);

    expect(doStream).not.toHaveBeenCalled();
  });
});

describe("budgetGuardMiddleware — wrapStream (ai@5 usage shape)", () => {
  it("records usage from an ai@5 finish part", async () => {
    const guard = new BudgetGuard(1.0);
    const mw = budgetGuardMiddleware(guard);

    const doStream = vi.fn(async () => ({
      stream: readableOf([
        { type: "text-delta", id: "1", delta: "hello" },
        {
          type: "finish",
          finishReason: "stop",
          usage: { inputTokens: 1000, outputTokens: 1000, totalTokens: 2000 },
        },
      ]),
    }));

    const result = await mw.wrapStream!({
      doGenerate: vi.fn() as never,
      doStream: doStream as never,
      params: fakeParams,
      model: fakeModel("gpt-4o"),
    });

    await drain((result as { stream: ReadableStream }).stream);

    const priced = pricing.resolvePrice("gpt-4o")!;
    expect(guard.spentUsd).toBeCloseTo(pricing.priceTokens(priced, 1000, 1000), 12);
  });
});

describe("budgetGuardMiddleware — wrapStream (no finish part / fail-closed)", () => {
  it("rejects when the stream ends with no usage-bearing finish part", async () => {
    // Verify: a stream that produces only text-delta chunks (no finish part) must
    // reject — the guard cannot meter spend it cannot see, so treating the call as
    // $0 / free would violate the package's fail-closed philosophy.
    const guard = new BudgetGuard(1.0);
    const mw = budgetGuardMiddleware(guard);
    const initialRemaining = guard.remainingUsd;

    const doStream = vi.fn(async () => ({
      stream: readableOf([
        { type: "text-delta", textDelta: "hello" },
        { type: "text-delta", textDelta: " world" },
        // No finish part — the stream closes cleanly without token usage.
      ]),
      rawCall: { rawPrompt: null, rawSettings: {} },
    }));

    const result = await mw.wrapStream!({
      doGenerate: vi.fn() as never,
      doStream: doStream as never,
      params: fakeParams,
      model: fakeModel("gpt-4o"),
    });

    // Consuming the stream must reject with the expected error.
    await expect(drain((result as { stream: ReadableStream }).stream)).rejects.toThrow(
      /stream completed without a token-usage finish part/,
    );

    // The call must NOT be metered as $0 / free.
    expect(guard.spentUsd).toBe(0);

    // The reservation must have been released — budget returns to pre-call level.
    expect(guard.remainingUsd).toBeCloseTo(initialRemaining, 12);
  });

  it("the normal finish-with-usage path still settles successfully", async () => {
    // Regression guard: ensure the fail-closed change does not break the happy path.
    const guard = new BudgetGuard(1.0);
    const mw = budgetGuardMiddleware(guard);

    const doStream = vi.fn(async () => ({
      stream: readableOf([
        { type: "text-delta", textDelta: "hello" },
        {
          type: "finish",
          finishReason: "stop",
          usage: { promptTokens: 500, completionTokens: 500 },
        },
      ]),
      rawCall: { rawPrompt: null, rawSettings: {} },
    }));

    const result = await mw.wrapStream!({
      doGenerate: vi.fn() as never,
      doStream: doStream as never,
      params: fakeParams,
      model: fakeModel("gpt-4o"),
    });

    // Must not throw.
    await drain((result as { stream: ReadableStream }).stream);

    const priced = pricing.resolvePrice("gpt-4o")!;
    expect(guard.spentUsd).toBeCloseTo(pricing.priceTokens(priced, 500, 500), 12);
    expect(guard.spentUsd).toBeGreaterThan(0);
  });
});

describe("BudgetExceeded message parity with the Python guard", () => {
  it("formats the message exactly like floe_guard.errors.BudgetExceeded", () => {
    const err = new BudgetExceeded(5.00125, 5.0);
    expect(err.message).toBe(
      "BUDGET EXCEEDED — call blocked (spent $5.001250 of $5.000000 ceiling)",
    );
  });
});

describe("BudgetGuard constructor validation", () => {
  it("rejects a non-finite limit so the guard can't fail open", () => {
    // NaN/Infinity would make check() never trigger — a silently disabled guard.
    expect(() => new BudgetGuard(NaN)).toThrow(RangeError);
    expect(() => new BudgetGuard(Infinity)).toThrow(RangeError);
    expect(() => new BudgetGuard(-1)).toThrow(RangeError);
  });

  it("accepts a finite, non-negative limit (including 0)", () => {
    expect(new BudgetGuard(0).limitUsd).toBe(0);
    expect(new BudgetGuard(5).limitUsd).toBe(5);
  });
});
