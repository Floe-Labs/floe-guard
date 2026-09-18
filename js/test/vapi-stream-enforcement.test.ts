import { describe, expect, it } from "vitest";
import { BudgetExceeded, BudgetGuard, StreamGuard, UnpriceableModelError, priceTokens, resolvePrice } from "../src/index.js";
import { VapiBudgetGuard, VapiUsageMissingError, type ChatCompletionChunkLike } from "../src/adapters/vapi.js";

const price = { inputCostPerToken: 0.001, outputCostPerToken: 0.001 };
const text = (content: string) => ({ choices: [{ delta: { content } }] });
const makeGuard = (limit = 0.01) => new BudgetGuard(limit, {
  priceOverrides: { m: price }, onBlock: () => {},
});
const adapter = (guard: BudgetGuard) => new VapiBudgetGuard(guard, {
  model: "m",
});
async function drain(stream: AsyncIterable<unknown>): Promise<void> {
  for await (const chunk of stream) void chunk;
}

describe("Vapi mid-stream enforcement", () => {
  it.each([
    { kind: "stream", heldUsd: 0 },
    { kind: "stream", heldUsd: 0.004 },
    { kind: "stream", heldUsd: 0.008 },
    { kind: "hold", heldUsd: 0.008 },
  ])("checks final usage against another $kind with a $heldUsd hold", async ({ kind, heldUsd }) => {
    const guard = makeGuard();
    const held = guard.reserve(heldUsd);
    const other = kind === "stream" ? new StreamGuard(guard, "m", { reserved: held }) : undefined;
    other?.feedTokens(8);
    let closed = false;
    const stream = adapter(guard).guardStream(async function* () {
      try { yield { usage: { prompt_tokens: 0, completion_tokens: 3 } }; }
      finally { closed = true; }
    }, { estimatedCost: 0 });
    await expect(stream.next()).rejects.toBeInstanceOf(BudgetExceeded);
    expect(closed).toBe(true);
    expect(guard.spentUsd).toBeCloseTo(0.003);
    expect(guard.spendLog).toHaveLength(1);
    if (other) other.close();
    else guard.release(held);
    expect(guard.spentUsd).toBeCloseTo(kind === "stream" ? 0.011 : 0.003);
  });

  it("counts prompt cost while awaiting the first provider chunk", async () => {
    const guard = makeGuard();
    let resume!: () => void;
    const pending = new Promise<void>(resolve => { resume = resolve; });
    const stream = adapter(guard).guardStream(async function* () {
      await pending;
      yield { usage: { prompt_tokens: 9, completion_tokens: 0 } };
    }, { estimatedCost: 0, promptTokens: 9 });
    const first = stream.next();
    try {
      expect(guard.remainingUsd).toBeCloseTo(0.001);
      expect(() => guard.reserveTool(0.002)).toThrow(BudgetExceeded);
    } finally { resume(); await first; await stream.return?.(); }
    expect(guard.spentUsd).toBeCloseTo(0.009);
    expect(guard.spendLog).toHaveLength(1);
  });

  it.each(["stream", "settle", "per-call"])("rejects negative cache pricing during %s and releases only its hold", mode => {
    const invalidPrice = { ...price, cacheReadCostPerToken: -0.001 };
    const guard = new BudgetGuard(0.01, {
      priceOverrides: mode === "per-call" ? undefined : { m: invalidPrice },
    });
    const other = guard.reserveTool(0.002);
    let opened = false;
    expect(() => {
      if (mode === "stream") {
        adapter(guard).guardStream(async function* () {
          opened = true;
          yield text("word");
        }, { estimatedCost: 0.003 });
      } else {
        const reserved = guard.reserve(0.003);
        guard.settle("m", 10, 10, {
          reserved, cacheReadInputTokens: 100,
          price: mode === "per-call" ? invalidPrice : undefined,
        });
      }
    }).toThrow(UnpriceableModelError);
    expect(opened).toBe(false);
    expect(guard.spentUsd).toBe(0);
    expect(guard.spendLog).toHaveLength(0);
    expect(guard.remainingUsd).toBeCloseTo(0.008, 12);
    guard.release(other);
  });

  it("applies final usage before the consumer can admit another call", async () => {
    const guard = makeGuard();
    const stream = adapter(guard).guardStream(async function* () {
      yield text("word");
      yield { choices: [], usage: { prompt_tokens: 8, completion_tokens: 1 } };
    });
    await stream.next();
    const final = await stream.next();
    expect(final.value?.usage).toBeDefined();
    try {
      expect(guard.spentUsd).toBeCloseTo(0.009, 12);
      expect(guard.remainingUsd).toBeCloseTo(0.001, 12);
      expect(() => guard.reserveTool(0.002)).toThrow(BudgetExceeded);
    } finally { await stream.return?.(); }
    expect(guard.spendLog).toHaveLength(1);
  });

  it("records an actual overrun before rejecting the final usage chunk", async () => {
    const guard = makeGuard();
    let closed = false;
    const stream = adapter(guard).guardStream(async function* () {
      try {
        yield text("word");
        yield { choices: [], usage: { prompt_tokens: 9, completion_tokens: 2 } };
      } finally { closed = true; }
    });
    await stream.next();
    await expect(stream.next()).rejects.toBeInstanceOf(BudgetExceeded);
    expect(closed).toBe(true);
    expect(guard.spentUsd).toBeCloseTo(0.011, 12);
    expect(guard.spendLog).toHaveLength(1);
    await stream.return?.();
    expect(guard.spendLog).toHaveLength(1);
  });

  it("keeps accrued stream spend visible to another tool call", async () => {
    const guard = makeGuard();
    const stream = adapter(guard).guardStream(async function* () { yield text("x".repeat(36)); });
    await stream.next();
    expect(() => guard.reserveTool(0.002)).toThrow(BudgetExceeded);
    await stream.return?.();
    expect(guard.spentUsd).toBeCloseTo(0.009, 12);
  });

  it("blocks on final reported usage that exceeds the estimate", async () => {
    const guard = makeGuard();
    const stream = adapter(guard).guardStream(async function* () {
      yield text("word");
      yield { usage: { prompt_tokens: 5, completion_tokens: 6 } };
    });
    await expect(drain(stream)).rejects.toBeInstanceOf(BudgetExceeded);
    expect(guard.spentUsd).toBeCloseTo(0.011, 12);
    expect(guard.spendLog).toHaveLength(1);
  });

  it("cleans up throw-before-start once and preserves another reservation", async () => {
    const guard = makeGuard();
    const other = guard.reserve(0.002);
    const stream = adapter(guard).guardStream(async function* () { yield text("word"); }, {
      estimatedCost: 0.004,
    });
    await expect(stream.throw!(new Error("cancel"))).rejects.toThrow("cancel");
    await stream.return?.();
    expect(guard.remainingUsd).toBeCloseTo(0.008, 12);
    guard.release(other);
  });

  it("does not silently activate an initially unpriceable fail-open stream", async () => {
    const overrides: Record<string, typeof price> = {};
    const guard = new BudgetGuard(1, { failClosed: false, priceOverrides: overrides });
    const stream = adapter(guard).guardStream(async function* () {
      overrides.m = price;
      yield text("word");
      yield { usage: { prompt_tokens: 100, completion_tokens: 50 } };
    }, { estimatedCost: 0.1 });
    await drain(stream);
    expect(guard.spentUsd).toBe(0);
    expect(guard.remainingUsd).toBe(1);
  });

  it("settles the crossing chunk, closes upstream, and does not forward it", async () => {
    const guard = makeGuard(0.0025);
    let closed = false;
    let generated = 0;
    async function* source() {
      try {
        for (let i = 0; i < 10; i++) { generated++; yield text("word"); }
      } finally { closed = true; }
    }
    const seen: unknown[] = [];
    await expect((async () => {
      for await (const chunk of adapter(guard).guardStream(source)) seen.push(chunk);
    })()).rejects.toBeInstanceOf(BudgetExceeded);
    expect(seen).toHaveLength(2);
    expect(generated).toBe(3);
    expect(closed).toBe(true);
    expect(guard.spentUsd).toBeCloseTo(0.003, 12);
    expect(guard.spendLog).toHaveLength(1);
  });

  it.each(["break", "error", "missing-usage"])("records partial usage on %s without consuming another hold", async exit => {
    const guard = makeGuard();
    const other = guard.reserveTool(0.002);
    async function* source() {
      yield text("word");
      if (exit === "error") throw new Error("provider failed");
    }
    const stream = adapter(guard).guardStream(source, { estimatedCost: 0.003 });
    if (exit === "break") { for await (const chunk of stream) { void chunk; break; } }
    else if (exit === "error") await expect(drain(stream)).rejects.toThrow("provider failed");
    else await expect(drain(stream)).rejects.toBeInstanceOf(VapiUsageMissingError);
    expect(guard.spentUsd).toBeCloseTo(0.001, 12);
    expect(guard.remainingUsd).toBeCloseTo(0.007, 12);
    expect(guard.spendLog).toHaveLength(1);
    guard.release(other);
  });

  it("releases an iterator returned before its first pull without opening upstream", async () => {
    const guard = makeGuard();
    let opened = false;
    const stream = adapter(guard).guardStream(() => {
      opened = true;
      return (async function* () { yield text("word"); })();
    }, { estimatedCost: 0.004, promptTokens: 9 });
    await stream.return?.();
    await stream.return?.();
    expect(opened).toBe(false);
    expect(guard.remainingUsd).toBe(0.01);
    expect(guard.spendLog).toHaveLength(0);
  });

  it("releases on source startup failure without charging the prompt estimate", async () => {
    const guard = makeGuard();
    const stream = adapter(guard).guardStream(() => { throw new Error("connect failed"); }, {
      estimatedCost: 0.004, promptTokens: 9,
    });
    await expect(drain(stream)).rejects.toThrow("connect failed");
    expect(guard.remainingUsd).toBe(0.01);
    expect(guard.spendLog).toHaveLength(0);
  });

  it("refuses an unknown model eagerly and releases its reservation", () => {
    const guard = makeGuard();
    const other = guard.reserve(0.002);
    expect(() => adapter(guard).guardStream(async function* () {}, {
      model: "unknown-vapi-model", estimatedCost: 0.004,
    })).toThrow(UnpriceableModelError);
    expect(guard.remainingUsd).toBeCloseTo(0.008, 12);
    guard.release(other);
  });

  it("meters function names and argument deltas, including multiple choices", async () => {
    const guard = makeGuard(0.0015);
    async function* source() {
      yield { choices: [
        { delta: { tool_calls: [{ function: { name: "find", arguments: "{}" } }] } },
        { delta: { content: "word" } },
      ] };
    }
    await expect(drain(adapter(guard).guardStream(source))).rejects.toBeInstanceOf(BudgetExceeded);
    expect(guard.spentUsd).toBeCloseTo(0.002, 12);
  });

  it("uses supplied prompt tokens and tokenizer for partial accounting", async () => {
    const guard = makeGuard();
    const stream = adapter(guard).guardStream(async function* () { yield text("x"); }, {
      promptTokens: 2, countTokens: () => 3,
    });
    for await (const chunk of stream) { void chunk; break; }
    expect(guard.spentUsd).toBeCloseTo(0.005, 12);
  });

  it.each([false, true])("reconciles cached final usage once (break after usage: %s)", async earlyBreak => {
    const guard = new BudgetGuard(1);
    const budget = new VapiBudgetGuard(guard, { model: "gpt-4o" });
    const final = { choices: [], usage: {
      prompt_tokens: 10_000, completion_tokens: 20,
      prompt_tokens_details: { cached_tokens: 9_000 },
    } };
    async function* source(): AsyncGenerator<ChatCompletionChunkLike> {
      yield text("hello"); yield final;
    }
    for await (const chunk of budget.guardStream(source)) {
      if (earlyBreak && chunk === final) break;
    }
    expect(guard.spentUsd).toBeCloseTo(priceTokens(resolvePrice("gpt-4o")!, 1_000, 20, {
      cacheReadInputTokens: 9_000,
    }), 12);
    expect(guard.spendLog).toHaveLength(1);
  });
});

it("StreamGuard preserves captured cache pricing when reconciling usage", () => {
  const guard = new BudgetGuard(1);
  const stream = new StreamGuard(guard, "gpt-4o");
  stream.feedText("hello");
  expect(stream.finish({ promptTokens: 1_000, completionTokens: 20, cacheReadInputTokens: 9_000 }))
    .toBeCloseTo(priceTokens(resolvePrice("gpt-4o")!, 1_000, 20, { cacheReadInputTokens: 9_000 }), 12);
});
