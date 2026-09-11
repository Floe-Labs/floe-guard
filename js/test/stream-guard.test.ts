import { describe, expect, it, vi } from "vitest";
import { BudgetExceeded, BudgetGuard, StreamGuard, UnpriceableModelError, guardStream, approxTokens } from "../src/index.js";

const MODEL = "stream-test";
const price = { inputCostPerToken: 0.0000025, outputCostPerToken: 0.00001 };
const makeGuard = (limit = 0.01) => new BudgetGuard(limit, {
  priceOverrides: { [MODEL]: price }, onBlock: () => {},
});

describe("StreamGuard", () => {
  it("captures price, tokenizer and label when constructed", () => {
    const guard = makeGuard(1);
    const options = {
      price: { inputCostPerToken: 0, outputCostPerToken: 0.01 },
      countTokens: () => 10,
      label: "original",
    };
    const stream = new StreamGuard(guard, "custom-model", options);
    options.price = { inputCostPerToken: 0, outputCostPerToken: 0 };
    options.countTokens = () => 0;
    options.label = "changed";
    stream.feedText("hello");
    expect(stream.finish()).toBeCloseTo(0.1, 12);
    expect(guard.spendLog[0]).toMatchObject({ completionTokens: 10, label: "original" });
  });

  it("copies manual price fields so mutation cannot change settlement", () => {
    const guard = makeGuard(1);
    const manualPrice = { inputCostPerToken: 0, outputCostPerToken: 0.01 };
    const stream = new StreamGuard(guard, "custom-model", { price: manualPrice });
    manualPrice.outputCostPerToken = 0;
    stream.feedTokens(10);
    expect(stream.finish()).toBeCloseTo(0.1, 12);
    expect(guard.spendLog[0].costUsd).toBeCloseTo(0.1, 12);
  });

  it("uses captured configuration to settle the crossing chunk before interruption", () => {
    const guard = makeGuard(0.05);
    const options = {
      price: { inputCostPerToken: 0, outputCostPerToken: 0.01 },
      countTokens: () => 10,
    };
    const stream = new StreamGuard(guard, "custom-model", options);
    options.price = { inputCostPerToken: 0, outputCostPerToken: 0 };
    options.countTokens = () => 0;
    expect(() => stream.feedText("hello")).toThrow(BudgetExceeded);
    expect(guard.spentUsd).toBeCloseTo(0.1, 12);
    expect(guard.spendLog).toHaveLength(1);
  });
  it("never calls fetch unless hosted sync was explicitly enabled", () => {
    const fetch = vi.fn(() => { throw new Error("unexpected network request"); });
    vi.stubGlobal("fetch", fetch);
    try {
      const guard = makeGuard(1);
      expect([...guardStream(guard, MODEL, ["hello"])]).toEqual(["hello"]);
      expect(() => [...guardStream(makeGuard(0), MODEL, ["hello"])]).toThrow(BudgetExceeded);
      expect(fetch).not.toHaveBeenCalled();
    } finally { vi.unstubAllGlobals(); }
  });

  it("releases token holds on fail-closed construction", () => {
    const guard = new BudgetGuard(1, { tokenLimit: 10 });
    const reserved = guard.reserve(0.1, { estimatedTokens: 10 });
    const warning = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      expect(() => new StreamGuard(guard, "unknown-model", { reserved })).toThrow(UnpriceableModelError);
      const replacement = guard.reserve(0.1, { estimatedTokens: 10 });
      guard.release(replacement);
      expect(guard.remainingUsd).toBe(1);
    } finally { warning.mockRestore(); }
  });

  it("counts parallel streams' overages once and preserves a surviving reservation", () => {
    const guard = makeGuard();
    const a = new StreamGuard(guard, MODEL, { reserved: guard.reserve(0.002) });
    const b = new StreamGuard(guard, MODEL, { reserved: guard.reserve(0.004) });
    a.feedTokens(500); // $0.005 accrued, including $0.003 beyond its hold
    b.feedTokens(500); // joint $0.010 fits, without double-counting either hold
    expect(() => a.feedTokens(10)).toThrow(BudgetExceeded);
    expect(guard.remainingUsd).toBeCloseTo(0.0009, 12); // b still holds $0.004
    b.close();
    expect(guard.spentUsd).toBeCloseTo(0.0101, 12);
    expect(guard.spendLog).toHaveLength(2);
  });

  it("notifies onBlock only after partial spending is recorded and the hold is released", () => {
    const notices: number[] = [];
    const guard = new BudgetGuard(0.0001, {
      priceOverrides: { [MODEL]: price },
      onBlock: spent => {
        expect(guard.spendLog).toHaveLength(1);
        expect(guard.remainingUsd).toBe(0);
        notices.push(spent);
      },
    });
    const stream = new StreamGuard(guard, MODEL, { reserved: guard.reserve(0.00005) });
    expect(() => stream.feedTokens(20)).toThrow(BudgetExceeded);
    expect(notices).toEqual([0.0002]);
  });

  it("releases token holds so future reservations can reuse the token capacity", () => {
    const guard = new BudgetGuard(1, { tokenLimit: 10, priceOverrides: { [MODEL]: price } });
    const stream = new StreamGuard(guard, MODEL, {
      reserved: guard.reserve(0.01, { estimatedTokens: 8 }),
    });
    stream.feedTokens(2);
    stream.close();
    const next = guard.reserve(0.01, { estimatedTokens: 8 });
    guard.release(next);
    expect(guard.spentTokens).toBe(2);
    expect(guard.remainingUsd).toBeCloseTo(0.99998, 12);
  });

  it("meters synchronous chunks before yielding the crossing chunk", () => {
    const guard = makeGuard(0.0001);
    const stream = guardStream(guard, MODEL, ["a".repeat(40), "b".repeat(40)]);
    expect(stream.next().value).toBe("a".repeat(40));
    expect(() => stream.next()).toThrow(BudgetExceeded);
    expect(guard.spendLog[0].completionTokens).toBe(20);
  });

  it("settles earlier chunks if the tokenizer throws", () => {
    const guard = makeGuard(1);
    const stream = guardStream(guard, MODEL, ["first", "bad"], {
      reserved: guard.reserve(0.1),
      countTokens: text => { if (text === "bad") throw new Error("tokenizer failed"); return 10; },
    });
    expect(() => [...stream]).toThrow("tokenizer failed");
    expect(guard.remainingUsd).toBeCloseTo(0.9999, 12);
  });

  it("still settles if source cleanup throws", async () => {
    const guard = makeGuard(1);
    async function* source() { try { yield "hello"; } finally { throw new Error("cleanup failed"); } }
    await expect((async () => {
      for await (const _chunk of guardStream(guard, MODEL, source(), { reserved: guard.reserve(0.1) })) break;
    })()).rejects.toThrow("cleanup failed");
    expect(guard.remainingUsd).toBeCloseTo(0.99999, 12);
  });
  it("rejects an extractor that returns a non-string instead of silently recording zero", () => {
    const guard = makeGuard(1);
    const chunks = [{ text: "hello" }];
    // @ts-expect-error JavaScript callers are not protected by static types.
    const stream = guardStream(guard, MODEL, chunks, {
      reserved: guard.reserve(0.1),
      getText: () => undefined,
    });
    expect(() => stream.next()).toThrow(TypeError);
    expect(guard.remainingUsd).toBe(1);
  });
  it.each([["", 0], ["abc", 1], ["abcdefgh", 2], ["😀😀😀😀😀😀😀😀", 2]] as const)(
    "matches Python's heuristic for %s", (text, expected) => expect(approxTokens(text)).toBe(expected),
  );

  it("uses a custom tokenizer and manual price", () => {
    const guard = makeGuard(1);
    const stream = new StreamGuard(guard, "custom-model", { price, countTokens: () => 7 });
    stream.feedText("x");
    expect(stream.finish()).toBeCloseTo(0.00007, 12);
  });

  it("settles its own reservation on a budget abort without clearing another hold", () => {
    const guard = makeGuard();
    const other = guard.reserve(0.004);
    const stream = new StreamGuard(guard, MODEL, { reserved: guard.reserve(0.003) });
    stream.feedTokens(600);
    expect(() => stream.feedTokens(10)).toThrow(BudgetExceeded);
    guard.release(other);
    expect(guard.remainingUsd).toBeCloseTo(0.0039, 12);
    stream.close();
    expect(guard.spendLog).toHaveLength(1);
  });

  it("settles partial spend on source failure", () => {
    const guard = makeGuard(1);
    function* source() { yield "a".repeat(40); throw new Error("network died"); }
    expect(() => [...guardStream(guard, MODEL, source(), { reserved: guard.reserve(0.1) })]).toThrow("network died");
    expect(guard.spentUsd).toBeCloseTo(0.0001, 12);
    expect(guard.remainingUsd).toBeCloseTo(0.9999, 12);
  });

  it("settles on normal exhaustion including prompt tokens", () => {
    const guard = makeGuard(1);
    const chunks = ["a".repeat(40), "b".repeat(40)];
    expect([...guardStream(guard, MODEL, chunks, { promptTokens: 50 })]).toEqual(chunks);
    expect(guard.spentUsd).toBeCloseTo(0.000325, 12);
    expect(guard.spendLog[0]).toMatchObject({ promptTokens: 50, completionTokens: 20 });
  });

  it("fails closed eagerly without consuming the source", () => {
    const guard = makeGuard(1);
    let started = false;
    function* source() { started = true; yield "x"; }
    const warning = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      expect(() => guardStream(guard, "unknown-model", source(), { reserved: guard.reserve(0.1) })).toThrow(UnpriceableModelError);
      expect(started).toBe(false);
      expect(guard.remainingUsd).toBe(1);
    } finally { warning.mockRestore(); }
  });

  it("refuses structured chunks without an extractor and frees the hold", () => {
    const guard = makeGuard(1);
    const stream = guardStream(guard, MODEL, [{ delta: "hi" }], { reserved: guard.reserve(0.1) });
    expect(() => stream.next()).toThrow(/getText/);
    expect(guard.remainingUsd).toBe(1);
  });

  it("extracts structured chunks and charges only the generated text", () => {
    const guard = makeGuard(1);
    const chunks = [{ delta: "hello" }];
    expect([...guardStream(guard, MODEL, chunks, { getText: c => c.delta })]).toEqual(chunks);
    expect(guard.spendLog[0].completionTokens).toBe(1);
  });

  it.each([NaN, Infinity, -1])("rejects invalid reservation %s", reserved => {
    expect(() => new StreamGuard(makeGuard(), MODEL, { reserved })).toThrow(RangeError);
  });

  it.each([NaN, Infinity])("rejects nonfinite token counts %s without corrupting previous accrual", count => {
    const guard = makeGuard(1);
    const stream = new StreamGuard(guard, MODEL);
    stream.feedTokens(10);
    expect(() => stream.feedTokens(count)).toThrow(RangeError);
    stream.close();
    expect(guard.spentUsd).toBeCloseTo(0.0001, 12);
  });

  it("direct use settles in finally after a consumer error", () => {
    const guard = makeGuard(1);
    const stream = new StreamGuard(guard, MODEL, { reserved: guard.reserve(0.1) });
    expect(() => {
      try { stream.feedTokens(10); throw new Error("consumer failed"); }
      finally { stream.close(); }
    }).toThrow("consumer failed");
    expect(guard.remainingUsd).toBeCloseTo(0.9999, 12);
  });

  it("leaves a never-started iterator's reservation with the caller", () => {
    const guard = makeGuard(1);
    const reserved = guard.reserve(0.1);
    const unused = guardStream(guard, MODEL, ["x"], { reserved });
    unused.return?.();
    expect(guard.remainingUsd).toBeCloseTo(0.9);
    expect(guard.spendLog).toHaveLength(0);
    guard.release(reserved);
    expect(guard.remainingUsd).toBe(1);
  });

  it("closes async sources and settles after an early consumer break", async () => {
    const guard = makeGuard(1);
    let closed = false;
    async function* source() { try { yield "hello"; yield "world"; } finally { closed = true; } }
    for await (const _chunk of guardStream(guard, MODEL, source(), { reserved: guard.reserve(0.1) })) break;
    expect(closed).toBe(true);
    expect(guard.remainingUsd).toBeCloseTo(0.99999, 12);
  });
  it("meters asynchronous chunks and closes their source on budget interruption", async () => {
    const guard = makeGuard(0.0001);
    let sourceClosed = false;
    async function* source() {
      try { yield "a".repeat(40); yield "b".repeat(40); yield "unreachable"; }
      finally { sourceClosed = true; }
    }
    const seen: string[] = [];
    await expect((async () => {
      for await (const chunk of guardStream(guard, MODEL, source())) seen.push(chunk);
    })()).rejects.toThrow(BudgetExceeded);
    expect(seen).toEqual(["a".repeat(40)]);
    expect(sourceClosed).toBe(true);
    expect(guard.spentUsd).toBeCloseTo(0.0002, 12);
    expect(guard.spendLog).toHaveLength(1);
  });
  it("settles partial usage and closes the source on an early consumer break", () => {
    const guard = makeGuard(1);
    let sourceClosed = false;
    function* source() {
      try { yield "a".repeat(40); yield "b".repeat(40); }
      finally { sourceClosed = true; }
    }
    for (const chunk of guardStream(guard, MODEL, source(), { reserved: guard.reserve(0.1) })) {
      expect(chunk).toHaveLength(40);
      break;
    }
    expect(sourceClosed).toBe(true);
    expect(guard.spentUsd).toBeCloseTo(0.0001, 12);
    expect(guard.remainingUsd).toBeCloseTo(0.9999, 12);
    expect(guard.spendLog).toHaveLength(1);
  });
  it.each([true, false])("handles an unknown model with failClosed=%s without leaking a hold", (failClosed) => {
    const guard = new BudgetGuard(1, { failClosed });
    const reserved = guard.reserve(0.1);
    const warning = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      if (failClosed) {
        expect(() => new StreamGuard(guard, "unknown-model", { reserved })).toThrow(UnpriceableModelError);
      } else {
        const stream = new StreamGuard(guard, "unknown-model", { reserved });
        stream.feedTokens(10000);
        expect(stream.finish()).toBe(0);
      }
      expect(guard.remainingUsd).toBe(1);
      expect(guard.spendLog).toHaveLength(0);
      expect(warning).toHaveBeenCalled();
    } finally { warning.mockRestore(); }
  });
  it("reconciles provider usage and releases USD and token reservations", () => {
    const guard = makeGuard(1);
    const reserved = guard.reserve(0.01, { estimatedTokens: 200 });
    const stream = new StreamGuard(guard, MODEL, { reserved, promptTokens: 100, label: "writer" });
    stream.feedText("a".repeat(40));
    expect(stream.completionTokens).toBe(10);
    expect(stream.finish({ promptTokens: 120, completionTokens: 37 })).toBeCloseTo(0.00067, 12);
    expect(guard.remainingUsd).toBeCloseTo(0.99933, 12);
    expect(guard.spendLog[0]).toMatchObject({ promptTokens: 120, completionTokens: 37, label: "writer" });
    expect(() => stream.feedTokens(1)).toThrow(/settled/);
    expect(() => stream.finish()).toThrow(/settled/);
  });
  it("shares the ceiling between parallel unreserved streams", () => {
    const guard = makeGuard();
    const a = new StreamGuard(guard, MODEL);
    const b = new StreamGuard(guard, MODEL);
    for (let i = 0; i < 50; i++) { a.feedTokens(10); b.feedTokens(10); }
    expect(() => a.feedTokens(10)).toThrow(BudgetExceeded);
    expect(() => b.feedTokens(10)).toThrow(BudgetExceeded);
    expect(guard.spentUsd).toBeCloseTo(0.0102, 12);
    expect(guard.spendLog).toHaveLength(2);
  });
  it("interrupts a runaway response and records the crossing chunk before throwing", () => {
    const guard = makeGuard();
    const stream = new StreamGuard(guard, MODEL);
    for (let i = 0; i < 100; i++) stream.feedTokens(10);
    expect(() => stream.feedTokens(10)).toThrow(BudgetExceeded);
    expect(guard.spentUsd).toBeCloseTo(0.0101, 12);
    expect(guard.spendLog).toHaveLength(1);
    expect(guard.spendLog[0].completionTokens).toBe(1010);
    expect(() => guard.check()).toThrow(BudgetExceeded);
  });
});
