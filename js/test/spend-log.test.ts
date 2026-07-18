import { describe, expect, it, vi } from "vitest";

import { BudgetExceeded, BudgetGuard } from "../src/index.js";

const MODEL = "gpt-4o"; // 1k in + 1k out = $0.0025 + $0.01 = $0.0125/call

describe("BudgetGuard.spendLog", () => {
  it("every priced call appends one event and the ledger sums to spentUsd", () => {
    const guard = new BudgetGuard(1.0);
    const n = 7;
    for (let i = 0; i < n; i++) {
      guard.record(MODEL, 1_000, 1_000);
    }
    const log = guard.spendLog;
    expect(log).toHaveLength(n);
    const total = log.reduce((sum, e) => sum + e.costUsd, 0);
    expect(total).toBeCloseTo(guard.spentUsd, 12);
  });

  it("llm event schema", () => {
    const guard = new BudgetGuard(1.0);
    const cost = guard.record(MODEL, 1_200, 350, { label: "researcher" });
    const [event] = guard.spendLog;
    expect(event.kind).toBe("llm");
    expect(event.modelOrTool).toBe(MODEL);
    expect(event.promptTokens).toBe(1_200);
    expect(event.completionTokens).toBe(350);
    expect(event.costUsd).toBeCloseTo(cost, 12);
    expect(event.label).toBe("researcher");
    expect(event.reserved).toBeUndefined(); // plain record(): no reservation to log
    expect(event.timestamp).toBeGreaterThan(0);
  });

  it("settle logs the reservation it settled", () => {
    const guard = new BudgetGuard(1.0);
    const handle = guard.reserve(0.05);
    guard.settle(MODEL, 1_000, 1_000, { reserved: handle });
    const [event] = guard.spendLog;
    expect(event.reserved).toBeCloseTo(0.05, 12);
  });

  it("recordTool accrues and logs a tool event", () => {
    const guard = new BudgetGuard(1.0);
    guard.record(MODEL, 1_000, 1_000);
    const returned = guard.recordTool("serpapi.search", 0.01, { label: "researcher" });
    expect(returned).toBeCloseTo(0.01, 12);
    expect(guard.spentUsd).toBeCloseTo(0.0125 + 0.01, 12);
    const toolEvent = guard.spendLog.at(-1)!;
    expect(toolEvent.kind).toBe("tool");
    expect(toolEvent.modelOrTool).toBe("serpapi.search");
    expect(toolEvent.promptTokens).toBeNull();
    expect(toolEvent.completionTokens).toBeNull();
    expect(toolEvent.costUsd).toBeCloseTo(0.01, 12);
    expect(toolEvent.label).toBe("researcher");
    const total = guard.spendLog.reduce((sum, e) => sum + e.costUsd, 0);
    expect(total).toBeCloseTo(guard.spentUsd, 12);
  });

  it("recordTool spend counts toward the ceiling", () => {
    const guard = new BudgetGuard(0.05, { onBlock: () => {} });
    guard.recordTool("scraper", 0.05);
    expect(() => guard.check()).toThrow(BudgetExceeded);
  });

  it("recordTool rejects bad costs", () => {
    const guard = new BudgetGuard(1.0);
    for (const bad of [Number.NaN, Number.POSITIVE_INFINITY, -0.01]) {
      expect(() => guard.recordTool("tool", bad)).toThrow(RangeError);
    }
    expect(guard.spendLog).toHaveLength(0);
  });

  it("the unpriceable skip path logs nothing", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      const guard = new BudgetGuard(1.0, { failClosed: false });
      guard.record("model-that-does-not-exist", 1_000, 1_000);
      expect(guard.spendLog).toHaveLength(0);
      expect(guard.spentUsd).toBe(0);
    } finally {
      warn.mockRestore();
    }
  });

  it("maxLogEvents is a ring buffer keeping the newest", () => {
    const guard = new BudgetGuard(10.0, { maxLogEvents: 3 });
    for (let i = 0; i < 5; i++) {
      guard.record(MODEL, 1_000, 1_000, { label: `call-${i}` });
    }
    expect(guard.spendLog.map((e) => e.label)).toEqual(["call-2", "call-3", "call-4"]);
    // The cap bounds the ledger only — totals still cover all 5 calls.
    expect(guard.spentUsd).toBeCloseTo(5 * 0.0125, 12);
  });

  it("maxLogEvents: 0 disables the ledger but not the totals", () => {
    const guard = new BudgetGuard(1.0, { maxLogEvents: 0 });
    guard.record(MODEL, 1_000, 1_000);
    expect(guard.spendLog).toHaveLength(0);
    expect(guard.exportLog()).toBe("");
    expect(guard.spentUsd).toBeCloseTo(0.0125, 12);
  });

  it("maxLogEvents validation", () => {
    for (const bad of [-1, 1.5, Number.NaN]) {
      expect(() => new BudgetGuard(1.0, { maxLogEvents: bad })).toThrow(RangeError);
    }
  });

  it("spendLog returns a snapshot copy", () => {
    const guard = new BudgetGuard(1.0);
    guard.record(MODEL, 1_000, 1_000);
    const snapshot = guard.spendLog;
    snapshot.length = 0;
    expect(guard.spendLog).toHaveLength(1);
  });
});

describe("BudgetGuard.exportLog", () => {
  it("emits stable, Python-compatible JSONL", () => {
    const guard = new BudgetGuard(1.0);
    const handle = guard.reserve(0.02);
    guard.settle(MODEL, 1_000, 1_000, { reserved: handle, label: "writer" });
    guard.record(MODEL, 500, 100);
    guard.recordTool("browser", 0.001);

    const out = guard.exportLog();
    expect(out.endsWith("\n")).toBe(true);
    const lines = out.trimEnd().split("\n");
    expect(lines).toHaveLength(3);

    const [settled, recorded, tool] = lines.map((l) => JSON.parse(l));
    // Fixed snake_case key order, optional fields present only when set — the
    // schema the Python package's export_log() emits field-for-field.
    expect(Object.keys(settled)).toEqual([
      "timestamp",
      "kind",
      "model_or_tool",
      "prompt_tokens",
      "completion_tokens",
      "cost_usd",
      "label",
      "reserved",
    ]);
    expect(Object.keys(recorded)).toEqual([
      "timestamp",
      "kind",
      "model_or_tool",
      "prompt_tokens",
      "completion_tokens",
      "cost_usd",
    ]);
    expect(tool.kind).toBe("tool");
    expect(tool.prompt_tokens).toBeNull();
    expect(tool.completion_tokens).toBeNull();
    const total = [settled, recorded, tool].reduce((sum, row) => sum + row.cost_usd, 0);
    expect(total).toBeCloseTo(guard.spentUsd, 12);
  });

  it("keeps unicode raw, matching Python's ensure_ascii=False", () => {
    const guard = new BudgetGuard(1.0);
    guard.record(MODEL, 10, 10, { label: "café-agent" });
    expect(guard.exportLog()).toContain("café-agent");
  });

  it("an empty ledger exports as the empty string", () => {
    expect(new BudgetGuard(1.0).exportLog()).toBe("");
  });
});
