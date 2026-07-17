import { describe, expect, it } from "vitest";

import { BudgetExceeded, BudgetGuard } from "../src/index.js";

const MODEL = "gpt-4o"; // 1k in + 1k out = $0.0125/call

describe("tool spend (reserveTool / settleTool / recordTool / toolCosts)", () => {
  it("reserveTool rejects non-finite and negative estimates", () => {
    const guard = new BudgetGuard(1.0);
    for (const bad of [Number.NaN, Number.POSITIVE_INFINITY, -0.01]) {
      expect(() => guard.reserveTool(bad)).toThrow(RangeError);
    }
    expect(guard.remainingUsd).toBeCloseTo(1.0, 12); // nothing was held
  });

  it("reserveTool rejects a missing estimate instead of silently falling back", () => {
    // reserve(undefined) means "use the last-cost prediction" — 0 on a fresh
    // guard, i.e. an unguarded tool call. reserveTool must fail loudly instead.
    const guard = new BudgetGuard(1.0);
    expect(() => guard.reserveTool(undefined as unknown as number)).toThrow(RangeError);
  });

  it("settleTool rejects bad amounts", () => {
    const guard = new BudgetGuard(1.0);
    for (const bad of [Number.NaN, Number.POSITIVE_INFINITY, -0.01]) {
      expect(() => guard.settleTool("apollo.people_lookup", bad)).toThrow(RangeError);
      expect(() => guard.settleTool("apollo.people_lookup", 0.01, { reserved: bad })).toThrow(
        RangeError,
      );
    }
    expect(guard.spendLog).toHaveLength(0);
    expect(guard.toolCosts).toEqual({});
  });

  it("toolCosts tallies per name", () => {
    const guard = new BudgetGuard(1.0);
    guard.recordTool("apollo.people_lookup", 0.02);
    guard.recordTool("apollo.people_lookup", 0.02);
    guard.recordTool("exa.search", 0.01);
    expect(guard.toolCosts["apollo.people_lookup"]).toBeCloseTo(0.04, 12);
    expect(guard.toolCosts["exa.search"]).toBeCloseTo(0.01, 12);
    expect(guard.spentUsd).toBeCloseTo(0.05, 12);
  });

  it("tokens and tools share one ceiling and the split is inspectable", () => {
    const guard = new BudgetGuard(1.0);
    guard.record(MODEL, 1_000, 1_000); // $0.0125 of tokens
    guard.recordTool("apollo.people_lookup", 0.02);
    expect(guard.spentUsd).toBeCloseTo(0.0325, 12);
    expect(guard.remainingUsd).toBeCloseTo(1.0 - 0.0325, 12);
    const toolTotal = Object.values(guard.toolCosts).reduce((s, c) => s + c, 0);
    expect(toolTotal).toBeCloseTo(0.02, 12);
    expect(guard.spentUsd - toolTotal).toBeCloseTo(0.0125, 12); // token side
  });

  it("toolCosts returns a snapshot copy", () => {
    const guard = new BudgetGuard(1.0);
    guard.recordTool("exa.search", 0.01);
    const snapshot = guard.toolCosts;
    snapshot["exa.search"] = 999;
    expect(guard.toolCosts["exa.search"]).toBeCloseTo(0.01, 12);
  });

  it('a "__proto__" tool name is stored as data, not prototype pollution', () => {
    const guard = new BudgetGuard(1.0);
    guard.recordTool("__proto__", 0.01);
    expect(guard.toolCosts["__proto__"]).toBeCloseTo(0.01, 12);
    expect(({} as Record<string, unknown>)["__proto__"]).not.toBe(0.01);
    expect(guard.spentUsd).toBeCloseTo(0.01, 12);
  });

  it("reserveTool blocks BEFORE the tool runs", () => {
    const guard = new BudgetGuard(0.01, { onBlock: () => {} });
    expect(() => guard.reserveTool(0.02)).toThrow(BudgetExceeded);
    expect(guard.remainingUsd).toBeCloseTo(0.01, 12);
    expect(guard.spentUsd).toBe(0);
  });

  it("reserveTool/settleTool round trip", () => {
    const guard = new BudgetGuard(1.0);
    const handle = guard.reserveTool(0.02);
    expect(handle).toBeCloseTo(0.02, 12);
    expect(guard.remainingUsd).toBeCloseTo(0.98, 12); // held while in flight
    const cost = guard.settleTool("apollo.people_lookup", 0.02, {
      reserved: handle,
      label: "prospector",
    });
    expect(cost).toBeCloseTo(0.02, 12);
    expect(guard.remainingUsd).toBeCloseTo(0.98, 12); // hold swapped for spend
    const [event] = guard.spendLog;
    expect(event.kind).toBe("tool");
    expect(event.modelOrTool).toBe("apollo.people_lookup");
    expect(event.reserved).toBeCloseTo(0.02, 12);
    expect(event.label).toBe("prospector");
  });

  it("release frees a tool reservation when the call fails", () => {
    const guard = new BudgetGuard(1.0);
    const handle = guard.reserveTool(0.02);
    guard.release(handle);
    expect(guard.remainingUsd).toBeCloseTo(1.0, 12);
    expect(guard.spendLog).toHaveLength(0);
  });

  it("a runaway tool loop dies at the ceiling (check predicts one call ahead)", () => {
    const guard = new BudgetGuard(0.01, { onBlock: () => {} });
    let calls = 0;
    expect(() => {
      for (let i = 0; i < 1_000; i++) {
        guard.check();
        guard.recordTool("apollo.people_lookup", 0.002);
        calls++;
      }
    }).toThrow(BudgetExceeded);
    expect(calls).toBe(5); // 5 × $0.002 == $0.01 — call 6 was blocked
    expect(guard.spentUsd).toBeLessThanOrEqual(guard.limitUsd + 1e-9);
  });

  it("mixed token + tool reservations hold the ceiling under interleaving", async () => {
    // JS is single-threaded, but the same check-then-act race exists across
    // awaits (issue #18): reserve before the await, settle after.
    const guard = new BudgetGuard(0.1, { onBlock: () => {} });
    guard.record(MODEL, 1_000, 1_000); // warm the LLM estimate

    let blocked = 0;
    const agents = Array.from({ length: 16 }, (_, i) => async () => {
      let reserved: number;
      try {
        reserved = i % 2 ? guard.reserve() : guard.reserveTool(0.0125);
      } catch (err) {
        if (!(err instanceof BudgetExceeded)) throw err;
        blocked++;
        return;
      }
      await new Promise((resolve) => setTimeout(resolve, 5)); // API latency
      if (i % 2) {
        guard.settle(MODEL, 1_000, 1_000, { reserved });
      } else {
        guard.settleTool("apollo.people_lookup", 0.0125, { reserved });
      }
    });
    await Promise.all(agents.map((run) => run()));

    expect(guard.spentUsd).toBeLessThanOrEqual(guard.limitUsd + 1e-9);
    expect(blocked).toBeGreaterThan(0);
    expect(guard.toolCosts["apollo.people_lookup"]).toBeGreaterThan(0);
  });
});
