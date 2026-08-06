/**
 * Tests for token ceilings and per-step budgets (issue #46) — the TS mirror of
 * `tests/test_token_step_budgets.py`.
 *
 * The feature is a second dimension (tokens) on the existing enforcement choke
 * point plus a step scope, so these tests also pin the backward-compat contract:
 * a USD-only guard behaves exactly as before, including `reserve()` still
 * returning a plain number.
 */

import { describe, expect, it } from "vitest";

import {
  BudgetExceeded,
  BudgetGuard,
  TokenBudgetExceeded,
  type BudgetReservation,
} from "../src/index.js";

const MODEL = "gpt-4o"; // $2.5e-6/input token, $1e-5/output token
const silent = { onBlock: () => {} };

describe("aggregate token ceiling", () => {
  it("hard-blocks when a call would cross the token limit", () => {
    const guard = new BudgetGuard(100, { tokenLimit: 1_000, ...silent });
    guard.record(MODEL, 400, 200); // 600 tokens accrued
    try {
      guard.check(undefined, { estimatedTokens: 500 }); // 600 + 500 > 1000
      throw new Error("expected TokenBudgetExceeded");
    } catch (err) {
      expect(err).toBeInstanceOf(TokenBudgetExceeded);
      expect((err as TokenBudgetExceeded).scope).toBe("aggregate");
      expect((err as TokenBudgetExceeded).limitTokens).toBe(1_000);
    }
  });

  it("reserve holds tokens in-flight and blocks the overshoot", () => {
    const guard = new BudgetGuard(100, { tokenLimit: 1_000, ...silent });
    const handle = guard.reserve(0.01, { estimatedTokens: 900 }) as BudgetReservation;
    expect(handle.tokens).toBe(900);
    expect(() => guard.reserve(0.01, { estimatedTokens: 200 })).toThrow(TokenBudgetExceeded);
  });

  it("accrues tokens from prompt + completion", () => {
    const guard = new BudgetGuard(100, { tokenLimit: 10_000 });
    guard.record(MODEL, 100, 50);
    expect(guard.spentTokens).toBe(150);
  });

  it("rejects a non-integer or negative tokenLimit", () => {
    expect(() => new BudgetGuard(1, { tokenLimit: 1.5 })).toThrow(RangeError);
    expect(() => new BudgetGuard(1, { tokenLimit: -1 })).toThrow(RangeError);
  });
});

describe("per-step caps", () => {
  it("token cap hard-blocks even with ample aggregate budget", () => {
    const guard = new BudgetGuard(100, silent);
    guard.step({ maxTokens: 500 }, (g) => {
      expect(g).toBe(guard); // callback receives the SAME guard
      g.record(MODEL, 300, 100); // 400 tokens into the step
      try {
        g.check(undefined, { estimatedTokens: 200 }); // 400 + 200 > 500
        throw new Error("expected TokenBudgetExceeded");
      } catch (err) {
        expect(err).toBeInstanceOf(TokenBudgetExceeded);
        expect((err as TokenBudgetExceeded).scope).toBe("step");
        expect((err as TokenBudgetExceeded).limitTokens).toBe(500);
      }
    });
  });

  it("USD cap hard-blocks with a plain BudgetExceeded", () => {
    const guard = new BudgetGuard(100, silent);
    guard.step({ maxUsd: 0.01 }, (g) => {
      let caught: unknown;
      try {
        g.reserve(0.02);
      } catch (err) {
        caught = err;
      }
      expect(caught).toBeInstanceOf(BudgetExceeded);
      expect(caught).not.toBeInstanceOf(TokenBudgetExceeded);
    });
  });

  it("frees the step cap on exit", () => {
    const guard = new BudgetGuard(100, silent);
    guard.step({ maxTokens: 100 }, (g) => {
      expect(() => g.check(undefined, { estimatedTokens: 200 })).toThrow(TokenBudgetExceeded);
    });
    // Outside the step no token ceiling applies.
    expect(() => guard.check(undefined, { estimatedTokens: 1_000_000 })).not.toThrow();
  });

  it("returns a BudgetReservation while a step is active, even USD-only", () => {
    const guard = new BudgetGuard(100);
    guard.step({ maxUsd: 1.0 }, (g) => {
      const handle = g.reserve(0.02);
      expect(typeof handle).toBe("object");
      g.settle(MODEL, 100, 100, { reserved: handle });
    });
  });

  it("settles a BudgetReservation after its owning step has exited", () => {
    // A token-aware handle may outlive the step() that made it; settling it once
    // the step is gone must drain the aggregate hold and accrue, not throw.
    const guard = new BudgetGuard(100, { tokenLimit: 20_000 });
    let handle!: BudgetReservation;
    guard.step({ maxTokens: 5_000 }, (g) => {
      handle = g.reserve(undefined, { estimatedTokens: 1_000 }) as BudgetReservation;
    });
    expect(() => guard.settle(MODEL, 400, 300, { reserved: handle })).not.toThrow();
    expect(guard.spentTokens).toBe(700);
  });

  it("nested steps: innermost blocks first", () => {
    const guard = new BudgetGuard(100, silent);
    guard.step({ maxTokens: 10_000 }, (outer) => {
      outer.step({ maxTokens: 200 }, (inner) => {
        try {
          inner.check(undefined, { estimatedTokens: 500 });
          throw new Error("expected TokenBudgetExceeded");
        } catch (err) {
          expect((err as TokenBudgetExceeded).limitTokens).toBe(200);
        }
      });
    });
  });
});

describe("advisory", () => {
  it("reports aggregate token utilization", () => {
    const guard = new BudgetGuard(100, { tokenLimit: 1_000 });
    guard.record(MODEL, 500, 100); // 600 / 1000
    const adv = guard.advisory();
    expect(adv.tokenUsedBps).toBe(6000);
    expect(adv.remainingTokens).toBe(400);
  });

  it("flags per-step near-limit before the hard block", () => {
    const guard = new BudgetGuard(100, { nearLimitBps: 8000 });
    guard.step({ maxTokens: 1_000 }, (g) => {
      g.record(MODEL, 500, 300); // 800 / 1000 = 80%
      const adv = g.advisory();
      expect(adv.nearLimit).toBe(true);
      expect(adv.stepRemainingTokens).toBe(200);
      expect(() => g.check(undefined, { estimatedTokens: 100 })).not.toThrow();
    });
  });

  it("leaves token/step fields null when the dimension is unused", () => {
    const adv = new BudgetGuard(1).advisory();
    expect(adv.tokenUsedBps).toBeNull();
    expect(adv.remainingTokens).toBeNull();
    expect(adv.stepRemainingUsd).toBeNull();
    expect(adv.stepRemainingTokens).toBeNull();
  });
});

describe("backward-compat: USD-only is unchanged", () => {
  it("reserve() still returns a plain number", () => {
    const guard = new BudgetGuard(1);
    const handle = guard.reserve(0.0125);
    expect(typeof handle).toBe("number");
    expect(handle).toBeCloseTo(0.0125);
    guard.settle(MODEL, 1_000, 1_000, { reserved: handle });
    expect(guard.spentUsd).toBeCloseTo(0.0125);
  });

  it("default reserve() returns number 0 on a fresh guard", () => {
    const guard = new BudgetGuard(1, silent);
    expect(guard.reserve()).toBe(0);
  });

  it("advisory keeps its old fields and null for the new ones", () => {
    const guard = new BudgetGuard(1);
    guard.record(MODEL, 100, 100);
    const adv = guard.advisory();
    expect(adv.spentUsd).toBeCloseTo(guard.spentUsd);
    expect(adv.tokenUsedBps).toBeNull();
    expect(adv.stepRemainingUsd).toBeNull();
  });

  it("a token block is terminal like BudgetExceeded (subclass)", () => {
    const err = new TokenBudgetExceeded(600, 500, "step");
    expect(err).toBeInstanceOf(BudgetExceeded);
  });
});
