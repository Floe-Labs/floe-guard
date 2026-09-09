/**
 * Pricing-resolution tests — kept in lockstep with tests/test_pricing.py so
 * the two packages resolve identically (the cost map itself is byte-identical
 * by CI guard; this covers the lookup logic).
 */

import { describe, expect, it } from "vitest";

import { costMapGeneratedAt, priceTokens, resolvePrice } from "../src/pricing";

describe("resolvePrice", () => {
  it("resolves a known model and its provider-prefixed form", () => {
    const bare = resolvePrice("gpt-4o");
    const prefixed = resolvePrice("openai/gpt-4o");
    expect(bare).not.toBeNull();
    expect(prefixed).not.toBeNull();
    expect(prefixed!.inputCostPerToken).toBe(bare!.inputCostPerToken);
  });

  it("returns null for an unknown model", () => {
    expect(resolvePrice("no-such-model-anywhere")).toBeNull();
  });

  it("bridges LiteLLM 'groq/<org>/<model>' ids to the vendored ChatGroq keys", () => {
    for (const model of [
      "groq/qwen/qwen3-32b",
      "groq/meta-llama/llama-4-scout-17b-16e-instruct",
      "groq/openai/gpt-oss-120b",
    ]) {
      const priced = resolvePrice(model);
      expect(priced, model).not.toBeNull();
      const chatGroq = resolvePrice(model.slice("groq/".length));
      expect(chatGroq, model).not.toBeNull();
      expect(priced!.inputCostPerToken).toBe(chatGroq!.inputCostPerToken);
    }
  });

  it("keeps bare multi-provider names unpriceable (anti-under-metering)", () => {
    expect(resolvePrice("qwen3-32b")).toBeNull();
    expect(resolvePrice("gpt-oss-120b")).toBeNull();
  });

  it("does not bridge unknown provider prefixes", () => {
    expect(resolvePrice("fireworks_ai/qwen/qwen3-32b")).toBeNull();
  });

  it("does not bridge openai/ or anthropic/ prefixes into Groq-priced keys", () => {
    // "openai/<model>" is LiteLLM's route for ANY OpenAI-compatible endpoint;
    // a multi-segment remainder is some other vendor's model → fail closed.
    expect(resolvePrice("openai/qwen/qwen3-32b")).toBeNull();
    expect(resolvePrice("anthropic/qwen/qwen3-32b")).toBeNull();
    expect(resolvePrice("openai/meta-llama/llama-4-scout-17b-16e-instruct")).toBeNull();
  });

  it("prices an unlisted dated snapshot at its alias entry", () => {
    const alias = resolvePrice("claude-opus-4-8");
    const dated = resolvePrice("claude-opus-4-8-20991231");
    expect(alias).not.toBeNull();
    expect(dated).not.toBeNull();
    expect(dated!.inputCostPerToken).toBe(alias!.inputCostPerToken);
    expect(resolvePrice("gpt-5.5-2099-01-01")).not.toBeNull();
    expect(resolvePrice("anthropic/claude-sonnet-5-20991231")).not.toBeNull();
  });

  it("prefers an exact dated key over the alias fallback", () => {
    const exact = resolvePrice("claude-sonnet-4-5-20250929");
    expect(exact).not.toBeNull();
    expect(exact!.source).toBe("cost_map");
  });

  it("matches overrides against the provider-stripped candidate", () => {
    const priced = resolvePrice("groq/my-model", {
      "my-model": { inputCostPerToken: 1e-6, outputCostPerToken: 2e-6 },
    });
    expect(priced).not.toBeNull();
    expect(priced!.source).toBe("override");
  });

  it("fails closed on a malformed override", () => {
    expect(
      resolvePrice("x", { x: { inputCostPerToken: NaN, outputCostPerToken: 1e-6 } }),
    ).toBeNull();
  });

  it("does not let an alias override shadow an exact dated map entry", () => {
    // gpt-4o-2024-05-13 has its own map entry at 2x the alias rate; an
    // alias-keyed override is a less-specific match and must not absorb it.
    const exact = resolvePrice("gpt-4o-2024-05-13");
    expect(exact).not.toBeNull();
    const priced = resolvePrice("gpt-4o-2024-05-13", {
      "gpt-4o": { inputCostPerToken: 2.5e-6, outputCostPerToken: 1e-5 },
    });
    expect(priced!.source).toBe("cost_map");
    expect(priced!.inputCostPerToken).toBe(exact!.inputCostPerToken);
    // The override still wins for the alias itself and for unlisted snapshots.
    expect(resolvePrice("gpt-4o", {
      "gpt-4o": { inputCostPerToken: 1e-9, outputCostPerToken: 2e-9 },
    })!.source).toBe("override");
    expect(resolvePrice("gpt-4o-2099-01-01", {
      "gpt-4o": { inputCostPerToken: 1e-9, outputCostPerToken: 2e-9 },
    })!.source).toBe("override");
  });

  it("only strips ASCII-digit date suffixes (parity with Python's re.ASCII)", () => {
    expect(resolvePrice("gpt-4o-٢٠٢٥٠١٠١")).toBeNull();
  });

  it("resolves claude-3-5-sonnet and claude-3-5-haiku without an override", () => {
    const expected: Record<string, [number, number]> = {
      "claude-3-5-sonnet-20241022": [3e-6, 1.5e-5],
      "claude-3-5-sonnet-20240620": [3e-6, 1.5e-5],
      "claude-3-5-haiku-20241022": [8e-7, 4e-6],
    };
    for (const [model, [inputCost, outputCost]] of Object.entries(expected)) {
      const priced = resolvePrice(model);
      expect(priced, model).not.toBeNull();
      expect(priced!.source).toBe("cost_map");
      expect(priced!.inputCostPerToken).toBe(inputCost);
      expect(priced!.outputCostPerToken).toBe(outputCost);
    }
  });
});

describe("costMapGeneratedAt / reserved keys", () => {
  it("returns a valid YYYY-MM-DD snapshot date", () => {
    const date = costMapGeneratedAt();
    expect(date).toBeDefined();
    expect(date).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    // must be a real calendar date
    expect(new Date(`${date}T00:00:00Z`).toISOString().slice(0, 10)).toBe(date);
  });

  it("never resolves reserved dunder keys as models", () => {
    expect(resolvePrice("__meta__")).toBeNull();
    expect(resolvePrice("__voice__")).toBeNull();
    // a real model still resolves
    expect(resolvePrice("gpt-4o")).not.toBeNull();
  });
});

describe("cache-aware priceTokens", () => {
  it("falls back to Anthropic-style multipliers when the model has no published cache rate", () => {
    const priced = resolvePrice("claude-3-5-sonnet-20241022");
    expect(priced).not.toBeNull();
    expect(priced!.cacheReadCostPerToken).toBeUndefined();
    const cost = priceTokens(priced!, 0, 0, {
      cacheCreationInputTokens: 100,
      cacheReadInputTokens: 1000,
      cacheCreationInputTokens1h: 200,
    });
    // 5m write 1.25x, 1h write 2.0x, read 0.1x of input
    const input = priced!.inputCostPerToken;
    expect(cost).toBeCloseTo(100 * input * 1.25 + 200 * input * 2.0 + 1000 * input * 0.1, 12);
  });

  it("prices cache-read at the model's published rate, not a one-size multiplier", () => {
    const anthropic = resolvePrice("claude-3-7-sonnet-20250219");
    const openai = resolvePrice("gpt-4o");
    expect(anthropic).not.toBeNull();
    expect(openai).not.toBeNull();
    expect(anthropic!.cacheReadCostPerToken).toBeCloseTo(3e-7, 12);
    expect(openai!.cacheReadCostPerToken).toBeCloseTo(1.25e-6, 12);

    expect(priceTokens(anthropic!, 0, 0, { cacheReadInputTokens: 1000 })).toBeCloseTo(
      1000 * anthropic!.cacheReadCostPerToken!,
      12,
    );
    expect(priceTokens(openai!, 0, 0, { cacheReadInputTokens: 1000 })).toBeCloseTo(
      1000 * openai!.cacheReadCostPerToken!,
      12,
    );
  });

  it("does not attach cache rates to override-sourced prices", () => {
    const ov = resolvePrice("gpt-4o", {
      "gpt-4o": { inputCostPerToken: 1e-9, outputCostPerToken: 2e-9 },
    });
    expect(ov!.source).toBe("override");
    expect(ov!.cacheReadCostPerToken).toBeUndefined();
    expect(ov!.cacheCreationCostPerToken).toBeUndefined();
  });

  it("gpt-4o with a 90% cache hit is ~0.55x the uncached prompt, not 1.0x", () => {
    // 10_000 prompt tokens, 90% cached, no completion. Python bills fresh at
    // input and cached at cache-read (0.5x for gpt-4o). JS used to bill all
    // 10_000 at the full input rate (~1.8x overcharge).
    const priced = resolvePrice("gpt-4o")!;
    const uncached = priceTokens(priced, 10_000, 0);
    const cached = priceTokens(priced, 1_000, 0, { cacheReadInputTokens: 9_000 });
    expect(cached / uncached).toBeCloseTo(0.55, 5);
  });
});
