/**
 * Pricing-resolution tests — kept in lockstep with tests/test_pricing.py so
 * the two packages resolve identically (the cost map itself is byte-identical
 * by CI guard; this covers the lookup logic).
 */

import { describe, expect, it } from "vitest";

import { resolvePrice } from "../src/pricing";

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
});
