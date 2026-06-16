/**
 * Offline token pricing from a vendored LiteLLM cost map.
 *
 * Mirrors `src/floe_guard/pricing.py`: BOTH the input and output per-token prices
 * must be finite numbers, otherwise the model is treated as unpriceable. A
 * half-valid entry would silently undercharge — so we refuse it (fail-closed).
 *
 * No network. The cost map (`cost_map.json`) is a snapshot of LiteLLM's
 * `model_prices_and_context_window.json`; refresh it on a schedule or estimates
 * drift as vendors change prices.
 */

import costMapJson from "./cost_map.json";

/** A user-supplied per-token price, in USD, for a model the map cannot price. */
export interface ManualPrice {
  inputCostPerToken: number;
  outputCostPerToken: number;
}

/** A resolved per-token price plus where it came from. */
export interface PricedModel {
  inputCostPerToken: number;
  outputCostPerToken: number;
  source: "override" | "cost_map";
}

interface CostMapEntry {
  input_cost_per_token?: unknown;
  output_cost_per_token?: unknown;
}

const COST_MAP = costMapJson as Record<string, CostMapEntry>;

/** Strip an optional `provider/` prefix (LiteLLM convention, e.g. `openai/gpt-4o`). */
function bareModel(model: string): string {
  const m = model.trim();
  const slash = m.lastIndexOf("/");
  return slash === -1 ? m : m.slice(slash + 1);
}

function bothFinite(a: unknown, b: unknown): boolean {
  return (
    typeof a === "number" &&
    typeof b === "number" &&
    Number.isFinite(a) &&
    Number.isFinite(b)
  );
}

/**
 * Resolve a model to its per-token price, or `null` if it cannot be priced.
 *
 * Overrides win, then the bundled cost map (looked up by bare name, then the raw
 * field). Fail-closed: both prices must be finite, else `null`.
 */
export function resolvePrice(
  model: string,
  overrides?: Record<string, ManualPrice>,
): PricedModel | null {
  const bare = bareModel(model);

  if (overrides) {
    const ov = overrides[bare] ?? overrides[model.trim()];
    if (ov !== undefined) {
      if (bothFinite(ov.inputCostPerToken, ov.outputCostPerToken)) {
        return {
          inputCostPerToken: ov.inputCostPerToken,
          outputCostPerToken: ov.outputCostPerToken,
          source: "override",
        };
      }
      return null;
    }
  }

  const entry = COST_MAP[bare] ?? COST_MAP[model.trim()];
  if (!entry) return null;

  const input = entry.input_cost_per_token;
  const output = entry.output_cost_per_token;
  if (!bothFinite(input, output)) return null;

  return {
    inputCostPerToken: input as number,
    outputCostPerToken: output as number,
    source: "cost_map",
  };
}

/** USD cost for token usage. Negative counts are clamped to zero. */
export function priceTokens(
  priced: PricedModel,
  promptTokens: number,
  completionTokens: number,
): number {
  const p = Math.max(0, promptTokens);
  const c = Math.max(0, completionTokens);
  const cost = p * priced.inputCostPerToken + c * priced.outputCostPerToken;
  if (!Number.isFinite(cost)) {
    // Defense-in-depth: resolvePrice already guarantees finite rates.
    throw new Error("Non-finite LLM cost — pricing entry is invalid");
  }
  return Math.max(0, cost);
}
