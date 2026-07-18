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

/**
 * The one `<provider>/` prefix that is safe to strip: the remainder of a
 * `groq/…` id is the ChatGroq id the map vendors (e.g. `groq/qwen/qwen3-32b` →
 * `qwen/qwen3-32b`). `openai/` and `anthropic/` are deliberately excluded:
 * their own model ids never contain slashes (single-segment remainders are
 * already covered by the bare-last-segment fallback), so a multi-segment
 * remainder under those prefixes is some OTHER vendor's model behind an
 * OpenAI-compatible endpoint (vLLM, OpenRouter, …) — bridging it into a
 * Groq-priced key would under-meter. Unknown prefixes fail closed the same way.
 */
const PROVIDER_PREFIXES = new Set(["groq"]);

/**
 * A trailing dated-snapshot suffix: Anthropic's `-20250929` or OpenAI's
 * `-2024-08-06`. Vendors resolve alias ids to dated snapshots in responses, so a
 * snapshot the map doesn't list yet prices at its alias entry (same model, same
 * rate) instead of failing closed. (`\d` is ASCII-only in JS; the Python regex
 * uses re.ASCII to match.)
 */
const DATE_SUFFIX = /-(?:\d{8}|\d{4}-\d{2}-\d{2})$/;

/**
 * Lookup keys for a model id in two specificity groups, deduplicated.
 *
 * Group 1 (exact): the raw id, the id with a known `provider/` first segment
 * stripped, the bare last segment. Group 2 (date-stripped): the same forms
 * with a trailing dated-snapshot suffix removed. Kept separate so a
 * less-specific date-stripped key (in overrides OR the map) can never shadow
 * an exact dated entry — e.g. an alias override must not absorb a snapshot
 * the map prices differently.
 */
function candidateGroups(model: string): [string[], string[]] {
  const m = model.trim();
  const base = [m];
  const firstSlash = m.indexOf("/");
  if (firstSlash !== -1 && PROVIDER_PREFIXES.has(m.slice(0, firstSlash))) {
    base.push(m.slice(firstSlash + 1));
  }
  const lastSlash = m.lastIndexOf("/");
  if (lastSlash !== -1) {
    base.push(m.slice(lastSlash + 1));
  }
  const exact: string[] = [];
  for (const cand of base) {
    if (cand && !exact.includes(cand)) exact.push(cand);
  }
  const stripped: string[] = [];
  for (const cand of exact) {
    const c = cand.replace(DATE_SUFFIX, "");
    if (c && !exact.includes(c) && !stripped.includes(c)) stripped.push(c);
  }
  return [exact, stripped];
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
 * Per specificity group (exact forms first, date-stripped fallbacks second):
 * overrides win, then the bundled cost map. Fail-closed: the first matching
 * entry must have finite prices, else `null`.
 */
export function resolvePrice(
  model: string,
  overrides?: Record<string, ManualPrice>,
): PricedModel | null {
  for (const cands of candidateGroups(model)) {
    if (overrides) {
      for (const cand of cands) {
        // Own-property check so a key like "constructor" can't pull a function
        // off Object.prototype and be treated as a price entry.
        const ov = Object.prototype.hasOwnProperty.call(overrides, cand)
          ? overrides[cand]
          : undefined;
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
    }

    for (const cand of cands) {
      const entry = Object.prototype.hasOwnProperty.call(COST_MAP, cand)
        ? COST_MAP[cand]
        : undefined;
      if (!entry) continue;

      const input = entry.input_cost_per_token;
      const output = entry.output_cost_per_token;
      if (!bothFinite(input, output)) return null;

      return {
        inputCostPerToken: input as number,
        outputCostPerToken: output as number,
        source: "cost_map",
      };
    }
  }
  return null;
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
