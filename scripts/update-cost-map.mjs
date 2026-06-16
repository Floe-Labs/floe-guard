/**
 * Refresh floe-guard's vendored LiteLLM cost map — BOTH copies — from the same
 * public source the Floe proxy uses. This keeps the open-source package's pricing
 * current without coupling it to the private monorepo.
 *
 * Writes (identically, so the cost-map-sync CI guard stays green):
 *   - src/floe_guard/cost_map.json   (Python package)
 *   - js/src/cost_map.json           (JS package)
 *
 * The transform mirrors the proxy's scripts/update-llm-cost-map.ts exactly, and
 * serialises with the same JSON.stringify(…, 2) so refreshes show up as clean
 * price diffs rather than reformatting noise. (A Python re-serialiser would emit
 * floats differently, e.g. 1e-06 vs 1e-7, and churn the whole file.)
 *
 * Run: node scripts/update-cost-map.mjs
 */
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const SOURCE_URL =
  process.env.LITELLM_COST_MAP_URL ??
  "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json";

// Providers floe-guard prices (matches the proxy's ROUTABLE_PROVIDERS).
const ROUTABLE_PROVIDERS = new Set(["openai", "anthropic"]);

/**
 * A model is vendored only if we can fully price it: a numeric input rate, a
 * routable provider, and either embedding mode (input-only) or chat mode WITH a
 * numeric output rate. Coercing a missing output rate to 0 would ship a chat
 * model that bills output free, which fail-closed pricing can't catch (0 is
 * finite). An excluded model is simply absent.
 */
function isUsable(v) {
  // Number.isFinite (not typeof === "number") so a NaN, or a huge upstream value
  // that JSON.parse turns into Infinity, is treated as unpriceable and dropped —
  // matching the fail-closed pricing paths.
  return (
    !!v &&
    Number.isFinite(v.input_cost_per_token) &&
    v.litellm_provider !== undefined &&
    ROUTABLE_PROVIDERS.has(v.litellm_provider) &&
    (v.mode === "embedding" ||
      (v.mode === "chat" && Number.isFinite(v.output_cost_per_token)))
  );
}

const res = await fetch(SOURCE_URL);
if (!res.ok) {
  throw new Error(`Failed to fetch LiteLLM cost map: HTTP ${res.status}`);
}
const raw = await res.json();

const entries = Object.entries(raw)
  .filter(([, v]) => isUsable(v))
  .sort(([a], [b]) => a.localeCompare(b));

// null-prototype: model keys come from remote JSON, so a "__proto__" (or similar)
// key is stored as plain data instead of mutating the object's prototype.
const out = Object.create(null);
for (const [k, v] of entries) {
  out[k] = {
    input_cost_per_token: v.input_cost_per_token,
    output_cost_per_token: v.mode === "embedding" ? 0 : v.output_cost_per_token,
    litellm_provider: v.litellm_provider,
    mode: v.mode,
  };
}

const json = `${JSON.stringify(out, null, 2)}\n`;
const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const targets = [
  join(root, "src", "floe_guard", "cost_map.json"),
  join(root, "js", "src", "cost_map.json"),
];
for (const dest of targets) {
  writeFileSync(dest, json);
}

console.log(
  `Wrote ${entries.length} models to ${targets.length} files (source: ${SOURCE_URL}).`,
);
