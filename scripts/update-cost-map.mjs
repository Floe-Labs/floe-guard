/**
 * Refresh floe-guard's vendored LiteLLM cost map — BOTH copies — from the same
 * public source the Floe proxy uses. This keeps the open-source package's pricing
 * current without coupling it to the private monorepo.
 *
 * Writes (identically, so the cost-map-sync CI guard stays green):
 *   - src/floe_guard/cost_map.json   (Python package)
 *   - js/src/cost_map.json           (JS package)
 *
 * The transform mirrors the proxy's scripts/update-llm-cost-map.ts (plus a
 * curated Groq allowlist for the LangChain/Groq integration), and
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

// Curated Groq models, vendored under their ChatGroq ids (upstream keys them
// "groq/<id>"; pricing.py/pricing.ts strip the known "groq/" prefix at lookup,
// so both id conventions resolve). Kept as an explicit allowlist rather than
// adding "groq" to ROUTABLE_PROVIDERS: fully-generic bare names (e.g.
// "qwen3-32b") are multi-provider, and pricing them at Groq's cheap rate would
// under-meter a spend guard — unlisted models stay unpriceable and fail closed.
//
// Groq deprecation schedule (keep entries until their shutdown date passes):
//   qwen/qwen3-32b + meta-llama/llama-4-scout-17b-16e-instruct — 2026-07-17
//   llama-3.1-8b-instant + llama-3.3-70b-versatile             — 2026-08-16
const GROQ_KEY_MAP = new Map([
  ["groq/llama-3.1-8b-instant", "llama-3.1-8b-instant"],
  ["groq/llama-3.3-70b-versatile", "llama-3.3-70b-versatile"],
  [
    "groq/meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
  ],
  ["groq/qwen/qwen3-32b", "qwen/qwen3-32b"],
  // Current production lineup (gpt-oss-120b/20b are Groq's recommended
  // replacements for the deprecating llamas). The "openai/" ChatGroq prefix is
  // safe: OpenAI's own API does not serve gpt-oss, so the key can't collide
  // with an OpenAI-routed id.
  ["groq/openai/gpt-oss-120b", "openai/gpt-oss-120b"],
  ["groq/openai/gpt-oss-20b", "openai/gpt-oss-20b"],
  ["groq/openai/gpt-oss-safeguard-20b", "openai/gpt-oss-safeguard-20b"],
]);

// Providers vendored under the BARE model id, with their "<provider>/" key
// prefix stripped (upstream keys Gemini as "gemini/gemini-2.5-flash", but the
// google-genai SDK and @ai-sdk/google both take "gemini-2.5-flash").
//
// A rule rather than a Groq-style allowlist: nothing but Google ships a
// "gemini-*" model, so there is no generic multi-vendor name to mis-claim here,
// and hand-listing would go stale on every Google launch. The bare key also
// serves LiteLLM's "gemini/<id>" convention through the resolver's
// bare-last-segment fallback, so pricing.py/pricing.ts need no change (and the
// two stay in lockstep by not moving at all).
//
// Vertex AI is deliberately NOT vendored. It serves the SAME model ids under the
// "vertex_ai-*" providers at DIFFERENT prices (gemini-2.0-flash-001: Vertex is
// 50% dearer), and a model id alone cannot say which billing path a call took —
// pricing both from one key would under-meter Vertex users, which is the exact
// failure a spend guard must not have. Those providers are not routable, so they
// are already excluded; Vertex callers pass price_overrides, and the Gemini
// adapter detects them via `client.vertexai`.
const PREFIX_STRIPPED_PROVIDERS = new Set(["gemini"]);

/** The key a model is vendored under: "gemini/gemini-2.5-flash" -> "gemini-2.5-flash". */
function vendoredKey(k) {
  const mapped = GROQ_KEY_MAP.get(k);
  if (mapped !== undefined) return mapped;
  const slash = k.indexOf("/");
  if (slash !== -1 && PREFIX_STRIPPED_PROVIDERS.has(k.slice(0, slash))) {
    return k.slice(slash + 1);
  }
  return k;
}

// Embedding families vendored under a zeroed output rate. Matched as an id
// PREFIX, not a substring: `includes("embedding")` would also accept a chat model
// named e.g. "foo-embedding-chat", re-opening the very hole this list closes.
// Covers every embedding entry the map ships today (text-embedding-3-*,
// text-embedding-ada-*, gemini-embedding-*); a new family is dropped, and warned
// about, until it is added here.
const EMBEDDING_ID_PREFIXES = ["text-embedding-", "gemini-embedding-"];

/**
 * Embedding mode zeroes the output rate, so trusting a WRONG `mode` ships a chat
 * model that bills output free — the precise hole fail-closed pricing cannot see.
 * Upstream does get this wrong: it lists `gemini/gemini-1.5-flash`, a chat model,
 * as `mode: "embedding"` with `output_cost_per_token: 0`.
 *
 * So `mode` alone is not enough authority to zero a price. Require the model id to
 * agree with it, by matching a known embedding family (see EMBEDDING_ID_PREFIXES).
 * A single wrong field then can't produce a free-output chat model, and an
 * embedding whose name doesn't match simply fails closed — the safe direction.
 */
function isEmbeddingModel(vendored, v) {
  return (
    v.mode === "embedding" &&
    EMBEDDING_ID_PREFIXES.some((prefix) => vendored.startsWith(prefix))
  );
}

/** Whether floe-guard prices this provider at all — see the three sets above. */
function isPricedProvider(k, v) {
  return (
    v.litellm_provider !== undefined &&
    (ROUTABLE_PROVIDERS.has(v.litellm_provider) ||
      PREFIX_STRIPPED_PROVIDERS.has(v.litellm_provider) ||
      GROQ_KEY_MAP.has(k))
  );
}

/**
 * A model is vendored only if we can fully price it: a positive input rate, a
 * routable provider, and either a VERIFIED embedding (input-only — see
 * isEmbeddingModel) or chat mode with a non-zero output rate. Coercing a missing
 * or zero output rate would ship a chat model that bills output free, which
 * fail-closed pricing can't catch (0 is finite). An excluded model is simply absent.
 *
 * A rate of 0 bills every call free, and fail-closed pricing cannot catch that:
 * 0 is finite, so resolve_price returns a valid entry and the guard meters the
 * call at $0 forever. Upstream ships these for free/experimental tiers
 * (gemini-exp-1206 is listed 0/0 on one of its two keys). Dropping them makes the
 * model unpriceable, which fails closed loudly — the behaviour a spend guard
 * wants. The one exception is an embedding's 0 OUTPUT rate: that is a real price,
 * not a missing one.
 */
function isUsable(k, v) {
  // Number.isFinite (not typeof === "number") so a NaN, or a huge upstream value
  // that JSON.parse turns into Infinity, is treated as unpriceable and dropped —
  // matching the fail-closed pricing paths.
  return (
    !!v &&
    Number.isFinite(v.input_cost_per_token) &&
    v.input_cost_per_token > 0 &&
    isPricedProvider(k, v) &&
    (isEmbeddingModel(vendoredKey(k), v) ||
      (v.mode === "chat" &&
        Number.isFinite(v.output_cost_per_token) &&
        v.output_cost_per_token > 0))
  );
}

const res = await fetch(SOURCE_URL);
if (!res.ok) {
  throw new Error(`Failed to fetch LiteLLM cost map: HTTP ${res.status}`);
}
const raw = await res.json();

const entries = Object.entries(raw)
  .filter(([k, v]) => isUsable(k, v))
  .map(([k, v]) => [vendoredKey(k), v])
  .sort(([a], [b]) => a.localeCompare(b));

// A curated Groq model that upstream dropped (or stopped fully pricing) would
// otherwise vanish from the vendored map with no signal — the refresh PR diff
// would just show a deletion. Warn so the reviewer knows the allowlist entry
// stopped resolving (expected once Groq's shutdown dates pass; see above).
const vendored = new Set(entries.map(([k]) => k));
for (const [src, dest] of GROQ_KEY_MAP) {
  if (!vendored.has(dest)) {
    console.warn(
      `WARNING: curated Groq model ${src} is missing or unpriceable upstream — dropped from the vendored map.`,
    );
  }
}

// The other half of EMBEDDING_ID_PREFIXES: a genuine embedding model from a
// priced provider whose id doesn't match a known family is dropped (fail-closed,
// the safe direction) — but silently, so a new Google/OpenAI embedding line would
// just never appear. Warn so the reviewer knows to extend the prefix list.
for (const [k, v] of Object.entries(raw)) {
  const key = vendoredKey(k);
  if (
    v?.mode === "embedding" &&
    isPricedProvider(k, v) &&
    !isEmbeddingModel(key, v)
  ) {
    console.warn(
      `WARNING: ${k} declares mode="embedding" but ${key} matches no known embedding ` +
        `family — dropped. Expected when upstream mislabels a chat model; if it ` +
        `really is an embedding, add its family to EMBEDDING_ID_PREFIXES.`,
    );
  }
}

// null-prototype: model keys come from remote JSON, so a "__proto__" (or similar)
// key is stored as plain data instead of mutating the object's prototype.
const out = Object.create(null);
for (const [k, v] of entries) {
  const entry = {
    input_cost_per_token: v.input_cost_per_token,
    // Same predicate as the filter — `k` is already the vendored key here. Zeroing
    // on raw `v.mode` would re-introduce the free-output hole for any entry whose
    // declared mode and id disagree.
    output_cost_per_token: isEmbeddingModel(k, v) ? 0 : v.output_cost_per_token,
    litellm_provider: v.litellm_provider,
    mode: v.mode,
  };
  const existing = out[k];
  if (existing === undefined) {
    out[k] = entry;
    continue;
  }
  // Two upstream keys collapsed onto one vendored key — upstream lists several
  // Gemini models BOTH bare and "gemini/"-prefixed. Plain assignment would let
  // the last one silently win, so resolve deterministically toward the dearer
  // rate PER BUCKET: picking one whole entry by total cost still under-meters a
  // prompt/completion mix whenever one duplicate has the higher input rate and
  // the other the higher output rate. Over-pricing stops the agent one call
  // early (safe); under-pricing lets a crossing call through (the failure this
  // package exists to prevent).
  const input_cost_per_token = Math.max(
    existing.input_cost_per_token,
    entry.input_cost_per_token,
  );
  const output_cost_per_token = Math.max(
    existing.output_cost_per_token,
    entry.output_cost_per_token,
  );
  const merged = {
    input_cost_per_token,
    output_cost_per_token,
    litellm_provider: existing.litellm_provider,
    // Both sides already passed isUsable, so a 0 output rate here means both were
    // VERIFIED embeddings. Anything else took a chat rate on at least one bucket
    // and must not keep a mode that reads as "output is free".
    mode: output_cost_per_token === 0 ? "embedding" : "chat",
  };
  out[k] = merged;
  console.warn(
    `NOTE: ${k} is listed more than once upstream — kept the dearer rate in each ` +
      `bucket (${input_cost_per_token}/${output_cost_per_token}).`,
  );
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
  // Object.keys(out), not entries.length: collapsed duplicate keys (see the
  // collision note above) mean fewer models are vendored than entries survived.
  `Wrote ${Object.keys(out).length} models to ${targets.length} files (source: ${SOURCE_URL}).`,
);
