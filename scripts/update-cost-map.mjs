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

// ── Pinned must-keep models ──────────────────────────────────────────────
// Models we always ship even if LiteLLM stops listing them at their real
// provider. Upstream dropped the claude-3-5 family from `litellm_provider:
// "anthropic"` (refresh 2026-08-24), which nulled them out of the map and broke
// pricing for the many users still on those models. Seeded into the LiteLLM
// data below so they flow through the same isUsable filter, dedup, and sort as
// everything else; a live upstream entry still wins (the seed only fills a gap).
// Prices are Anthropic public list rates.
const PINNED_MODELS = {
  "claude-3-5-sonnet-20241022": {
    input_cost_per_token: 0.000003,
    output_cost_per_token: 0.000015,
    litellm_provider: "anthropic",
    mode: "chat",
  },
  "claude-3-5-sonnet-20240620": {
    input_cost_per_token: 0.000003,
    output_cost_per_token: 0.000015,
    litellm_provider: "anthropic",
    mode: "chat",
  },
  "claude-3-5-haiku-20241022": {
    input_cost_per_token: 0.0000008,
    output_cost_per_token: 0.000004,
    litellm_provider: "anthropic",
    mode: "chat",
  },
};

// ── Per-unit leg rates (STT / TTS / telephony / SMS / OCR / GPU / avatar) ────
//
// The token map above is fetched from LiteLLM; this one is NOT — LiteLLM does
// not carry list prices for any of these legs, so they are hand-curated from
// each vendor's public pricing page and injected under the reserved "__legs__"
// key (see src/floe_guard/pricing.py, which splits it back out so the token
// resolver never sees it). They are a DRIFT-PRONE SNAPSHOT: vendors change these
// far more often than this file is refreshed, so treat them as an estimate and
// re-verify against the live pricing page before trusting a figure. Telephony
// and SMS are US-only in v1.
//
// The section was called "__voice__" until P1.11. The mechanism was never
// voice-specific — it prices any leg billed in a unit other than tokens — and
// once it grew SMS, OCR, GPU and avatar rates the old name was actively
// misleading. pricing.py reads "__legs__" first and falls back to "__voice__",
// so the rename is non-breaking for a map generated before it.
//
// The line between this map and the token map is the BILLING UNIT, not the
// modality: per-token spend lives in the flat model map, per-anything-else lives
// here. LLM inference is therefore NOT a leg, while a rented GPU-second is.
//
// TIERS ARE SEPARATE KEYS. Where a vendor publishes volume or plan tiers, each
// tier ships as its own entry (e.g. "...-high-volume") so choosing one is a
// choice of KEY by the caller, never a curation guess baked into a single
// number. The bundled rate is a starting point the user confirms or replaces
// with their own rate card — it is not an assertion about their bill.
//
// EVERY ENTRY CITES A SOURCE. Each block below names the public price-list URL
// it was read from and the date it was read. An entry with no public source does
// not ship — see the TODO(...) notes, which are deliberately left unpriced
// rather than guessed. A guessed rate is worse than no rate: an unpriced leg
// fails closed and is visible, a wrong one silently mis-bills.
//
// Units are canonical per mode and a mismatch fails closed downstream:
//   stt       -> usd_per_second       tts    -> usd_per_1k_chars
//   telephony -> usd_per_minute       sms    -> usd_per_segment
//   ocr       -> usd_per_page         gpu    -> usd_per_gpu_second
//   avatar    -> usd_per_minute
// Where a vendor lists a different unit, the arithmetic to convert is shown.
const VOICE_RATES = {
  // Deepgram Nova-3 streaming, billed per minute -> ÷60 for $/sec.
  "deepgram-nova-3": {
    mode: "stt",
    unit: "usd_per_second",
    rate: 0.0001283333, // $0.0077/min mono ÷ 60
    provider: "deepgram",
  },
  "deepgram-nova-3-multilingual": {
    mode: "stt",
    unit: "usd_per_second",
    rate: 0.0001533333, // $0.0092/min ÷ 60
    provider: "deepgram",
  },
  "deepgram-nova-3-base": {
    mode: "stt",
    unit: "usd_per_second",
    rate: 0.0002416667, // $0.0145/min ÷ 60
    provider: "deepgram",
  },
  // AssemblyAI Universal Streaming. Note: ~$0.0042/min effective once you
  // include the per-session idle/overhead billing; the list rate is $0.0025/min.
  "assemblyai-universal-streaming": {
    mode: "stt",
    unit: "usd_per_second",
    rate: 0.0000416667, // $0.0025/min ÷ 60
    provider: "assemblyai",
  },
  // ElevenLabs bills in credits/char: Multilingual v2 = 1 credit/char,
  // Flash/Turbo = 0.5 credit/char. On the standard $/credit these list at
  // ~$0.10 and ~$0.05 per 1k chars respectively.
  "elevenlabs-multilingual-v2": {
    mode: "tts",
    unit: "usd_per_1k_chars",
    rate: 0.1, // 1 credit/char
    provider: "elevenlabs",
  },
  "elevenlabs-flash-v2.5": {
    mode: "tts",
    unit: "usd_per_1k_chars",
    rate: 0.05, // 0.5 credit/char
    provider: "elevenlabs",
  },
  "elevenlabs-turbo-v2.5": {
    mode: "tts",
    unit: "usd_per_1k_chars",
    rate: 0.05, // 0.5 credit/char
    provider: "elevenlabs",
  },
  // Cartesia is billed per minute of audio; converted to $/1k-chars at an
  // assumed 1000 chars/min of synthesised speech (≈150 wpm). Sonic ≈ $0.03/min;
  // the phone-optimised "Line" product ≈ $0.06/min.
  "cartesia-sonic": {
    mode: "tts",
    unit: "usd_per_1k_chars",
    rate: 0.03, // $0.03/min ÷ 1000 chars/min = $0.03/1k chars
    provider: "cartesia",
  },
  "cartesia-line": {
    mode: "telephony",
    unit: "usd_per_minute",
    rate: 0.06, // Cartesia Line — telephony, $0.06/min list rate (per-minute)
    provider: "cartesia",
  },
  // Rime is billed per minute; converted at the same 1000 chars/min assumption.
  "rime-mist-v2": {
    mode: "tts",
    unit: "usd_per_1k_chars",
    rate: 0.03, // $0.030/min ÷ 1000 chars/min = $0.03/1k chars
    provider: "rime",
  },
  // Twilio Programmable Voice, US only. Outbound lists as a $0.013–0.014/min
  // range; we keep the top of the range because over-pricing a spend guard
  // stops one call early (safe) while under-pricing lets a crossing call through.
  "twilio-us-inbound-local": {
    mode: "telephony",
    unit: "usd_per_minute",
    rate: 0.0085, // US inbound to a local number
    provider: "twilio",
  },
  "twilio-us-outbound-local": {
    mode: "telephony",
    unit: "usd_per_minute",
    rate: 0.014, // $0.013–0.014/min — top of range
    provider: "twilio",
  },
  "twilio-us-sip-inbound": {
    mode: "telephony",
    unit: "usd_per_minute",
    rate: 0.004, // US SIP inbound
    provider: "twilio",
  },
  // TODO(telnyx): unverified TELEPHONY list rate — deferred. Do not invent a
  // number; add a telnyx-us-* telephony entry only after confirming the current
  // per-minute list price. (Telnyx SMS below IS sourced and does ship.)

  // ── SMS (per segment, US) ─────────────────────────────────────────────────
  // Billed per SEGMENT, not per message: GSM-7 splits at 153 chars per segment
  // for concatenated messages (67 for UCS-2), so one "message" can be several
  // billable units. The caller passes segment count, not message count.
  //
  // These are the base list rates and EXCLUDE US carrier passthrough fees
  // ($0.0025–$0.007/segment depending on carrier and number type), which neither
  // vendor includes in the headline number. A real invoice will therefore exceed
  // this; the guard under-states SMS spend by the carrier fee. Documented rather
  // than fudged, because inventing a blended number would be a fabricated price.
  //
  // Source: https://www.twilio.com/en-us/sms/pricing/us — retrieved 2026-09-18
  // ("$0.0083 per outbound/inbound message, plus carrier fees").
  "twilio-sms-us-outbound": {
    mode: "sms",
    unit: "usd_per_segment",
    rate: 0.0083, // $0.0083/segment list, ex carrier fees
    provider: "twilio",
  },
  "twilio-sms-us-inbound": {
    mode: "sms",
    unit: "usd_per_segment",
    rate: 0.0083, // Twilio lists the same rate both directions
    provider: "twilio",
  },
  // Source: https://telnyx.com/pricing/messaging — retrieved 2026-09-18
  // ("$0.004 per message part" outbound and inbound, plus carrier fees).
  "telnyx-sms-us-outbound": {
    mode: "sms",
    unit: "usd_per_segment",
    rate: 0.004, // $0.004/message part list, ex carrier fees
    provider: "telnyx",
  },
  "telnyx-sms-us-inbound": {
    mode: "sms",
    unit: "usd_per_segment",
    rate: 0.004,
    provider: "telnyx",
  },

  // ── GPU (per GPU-second, per accelerator class) ───────────────────────────
  // Modal bills per second natively, so these are list rates with no conversion.
  // Source: https://modal.com/pricing — retrieved 2026-09-18.
  "modal-b300": { mode: "gpu", unit: "usd_per_gpu_second", rate: 0.001972, provider: "modal" },
  "modal-b200": { mode: "gpu", unit: "usd_per_gpu_second", rate: 0.001736, provider: "modal" },
  "modal-h200-sxm": { mode: "gpu", unit: "usd_per_gpu_second", rate: 0.001261, provider: "modal" },
  "modal-h100-sxm5": { mode: "gpu", unit: "usd_per_gpu_second", rate: 0.001097, provider: "modal" },
  "modal-rtx-pro-6000": {
    mode: "gpu",
    unit: "usd_per_gpu_second",
    rate: 0.000842,
    provider: "modal",
  },
  "modal-a100-80gb": { mode: "gpu", unit: "usd_per_gpu_second", rate: 0.000694, provider: "modal" },
  "modal-a100-40gb": { mode: "gpu", unit: "usd_per_gpu_second", rate: 0.000583, provider: "modal" },
  "modal-l40s": { mode: "gpu", unit: "usd_per_gpu_second", rate: 0.000542, provider: "modal" },
  "modal-a10": { mode: "gpu", unit: "usd_per_gpu_second", rate: 0.000306, provider: "modal" },
  "modal-l4": { mode: "gpu", unit: "usd_per_gpu_second", rate: 0.000222, provider: "modal" },
  "modal-t4": { mode: "gpu", unit: "usd_per_gpu_second", rate: 0.000164, provider: "modal" },
  // TODO(runpod): deliberately NOT vendored. Runpod quotes per hour across two
  // tiers (on-demand Pods vs the dearer serverless), and the bundled map cannot
  // know which one a caller is on — vendoring either would be a curation guess
  // about someone else's bill. A Runpod user adds it through their own rate
  // card, which is the mechanism for every vendor-specific or negotiated rate.

  // ── OCR (per page) ────────────────────────────────────────────────────────
  // Vendors quote per 1,000 pages; stored per PAGE (÷1000) so the caller passes
  // a page count and never has to remember the thousand-factor. Getting this
  // wrong is a 1000x mis-bill, which is why the unit is explicit and checked.
  //
  // Both vendors are volume-tiered, and EVERY tier ships as its own key. The map
  // cannot know what monthly volume a caller is at, so picking one tier for them
  // would be a guess about their bill; picking the key is theirs to make, and a
  // negotiated rate goes in their rate card instead.
  //
  // Source: https://cloud.google.com/vision/pricing — retrieved 2026-09-18
  // ($1.50 per 1,000 units for 1,001–5,000,000/month; $0.60 above 5M; first
  // 1,000/month free).
  "gcp-vision-text-detection": {
    mode: "ocr",
    unit: "usd_per_page",
    rate: 0.0015, // $1.50/1k pages ÷ 1000
    provider: "gcp",
  },
  "gcp-vision-document-text-detection": {
    mode: "ocr",
    unit: "usd_per_page",
    rate: 0.0015, // same tier pricing as TEXT_DETECTION
    provider: "gcp",
  },
  "gcp-vision-text-detection-high-volume": {
    mode: "ocr",
    unit: "usd_per_page",
    rate: 0.0006, // $0.60/1k pages ÷ 1000, above 5,000,000 units/month
    provider: "gcp",
  },
  "gcp-vision-document-text-detection-high-volume": {
    mode: "ocr",
    unit: "usd_per_page",
    rate: 0.0006, // $0.60/1k pages ÷ 1000, above 5,000,000 units/month
    provider: "gcp",
  },
  // Source: https://aws.amazon.com/textract/pricing/ — retrieved 2026-09-18
  // (US West (Oregon), the region the page prints: DetectDocumentText $0.0015/pg
  // for the first 1M pages/month, $0.0006/pg above; Forms $0.05/pg; Tables
  // $0.015/pg). Textract is priced per region — a caller outside us-west-2 may
  // pay a different rate.
  "aws-textract-detect-document-text": {
    mode: "ocr",
    unit: "usd_per_page",
    rate: 0.0015, // $1.50/1k pages, first 1M/month
    provider: "aws",
  },
  "aws-textract-detect-document-text-high-volume": {
    mode: "ocr",
    unit: "usd_per_page",
    rate: 0.0006, // $0.60/1k pages, above 1M pages/month
    provider: "aws",
  },
  "aws-textract-analyze-forms": {
    mode: "ocr",
    unit: "usd_per_page",
    rate: 0.05, // $50/1k pages
    provider: "aws",
  },
  "aws-textract-analyze-tables": {
    mode: "ocr",
    unit: "usd_per_page",
    rate: 0.015, // $15/1k pages
    provider: "aws",
  },
  // TODO(mistral-ocr): Mistral's pricing page states "OCR is per 1,000 pages"
  // but publishes no figure. Unpriced until a public number exists.
  // TODO(azure-document-intelligence): Azure's pricing page renders every
  // Document Intelligence rate as "$-" and defers to the signed-in calculator.
  // Unpriced until a public number exists.

  // ── Avatar (per minute of generated video) ────────────────────────────────
  // An avatar bills per minute of video and no voice framework emits a metric
  // for it, so the caller derives minutes from call duration. This is the entry
  // the livekit-full-bill-reconcile cookbook resolves instead of asking the
  // operator to hand-set a rate.
  //
  // Tavus is plan-tiered and each plan ships as its own key — no bare "tavus-cvi"
  // implying a default, because which plan you are on is a fact about you, not
  // about Tavus.
  //
  // CAVEAT the linear model cannot express: usage rounds to 6s with a 30s minimum
  // per conversation, so a short call bills MORE than duration x rate. A shop
  // with many brief calls should put its own effective rate in a rate card.
  // Source: https://www.tavus.io/pricing — retrieved 2026-09-18.
  "tavus-cvi-starter": {
    mode: "avatar",
    unit: "usd_per_minute",
    rate: 0.37, // Starter overage, beyond 100 included min/month
    provider: "tavus",
  },
  "tavus-cvi-growth": {
    mode: "avatar",
    unit: "usd_per_minute",
    rate: 0.32, // Growth overage, beyond 1,250 included min/month
    provider: "tavus",
  },
  "tavus-cvi-business": {
    mode: "avatar",
    unit: "usd_per_minute",
    rate: 0.26, // Business overage, beyond 4,000 included min/month
    provider: "tavus",
  },
  // TODO(heygen): publishes only a monthly credit bundle, no per-minute rate.
  // TODO(simli): pricing page is a 404 as of 2026-09-18.
  // TODO(beyond-presence): no public per-minute list price found.
};

// ── Provenance stamping ─────────────────────────────────────────────────────
//
// Every rate curated in P1.11 carries the URL it was read from and the date it
// was read, so a resolved rate can tell the caller where its number came from
// and how stale it is. Applied as a loop rather than two extra lines on each of
// 26 literals: the SOURCE_BY_PROVIDER table is then the single place a URL is
// written, and it cannot disagree with itself.
//
// The 13 pre-P1.11 voice rates are deliberately NOT stamped. There is no
// verified URL on record for them, and back-filling a plausible-looking one
// would launder a guess into a citation. Absent provenance means UNVERIFIED and
// the resolver reports it as such.
const LEG_RATE_RETRIEVED_AT = "2026-09-18";
const SOURCE_BY_PROVIDER = {
  twilio: "https://www.twilio.com/en-us/sms/pricing/us",
  telnyx: "https://telnyx.com/pricing/messaging",
  modal: "https://modal.com/pricing",
  gcp: "https://cloud.google.com/vision/pricing",
  aws: "https://aws.amazon.com/textract/pricing/",
  tavus: "https://www.tavus.io/pricing",
};
const SOURCED_IN_P111 = new Set([
  "twilio-sms-us-outbound",
  "twilio-sms-us-inbound",
  "telnyx-sms-us-outbound",
  "telnyx-sms-us-inbound",
  "modal-b300",
  "modal-b200",
  "modal-h200-sxm",
  "modal-h100-sxm5",
  "modal-rtx-pro-6000",
  "modal-a100-80gb",
  "modal-a100-40gb",
  "modal-l40s",
  "modal-a10",
  "modal-l4",
  "modal-t4",
  "gcp-vision-text-detection",
  "gcp-vision-document-text-detection",
  "gcp-vision-text-detection-high-volume",
  "gcp-vision-document-text-detection-high-volume",
  "aws-textract-detect-document-text",
  "aws-textract-detect-document-text-high-volume",
  "aws-textract-analyze-forms",
  "aws-textract-analyze-tables",
  "tavus-cvi-starter",
  "tavus-cvi-growth",
  "tavus-cvi-business",
]);
for (const [k, v] of Object.entries(VOICE_RATES)) {
  if (!SOURCED_IN_P111.has(k)) continue;
  const url = SOURCE_BY_PROVIDER[v.provider];
  if (!url) throw new Error(`No source URL for provider ${v.provider} (entry ${k})`);
  VOICE_RATES[k] = { ...v, source_url: url, retrieved_at: LEG_RATE_RETRIEVED_AT };
}

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

// Seed pinned must-keep models when upstream has dropped them; a live upstream
// entry wins. They then flow through isUsable/dedup/sort like any other model.
for (const [k, v] of Object.entries(PINNED_MODELS)) {
  if (!(k in raw)) raw[k] = v;
}

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
// Provenance/freshness of this snapshot. pricing.py / pricing.ts split this
// reserved key back out (like __voice__), so it never reaches the token resolver;
// it's surfaced via cost_map_generated_at() / costMapGeneratedAt().
out.__meta__ = {
  generated_at: new Date().toISOString().slice(0, 10),
  source:
    "LiteLLM public model prices (bundled snapshot); per-unit leg rates " +
    "(stt/tts/telephony/sms/ocr/gpu/avatar) hand-curated from vendor list pages, " +
    `leg rates last retrieved ${LEG_RATE_RETRIEVED_AT}`,
};
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
  // Per-model prompt-cache rates, when upstream publishes them and they are
  // finite and positive (mirrors the input/output finiteness handling). Kept
  // optional so a model with no published cache rate simply omits them, and
  // pricing falls back to a single conservative multiplier (currently Anthropic's
  // ~0.1x read ratio, applied to all providers) — see _CACHE_READ_MULTIPLIER.
  if (
    Number.isFinite(v.cache_read_input_token_cost) &&
    v.cache_read_input_token_cost > 0
  ) {
    entry.cache_read_input_token_cost = v.cache_read_input_token_cost;
  }
  if (
    Number.isFinite(v.cache_creation_input_token_cost) &&
    v.cache_creation_input_token_cost > 0
  ) {
    entry.cache_creation_input_token_cost = v.cache_creation_input_token_cost;
  }
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
  // Merge cache rates toward the dearer rate too (Math.max), matching the
  // under-metering-averse merge above. A rate present on only one side wins as-is;
  // absent on both, the field stays omitted.
  const cacheRead = Math.max(
    existing.cache_read_input_token_cost ?? -Infinity,
    entry.cache_read_input_token_cost ?? -Infinity,
  );
  if (Number.isFinite(cacheRead)) merged.cache_read_input_token_cost = cacheRead;
  const cacheCreation = Math.max(
    existing.cache_creation_input_token_cost ?? -Infinity,
    entry.cache_creation_input_token_cost ?? -Infinity,
  );
  if (Number.isFinite(cacheCreation)) merged.cache_creation_input_token_cost = cacheCreation;
  out[k] = merged;
  console.warn(
    `NOTE: ${k} is listed more than once upstream — kept the dearer rate in each ` +
      `bucket (${input_cost_per_token}/${output_cost_per_token}).`,
  );
}

// Inject the hand-curated per-unit leg rates under the reserved "__legs__" key,
// last, so a token refresh preserves them (they are not in the fetched LiteLLM
// data). pricing.py splits this key back out, so it never reaches the token
// resolver. Renamed from "__voice__" in P1.11; the loaders read the new name
// first and fall back to the old one, so an older vendored map still resolves.
out.__legs__ = VOICE_RATES;

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
