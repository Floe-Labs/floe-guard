/**
 * User-supplied rate card — the rates YOU actually pay.
 *
 * The bundled cost map is a snapshot of each vendor's PUBLIC LIST price. Almost
 * nobody pays list: volume tiers, negotiated contracts, committed-spend
 * discounts, carrier passthrough fees and plan minimums all move the real
 * number. A COGS tool that reports list prices as if they were your costs is
 * guessing, so the map is a starting point the user confirms or replaces — never
 * an assertion about a bill.
 *
 * Resolution order (see `voice-pricing.ts`):
 *
 *     per-call override -> rate card -> bundled cost map -> UnpriceableLegError
 *
 * A rate card can both OVERRIDE a bundled vendor and ADD one the map has never
 * heard of, which is how vendors that publish no usable figure (Mistral OCR,
 * Azure Document Intelligence, HeyGen, Simli, Beyond Presence) get priced.
 *
 * Entries use the same schema as the bundled `__legs__` section:
 *
 *     { "acme-ocr": { "mode": "ocr", "unit": "usd_per_page", "rate": 0.0009 } }
 *
 * `confirmed` is the difference between a number someone checked and a number we
 * guessed. Bundled rates resolve as `confirmed: false` (list price, unverified);
 * rate-card rates default to `confirmed: true` because you typed them.
 *
 * DIFFERENCE FROM THE PYTHON PACKAGE: `FLOE_RATE_CARD` here holds inline JSON,
 * not a file path. This package must work in runtimes with no filesystem, and
 * reading a file would force `node:fs` and make rate resolution async — which
 * would turn every synchronous pricing call in the SDK inside out. Pass an
 * object to {@link setRateCard} for anything more elaborate.
 */

/** Environment variable holding rate-card JSON (inline, not a path). */
export const RATE_CARD_ENV = "FLOE_RATE_CARD";

/** One user-declared rate. Mirrors the bundled `__legs__` entry schema. */
export interface RateCardEntry {
  mode: string;
  unit: string;
  rate: number;
  provider?: string;
  source_url?: string;
  retrieved_at?: string;
  confirmed?: boolean;
}

// The only keys an entry may carry. An unknown key is rejected rather than
// ignored: a typo'd "price" would otherwise leave the real rate undefined and
// silently fall through to a list price.
const ALLOWED_ENTRY_KEYS = new Set([
  "mode",
  "unit",
  "rate",
  "provider",
  "source_url",
  "retrieved_at",
  "confirmed",
]);
const REQUIRED_ENTRY_KEYS = ["mode", "unit", "rate"] as const;

let RATE_CARD: Record<string, RateCardEntry> = {};

function envValue(name: string): string | undefined {
  // The package compiles with `types: []` (no @types/node), so `process` is not
  // globally typed and may not exist at all in a browser.
  const p = (globalThis as { process?: { env?: Record<string, string | undefined> } }).process;
  return p?.env?.[name];
}

/**
 * Reject a malformed entry loudly. Shape only.
 *
 * The mode/unit CONSISTENCY check (`ocr` must be `usd_per_page` etc.) lives in
 * `voice-pricing.ts`, which owns that table — checking it here would make this
 * module import the thing that imports it.
 */
function validateEntry(key: string, entry: unknown): asserts entry is RateCardEntry {
  if (typeof entry !== "object" || entry === null || Array.isArray(entry)) {
    throw new TypeError(`Rate card entry '${key}' must be an object.`);
  }
  const e = entry as Record<string, unknown>;
  const extra = Object.keys(e)
    .filter((k) => !ALLOWED_ENTRY_KEYS.has(k))
    .sort();
  if (extra.length) {
    throw new TypeError(
      `Rate card entry '${key}' has unknown field(s): ${extra.join(", ")}. ` +
        `Allowed: ${[...ALLOWED_ENTRY_KEYS].sort().join(", ")}.`,
    );
  }
  const missing = REQUIRED_ENTRY_KEYS.filter((k) => !(k in e));
  if (missing.length) {
    throw new TypeError(`Rate card entry '${key}' is missing required field(s): ${missing.join(", ")}.`);
  }
  for (const field of ["mode", "unit"] as const) {
    if (typeof e[field] !== "string" || !e[field]) {
      throw new TypeError(`Rate card entry '${key}': ${field} must be a non-empty string.`);
    }
  }
  const rate = e.rate;
  if (typeof rate !== "number" || !Number.isFinite(rate) || rate < 0) {
    throw new RangeError(
      `Rate card entry '${key}': rate must be a finite, non-negative number, got ${String(rate)}.`,
    );
  }
  if ("confirmed" in e && typeof e.confirmed !== "boolean") {
    throw new TypeError(`Rate card entry '${key}': confirmed must be true or false.`);
  }
  for (const field of ["provider", "source_url", "retrieved_at"] as const) {
    if (field in e && typeof e[field] !== "string") {
      throw new TypeError(`Rate card entry '${key}': ${field} must be a string.`);
    }
  }
}

/**
 * Read and validate a rate card from an object or from JSON text. When omitted,
 * `$FLOE_RATE_CARD` is consulted; unset yields an empty card (the zero-config
 * default). Throws on unreadable or malformed input — deliberately loud, because
 * silently falling back to list prices produces confident, wrong COGS.
 */
export function loadRateCard(source?: unknown): Record<string, RateCardEntry> {
  let input = source;
  if (input === undefined || input === null) {
    const fromEnv = envValue(RATE_CARD_ENV);
    if (!fromEnv) return {};
    input = fromEnv;
  }
  let raw: unknown;
  if (typeof input === "string") {
    try {
      raw = JSON.parse(input);
    } catch (e) {
      throw new TypeError(
        `Rate card is not valid JSON: ${(e as Error).message}. ` +
          `${RATE_CARD_ENV} holds inline JSON in this package, not a file path.`,
      );
    }
  } else {
    raw = input;
  }
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
    throw new TypeError("Rate card must be a JSON object mapping vendor keys to entries.");
  }
  const out: Record<string, RateCardEntry> = {};
  for (const [key, entry] of Object.entries(raw as Record<string, unknown>)) {
    validateEntry(key, entry);
    out[key] = entry;
  }
  return out;
}

/** Install a rate card for this process and return it. `{}` clears it. */
export function setRateCard(source?: unknown): Record<string, RateCardEntry> {
  RATE_CARD = loadRateCard(source);
  return RATE_CARD;
}

/** The rate card in force. Empty when none is configured. */
export function currentRateCard(): Record<string, RateCardEntry> {
  return RATE_CARD;
}

// Load once at module init so `FLOE_RATE_CARD=... node agent.js` needs no code
// change. A broken card throws here rather than at the first priced leg, which
// is the difference between a startup failure and a wrong invoice.
RATE_CARD = loadRateCard();
