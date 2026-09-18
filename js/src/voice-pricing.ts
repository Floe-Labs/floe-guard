/**
 * Offline pricing for per-unit legs from the vendored map.
 *
 * The per-unit twin of `pricing.ts`. The token cost map is per-token; these legs
 * are billed in anything but tokens — per second (STT), per 1k chars (TTS), per
 * minute (telephony, avatar), per segment (SMS), per page (OCR) and per
 * GPU-second (GPU) — so their rates live under the reserved `"__legs__"` key of
 * `cost_map.json` with their own schema:
 *
 *     "deepgram-nova-3": {
 *       "mode": "stt",              // discriminator: see LegMode
 *       "unit": "usd_per_second",   // explicit unit — a mismatch fails closed
 *       "rate": 0.0001283333,       // USD in that unit
 *       "provider": "deepgram"
 *     }
 *
 * The section was called `"__voice__"` until P1.11. The mechanism was never
 * voice-specific, and once it grew SMS, OCR, GPU and avatar rates the old name
 * was actively misleading — so it is now `"__legs__"`. The loader reads the new
 * name first and falls back to the old one, which is what makes the rename
 * non-breaking for a `cost_map.json` generated before it.
 *
 * The line between this map and the token map is the BILLING UNIT, not the
 * modality: per-token spend lives in the flat model map, per-anything-else lives
 * here. LLM inference is therefore NOT a leg, while a rented GPU-second is.
 *
 * Same fail-closed contract as token pricing: a vendor absent from the map (or an
 * entry whose `unit`/`mode` does not match the leg it is asked to price) is
 * unpriceable — {@link lookupVoiceRate} returns `null` and the adapter throws
 * {@link UnpriceableVoiceError} rather than silently metering the leg at $0. Pass a
 * per-unit override to enforce a leg the map cannot price.
 *
 * No network. The rates are a **drift-prone snapshot** of each vendor's public
 * list price — refresh them (`scripts/update-cost-map.mjs`, which names the
 * source URL and retrieval date for every entry) like the token map, or
 * estimates drift as vendors change prices. Telephony and SMS are **US-only in
 * v1**, and several rates are volume- or region-tiered: the map vendors the
 * DEAREST public tier, because over-pricing a spend guard stops one call early
 * while under-pricing lets a crossing call through.
 *
 * This is a faithful port of `src/floe_guard/voice_pricing.py`.
 */

import costMapJson from "./cost_map.json";
import { UnpriceableVoiceError } from "./errors.js";
import { LEG_MODES, unitForMode, type LegMode } from "./leg-units.js";
import { currentRateCard } from "./rate-card.js";

/**
 * The legs the bundled map can price per unit.
 *
 * Named for the leg rather than for voice — the pricing mechanism is not
 * voice-specific — but the MEMBERS are deliberately unchanged: each one needs a
 * unit in `unitForMode` and an entry shape in the cost map, so widening this is
 * a pricing change, not a rename. {@link VoiceMode} remains as a deprecated
 * alias.
 */
export type { LegMode };

/**
 * Deprecated alias for {@link LegMode}. Prefer the leg-shaped name. It now spans
 * legs that are not voice at all; the name is kept only for compatibility.
 */
export type VoiceMode = LegMode;

interface VoiceMapEntry {
  mode?: unknown;
  unit?: unknown;
  rate?: unknown;
  provider?: unknown;
  // Provenance, present only on rates curated with a citation (P1.11 onward).
  // Absent means UNVERIFIED — see VoiceRate.
  source_url?: unknown;
  retrieved_at?: unknown;
  confirmed?: unknown;
}

/**
 * The vendored per-unit leg rates — the reserved `"__legs__"` section of the
 * cost map.
 *
 * Renamed from `"__voice__"` in P1.11, when the section grew SMS, OCR, GPU and
 * avatar rates and the old name became actively misleading. Reading new-name
 * first and old-name second is what makes the rename non-breaking: a
 * cost_map.json generated before the rename still resolves, so an older vendored
 * map (or a user's pinned copy) keeps working untouched.
 */
const rawCostMap = costMapJson as Record<string, unknown>;
const VOICE_MAP = (rawCostMap["__legs__"] ?? rawCostMap["__voice__"] ?? {}) as Record<
  string,
  VoiceMapEntry
>;

// Re-exported from leg-units, which owns the table. It lives there so the
// rate-card validator can check mode/unit consistency without importing this
// module, which imports IT. See leg-units for why that cycle mattered.
export { unitForMode, LEG_MODES };

/**
 * A resolved per-unit leg rate plus where it came from.
 *
 * The provenance fields are the difference between a number and an auditable
 * number. A COGS figure built on an unverified list price should be presentable
 * as exactly that — "$1.50/1k pages, Google's list page, read 2026-09-18, not
 * yet confirmed by you" — so a console can ask for confirmation instead of
 * quietly reporting a guess as a cost.
 */
export interface VoiceRate {
  mode: string;
  unit: string;
  rate: number;
  /**
   * Where the number came from, cheapest-trust first: `cost_map` (public list
   * price) < `rate_card` (you declared it) < `override` (passed at the call
   * site). A later vendor-invoice read supersedes all of them at reconcile time,
   * and the gap between the estimate and that actual is the variance signal.
   */
  source: "override" | "rate_card" | "cost_map";
  provider?: string;
  /**
   * Public price-list URL and the date it was read. Present only on rates
   * curated with a citation; absent means UNVERIFIED and should be shown as such
   * rather than rendered as a blank.
   */
  source_url?: string;
  retrieved_at?: string;
  /**
   * Has a human accepted this number as what they actually pay? Bundled list
   * prices are `false` until confirmed; a rate you declared is `true`.
   */
  confirmed: boolean;
}

function finiteNonNegative(value: unknown): value is number {
  // typeof excludes booleans (Python's _finite_non_negative excludes bool too).
  return typeof value === "number" && Number.isFinite(value) && value >= 0;
}

/**
 * Resolve a voice vendor to its per-unit rate, or `null` if unpriceable.
 *
 * The pure resolver (the voice twin of {@link resolvePrice}): it never throws.
 * Fail-closed on every mismatch — a missing vendor, an entry whose `mode` is not
 * `mode`, an entry whose `unit` is not the canonical unit for `mode`, or a
 * non-finite/negative rate — so a schema slip can never surface as a silent,
 * valid-looking price.
 */
export function lookupVoiceRate(model: string | null | undefined, mode: VoiceMode): number | null {
  const found = lookupLegEntry(model, mode);
  return found === null ? null : (found.entry.rate as number);
}

/** Fail-closed on every mismatch: wrong mode, wrong unit, or a bad rate. */
function entryIsUsable(entry: unknown, mode: VoiceMode): entry is VoiceMapEntry {
  if (typeof entry !== "object" || entry === null) return false;
  const e = entry as VoiceMapEntry;
  if (e.mode !== mode) return false;
  if (e.unit !== unitForMode[mode]) return false;
  return finiteNonNegative(e.rate);
}

/**
 * The entry that prices this leg and which source it came from.
 *
 * Rate card first, bundled map second — your negotiated rate beats a list price.
 *
 * One deliberate asymmetry: if the rate card declares this exact vendor AND this
 * exact leg but the declaration is broken (wrong unit, bad rate), this returns
 * `null` instead of falling through to the bundled price. You said what this leg
 * costs you; silently substituting a list price you never chose would be a
 * confident wrong number, which is the failure this module exists to prevent. A
 * card entry for a DIFFERENT mode is not a declaration about this leg, so that
 * falls through normally.
 */
function lookupLegEntry(
  model: string | null | undefined,
  mode: VoiceMode,
): { entry: VoiceMapEntry; source: "rate_card" | "cost_map" } | null {
  if (model === null || model === undefined) return null;
  const card = currentRateCard();
  const declared = Object.prototype.hasOwnProperty.call(card, model) ? card[model] : undefined;
  if (declared && declared.mode === mode) {
    return entryIsUsable(declared, mode) ? { entry: declared, source: "rate_card" } : null;
  }
  const entry = Object.prototype.hasOwnProperty.call(VOICE_MAP, model)
    ? VOICE_MAP[model]
    : undefined;
  if (entryIsUsable(entry, mode)) return { entry, source: "cost_map" };
  return null;
}

/**
 * Resolve a leg's per-unit rate, fail-closed. An `override` wins over the map.
 *
 * Throws {@link UnpriceableVoiceError} when neither an override nor the bundled
 * map can price the leg — the guard refuses to meter spend it cannot measure.
 */
export function resolveVoiceRate(
  model: string | null | undefined,
  mode: VoiceMode,
  override?: number | null,
): VoiceRate {
  if (override !== undefined && override !== null) {
    if (!finiteNonNegative(override)) {
      throw new RangeError(
        `voice ${mode} override must be a finite, non-negative number, got ${override}`,
      );
    }
    // Passed at the call site by a human who knows what this leg costs, so it is
    // confirmed by construction.
    return {
      mode,
      unit: unitForMode[mode],
      rate: override,
      source: "override",
      confirmed: true,
    };
  }
  const found = lookupLegEntry(model, mode);
  if (found === null) throw new UnpriceableVoiceError(model ?? null, mode);
  const { entry, source } = found;
  const str = (v: unknown): string | undefined => (typeof v === "string" ? v : undefined);
  return {
    mode,
    unit: unitForMode[mode],
    rate: entry.rate as number,
    source,
    ...(str(entry.provider) !== undefined ? { provider: str(entry.provider) } : {}),
    ...(str(entry.source_url) !== undefined ? { source_url: str(entry.source_url) } : {}),
    ...(str(entry.retrieved_at) !== undefined ? { retrieved_at: str(entry.retrieved_at) } : {}),
    // A declared rate is confirmed unless the card says otherwise (an imported or
    // draft card can mark entries unconfirmed); a bundled list price is
    // UNCONFIRMED until a human accepts it.
    confirmed: typeof entry.confirmed === "boolean" ? entry.confirmed : source === "rate_card",
  };
}

/**
 * USD for one leg. `quantity` is seconds (stt), characters (tts), or minutes
 * (telephony); negative quantities clamp to zero.
 */
export function voiceLegCost(mode: VoiceMode, quantity: number, rate: number): number {
  // Reject non-finite quantities: Math.max(0, NaN) is NaN, so a NaN/Infinity
  // quantity would return a non-finite USD amount that then poisons the guard's
  // running total (NaN disables every ceiling comparison). Negatives still clamp.
  if (!Number.isFinite(quantity)) {
    throw new RangeError(`voice ${mode} quantity must be a finite number, got ${quantity}`);
  }
  const q = Math.max(0, quantity);
  // Every mode except tts is a plain quantity * rate; the unit is what differs,
  // and the unit was already checked when the rate was resolved. tts is the one
  // arm with arithmetic, because its rate is quoted per 1000 chars.
  if (
    mode === "stt" || // seconds * $/second
    mode === "telephony" || // minutes * $/minute
    mode === "sms" || // segments * $/segment
    mode === "ocr" || // pages * $/page
    mode === "gpu" || // gpu-seconds * $/gpu-second
    mode === "avatar" // minutes * $/minute
  ) {
    return q * rate;
  }
  if (mode === "tts") return (q / 1000) * rate; // chars / 1000 * $/1k-chars
  throw new RangeError(`unknown voice mode ${mode}`);
}

/**
 * USD for one voice leg, or `null` when the leg is unconfigured (skip it).
 *
 * The single entry point the adapters use. A leg with neither a vendor (`model`)
 * nor an `override` is unconfigured — return `null` so the token-only contract is
 * preserved and nothing is metered. A leg that *is* configured but cannot be
 * priced throws {@link UnpriceableVoiceError} (via {@link resolveVoiceRate}) rather
 * than accruing a silent $0.
 */
export function priceVoiceLeg(
  mode: VoiceMode,
  quantity: number,
  options: { model?: string | null; override?: number | null } = {},
): number | null {
  const model = options.model ?? null;
  const override = options.override ?? null;
  if (model === null && override === null) return null;
  const resolved = resolveVoiceRate(model, mode, override);
  return voiceLegCost(mode, quantity, resolved.rate);
}
