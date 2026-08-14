/**
 * Offline pricing for voice legs (STT / TTS / telephony) from the vendored map.
 *
 * The voice twin of `pricing.ts`. The token cost map is per-token; a voice
 * pipeline's bill is per-second (STT), per-1k-chars (TTS), and per-minute
 * (telephony) instead, so the voice rates live under the reserved `"__voice__"`
 * key of `cost_map.json` with their own schema:
 *
 *     "deepgram-nova-3": {
 *       "mode": "stt",              // discriminator: "stt" | "tts" | "telephony"
 *       "unit": "usd_per_second",   // explicit unit — a mismatch fails closed
 *       "rate": 0.0001283333,       // USD in that unit
 *       "provider": "deepgram"
 *     }
 *
 * Same fail-closed contract as token pricing: a vendor absent from the map (or an
 * entry whose `unit`/`mode` does not match the leg it is asked to price) is
 * unpriceable — {@link lookupVoiceRate} returns `null` and the adapter throws
 * {@link UnpriceableVoiceError} rather than silently metering the leg at $0. Pass a
 * per-unit override to enforce a leg the map cannot price.
 *
 * No network. The rates are a **drift-prone snapshot** of each vendor's public
 * list price — refresh them (`scripts/update-cost-map.mjs`) like the token map, or
 * estimates drift as vendors change prices. Telephony is **US-only in v1**.
 *
 * This is a faithful port of `src/floe_guard/voice_pricing.py`.
 */

import costMapJson from "./cost_map.json";
import { UnpriceableVoiceError } from "./errors.js";

export type VoiceMode = "stt" | "tts" | "telephony";

interface VoiceMapEntry {
  mode?: unknown;
  unit?: unknown;
  rate?: unknown;
  provider?: unknown;
}

/** The vendored voice rates — the reserved `"__voice__"` section of the cost map. */
const VOICE_MAP = ((costMapJson as Record<string, unknown>)["__voice__"] ?? {}) as Record<
  string,
  VoiceMapEntry
>;

/**
 * The one unit each leg is billed in. An entry whose `unit` disagrees with its
 * leg is a schema mismatch and fails closed — a Deepgram $/min figure stored
 * without the ÷60 conversion would over-bill 60x if it were silently accepted.
 */
export const unitForMode: Record<VoiceMode, string> = {
  stt: "usd_per_second",
  tts: "usd_per_1k_chars",
  telephony: "usd_per_minute",
};

/** A resolved per-unit voice rate plus where it came from. */
export interface VoiceRate {
  mode: string;
  unit: string;
  rate: number;
  source: "override" | "cost_map";
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
  if (model === null || model === undefined) return null;
  const entry = Object.prototype.hasOwnProperty.call(VOICE_MAP, model)
    ? VOICE_MAP[model]
    : undefined;
  if (!entry) return null;
  if (entry.mode !== mode) return null;
  if (entry.unit !== unitForMode[mode]) return null;
  const rate = entry.rate;
  if (!finiteNonNegative(rate)) return null;
  return rate;
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
    return { mode, unit: unitForMode[mode], rate: override, source: "override" };
  }
  const rate = lookupVoiceRate(model, mode);
  if (rate === null) throw new UnpriceableVoiceError(model ?? null, mode);
  return { mode, unit: unitForMode[mode], rate, source: "cost_map" };
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
  if (mode === "stt") return q * rate; // seconds * $/second
  if (mode === "tts") return (q / 1000) * rate; // chars / 1000 * $/1k-chars
  if (mode === "telephony") return q * rate; // minutes * $/minute
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
