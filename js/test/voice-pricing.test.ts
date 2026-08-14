/**
 * Offline voice pricing — fail-closed resolution and per-unit math.
 *
 * The voice twin of the token pricing tests: STT is billed per second, TTS per 1k
 * chars, telephony per minute, and every schema/vendor mismatch fails closed
 * (`UnpriceableVoiceError`) rather than metering a leg at a silent $0.
 *
 * Mirrors `tests/test_voice_pricing.py`.
 */

import { describe, expect, it } from "vitest";

import { UnpriceableVoiceError } from "../src/index.js";
import {
  lookupVoiceRate,
  priceVoiceLeg,
  resolveVoiceRate,
  unitForMode,
  voiceLegCost,
} from "../src/voice-pricing.js";

describe("lookupVoiceRate — resolution", () => {
  it("resolves a known STT vendor from the cost map", () => {
    // $0.0077/min mono ÷ 60 = $0.0001283333/sec.
    expect(lookupVoiceRate("deepgram-nova-3", "stt")).toBeCloseTo(0.0077 / 60, 7);
  });

  it("resolves known TTS vendors from the cost map", () => {
    expect(lookupVoiceRate("elevenlabs-multilingual-v2", "tts")).toBeCloseTo(0.1, 9);
    expect(lookupVoiceRate("elevenlabs-flash-v2.5", "tts")).toBeCloseTo(0.05, 9);
  });

  it("resolves a known telephony vendor from the cost map", () => {
    expect(lookupVoiceRate("twilio-us-inbound-local", "telephony")).toBeCloseTo(0.0085, 9);
  });

  it("an unknown or null vendor is unpriceable (null, no throw)", () => {
    expect(lookupVoiceRate("no-such-vendor-anywhere", "stt")).toBeNull();
    expect(lookupVoiceRate(null, "stt")).toBeNull();
    expect(lookupVoiceRate(undefined, "stt")).toBeNull();
  });

  it("a mode mismatch fails closed", () => {
    // An STT entry asked to price a TTS leg is a schema mismatch — refuse rather
    // than mis-bill a per-second rate as if it were per-1k-chars.
    expect(lookupVoiceRate("deepgram-nova-3", "tts")).toBeNull();
    expect(lookupVoiceRate("elevenlabs-flash-v2.5", "stt")).toBeNull();
    expect(lookupVoiceRate("twilio-us-inbound-local", "stt")).toBeNull();
  });
});

describe("resolveVoiceRate — fail-closed + overrides", () => {
  it("throws the fail-closed error for an unknown vendor", () => {
    let err: unknown;
    try {
      resolveVoiceRate("mystery-tts", "tts");
    } catch (e) {
      err = e;
    }
    expect(err).toBeInstanceOf(UnpriceableVoiceError);
    expect((err as UnpriceableVoiceError).vendor).toBe("mystery-tts");
    expect((err as UnpriceableVoiceError).mode).toBe("tts");
  });

  it("an override wins over the cost map", () => {
    const resolved = resolveVoiceRate("deepgram-nova-3", "stt", 0.0002);
    expect(resolved.source).toBe("override");
    expect(resolved.rate).toBe(0.0002);
    expect(resolved.unit).toBe(unitForMode.stt);
  });

  it("an override prices a vendor the map cannot", () => {
    const resolved = resolveVoiceRate("some-brand-new-tts", "tts", 0.07);
    expect(resolved.source).toBe("override");
    expect(resolved.rate).toBe(0.07);
  });

  it("a non-finite or negative override throws", () => {
    expect(() => resolveVoiceRate("x", "stt", Number.NaN)).toThrow(RangeError);
    expect(() => resolveVoiceRate("x", "stt", -1.0)).toThrow(RangeError);
  });
});

describe("voiceLegCost — per-unit math", () => {
  it("bills each unit correctly", () => {
    expect(voiceLegCost("stt", 10.0, 0.0001)).toBeCloseTo(0.001, 9); // seconds * $/sec
    expect(voiceLegCost("tts", 2000, 0.05)).toBeCloseTo(0.1, 9); // chars / 1000 * $/1k
    expect(voiceLegCost("telephony", 3.0, 0.0085)).toBeCloseTo(0.0255, 9); // min * $/min
  });

  it("clamps a negative quantity to zero", () => {
    expect(voiceLegCost("stt", -5.0, 0.01)).toBe(0);
  });
});

describe("priceVoiceLeg — entry point", () => {
  it("skips (null) when the leg is unconfigured", () => {
    // Neither a vendor nor an override — the leg is un-metered (token-only
    // contract preserved), NOT a fail-closed throw.
    expect(priceVoiceLeg("stt", 10.0)).toBeNull();
  });

  it("fails closed when configured but unpriceable", () => {
    expect(() => priceVoiceLeg("tts", 1000, { model: "ghost-vendor" })).toThrow(
      UnpriceableVoiceError,
    );
  });

  it("prices a configured, known leg", () => {
    // deepgram STT over 60s ≈ 60 * (0.0077/60) = $0.0077.
    expect(priceVoiceLeg("stt", 60, { model: "deepgram-nova-3" })).toBeCloseTo(0.0077, 6);
  });
});
