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

describe("P1.11 — SMS / OCR / GPU / avatar", () => {
  it("every mode has a unit", () => {
    // A mode with no unit fails closed on EVERY call, which is worse than not
    // having the mode at all. This is the bar for adding one.
    expect(Object.keys(unitForMode).sort()).toEqual(
      ["avatar", "gpu", "ocr", "sms", "stt", "telephony", "tts"].sort(),
    );
  });

  it("bills each new mode in its own unit", () => {
    expect(voiceLegCost("sms", 3, 0.0083)).toBeCloseTo(0.0249, 9);
    expect(voiceLegCost("ocr", 200, 0.0015)).toBeCloseTo(0.3, 9);
    expect(voiceLegCost("gpu", 90, 0.001097)).toBeCloseTo(0.09873, 9);
    expect(voiceLegCost("avatar", 2.5, 0.37)).toBeCloseTo(0.925, 9);
  });

  it("resolves a vendor in each new modality", () => {
    expect(lookupVoiceRate("twilio-sms-us-outbound", "sms")).toBeCloseTo(0.0083, 9);
    expect(lookupVoiceRate("telnyx-sms-us-outbound", "sms")).toBeCloseTo(0.004, 9);
    expect(lookupVoiceRate("modal-h100-sxm5", "gpu")).toBeCloseTo(0.001097, 9);
    expect(lookupVoiceRate("aws-textract-detect-document-text", "ocr")).toBeCloseTo(0.0015, 9);
    expect(lookupVoiceRate("tavus-cvi-starter", "avatar")).toBeCloseTo(0.37, 9);
  });

  it("stores OCR per PAGE, not per thousand pages", () => {
    // Vendors quote per 1,000 pages. Getting this wrong is a 1000x mis-bill.
    expect(priceVoiceLeg("ocr", 1000, { model: "gcp-vision-document-text-detection" })).toBeCloseTo(
      1.5,
      9,
    );
  });

  it("ships volume tiers as separate keys", () => {
    // Which tier you are on is a fact about YOU, so it is a choice of key rather
    // than a curation guess baked into one number.
    expect(lookupVoiceRate("gcp-vision-document-text-detection", "ocr")).toBeCloseTo(0.0015, 9);
    expect(lookupVoiceRate("gcp-vision-document-text-detection-high-volume", "ocr")).toBeCloseTo(
      0.0006,
      9,
    );
    expect(lookupVoiceRate("tavus-cvi-growth", "avatar")).toBeCloseTo(0.32, 9);
    expect(lookupVoiceRate("tavus-cvi-business", "avatar")).toBeCloseTo(0.26, 9);
    // No bare "tavus-cvi" implying a default plan.
    expect(lookupVoiceRate("tavus-cvi", "avatar")).toBeNull();
  });

  it("resolves a bundled rate as UNCONFIRMED, with its citation", () => {
    // A list price is not a cost. It carries provenance and confirmed=false so a
    // console can ask the user to confirm or correct it, rather than reporting a
    // guess as though it were their bill.
    const resolved = resolveVoiceRate("tavus-cvi-starter", "avatar");
    expect(resolved.source).toBe("cost_map");
    expect(resolved.confirmed).toBe(false);
    expect(resolved.provider).toBe("tavus");
    expect(resolved.source_url).toBe("https://www.tavus.io/pricing");
    expect(resolved.retrieved_at).toBe("2026-09-18");
  });

  it("invents no provenance for the pre-P1.11 rates", () => {
    // Back-filling a plausible-looking URL would launder a guess into a
    // citation, so absent provenance is reported honestly as absent.
    const resolved = resolveVoiceRate("deepgram-nova-3", "stt");
    expect(resolved.source_url).toBeUndefined();
    expect(resolved.confirmed).toBe(false);
  });

  it("does not vendor runpod", () => {
    // Per-hour across two tiers, and the map cannot know which one a caller is
    // on. Runpod users add it through their own rate card.
    expect(lookupVoiceRate("runpod-h100-sxm", "gpu")).toBeNull();
  });
});

describe("voiceLegCost — non-finite quantities fail closed", () => {
  it.each([Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY])(
    "throws rather than returning a non-finite cost (%s)",
    (q) => {
      // A NaN/Infinity quantity would poison the guard's running total.
      expect(() => voiceLegCost("stt", q, 0.0001)).toThrow(RangeError);
      expect(() => priceVoiceLeg("stt", q, { model: "deepgram-nova-3" })).toThrow(RangeError);
    },
  );
});
