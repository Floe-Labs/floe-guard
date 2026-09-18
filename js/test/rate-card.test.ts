/**
 * User-supplied rate cards — the rates you actually pay beat the bundled list.
 *
 * The bundled map holds PUBLIC LIST prices. Almost nobody pays list, so the
 * load-bearing behaviour is that a declared rate wins, that a declared rate for
 * a vendor the map has never heard of works at all, and that a BROKEN
 * declaration fails closed instead of silently reverting to a list price the
 * user never chose.
 *
 * Mirrors `tests/test_rate_card.py`.
 */

import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { UnpriceableVoiceError } from "../src/index.js";
import { currentRateCard, loadRateCard, setRateCard } from "../src/rate-card.js";
import { lookupVoiceRate, priceVoiceLeg, resolveVoiceRate } from "../src/voice-pricing.js";

// Never leak a card between tests — it is module-global state.
beforeEach(() => setRateCard({}));
afterEach(() => setRateCard({}));

describe("the zero-config default", () => {
  it("is an empty card, and the bundled list price still resolves", () => {
    expect(currentRateCard()).toEqual({});
    expect(lookupVoiceRate("gcp-vision-document-text-detection", "ocr")).toBeCloseTo(0.0015, 9);
  });
});

describe("a declared rate wins", () => {
  it("beats the bundled list price", () => {
    // You negotiated $0.0004/page; the list is $0.0015.
    setRateCard({
      "gcp-vision-document-text-detection": { mode: "ocr", unit: "usd_per_page", rate: 0.0004 },
    });
    const resolved = resolveVoiceRate("gcp-vision-document-text-detection", "ocr");
    expect(resolved.rate).toBeCloseTo(0.0004, 9);
    expect(resolved.source).toBe("rate_card");
    expect(resolved.confirmed).toBe(true); // you typed it
  });

  it("prices a vendor the map has never heard of", () => {
    // How Mistral OCR / Azure DI / HeyGen / Simli get priced at all: they
    // publish no usable public figure, so nothing ships for them by design.
    expect(lookupVoiceRate("mistral-ocr", "ocr")).toBeNull();
    setRateCard({ "mistral-ocr": { mode: "ocr", unit: "usd_per_page", rate: 0.001 } });
    expect(priceVoiceLeg("ocr", 250, { model: "mistral-ocr" })).toBeCloseTo(0.25, 9);
  });

  it("still loses to a per-call override", () => {
    setRateCard({ acme: { mode: "sms", unit: "usd_per_segment", rate: 0.002 } });
    const resolved = resolveVoiceRate("acme", "sms", 0.001);
    expect(resolved.source).toBe("override");
    expect(resolved.rate).toBeCloseTo(0.001, 9);
  });
});

describe("a broken declaration fails closed", () => {
  it("does NOT silently fall back to the list price", () => {
    // The dangerous direction, pinned. The user said what this leg costs them;
    // quietly substituting Google's list price would be a confident wrong number.
    setRateCard({
      "gcp-vision-document-text-detection": {
        mode: "ocr",
        unit: "usd_per_1k_pages", // wrong: ocr must be usd_per_page
        rate: 0.4,
      },
    });
    expect(() => resolveVoiceRate("gcp-vision-document-text-detection", "ocr")).toThrow(
      UnpriceableVoiceError,
    );
  });

  it("but a card entry for ANOTHER mode does not shadow this leg", () => {
    setRateCard({
      "gcp-vision-document-text-detection": { mode: "avatar", unit: "usd_per_minute", rate: 9 },
    });
    expect(lookupVoiceRate("gcp-vision-document-text-detection", "ocr")).toBeCloseTo(0.0015, 9);
  });
});

describe("confirmation state", () => {
  it("survives the round trip when explicitly false", () => {
    // An imported or draft card can mark entries unconfirmed so a console flags
    // them for review before anyone trusts the margin they produce.
    setRateCard({
      acme: { mode: "sms", unit: "usd_per_segment", rate: 0.002, confirmed: false },
    });
    expect(resolveVoiceRate("acme", "sms").confirmed).toBe(false);
  });
});

describe("loading", () => {
  it("accepts inline JSON as well as an object", () => {
    const entry = { acme: { mode: "sms", unit: "usd_per_segment", rate: 0.002 } };
    expect(loadRateCard(JSON.stringify(entry))).toEqual(entry);
    expect(loadRateCard(entry)).toEqual(entry);
  });

  it.each<[unknown, string]>([
    [{ acme: { mode: "ocr", unit: "usd_per_page" } }, "no rate"],
    [{ acme: { mode: "ocr", unit: "usd_per_page", rate: -1 } }, "negative"],
    [{ acme: { mode: "ocr", unit: "usd_per_page", rate: Number.POSITIVE_INFINITY } }, "non-finite"],
    [{ acme: { mode: "ocr", unit: "usd_per_page", rate: 1, prive: "typo" } }, "unknown key"],
    [{ acme: { mode: "", unit: "usd_per_page", rate: 1 } }, "empty mode"],
    [{ acme: { mode: "ocr", unit: "usd_per_page", rate: 1, confirmed: "yes" } }, "not a bool"],
    [{ acme: "0.002" }, "not an object"],
  ])("throws rather than skipping a malformed card (%#: %s)", (bad) => {
    // Loud, not lenient. A skipped rate card means your declared costs vanish
    // and list prices quietly take their place.
    expect(() => loadRateCard(bad)).toThrow();
  });

  it("rejects JSON that is not an object", () => {
    expect(() => loadRateCard("[1,2,3]")).toThrow(/must be a JSON object/);
  });
});
