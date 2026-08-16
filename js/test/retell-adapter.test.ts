/**
 * RetellBudgetGuard — reserve when a `response_required` arrives, settle on real
 * token usage after `content_complete`, release when a newer `response_id`
 * interrupts, and price STT/TTS/telephony legs via the voice cost map.
 *
 * Fake Retell interaction events drive the adapter — no `ws` server, no Retell SDK,
 * no network, no keys. Everything is a plain parsed-message shape.
 */

import { describe, expect, it } from "vitest";

import { BudgetGuard, UnpriceableVoiceError, gates, priceVoiceLeg } from "../src/index.js";
import {
  RetellBudgetGuard,
  type RetellResponseRequiredEvent,
} from "../src/adapters/retell.js";

const PRICE = { inputCostPerToken: 1e-6, outputCostPerToken: 2e-6 };

/** A `response_required` interaction event, as Retell delivers it over the WS. */
function responseRequired(responseId: number): RetellResponseRequiredEvent {
  return {
    interaction_type: "response_required",
    response_id: responseId,
    transcript: [{ role: "user", content: "hi" }],
  };
}

describe("RetellBudgetGuard — reserve before the LLM turn", () => {
  it("blocks an over-budget turn (admitted: false, carries the BudgetExceeded)", () => {
    const guard = new BudgetGuard(1.0, { priceOverrides: { m: PRICE } });
    guard.recordTool("prior", 1.0); // spend the ceiling
    const budget = new RetellBudgetGuard(guard, { model: "m" });

    const decision = budget.beginTurn(responseRequired(1));
    expect(decision.admitted).toBe(false);
    if (!decision.admitted) {
      expect(decision.error.name).toBe("BudgetExceeded");
      expect(decision.responseId).toBe(1);
    }
  });

  it("admits an under-budget turn", () => {
    const guard = new BudgetGuard(1.0, { priceOverrides: { m: PRICE } });
    const budget = new RetellBudgetGuard(guard, { model: "m" });

    const decision = budget.beginTurn(responseRequired(1));
    expect(decision.admitted).toBe(true);
    expect(decision.responseId).toBe(1);
  });
});

describe("RetellBudgetGuard — settle on real usage", () => {
  it("accrues the priced LLM cost after content_complete and frees the hold", () => {
    const guard = new BudgetGuard(1.0, { priceOverrides: { m: PRICE } });
    const budget = new RetellBudgetGuard(guard, { model: "m" });

    budget.beginTurn(responseRequired(1));
    budget.settleTurn(1, { promptTokens: 1000, completionTokens: 500 });

    // 1000 * 1e-6 + 500 * 2e-6 = 0.002
    expect(guard.spentUsd).toBeCloseTo(0.002, 12);
    // Reservation consumed (not leaked): remaining == limit - spent, no held slice.
    expect(guard.remainingUsd).toBeCloseTo(1.0 - 0.002, 12);
    expect(guard.spendLog).toHaveLength(1);
    expect(guard.spendLog[0]!.modelOrTool).toBe("m");
  });

  it("settles a turn that was never begun (no open slot) as a plain record", () => {
    const guard = new BudgetGuard(1.0, { priceOverrides: { m: PRICE } });
    const budget = new RetellBudgetGuard(guard, { model: "m" });

    budget.settleTurn(99, { promptTokens: 1000, completionTokens: 0 });
    expect(guard.spentUsd).toBeCloseTo(0.001, 12);
  });
});

describe("RetellBudgetGuard — release on interrupt", () => {
  it("a newer response_id releases the prior open turn's reservation", () => {
    const guard = new BudgetGuard(1.0, { priceOverrides: { m: PRICE } });
    const budget = new RetellBudgetGuard(guard, { model: "m" });

    // Turn 1: settle so the guard's next-call estimate becomes non-zero (0.002),
    // making the following turn's reservation observable in remainingUsd.
    budget.beginTurn(responseRequired(1));
    budget.settleTurn(1, { promptTokens: 1000, completionTokens: 500 });
    const afterTurn1 = guard.remainingUsd; // 1.0 - 0.002

    // Turn 2: reserves the 0.002 estimate — remaining drops.
    budget.beginTurn(responseRequired(2));
    expect(guard.remainingUsd).toBeCloseTo(afterTurn1 - 0.002, 12);

    // Turn 3 (higher response_id) arrives before turn 2 settles — turn 2 is
    // interrupted and its hold released. Turn 3 then holds its own 0.002.
    budget.beginTurn(responseRequired(3));
    expect(guard.remainingUsd).toBeCloseTo(afterTurn1 - 0.002, 12); // only turn 3 held
    expect(guard.spentUsd).toBeCloseTo(0.002, 12); // interrupt accrued no new spend
  });

  it("close releases a still-open turn's reservation", () => {
    const guard = new BudgetGuard(1.0, { priceOverrides: { m: PRICE } });
    const budget = new RetellBudgetGuard(guard, { model: "m" });

    budget.beginTurn(responseRequired(1));
    budget.settleTurn(1, { promptTokens: 1000, completionTokens: 500 });
    const afterTurn1 = guard.remainingUsd;

    budget.beginTurn(responseRequired(2)); // holds 0.002, never settles
    expect(guard.remainingUsd).toBeCloseTo(afterTurn1 - 0.002, 12);

    budget.close();
    expect(guard.remainingUsd).toBeCloseTo(afterTurn1, 12);
  });
});

describe("RetellBudgetGuard — pre-call admission via gates.retell", () => {
  it("admits with available budget and rejects when exhausted", () => {
    const guard = new BudgetGuard(1.0);
    const budget = new RetellBudgetGuard(guard, { model: "m" });

    const admit = budget.admitCall({ admit: { metadata: { plan: "free" } } });
    expect(admit).toEqual(gates.retell(guard, { admit: { metadata: { plan: "free" } } }));
    expect(admit.call_inbound.reject).toBeUndefined();

    guard.recordTool("prior", 1.0); // spend the ceiling
    expect(budget.admitCall()).toEqual({ call_inbound: { reject: true } });
  });
});

describe("RetellBudgetGuard — voice legs price via the cost map", () => {
  it("meters STT (per second), TTS (per 1k chars) and telephony (per minute)", () => {
    const guard = new BudgetGuard(10.0, { priceOverrides: { m: PRICE } });
    const budget = new RetellBudgetGuard(guard, {
      model: "m",
      sttModel: "deepgram-nova-3",
      ttsModel: "elevenlabs-flash-v2.5",
      telephony: "twilio-us-inbound-local",
    });

    const stt = budget.meterStt(30);
    const tts = budget.meterTts(1_200);
    const tel = budget.meterTelephony(2.5);

    expect(stt).toBeCloseTo(priceVoiceLeg("stt", 30, { model: "deepgram-nova-3" })!, 12);
    expect(tts).toBeCloseTo(priceVoiceLeg("tts", 1_200, { model: "elevenlabs-flash-v2.5" })!, 12);
    expect(tel).toBeCloseTo(
      priceVoiceLeg("telephony", 2.5, { model: "twilio-us-inbound-local" })!,
      12,
    );
    expect(guard.toolCosts["retell-stt"]).toBeCloseTo(stt!, 12);
    expect(guard.toolCosts["retell-tts"]).toBeCloseTo(tts!, 12);
    expect(guard.toolCosts["retell-telephony"]).toBeCloseTo(tel!, 12);
  });

  it("leaves an unconfigured leg un-metered (token-only contract)", () => {
    const guard = new BudgetGuard(10.0, { priceOverrides: { m: PRICE } });
    const budget = new RetellBudgetGuard(guard, { model: "m" });

    expect(budget.meterStt(30)).toBeNull();
    expect(guard.toolCosts["retell-stt"]).toBeUndefined();
    expect(guard.spentUsd).toBe(0);
  });

  it("fails closed on a vendor the voice map cannot price", () => {
    const guard = new BudgetGuard(10.0, { priceOverrides: { m: PRICE } });
    const budget = new RetellBudgetGuard(guard, { model: "m", sttModel: "no-such-vendor" });

    expect(() => budget.meterStt(1)).toThrow(UnpriceableVoiceError);
  });
});
