/**
 * RetellBudgetGuard demo — STUBBED, no keys, no network, no WS server.
 *
 *   node examples/retell_voice_cost.mjs   (run `npm run build` first)
 *
 * Fabricates a Retell custom-LLM `response_required` interaction event plus fake LLM
 * token usage, drives the adapter through one voice turn, and prints:
 *   1. a pre-call admission decision (Retell's `call_inbound` webhook shape), and
 *   2. a per-leg call-cost receipt (LLM tokens + STT + TTS + telephony).
 *
 * Scope is pre-call admission + per-turn settlement — nothing here cuts a turn off
 * partway. Every rate is offline (the bundled voice cost map); no vendor is called.
 */

import { BudgetGuard } from "../dist/index.js";
import { RetellBudgetGuard } from "../dist/adapters/retell.js";

// --- Wiring ------------------------------------------------------------------

const guard = new BudgetGuard(1.0, {
  // gpt-4o-mini isn't in the token map here, so price it explicitly.
  priceOverrides: {
    "gpt-4o-mini": { inputCostPerToken: 0.15e-6, outputCostPerToken: 0.6e-6 },
  },
});

const budget = new RetellBudgetGuard(guard, {
  model: "gpt-4o-mini",
  sttModel: "deepgram-nova-3", // priced per second from the voice map
  ttsModel: "elevenlabs-flash-v2.5", // priced per 1k characters
  telephony: "twilio-us-inbound-local", // priced per minute
});

// 1) Pre-call admission — Retell's inbound-call webhook. Would we even accept this
//    call, given the budget left? (budget-exhausted -> { call_inbound: { reject: true } })
const inbound = budget.admitCall({
  estimatedCallUsd: 0.25,
  admit: { dynamic_variables: { plan: "free" } },
});
console.log("=== Pre-call admission (Retell call_inbound webhook) ===");
console.log(`  budget remaining : $${guard.remainingUsd.toFixed(6)}`);
console.log(`  decision         : ${inbound.call_inbound.reject ? "REJECT" : "ADMIT"}`);
console.log(`  webhook response : ${JSON.stringify(inbound)}`);
console.log();

// --- One voice turn ----------------------------------------------------------

// Retell sends a `response_required` interaction event over the WS.
const event = {
  interaction_type: "response_required",
  response_id: 1,
  transcript: [{ role: "user", content: "What's my balance?" }],
};

// Reserve BEFORE the LLM call for this response_id.
const turn = budget.beginTurn(event);
if (!turn.admitted) {
  // Budget spent — send a wrap-up response and hang up (no LLM call).
  console.log("Turn blocked:", budget.response(event.response_id, "Out of budget.", {
    complete: true,
    endCall: true,
  }));
  process.exit(0);
}

// Your custom LLM streams a reply; we fake its text + real token usage.
const reply = "Your balance is $42.00.";
const usage = { promptTokens: 1_200, completionTokens: 350 };
// (In a real server you'd `ws.send` each partial, ending with content_complete.)
budget.response(event.response_id, reply, { complete: true });

// Settle the turn's real token usage after content_complete.
budget.settleTurn(event.response_id, usage);

// The socket never sees STT / TTS / telephony — meter them explicitly.
budget.meterStt(8.4); // 8.4 s of caller audio transcribed
budget.meterTts(reply.length); // characters synthesized back
budget.meterTelephony(1.5); // 1.5 min of call time

// --- Receipt -----------------------------------------------------------------

const tools = guard.toolCosts;
const toolTotal = Object.values(tools).reduce((a, b) => a + b, 0);
const llmCost = guard.spentUsd - toolTotal;

console.log("=== Call-cost receipt (all offline) ===");
console.log(`  LLM (gpt-4o-mini, 1200+350 tok)      : $${llmCost.toFixed(6)}`);
console.log(`  STT (deepgram-nova-3, 8.4 s)         : $${(tools["retell-stt"] ?? 0).toFixed(6)}`);
console.log(`  TTS (elevenlabs-flash-v2.5, ${String(reply.length).padStart(2)} ch)   : $${(tools["retell-tts"] ?? 0).toFixed(6)}`);
console.log(`  Telephony (twilio inbound, 1.5 min)  : $${(tools["retell-telephony"] ?? 0).toFixed(6)}`);
console.log("  " + "-".repeat(44));
console.log(`  TOTAL spent                          : $${guard.spentUsd.toFixed(6)}`);
console.log(`  Budget remaining                     : $${guard.remainingUsd.toFixed(6)}`);
