/**
 * VapiBudgetGuard demo — STUBBED, no keys, no network.
 *
 *   node examples/vapi_voice_cost.mjs   (run `npm run build` first)
 *
 * Fabricates a Vapi custom-LLM request and a fake upstream OpenAI SSE completion,
 * drives the adapter through one model turn, and prints:
 *   1. a pre-call admission decision (Vapi's assistant-request webhook shape), and
 *   2. a per-leg call-cost receipt (LLM tokens + STT + TTS + telephony).
 *
 * Scope is pre-call admission + per-turn settlement — nothing here cuts a turn (or
 * a stream) off partway. Every rate is offline (the bundled voice cost map); no
 * vendor and no LLM is called.
 */

import { BudgetGuard } from "../dist/index.js";
import { VapiBudgetGuard } from "../dist/adapters/vapi.js";

// --- A fake upstream OpenAI SSE stream (what your /chat/completions would proxy) ---
// Content chunks carry no usage; the FINAL empty-choices chunk carries usage, exactly
// as stream_options:{ include_usage: true } produces. Without that flag there is no
// usage chunk and the adapter fails loudly (VapiUsageMissingError) — never $0.
async function* fakeUpstreamStream() {
  for (const piece of ["Sure, ", "here's ", "your ", "answer."]) {
    yield { choices: [{ delta: { content: piece } }] };
  }
  yield { choices: [], usage: { prompt_tokens: 1_200, completion_tokens: 350 } };
}

// --- Wiring ------------------------------------------------------------------

const guard = new BudgetGuard(1.0, {
  // gemini-2.0-flash isn't in the token map here, so price it explicitly.
  priceOverrides: {
    "gemini-2.0-flash": { inputCostPerToken: 0.3e-6, outputCostPerToken: 2.5e-6 },
  },
});

const budget = new VapiBudgetGuard(guard, {
  sttModel: "deepgram-nova-3", // priced per second from the voice map
  ttsModel: "elevenlabs-flash-v2.5", // priced per 1k characters
  telephony: "twilio-us-inbound-local", // priced per minute
});

// 1) Pre-call admission — Vapi's assistant-request webhook, answered from the budget.
console.log("=== assistant-request admission (Vapi webhook shape) ===");
const admit = budget.assistantRequest({
  assistantId: "asst_demo_123",
  estimatedCallUsd: 0.25,
  errorMessage: "Sorry, this agent is out of budget right now.",
});
console.log(`  budget remaining : $${guard.remainingUsd.toFixed(6)}`);
console.log(`  response         : ${JSON.stringify(admit)}`);
console.log(`  decision         : ${"error" in admit ? "REJECT (spoken error)" : "ADMIT"}`);
console.log();

// --- One model turn via the custom-LLM proxy ---------------------------------

// The Vapi request points at YOUR /chat/completions with { stream: true }. The
// adapter reserves before the upstream call, forwards each chunk, and settles on
// the real token usage carried by the final chunk.
const model = "gemini-2.0-flash"; // Vapi sends the model in the request body
const sse = budget.guardStream(() => fakeUpstreamStream(), { model });

let text = "";
for await (const chunk of sse) {
  text += chunk.choices?.[0]?.delta?.content ?? "";
}

// The other legs never reach the custom-LLM proxy — meter them explicitly (as you
// would from Vapi's end-of-call-report).
budget.meterStt(8.4); // 8.4 s of caller audio
budget.meterTts(240); // 240 synthesized characters
budget.meterTelephony(1.5); // 1.5 min of call time

// --- Receipt -----------------------------------------------------------------

const tools = guard.toolCosts;
const toolTotal = Object.values(tools).reduce((a, b) => a + b, 0);
const llmCost = guard.spentUsd - toolTotal;

console.log("=== Call-cost receipt (all offline) ===");
console.log(`  assistant reply                      : "${text}"`);
console.log(`  LLM (gemini-2.0-flash, 1200+350 tok) : $${llmCost.toFixed(6)}`);
console.log(`  STT (deepgram-nova-3, 8.4 s)         : $${(tools["vapi-stt"] ?? 0).toFixed(6)}`);
console.log(`  TTS (elevenlabs-flash-v2.5, 240 ch)  : $${(tools["vapi-tts"] ?? 0).toFixed(6)}`);
console.log(`  Telephony (twilio inbound, 1.5 min)  : $${(tools["vapi-telephony"] ?? 0).toFixed(6)}`);
console.log("  " + "-".repeat(44));
console.log(`  TOTAL spent                          : $${guard.spentUsd.toFixed(6)}`);
console.log(`  Budget remaining                     : $${guard.remainingUsd.toFixed(6)}`);
