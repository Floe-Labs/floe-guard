/**
 * LiveKitBudgetGuard demo — STUBBED, no keys, no network.
 *
 *   node examples/livekit_voice_cost.mjs   (run `npm run build` first)
 *
 * Fabricates a LiveKit-shaped session (a typed emitter) and agent (an `llmNode`),
 * drives the adapter through one voice turn, and prints:
 *   1. a pre-call admission decision (from the budget gate), and
 *   2. a per-leg call-cost receipt (LLM tokens + STT + TTS + telephony).
 *
 * Scope is pre-call admission + per-turn settlement — nothing here cuts a turn off
 * partway. Every rate is offline (the bundled voice cost map); no vendor is called.
 */

import { BudgetGuard } from "../dist/index.js";
import { LiveKitBudgetGuard } from "../dist/adapters/livekit.js";

// --- Fakes standing in for @livekit/agents (so this runs with zero credentials) ---

/** Minimal typed-emitter stand-in for `AgentSession`. */
class FakeSession {
  #listeners = {};
  on(event, listener) {
    (this.#listeners[event] ??= []).push(listener);
    return this;
  }
  emit(event, ev) {
    for (const listener of this.#listeners[event] ?? []) listener(ev);
  }
}

/** A web ReadableStream of chunks, as `Agent.llmNode` returns. */
function streamOf(chunks) {
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(chunk);
      controller.close();
    },
  });
}

async function drain(stream) {
  const reader = stream.getReader();
  for (;;) {
    const { done } = await reader.read();
    if (done) break;
  }
}

/** An agent whose `llmNode` returns a short reply stream. */
const agent = {
  async llmNode() {
    return streamOf(["Sure, ", "here's ", "your ", "answer."]);
  },
};

// --- Wiring ------------------------------------------------------------------

const guard = new BudgetGuard(1.0, {
  // gemini-2.0-flash isn't in the token map here, so price it explicitly.
  priceOverrides: {
    "gemini-2.0-flash": { inputCostPerToken: 0.3e-6, outputCostPerToken: 2.5e-6 },
  },
});

const budget = new LiveKitBudgetGuard(guard, {
  model: "gemini-2.0-flash",
  sttModel: "deepgram-nova-3", // priced per second from the voice map
  ttsModel: "elevenlabs-flash-v2.5", // priced per 1k characters
  telephony: "twilio-us-inbound-local", // priced per minute
});

// 1) Pre-call admission — would we even start this call, given the budget left?
const estimatedCallUsd = 0.25;
const admit = budget.admitCall({ estimatedCallUsd });
console.log("=== Pre-call admission ===");
console.log(`  budget remaining : $${guard.remainingUsd.toFixed(6)}`);
console.log(`  estimated call   : $${estimatedCallUsd.toFixed(6)}`);
console.log(`  decision         : ${admit ? "ADMIT" : "REJECT"}`);
console.log();

const session = new FakeSession();
budget.attach(session, agent);

// --- One voice turn ----------------------------------------------------------

// Reserve before the LLM turn, then run it (drain the reply stream).
const stream = await agent.llmNode();
await drain(stream);

// Real usage lands after the turn — the non-deprecated `metrics_collected` event.
session.emit("metrics_collected", {
  metrics: { type: "llm_metrics", promptTokens: 1_200, completionTokens: 350 },
});
session.emit("metrics_collected", {
  metrics: { type: "stt_metrics", audioDurationMs: 8_400 }, // 8.4 s of caller audio
});
session.emit("metrics_collected", {
  metrics: { type: "tts_metrics", charactersCount: 240 }, // synthesized reply
});

// Telephony has no LiveKit metric — the transport meters it per minute.
budget.meterTelephony(1.5);

// --- Receipt -----------------------------------------------------------------

const tools = guard.toolCosts;
const toolTotal = Object.values(tools).reduce((a, b) => a + b, 0);
const llmCost = guard.spentUsd - toolTotal;

console.log("=== Call-cost receipt (all offline) ===");
console.log(`  LLM (gemini-2.0-flash, 1200+350 tok) : $${llmCost.toFixed(6)}`);
console.log(`  STT (deepgram-nova-3, 8.4 s)         : $${(tools["livekit-stt"] ?? 0).toFixed(6)}`);
console.log(`  TTS (elevenlabs-flash-v2.5, 240 ch)  : $${(tools["livekit-tts"] ?? 0).toFixed(6)}`);
console.log(`  Telephony (twilio inbound, 1.5 min)  : $${(tools["livekit-telephony"] ?? 0).toFixed(6)}`);
console.log("  " + "-".repeat(44));
console.log(`  TOTAL spent                          : $${guard.spentUsd.toFixed(6)}`);
console.log(`  Budget remaining                     : $${guard.remainingUsd.toFixed(6)}`);
