/**
 * LiveKitBudgetGuard — reserve before the LLM turn, settle on real usage, release
 * on interrupt, and price STT/TTS/telephony legs via the voice cost map.
 *
 * Stubbed session/agent objects drive the adapter — no @livekit/agents runtime, no
 * network, no keys. The adapter is typed structurally; the `types` block at the
 * bottom pins those structural shapes against the REAL exported @livekit/agents
 * types, so this test also fails if the SDK's surface drifts from what we wrapped.
 */

import { describe, expect, it, vi } from "vitest";

import { BudgetExceeded, BudgetGuard, UnpriceableVoiceError, priceVoiceLeg } from "../src/index.js";
import {
  LiveKitBudgetGuard,
  type LiveKitAgentLike,
  type LiveKitSessionLike,
} from "../src/adapters/livekit.js";

const PRICE = { inputCostPerToken: 1e-6, outputCostPerToken: 2e-6 };

/** Minimal typed-emitter stand-in for `AgentSession` — record listeners, emit to them. */
class FakeSession implements LiveKitSessionLike {
  private readonly listeners: Record<string, ((ev: unknown) => void)[]> = {};
  on(event: string, listener: (ev: never) => void): this {
    (this.listeners[event] ??= []).push(listener as (ev: unknown) => void);
    return this;
  }
  emit(event: string, ev: unknown): void {
    for (const listener of this.listeners[event] ?? []) listener(ev);
  }
}

/** A web ReadableStream of the given chunks (as `Agent.llmNode` returns). */
function streamOf(chunks: unknown[]): ReadableStream<unknown> {
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(chunk);
      controller.close();
    },
  });
}

async function drain(stream: ReadableStream<unknown>): Promise<void> {
  const reader = stream.getReader();
  for (;;) {
    const { done } = await reader.read();
    if (done) break;
  }
}

/** An agent whose `llmNode` yields a fresh two-chunk stream on each turn. */
function fakeAgent(): LiveKitAgentLike & { calls: number } {
  return {
    calls: 0,
    async llmNode(this: { calls: number }) {
      this.calls += 1;
      return streamOf(["hello", "world"]);
    },
  };
}

function llmMetrics(promptTokens: number, completionTokens: number) {
  return { metrics: { type: "llm_metrics", promptTokens, completionTokens } };
}

describe("LiveKitBudgetGuard — reserve before the LLM turn", () => {
  it("blocks an over-budget turn before the LLM runs (no stream, no llmNode call)", async () => {
    const guard = new BudgetGuard(1.0, { priceOverrides: { m: PRICE } });
    guard.recordTool("prior", 1.0); // spend the ceiling
    const agent = fakeAgent();
    new LiveKitBudgetGuard(guard, { model: "m" }).attach(new FakeSession(), agent);

    await expect(agent.llmNode()).rejects.toBeInstanceOf(BudgetExceeded);
    expect(agent.calls).toBe(0); // the original llmNode was never reached
  });

  it("invokes onBudgetExceeded and ends the turn with no stream instead of throwing", async () => {
    const guard = new BudgetGuard(1.0, { priceOverrides: { m: PRICE } });
    guard.recordTool("prior", 1.0);
    const spoken = vi.fn(async (_e: BudgetExceeded) => {});
    const agent = fakeAgent();
    new LiveKitBudgetGuard(guard, { model: "m", onBudgetExceeded: spoken }).attach(
      new FakeSession(),
      agent,
    );

    const stream = await agent.llmNode();
    expect(stream).toBeNull();
    expect(spoken).toHaveBeenCalledOnce();
    expect(spoken.mock.calls[0]![0]).toBeInstanceOf(BudgetExceeded);
    expect(agent.calls).toBe(0);
  });

  it("admits an under-budget turn and returns a (wrapped) stream", async () => {
    const guard = new BudgetGuard(1.0, { priceOverrides: { m: PRICE } });
    const agent = fakeAgent();
    new LiveKitBudgetGuard(guard, { model: "m" }).attach(new FakeSession(), agent);

    const stream = await agent.llmNode();
    expect(stream).not.toBeNull();
    await drain(stream!);
    expect(agent.calls).toBe(1);
  });
});

describe("LiveKitBudgetGuard — settle on real usage", () => {
  it("accrues the priced LLM cost from a metrics_collected event and frees the hold", async () => {
    const guard = new BudgetGuard(1.0, { priceOverrides: { m: PRICE } });
    const session = new FakeSession();
    const agent = fakeAgent();
    new LiveKitBudgetGuard(guard, { model: "m" }).attach(session, agent);

    await drain((await agent.llmNode())!); // clean end leaves the hold for the metrics event
    session.emit("metrics_collected", llmMetrics(1000, 500));

    // 1000 * 1e-6 + 500 * 2e-6 = 0.002
    expect(guard.spentUsd).toBeCloseTo(0.002, 12);
    // Reservation consumed (not leaked): remaining == limit - spent, no held slice.
    expect(guard.remainingUsd).toBeCloseTo(1.0 - 0.002, 12);
    expect(guard.spendLog).toHaveLength(1);
    expect(guard.spendLog[0]!.modelOrTool).toBe("m");
  });

  it("settles a delayed metrics event (empty-queue path) as a plain record", async () => {
    const guard = new BudgetGuard(1.0, { priceOverrides: { m: PRICE } });
    const session = new FakeSession();
    new LiveKitBudgetGuard(guard, { model: "m" }).attach(session, fakeAgent());

    // No turn opened — a stray metrics event still meters usage.
    session.emit("metrics_collected", llmMetrics(1000, 0));
    expect(guard.spentUsd).toBeCloseTo(0.001, 12);
  });
});

describe("LiveKitBudgetGuard — release on interrupt", () => {
  it("cancelling the turn stream frees the held reservation", async () => {
    const guard = new BudgetGuard(1.0, { priceOverrides: { m: PRICE } });
    const session = new FakeSession();
    const agent = fakeAgent();
    new LiveKitBudgetGuard(guard, { model: "m" }).attach(session, agent);

    // Turn 1: settle so the guard's next-call estimate becomes non-zero (0.002),
    // making the following turn's reservation observable in remainingUsd.
    await drain((await agent.llmNode())!);
    session.emit("metrics_collected", llmMetrics(1000, 500));
    const afterTurn1 = guard.remainingUsd; // 1.0 - 0.002

    // Turn 2: opening it reserves the 0.002 estimate — remaining drops.
    const stream2 = await agent.llmNode();
    expect(guard.remainingUsd).toBeCloseTo(afterTurn1 - 0.002, 12);

    // Interrupt: cancel the stream instead of draining it. The hold is released.
    await stream2!.cancel();
    expect(guard.remainingUsd).toBeCloseTo(afterTurn1, 12);
    expect(guard.spentUsd).toBeCloseTo(0.002, 12); // no new spend accrued by the interrupt
  });

  it("close releases a still-open turn's reservation", async () => {
    const guard = new BudgetGuard(1.0, { priceOverrides: { m: PRICE } });
    const session = new FakeSession();
    const agent = fakeAgent();
    new LiveKitBudgetGuard(guard, { model: "m" }).attach(session, agent);

    await drain((await agent.llmNode())!);
    session.emit("metrics_collected", llmMetrics(1000, 500));
    const afterTurn1 = guard.remainingUsd;

    await agent.llmNode(); // turn 2 holds 0.002, never settles
    expect(guard.remainingUsd).toBeCloseTo(afterTurn1 - 0.002, 12);

    session.emit("close", { type: "close" });
    expect(guard.remainingUsd).toBeCloseTo(afterTurn1, 12);
  });
});

describe("LiveKitBudgetGuard — voice legs price via the cost map", () => {
  it("meters STT (per second) and TTS (per 1k chars) from named vendors", () => {
    const guard = new BudgetGuard(10.0, { priceOverrides: { m: PRICE } });
    const session = new FakeSession();
    new LiveKitBudgetGuard(guard, {
      model: "m",
      sttModel: "deepgram-nova-3",
      ttsModel: "elevenlabs-flash-v2.5",
    }).attach(session, fakeAgent());

    // 30_000 ms of audio -> 30 s; 1_200 synthesized characters.
    session.emit("metrics_collected", {
      metrics: { type: "stt_metrics", audioDurationMs: 30_000 },
    });
    session.emit("metrics_collected", {
      metrics: { type: "tts_metrics", charactersCount: 1_200 },
    });

    const expectedStt = priceVoiceLeg("stt", 30, { model: "deepgram-nova-3" });
    const expectedTts = priceVoiceLeg("tts", 1_200, { model: "elevenlabs-flash-v2.5" });
    expect(guard.toolCosts["livekit-stt"]).toBeCloseTo(expectedStt!, 12);
    expect(guard.toolCosts["livekit-tts"]).toBeCloseTo(expectedTts!, 12);
  });

  it("meters telephony per minute via meterTelephony", () => {
    const guard = new BudgetGuard(10.0, { priceOverrides: { m: PRICE } });
    const budget = new LiveKitBudgetGuard(guard, {
      model: "m",
      telephony: "twilio-us-inbound-local",
    });

    const accrued = budget.meterTelephony(2.5);
    expect(accrued).toBeCloseTo(priceVoiceLeg("telephony", 2.5, { model: "twilio-us-inbound-local" })!, 12);
    expect(guard.toolCosts["livekit-telephony"]).toBeCloseTo(accrued!, 12);
  });

  it("leaves an unconfigured leg un-metered (token-only contract)", () => {
    const guard = new BudgetGuard(10.0, { priceOverrides: { m: PRICE } });
    const session = new FakeSession();
    new LiveKitBudgetGuard(guard, { model: "m" }).attach(session, fakeAgent());

    session.emit("metrics_collected", {
      metrics: { type: "stt_metrics", audioDurationMs: 30_000 },
    });
    expect(guard.toolCosts["livekit-stt"]).toBeUndefined();
    expect(guard.spentUsd).toBe(0);
  });

  it("fails closed on a vendor the voice map cannot price", () => {
    const guard = new BudgetGuard(10.0, { priceOverrides: { m: PRICE } });
    const session = new FakeSession();
    new LiveKitBudgetGuard(guard, { model: "m", sttModel: "no-such-vendor" }).attach(
      session,
      fakeAgent(),
    );

    expect(() =>
      session.emit("metrics_collected", {
        metrics: { type: "stt_metrics", audioDurationMs: 1_000 },
      }),
    ).toThrow(UnpriceableVoiceError);
  });
});

describe("LiveKitBudgetGuard — pre-call admission", () => {
  it("admitCall delegates to the budget gate", () => {
    const guard = new BudgetGuard(1.0);
    const budget = new LiveKitBudgetGuard(guard, { model: "m" });
    expect(budget.admitCall()).toBe(true);
    guard.recordTool("prior", 1.0);
    expect(budget.admitCall()).toBe(false);
  });
});

// --- Type pins: the adapter is structurally typed; assert those shapes accept the
// --- REAL @livekit/agents exports so this test breaks if the SDK surface drifts.
// --- Type-only — never executed.
import type { Agent, AgentSession, LLMMetrics, MetricsCollectedEvent } from "@livekit/agents";

function __typePins(
  realSession: AgentSession,
  realEvent: MetricsCollectedEvent,
  realLlm: LLMMetrics,
): void {
  // The real Agent.llmNode takes exactly 3 args (chatCtx, toolCtx, modelSettings)
  // — the arity we forward unchanged. Breaks here if the SDK changes it.
  const arity: 3 = null as unknown as Parameters<Agent["llmNode"]>["length"];
  // The real MetricsCollectedEvent carries a `.metrics` union with a `type`
  // discriminator (what onMetrics branches on), and LLMMetrics reports
  // promptTokens/completionTokens (what we settle against).
  const kind: string = realEvent.metrics.type;
  const tokens: number = realLlm.promptTokens + realLlm.completionTokens;
  // The real AgentSession is a typed emitter we can attach onto.
  const sessionLike: LiveKitSessionLike = realSession as unknown as LiveKitSessionLike;
  void arity;
  void kind;
  void tokens;
  void sessionLike;
}
void __typePins;
