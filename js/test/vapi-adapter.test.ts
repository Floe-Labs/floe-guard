/**
 * VapiBudgetGuard — reserve before the model turn, settle on the real OpenAI usage,
 * release on error/abort, admit the call via the assistant-request gate, and price
 * the STT/TTS/telephony legs the custom-LLM proxy never sees.
 *
 * Fabricated OpenAI-shaped completions / SSE chunk streams drive the adapter — no
 * Vapi SDK, no `openai` runtime, no network, no keys. The adapter is typed
 * structurally against the OpenAI wire format Vapi speaks.
 */

import { describe, expect, it } from "vitest";

import { BudgetExceeded, BudgetGuard, UnpriceableVoiceError, priceVoiceLeg } from "../src/index.js";
import {
  VapiBudgetGuard,
  VapiUsageMissingError,
  type ChatCompletionChunkLike,
  type ChatCompletionLike,
} from "../src/adapters/vapi.js";

const PRICE = { inputCostPerToken: 1e-6, outputCostPerToken: 2e-6 };

/** An OpenAI-format non-streaming completion carrying a usage block. */
function completion(promptTokens: number, completionTokens: number) {
  return {
    id: "chatcmpl-x",
    choices: [{ message: { role: "assistant", content: "hi" } }],
    usage: { prompt_tokens: promptTokens, completion_tokens: completionTokens },
  };
}

/**
 * A fake OpenAI SSE stream: content chunks (no usage), then — if `usage` is given —
 * a final empty-choices chunk carrying it, exactly as `stream_options:{include_usage:true}`
 * produces. Omit `usage` to model a stream WITHOUT include_usage.
 */
async function* sseStream(
  contentChunks: number,
  usage?: { prompt_tokens: number; completion_tokens: number },
): AsyncGenerator<ChatCompletionChunkLike> {
  for (let i = 0; i < contentChunks; i++) {
    yield { choices: [{ delta: { content: "tok" } }] } as ChatCompletionChunkLike;
  }
  if (usage !== undefined) {
    yield { choices: [], usage } as ChatCompletionChunkLike;
  }
}

async function drainStream(stream: AsyncIterable<unknown>): Promise<number> {
  let n = 0;
  for await (const _ of stream) n += 1;
  return n;
}

describe("VapiBudgetGuard — reserve before the model turn", () => {
  it("blocks an over-budget non-streaming turn before the upstream call runs", async () => {
    const guard = new BudgetGuard(1.0, { priceOverrides: { m: PRICE } });
    guard.recordTool("prior", 1.0); // spend the ceiling
    const budget = new VapiBudgetGuard(guard, { model: "m" });

    let ran = false;
    await expect(
      budget.guardCompletion(() => {
        ran = true;
        return completion(1000, 500);
      }),
    ).rejects.toBeInstanceOf(BudgetExceeded);
    expect(ran).toBe(false); // upstream never reached
  });

  it("blocks an over-budget streaming turn synchronously, before the stream opens", () => {
    const guard = new BudgetGuard(1.0, { priceOverrides: { m: PRICE } });
    guard.recordTool("prior", 1.0);
    const budget = new VapiBudgetGuard(guard, { model: "m" });

    let opened = false;
    // guardStream reserves eagerly, so the block throws right here — not lazily on
    // first pull. The handler learns before piping anything to Vapi.
    expect(() =>
      budget.guardStream(() => {
        opened = true;
        return sseStream(3, { prompt_tokens: 10, completion_tokens: 5 });
      }),
    ).toThrow(BudgetExceeded);
    expect(opened).toBe(false);
  });
});

describe("VapiBudgetGuard — settle on real usage", () => {
  it("accrues the priced LLM cost from a non-streaming completion and frees the hold", async () => {
    const guard = new BudgetGuard(1.0, { priceOverrides: { m: PRICE } });
    const budget = new VapiBudgetGuard(guard, { model: "m" });

    const out = await budget.guardCompletion(() => completion(1000, 500));
    expect(out.usage?.completion_tokens).toBe(500); // returned untouched

    // 1000 * 1e-6 + 500 * 2e-6 = 0.002
    expect(guard.spentUsd).toBeCloseTo(0.002, 12);
    expect(guard.remainingUsd).toBeCloseTo(1.0 - 0.002, 12); // reservation consumed, not leaked
    expect(guard.spendLog).toHaveLength(1);
    expect(guard.spendLog[0]!.modelOrTool).toBe("m");
  });

  it("settles a streaming turn on the final chunk's usage and passes chunks through", async () => {
    const guard = new BudgetGuard(1.0, { priceOverrides: { m: PRICE } });
    const budget = new VapiBudgetGuard(guard, { model: "m" });

    const stream = budget.guardStream(() =>
      sseStream(4, { prompt_tokens: 1000, completion_tokens: 500 }),
    );
    const yielded = await drainStream(stream);

    expect(yielded).toBe(5); // 4 content chunks + 1 usage chunk, all forwarded
    expect(guard.spentUsd).toBeCloseTo(0.002, 12);
    expect(guard.remainingUsd).toBeCloseTo(1.0 - 0.002, 12);
  });

  it("uses the per-call model (Vapi's request model) over the constructor default", async () => {
    const guard = new BudgetGuard(1.0, { priceOverrides: { "req-model": PRICE } });
    const budget = new VapiBudgetGuard(guard); // no default model

    await budget.guardCompletion(() => completion(1000, 0), { model: "req-model" });
    expect(guard.spendLog[0]!.modelOrTool).toBe("req-model");
  });
});

describe("VapiBudgetGuard — release on error / abort", () => {
  it("releases the hold when the upstream completion call throws", async () => {
    const guard = new BudgetGuard(1.0, { priceOverrides: { m: PRICE } });
    const budget = new VapiBudgetGuard(guard, { model: "m" });

    // Prime a non-zero next-call estimate so the reservation is observable.
    await budget.guardCompletion(() => completion(1000, 500));
    const afterTurn1 = guard.remainingUsd; // 1.0 - 0.002

    await expect(
      budget.guardCompletion(() => Promise.reject(new Error("upstream 500"))),
    ).rejects.toThrow("upstream 500");
    expect(guard.remainingUsd).toBeCloseTo(afterTurn1, 12); // hold released
    expect(guard.spentUsd).toBeCloseTo(0.002, 12); // no new spend
  });

  it("releases the hold when the caller aborts the stream early (break)", async () => {
    const guard = new BudgetGuard(1.0, { priceOverrides: { m: PRICE } });
    const budget = new VapiBudgetGuard(guard, { model: "m" });

    await budget.guardCompletion(() => completion(1000, 500)); // prime estimate 0.002
    const afterTurn1 = guard.remainingUsd;

    const stream = budget.guardStream(() =>
      sseStream(10, { prompt_tokens: 1000, completion_tokens: 500 }),
    );
    // Break after one chunk — for-await calls stream.return(), unwinding to the
    // generator's finally, which releases the still-open hold.
    for await (const _ of stream) break;

    expect(guard.remainingUsd).toBeCloseTo(afterTurn1, 12);
    expect(guard.spentUsd).toBeCloseTo(0.002, 12); // aborted turn metered nothing
  });
});

describe("VapiBudgetGuard — missing usage fails loudly", () => {
  it("throws VapiUsageMissingError and releases the hold for a non-streaming completion with no usage", async () => {
    const guard = new BudgetGuard(1.0, { priceOverrides: { m: PRICE } });
    const budget = new VapiBudgetGuard(guard, { model: "m" });
    await budget.guardCompletion(() => completion(1000, 500)); // prime estimate
    const afterTurn1 = guard.remainingUsd;

    const noUsage: ChatCompletionLike = { usage: null }; // completion with no usable usage
    await expect(budget.guardCompletion(() => noUsage)).rejects.toBeInstanceOf(
      VapiUsageMissingError,
    );
    expect(guard.remainingUsd).toBeCloseTo(afterTurn1, 12); // released, not metered at $0
    expect(guard.spentUsd).toBeCloseTo(0.002, 12);
  });

  it("throws VapiUsageMissingError for a stream with no include_usage chunk", async () => {
    const guard = new BudgetGuard(1.0, { priceOverrides: { m: PRICE } });
    const budget = new VapiBudgetGuard(guard, { model: "m" });

    const stream = budget.guardStream(() => sseStream(3)); // no usage chunk emitted
    await expect(drainStream(stream)).rejects.toBeInstanceOf(VapiUsageMissingError);
    expect(guard.spentUsd).toBe(0); // nothing metered
  });
});

describe("VapiBudgetGuard — assistant-request admission via gates.vapi", () => {
  it("admits an under-budget call with the assistant/assistantId shape", () => {
    const guard = new BudgetGuard(1.0);
    const budget = new VapiBudgetGuard(guard);
    expect(budget.assistantRequest({ assistantId: "asst_123" })).toEqual({
      assistantId: "asst_123",
    });
  });

  it("rejects an exhausted call with a spoken error", () => {
    const guard = new BudgetGuard(1.0);
    guard.recordTool("prior", 1.0);
    const budget = new VapiBudgetGuard(guard);
    expect(
      budget.assistantRequest({ assistantId: "asst_123", errorMessage: "Out of budget." }),
    ).toEqual({ error: "Out of budget." });
  });
});

describe("VapiBudgetGuard — voice legs price via the cost map", () => {
  it("meters STT, TTS, and telephony from named vendors", () => {
    const guard = new BudgetGuard(10.0);
    const budget = new VapiBudgetGuard(guard, {
      sttModel: "deepgram-nova-3",
      ttsModel: "elevenlabs-flash-v2.5",
      telephony: "twilio-us-inbound-local",
    });

    const stt = budget.meterStt(30); // 30 s
    const tts = budget.meterTts(1_200); // 1.2k chars
    const tel = budget.meterTelephony(2.5); // 2.5 min

    expect(stt).toBeCloseTo(priceVoiceLeg("stt", 30, { model: "deepgram-nova-3" })!, 12);
    expect(tts).toBeCloseTo(priceVoiceLeg("tts", 1_200, { model: "elevenlabs-flash-v2.5" })!, 12);
    expect(tel).toBeCloseTo(priceVoiceLeg("telephony", 2.5, { model: "twilio-us-inbound-local" })!, 12);
    expect(guard.toolCosts["vapi-stt"]).toBeCloseTo(stt!, 12);
    expect(guard.toolCosts["vapi-tts"]).toBeCloseTo(tts!, 12);
    expect(guard.toolCosts["vapi-telephony"]).toBeCloseTo(tel!, 12);
  });

  it("leaves an unconfigured leg un-metered (token-only contract)", () => {
    const guard = new BudgetGuard(10.0);
    const budget = new VapiBudgetGuard(guard); // no voice vendors configured
    expect(budget.meterStt(30)).toBeNull();
    expect(guard.spentUsd).toBe(0);
  });

  it("fails closed on a vendor the voice map cannot price", () => {
    const guard = new BudgetGuard(10.0);
    const budget = new VapiBudgetGuard(guard, { sttModel: "no-such-vendor" });
    expect(() => budget.meterStt(1)).toThrow(UnpriceableVoiceError);
  });
});

describe("VapiBudgetGuard — no double release", () => {
  it("does not double-release when a streaming settle throws (non-priceable model)", async () => {
    const guard = new BudgetGuard(1.0); // failClosed default; "mystery-model" unpriceable
    guard.recordTool("prior", 0.1); // non-zero next-call estimate → the reserve holds 0.10
    const before = guard.remainingUsd; // 0.90
    const budget = new VapiBudgetGuard(guard, { model: "mystery-model" });
    const stream = budget.guardStream(() =>
      (async function* () {
        yield { usage: { prompt_tokens: 100, completion_tokens: 50 } };
      })(),
    );
    // Draining triggers settle("mystery-model", …) → UnpriceableModelError, which
    // releases the reservation ITSELF. The finally must not release it again — a
    // double release would drive `reserved` negative and inflate remainingUsd.
    await expect(drainStream(stream)).rejects.toThrow();
    expect(guard.remainingUsd).toBe(before); // released exactly once — ceiling intact
  });
});
