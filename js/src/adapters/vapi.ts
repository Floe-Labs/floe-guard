/**
 * Vapi custom-LLM adapter (no SDK dependency — typed structurally).
 *
 * Vapi's **custom-LLM** feature points Vapi's model leg at YOUR OpenAI-compatible
 * `POST /chat/completions` endpoint (docs.vapi.ai/customization/custom-llm/using-your-server).
 * Vapi sends an OpenAI-format request (`{ model, messages, temperature, tools?,
 * stream }`); your endpoint proxies it to an upstream LLM and returns an
 * OpenAI-compatible completion — a single JSON object, or an SSE stream of chunks.
 *
 * Like the LiveKit adapter, a Vapi call has no single wrap point that sees every
 * cost: the custom-LLM proxy only sees the **model leg**. So this adapter has three
 * jobs, and only the first is automatic:
 *
 *   1. **Guard the model turn** — {@link VapiBudgetGuard.guardCompletion} (JSON)
 *      and {@link VapiBudgetGuard.guardStream} (SSE) reserve the estimated cost
 *      BEFORE the upstream call, meter the **real** `usage` after, and release the
 *      hold on error/abort. Reserving first is what refuses a turn before its spend
 *      lands: an over-budget turn throws {@link BudgetExceeded} instead of proxying.
 *   2. **Admit the call** — {@link VapiBudgetGuard.assistantRequest} answers Vapi's
 *      `assistant-request` webhook from the remaining budget (exhausted → a spoken
 *      error; else hand back `assistant`/`assistantId`), delegating to {@link gates.vapi}.
 *   3. **Meter the other legs** — the proxy never sees STT/TTS/telephony, so
 *      {@link VapiBudgetGuard.meterStt} / {@link VapiBudgetGuard.meterTts} /
 *      {@link VapiBudgetGuard.meterTelephony} accrue them explicitly (the Vapi twin
 *      of LiveKit's `.meterTelephony`), priced from the bundled voice cost map or a
 *      per-unit override.
 *
 *     import { BudgetGuard } from "floe-guard";
 *     import { VapiBudgetGuard } from "floe-guard/adapters/vapi";
 *
 *     const guard = new BudgetGuard(1.0);
 *     const budget = new VapiBudgetGuard(guard, {
 *       sttModel: "deepgram-nova-3",
 *       ttsModel: "elevenlabs-flash-v2.5",
 *       telephony: "twilio-us-inbound-local",
 *     });
 *
 *     // POST /chat/completions — the custom-LLM endpoint Vapi calls each turn.
 *     app.post("/chat/completions", async (req, reply) => {
 *       const { model, messages, tools, stream } = req.body;
 *       try {
 *         if (stream) {
 *           // Upstream MUST set stream_options:{ include_usage: true } — see below.
 *           const sse = budget.guardStream(
 *             () => openai.chat.completions.create({ model, messages, tools, stream: true,
 *               stream_options: { include_usage: true } }),
 *             { model },
 *           );
 *           return reply.sse(sse); // pipe chunks straight through
 *         }
 *         const completion = await budget.guardCompletion(
 *           () => openai.chat.completions.create({ model, messages, tools }),
 *           { model },
 *         );
 *         return reply.send(completion);
 *       } catch (err) {
 *         if (err instanceof BudgetExceeded) return reply.code(402).send({ error: String(err) });
 *         throw err;
 *       }
 *     });
 *
 * ## The streaming usage requirement (read this)
 *
 * OpenAI-style SSE omits `usage` from every chunk **unless** the caller sets
 * `stream_options: { include_usage: true }` on the upstream request — with it, a
 * final chunk (empty `choices`) carries the token `usage`. This adapter meters the
 * model turn from that real `usage`; if a stream ends with no usage anywhere,
 * {@link VapiBudgetGuard.guardStream} **fails loudly** ({@link VapiUsageMissingError})
 * and releases the hold rather than silently metering the turn at $0. Set
 * `include_usage: true` on your upstream streaming call.
 *
 * Scope is strictly **pre-call admission plus per-turn settlement**. There is no
 * mid-call intervention: an admitted turn runs to completion; nothing here cuts a
 * turn — or a stream — off partway.
 */

import { BudgetExceeded, FloeGuardError } from "../errors.js";
import { vapi as vapiGate } from "../gates.js";
import type { BudgetGuard, ReservationHandle } from "../guard.js";
import { priceVoiceLeg } from "../voice-pricing.js";

/**
 * The OpenAI `usage` block we settle against. Vapi speaks the OpenAI wire format,
 * so the fields are snake_case (`prompt_tokens` / `completion_tokens`). Typed
 * structurally — the adapter carries no dependency on `openai` or any Vapi SDK.
 */
export interface OpenAiUsageLike {
  readonly prompt_tokens: number;
  readonly completion_tokens: number;
}

/** A non-streaming OpenAI `ChatCompletion` — we read its `usage`. */
export interface ChatCompletionLike {
  readonly usage?: OpenAiUsageLike | null;
}

/**
 * One OpenAI `ChatCompletionChunk` from an SSE stream. Only the final chunk carries
 * `usage`, and only when the upstream request set
 * `stream_options: { include_usage: true }` — see the module docstring.
 */
export interface ChatCompletionChunkLike {
  readonly usage?: OpenAiUsageLike | null;
}

/**
 * Thrown when a guarded stream ends with no token `usage` to settle against.
 *
 * Almost always the fix is `stream_options: { include_usage: true }` on the
 * upstream streaming request (OpenAI omits `usage` from SSE without it). We refuse
 * rather than meter the turn at $0 — "we cannot cap what we cannot measure". Extends
 * {@link FloeGuardError} so it is caught by the same family as the priced errors;
 * it is adapter-local (not part of the shared `errors.ts` cross-language family).
 */
export class VapiUsageMissingError extends FloeGuardError {
  readonly model: string;

  constructor(model: string) {
    super(
      `Vapi custom-LLM stream for model '${model}' ended with no token usage to ` +
        `settle against. OpenAI-style SSE only includes usage when the upstream ` +
        `request sets stream_options:{ include_usage: true } — set it, or the guard ` +
        `cannot meter the turn (it refuses rather than accrue a silent $0).`,
    );
    this.name = "VapiUsageMissingError";
    this.model = model;
  }
}

export interface VapiBudgetGuardOptions {
  /**
   * Default model id to settle LLM cost against when a per-call `model` is not
   * passed to {@link VapiBudgetGuard.guardCompletion} / `guardStream`. Vapi's
   * request carries the model, so passing it per-call is usual; this is the
   * fallback. Must be priceable via the bundled cost map or the guard's
   * `priceOverrides`.
   */
  model?: string;
  /** Voice-map vendor key for the STT leg (e.g. `"deepgram-nova-3"`). */
  sttModel?: string;
  /** Voice-map vendor key for the TTS leg (e.g. `"elevenlabs-flash-v2.5"`). */
  ttsModel?: string;
  /** Voice-map vendor key for the telephony leg (e.g. `"twilio-us-inbound-local"`). */
  telephony?: string;
  /** Per-second STT override — wins over `sttModel`. Omit both to leave STT un-metered. */
  sttUsdPerSecond?: number;
  /** Per-1k-chars TTS override — wins over `ttsModel`. Omit both to leave TTS un-metered. */
  ttsUsdPer1kChars?: number;
  /** Per-minute telephony override — wins over `telephony`. */
  telephonyUsdPerMinute?: number;
}

/** A source of an OpenAI completion — a thunk, so we reserve BEFORE the call runs. */
type CompletionSource<T> = () => Promise<T> | T;

/** A source of an SSE chunk stream — a thunk, for the same reserve-first reason. */
type StreamSource<C> = () => AsyncIterable<C> | Promise<AsyncIterable<C>>;

/**
 * Enforce a {@link BudgetGuard} ceiling on a Vapi custom-LLM endpoint: reserve
 * before the model turn, settle on the real OpenAI `usage`, release on error/abort,
 * and meter the STT/TTS/telephony legs the proxy never sees.
 */
export class VapiBudgetGuard {
  private readonly guard: BudgetGuard;
  private readonly model?: string;
  private readonly sttModel?: string;
  private readonly ttsModel?: string;
  private readonly telephony?: string;
  private readonly sttUsdPerSecond?: number;
  private readonly ttsUsdPer1kChars?: number;
  private readonly telephonyUsdPerMinute?: number;

  constructor(guard: BudgetGuard, options: VapiBudgetGuardOptions = {}) {
    this.guard = guard;
    this.model = options.model;
    this.sttModel = options.sttModel;
    this.ttsModel = options.ttsModel;
    this.telephony = options.telephony;
    this.sttUsdPerSecond = options.sttUsdPerSecond;
    this.ttsUsdPer1kChars = options.ttsUsdPer1kChars;
    this.telephonyUsdPerMinute = options.telephonyUsdPerMinute;
  }

  /**
   * Answer Vapi's `assistant-request` webhook from the remaining budget.
   *
   * Budget exhausted → `{ error }` (Vapi speaks it, then ends the call); otherwise
   * admits with `{ assistantId }` (precedence) or `{ assistant }`. A thin wrapper
   * over {@link gates.vapi} — the same webhook contract the hosted gateway serves,
   * so the paid upgrade is a URL swap, not a rewrite. Respond within ~7.5 s.
   *
   * This is coarse, non-binding pre-call admission (a budget read, no reservation);
   * the binding hard-stop is the per-turn reserve in {@link guardCompletion} /
   * {@link guardStream}. Pass `estimatedCallUsd` (e.g. `$/min × expected minutes`)
   * to reject earlier, when the remaining budget can't cover the call.
   *
   * @throws RangeError when admitted but neither `assistant` nor `assistantId` was
   *   given — there'd be nothing to hand Vapi (surfaced by {@link gates.vapi}).
   */
  assistantRequest(
    options: {
      assistant?: Record<string, unknown>;
      assistantId?: string;
      errorMessage?: string;
      estimatedCallUsd?: number;
    } = {},
  ): Record<string, unknown> {
    return vapiGate(this.guard, options);
  }

  /**
   * Guard a **non-streaming** model turn: reserve, run the upstream completion,
   * settle on its real `usage`, release the hold on error.
   *
   * Reserving first throws {@link BudgetExceeded} BEFORE `run` is called when the
   * turn would cross the ceiling, so an over-budget turn never reaches the upstream
   * LLM. The completion is returned untouched for the handler to forward to Vapi. A
   * completion with no `usage` fails loudly ({@link VapiUsageMissingError}) and
   * releases the hold rather than metering $0.
   */
  async guardCompletion<T extends ChatCompletionLike>(
    run: CompletionSource<T>,
    options: { model?: string; estimatedCost?: number } = {},
  ): Promise<T> {
    const model = this.resolveModel(options.model);
    // Synchronous at the top of the async body: a block rejects the returned
    // promise with BudgetExceeded before `run` is ever called.
    const reserved = this.guard.reserve(options.estimatedCost);

    let completion: T;
    try {
      completion = await run();
    } catch (err) {
      this.guard.release(reserved);
      throw err;
    }

    const usage = readUsage(completion.usage);
    if (usage === null) {
      this.guard.release(reserved);
      throw new VapiUsageMissingError(model);
    }
    this.guard.settle(model, usage.prompt, usage.completion, { reserved });
    return completion;
  }

  /**
   * Guard a **streaming** model turn: reserve, then wrap the SSE chunk stream so it
   * settles on the final chunk's `usage`, releasing the hold on error or early abort.
   *
   * Reserving first throws {@link BudgetExceeded} BEFORE the stream is opened when
   * the turn would cross the ceiling (this method throws synchronously — the handler
   * learns immediately, before piping anything to Vapi). Chunks pass through
   * untouched; the returned async iterable is what you forward to the SSE response.
   *
   * **Usage requirement:** OpenAI SSE only carries `usage` when the upstream request
   * set `stream_options: { include_usage: true }`. A stream that ends with no usage
   * anywhere fails loudly ({@link VapiUsageMissingError}) and releases the hold — it
   * is not metered at $0. Breaking out of the consuming loop early (abort/interrupt)
   * also releases the hold; nothing is metered for an aborted turn.
   *
   * The returned iterable holds the reservation until it is consumed to completion,
   * or returned/aborted — pipe it straight to the response so the hold cannot leak.
   */
  guardStream<C extends ChatCompletionChunkLike>(
    run: StreamSource<C>,
    options: { model?: string; estimatedCost?: number } = {},
  ): AsyncIterableIterator<C> {
    const model = this.resolveModel(options.model);
    // Eager reserve (outside the generator) so a block throws synchronously here,
    // not lazily on first pull — the handler refuses the turn before streaming.
    const reserved = this.guard.reserve(options.estimatedCost);
    return this.iterateStream(run, model, reserved);
  }

  private async *iterateStream<C extends ChatCompletionChunkLike>(
    run: StreamSource<C>,
    model: string,
    reserved: ReservationHandle,
  ): AsyncIterableIterator<C> {
    let usage: { prompt: number; completion: number } | null = null;
    let settled = false;
    try {
      const stream = await run();
      for await (const chunk of stream) {
        // Last non-null usage wins — the final (empty-choices) chunk carries it.
        usage = readUsage(chunk.usage) ?? usage;
        yield chunk;
      }
      // Clean end: settle on real usage, or fail loudly if none was ever seen.
      if (usage === null) throw new VapiUsageMissingError(model);
      // Mark settled BEFORE the call: guard.settle owns the reservation on every
      // exit — it releases the hold on its own failure paths (unpriceable model,
      // priceTokens error) before throwing — so the finally must not release it a
      // second time (a double release drives `reserved` negative and weakens the
      // ceiling for other in-flight turns).
      settled = true;
      this.guard.settle(model, usage.prompt, usage.completion, { reserved });
    } finally {
      // Any exit before settle — error mid-stream, missing usage, or an early
      // consumer abort (for-await break -> generator.return()) — frees the hold.
      if (!settled) this.guard.release(reserved);
    }
  }

  /** Accrue STT spend for `seconds` of transcribed audio (per second). See {@link meterTelephony}. */
  meterStt(seconds: number): number | null {
    const cost = priceVoiceLeg("stt", seconds, {
      model: this.sttModel,
      override: this.sttUsdPerSecond,
    });
    if (cost !== null) this.guard.recordTool("vapi-stt", cost);
    return cost;
  }

  /** Accrue TTS spend for `characters` of synthesized speech (per 1k chars). See {@link meterTelephony}. */
  meterTts(characters: number): number | null {
    const cost = priceVoiceLeg("tts", characters, {
      model: this.ttsModel,
      override: this.ttsUsdPer1kChars,
    });
    if (cost !== null) this.guard.recordTool("vapi-tts", cost);
    return cost;
  }

  /**
   * Accrue telephony spend for `minutes` of call time (per minute).
   *
   * The custom-LLM proxy sees only the model leg, so the caller drives this (and
   * `meterStt` / `meterTts`) explicitly — from Vapi's `end-of-call-report` webhook,
   * or as the call accrues. This is per-unit accrual, not live line-cutting: the
   * guard meters the leg, it does not cut the call mid-turn. Priced from the voice
   * map when the vendor is set, or the per-unit override; returns `null` (no-op)
   * when the leg is unconfigured, and fails closed ({@link UnpriceableVoiceError})
   * on a vendor the voice map cannot price. Returns the USD accrued, if any.
   */
  meterTelephony(minutes: number): number | null {
    const cost = priceVoiceLeg("telephony", minutes, {
      model: this.telephony,
      override: this.telephonyUsdPerMinute,
    });
    if (cost !== null) this.guard.recordTool("vapi-telephony", cost);
    return cost;
  }

  /**
   * The model to settle against: the per-call override (Vapi's request model) or
   * the constructor default. Fails loudly if neither is set — the guard cannot
   * price a turn it cannot name.
   */
  private resolveModel(override?: string): string {
    const model = override ?? this.model;
    if (model === undefined || model === null || model === "") {
      throw new RangeError(
        "no model to settle the Vapi turn against: pass { model } to " +
          "guardCompletion/guardStream (Vapi's request model), or set `model` on the " +
          "VapiBudgetGuard constructor.",
      );
    }
    return model;
  }
}

// Re-export so a handler can `catch (e) { if (e instanceof BudgetExceeded) ... }`
// without a second import path.
export { BudgetExceeded };

/**
 * Read an OpenAI `usage` block into settled prompt/completion counts, or `null`
 * when absent/malformed. Both fields must be finite numbers — a partial or
 * non-numeric usage is treated as no usage (fail-closed), never coerced to 0.
 */
function readUsage(
  usage: OpenAiUsageLike | null | undefined,
): { prompt: number; completion: number } | null {
  if (usage === null || usage === undefined) return null;
  const prompt = usage.prompt_tokens;
  const completion = usage.completion_tokens;
  if (typeof prompt !== "number" || !Number.isFinite(prompt)) return null;
  if (typeof completion !== "number" || !Number.isFinite(completion)) return null;
  return { prompt, completion };
}
