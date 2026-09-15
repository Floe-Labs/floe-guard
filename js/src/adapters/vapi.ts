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
 *      BEFORE the upstream call and meter the **real** `usage` after. Streaming
 *      also checks generated deltas and records partial spend on error/abort.
 *      Reserving first is what refuses a turn before its spend
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
 * after recording estimated partial spend and settling the hold. Set
 * `include_usage: true` on your upstream streaming call.
 *
 * Streaming enforces the LLM leg chunk-wise via StreamGuard. Closing the source
 * iterator requests cancellation; actual provider cancellation depends on the
 * source. This does not end the Vapi call or stop STT/TTS/telephony billing.
 */

import { BudgetExceeded, FloeGuardError } from "../errors.js";
import { vapi as vapiGate } from "../gates.js";
import type { BudgetGuard, ReservationHandle } from "../guard.js";
import { StreamGuard } from "../stream.js";
import { priceVoiceLeg } from "../voice-pricing.js";

/**
 * The OpenAI `usage` block we settle against. Vapi speaks the OpenAI wire format,
 * so the fields are snake_case (`prompt_tokens` / `completion_tokens`). Typed
 * structurally — the adapter carries no dependency on `openai` or any Vapi SDK.
 */
export interface OpenAiUsageLike {
  readonly prompt_tokens: number;
  readonly completion_tokens: number;
  readonly prompt_tokens_details?: { readonly cached_tokens?: number };
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
  readonly choices?: ReadonlyArray<{
    readonly delta?: {
      readonly content?: string | null;
      readonly refusal?: string | null;
      readonly function_call?: { readonly name?: string; readonly arguments?: string };
      readonly tool_calls?: ReadonlyArray<{
        readonly function?: { readonly name?: string; readonly arguments?: string };
      }>;
    };
  }>;
}

/**
 * Thrown when a guarded stream ends with no token `usage` to settle against.
 *
 * Almost always the fix is `stream_options: { include_usage: true }` on the
 * upstream streaming request (OpenAI omits `usage` from SSE without it). We refuse
 * rather than claim estimated partial spend is authoritative usage. Extends
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
        `cannot reconcile the turn to provider usage. Streaming records partial estimates.`,
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
 * before the model turn, enforce streaming deltas, settle on real OpenAI `usage`,
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
    this.guard.settle(model, usage.prompt, usage.completion, {
      reserved,
      cacheReadInputTokens: usage.cacheRead,
    });
    return completion;
  }

  /**
   * Guard a **streaming** model turn: reserve, meter before forwarding each chunk,
   * then reconcile estimates to the final chunk's `usage` (including cached input).
   *
   * Reserving first throws {@link BudgetExceeded} BEFORE the stream is opened when
   * the turn would cross the ceiling (this method throws synchronously — the handler
   * learns immediately, before piping anything to Vapi). Unpriceable models are
   * also rejected eagerly unless fail-open. A crossing chunk is billed but not
   * forwarded; BudgetExceeded is raised during iteration after partial settlement.
   *
   * **Usage requirement:** OpenAI SSE only carries `usage` when the upstream request
   * set `stream_options: { include_usage: true }`. A stream that ends with no usage
   * anywhere fails loudly ({@link VapiUsageMissingError}) after recording estimated
   * partial spend. Error/early-break paths also settle generated usage. Text and
   * tool-call fragments use ~4 characters/token by default, overridable with
   * `countTokens`. Supply `promptTokens` for input accounting before final usage.
   *
   * The returned iterable holds the reservation until it is consumed to completion,
   * or explicitly returned/thrown into, including before its first pull. Abandoning
   * an iterator without closing it still holds its reservation.
   */
  guardStream<C extends ChatCompletionChunkLike>(
    run: StreamSource<C>,
    options: {
      model?: string;
      estimatedCost?: number;
      /** Input-token estimate for partial accounting; defaults to zero. */
      promptTokens?: number;
      /** Tokenizer for visible output deltas; defaults to approxTokens. */
      countTokens?: (delta: string) => number;
    } = {},
  ): AsyncIterableIterator<C> {
    const model = this.resolveModel(options.model);
    // Eager reserve (outside the generator) so a block throws synchronously here,
    // not lazily on first pull — the handler refuses the turn before streaming.
    const reserved = this.guard.reserve(options.estimatedCost);
    const meter = new StreamGuard(this.guard, model, {
      reserved, promptTokens: options.promptTokens, countTokens: options.countTokens,
    });
    return this.enforceStream(run, model, reserved, meter);
  }

  private enforceStream<C extends ChatCompletionChunkLike>(
    run: StreamSource<C>, model: string, reserved: ReservationHandle, meter: StreamGuard,
  ): AsyncIterableIterator<C> {
    const guard = this.guard;
    let started = false;
    let released = false;
    async function* iterate(): AsyncIterableIterator<C> {
      started = true;
      let opened = false;
      let settled = false;
      try {
        const source = await run();
        opened = true;
        for await (const chunk of source) {
          const usage = readUsage(chunk.usage);
          if (usage !== null) {
            // Own cleanup before settlement, which releases even if pricing fails.
            settled = true;
            meter.finish({
              promptTokens: usage.prompt, completionTokens: usage.completion,
              cacheReadInputTokens: usage.cacheRead,
            });
            if (guard.spentUsd > guard.limitUsd + 1e-12) guard._blockStream();
            yield chunk;
            // OpenAI's usage-bearing chunk is final. Close the source on resumption.
            return;
          }
          meter.feedText(chunkText(chunk));
          yield chunk;
        }
        throw new VapiUsageMissingError(model);
      } finally {
        if (!opened) guard.release(reserved);
        else if (!settled) meter.close();
      }
    }
    const iterator = iterate();
    // An async generator's finally does not run for return()/throw() before next().
    const releaseUnstarted = () => {
      if (!started && !released) { released = true; guard.release(reserved); }
    };
    return {
      next: () => iterator.next(),
      return: value => { releaseUnstarted(); return iterator.return!(value); },
      throw: error => { releaseUnstarted(); return iterator.throw!(error); },
      [Symbol.asyncIterator]() { return this; },
    };
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
): { prompt: number; completion: number; cacheRead: number } | null {
  if (usage === null || usage === undefined) return null;
  const prompt = usage.prompt_tokens;
  const completion = usage.completion_tokens;
  if (typeof prompt !== "number" || !Number.isFinite(prompt)) return null;
  if (typeof completion !== "number" || !Number.isFinite(completion)) return null;
  const promptClamped = Math.max(0, prompt);
  const completionClamped = Math.max(0, completion);
  const cachedRaw = usage.prompt_tokens_details?.cached_tokens;
  const cached =
    typeof cachedRaw === "number" && Number.isFinite(cachedRaw) ? Math.max(0, cachedRaw) : 0;
  const cacheRead = Math.min(cached, promptClamped);
  return { prompt: promptClamped - cacheRead, completion: completionClamped, cacheRead };
}

/** Visible generated text across all choices, including tool-call fragments. */
function chunkText(chunk: ChatCompletionChunkLike): string {
  let text = "";
  for (const choice of chunk.choices ?? []) {
    const delta = choice.delta;
    if (!delta) continue;
    text += delta.content ?? "";
    text += delta.refusal ?? "";
    text += delta.function_call?.name ?? "";
    text += delta.function_call?.arguments ?? "";
    for (const call of delta.tool_calls ?? []) {
      text += call.function?.name ?? "";
      text += call.function?.arguments ?? "";
    }
  }
  return text;
}
