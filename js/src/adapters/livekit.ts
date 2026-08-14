/**
 * LiveKit Agents adapter (optional peer: `npm i @livekit/agents`).
 *
 * Like the Python `LiveKitBudgetGuard`, a LiveKit `AgentSession` (STT -> LLM ->
 * TTS) has no single call site to wrap: turns fire for the life of a call. The
 * two enforcement points are the agent's LLM turn hook (before the LLM call, so a
 * turn is admitted or blocked before its TTS/audio spend piles on) and the
 * session's usage/metrics event (real usage, after). {@link LiveKitBudgetGuard}
 * holds the reservation state across both.
 *
 *     import { BudgetGuard } from "floe-guard";
 *     import { LiveKitBudgetGuard } from "floe-guard/adapters/livekit";
 *
 *     const guard = new BudgetGuard(1.0, {
 *       priceOverrides: {
 *         "gemini-2.0-flash": { inputCostPerToken: 0.30e-6, outputCostPerToken: 2.5e-6 },
 *       },
 *     });
 *     const budget = new LiveKitBudgetGuard(guard, { model: "gemini-2.0-flash" });
 *
 *     const session = new AgentSession({ ... });
 *     budget.attach(session, agent);   // wire reserve / settle / release
 *     await session.start({ agent, room });
 *
 * ## Why agents-js, not a blind port of the Python adapter
 *
 * The Node SDK (`@livekit/agents`) differs from Python and was verified against
 * the installed types (v1.6.x):
 *
 * - **LLM turn hook.** Python wraps `agent.llm_node`, an async generator. The JS
 *   analog is `Agent.llmNode(chatCtx, toolCtx, modelSettings):
 *   Promise<ReadableStream<ChatChunk | string | FlushSentinel> | null>` — it
 *   returns a web `ReadableStream`, not a generator. We reserve BEFORE calling the
 *   original, then wrap the returned stream so a consumer cancel (interrupt)
 *   releases the hold — the JS twin of Python's generator `except BaseException`.
 *
 * - **Settlement event.** Session-level `metrics_collected` is *deprecated in
 *   Python* (→ `session_usage_updated` + `ChatMessage.metrics`). In **agents-js**
 *   it is **not** deprecated: `AgentSession` (a `TypedEmitter<AgentSessionCallbacks>`)
 *   still emits `metrics_collected` with a `MetricsCollectedEvent { metrics:
 *   AgentMetrics }`, and `AgentMetrics` is a discriminated union keyed by `type`
 *   (`'llm_metrics' | 'stt_metrics' | 'tts_metrics' | ...`) — not the `isinstance`
 *   classes Python uses. We settle on that live event.
 *
 * LiveKit's `LLMMetrics` does not name the served model in a way we rely on, so
 * cost is settled against the `model` passed here (as in Python). STT/TTS spend —
 * often a voice agent's larger bill — is metered whenever you name the vendor:
 * pass `sttModel` / `ttsModel` and the rate comes from the bundled voice cost map,
 * or pass a per-unit override (`sttUsdPerSecond` / `ttsUsdPer1kChars`) which wins
 * over the map. Telephony (per-minute) is metered via {@link meterTelephony},
 * since LiveKit emits no telephony metric. A leg with neither a vendor nor an
 * override is left un-metered (the token-only contract); a vendor the voice map
 * cannot price fails closed ({@link UnpriceableVoiceError}).
 *
 * Scope is strictly **pre-turn / pre-call admission plus per-turn settlement**.
 * There is no mid-call intervention: an admitted turn runs to completion; nothing
 * here cuts a turn off partway.
 */

import { BudgetExceeded } from "../errors.js";
import { preCall } from "../gates.js";
import type { BudgetGuard, ReservationHandle } from "../guard.js";
import { priceVoiceLeg } from "../voice-pricing.js";

/**
 * The `LLMMetrics` fields we settle against (agents-js discriminated-union member
 * `type: 'llm_metrics'`). Typed structurally so the adapter carries no hard
 * dependency on `@livekit/agents` — the same idiom the Vercel-AI middleware uses
 * for `ai`. The test file pins these shapes against the real exported types.
 */
interface LlmMetricsLike {
  readonly type: "llm_metrics";
  readonly promptTokens: number;
  readonly completionTokens: number;
}

/** `STTMetrics` fields we meter against (`type: 'stt_metrics'`). Duration is ms. */
interface SttMetricsLike {
  readonly type: "stt_metrics";
  /** Duration of the pushed audio in **milliseconds** (LiveKit's `audioDurationMs`). */
  readonly audioDurationMs: number;
}

/** `TTSMetrics` fields we meter against (`type: 'tts_metrics'`). */
interface TtsMetricsLike {
  readonly type: "tts_metrics";
  readonly charactersCount: number;
}

/** Any member of LiveKit's `AgentMetrics` union — we branch on `type`. */
type AgentMetricsLike =
  | LlmMetricsLike
  | SttMetricsLike
  | TtsMetricsLike
  | { readonly type: string };

/** The `MetricsCollectedEvent` payload (`{ metrics: AgentMetrics }`). */
interface MetricsCollectedEventLike {
  readonly metrics: AgentMetricsLike;
}

/**
 * The slice of `AgentSession` we wire onto — its typed event emitter. Structural,
 * so a stubbed session in a test (or the real `AgentSession`) both satisfy it.
 */
export interface LiveKitSessionLike {
  on(event: "metrics_collected", listener: (ev: MetricsCollectedEventLike) => void): unknown;
  on(event: "close", listener: (ev: unknown) => void): unknown;
}

/** A web `ReadableStream` (Node 18+ global), as returned by `Agent.llmNode`. */
type LlmNodeStream = ReadableStream<unknown>;

/**
 * The slice of `Agent` we wrap — its `llmNode` turn hook. The real signature is
 * `llmNode(chatCtx, toolCtx, modelSettings): Promise<ReadableStream<...> | null>`;
 * we forward the arguments unchanged, so an upstream signature change can't break
 * the wrapper.
 */
export interface LiveKitAgentLike {
  llmNode: (...args: unknown[]) => Promise<LlmNodeStream | null>;
}

/** Callback invoked when a turn is blocked, so the bot can speak a wrap-up line. */
export type OnBudgetExceeded = (error: BudgetExceeded) => Promise<void> | void;

export interface LiveKitBudgetGuardOptions {
  /**
   * Model id to settle LLM cost against (LiveKit's `LLMMetrics` carries no model
   * name we rely on). Must be priceable via the bundled cost map or the guard's
   * `priceOverrides`.
   */
  model: string;
  /**
   * Invoked with the {@link BudgetExceeded} when a turn is blocked, so a bot can
   * speak a graceful "wrapping up" line before the turn ends silently (the LLM
   * hook returns no stream). If omitted, the {@link BudgetExceeded} propagates out
   * of `llmNode`.
   */
  onBudgetExceeded?: OnBudgetExceeded;
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

/**
 * One `llmNode` invocation awaiting (or already past) its `LLMMetrics` event.
 *
 * `open` means the USD amount is still held on the {@link BudgetGuard}. Early
 * release (next turn, interrupt, close) clears the guard hold but leaves the slot
 * in the FIFO queue so a delayed metrics event settles against this turn's slot
 * (with `amount` 0) instead of stealing a later turn's hold.
 */
interface TurnSlot {
  amount: ReservationHandle;
  open: boolean;
}

/**
 * Enforce a {@link BudgetGuard} ceiling on a LiveKit `AgentSession`, one turn at
 * a time: reserve before the LLM turn, settle on real usage, release on interrupt.
 */
export class LiveKitBudgetGuard {
  private readonly guard: BudgetGuard;
  private readonly model: string;
  private readonly onBudgetExceeded?: OnBudgetExceeded;
  private readonly sttModel?: string;
  private readonly ttsModel?: string;
  private readonly telephony?: string;
  private readonly sttUsdPerSecond?: number;
  private readonly ttsUsdPer1kChars?: number;
  private readonly telephonyUsdPerMinute?: number;

  private pending = false;
  private active: TurnSlot | null = null;
  /** FIFO of turns that may still emit `LLMMetrics` (including early-released). */
  private readonly slots: TurnSlot[] = [];

  constructor(guard: BudgetGuard, options: LiveKitBudgetGuardOptions) {
    this.guard = guard;
    this.model = options.model;
    this.onBudgetExceeded = options.onBudgetExceeded;
    this.sttModel = options.sttModel;
    this.ttsModel = options.ttsModel;
    this.telephony = options.telephony;
    this.sttUsdPerSecond = options.sttUsdPerSecond;
    this.ttsUsdPer1kChars = options.ttsUsdPer1kChars;
    this.telephonyUsdPerMinute = options.telephonyUsdPerMinute;
    // Bound once so the emitter calls them with the right `this`.
    this.onMetrics = this.onMetrics.bind(this);
    this.onClose = this.onClose.bind(this);
  }

  /**
   * Wire reserve (`agent.llmNode`), settle/meter (`metrics_collected`) and release
   * (`close`) onto a session + agent pair. Reserving before the LLM turn is what
   * blocks a turn before its TTS/audio spend piles on.
   */
  attach(session: LiveKitSessionLike, agent: LiveKitAgentLike): void {
    // Bind so the original keeps its `this` when we call it from the wrapper.
    const origLlmNode = agent.llmNode.bind(agent);

    const guardedLlmNode = async (...args: unknown[]): Promise<LlmNodeStream | null> => {
      // A previous turn that never settled still holds its reservation — release
      // the guard hold before opening this one, but keep a queue slot so delayed
      // metrics cannot steal this turn's hold.
      if (this.pending) this.earlyReleaseActive();

      let handle: ReservationHandle;
      try {
        handle = this.guard.reserve(); // throws BudgetExceeded before the turn runs
      } catch (err) {
        this.clearActive();
        if (err instanceof BudgetExceeded && this.onBudgetExceeded !== undefined) {
          // Let the bot speak a wrap-up line; the turn ends with no stream.
          await this.onBudgetExceeded(err);
          return null;
        }
        throw err;
      }

      const slot: TurnSlot = { amount: handle, open: true };
      this.slots.push(slot);
      this.active = slot;
      this.pending = true;

      let stream: LlmNodeStream | null;
      try {
        stream = await origLlmNode(...args);
      } catch (err) {
        // Release only if this invocation still owns the open hold — a later turn
        // may already have early-released ours and reserved its own.
        if (this.active === slot) this.earlyReleaseActive();
        throw err;
      }

      // No stream to wrap (realtime / text-only turn): leave the slot open for the
      // metrics event to settle against.
      if (stream === null) return null;
      return this.guardStream(stream, slot);
    };

    agent.llmNode = guardedLlmNode;
    session.on("metrics_collected", this.onMetrics);
    session.on("close", this.onClose);
  }

  /**
   * Pre-call admission decision (`true` = admit, `false` = reject), delegating to
   * {@link preCall} on the wrapped guard's remaining budget. A coarse,
   * non-binding preflight — the binding hard-stop is the per-turn reserve wired by
   * {@link attach}. Pass `estimatedCallUsd` (e.g. `$/min × expected minutes`) to
   * reject earlier, when the remaining budget can't cover the call.
   */
  admitCall(options: { estimatedCallUsd?: number } = {}): boolean {
    return preCall(this.guard, options);
  }

  /**
   * Accrue telephony spend for `minutes` of call time (per-minute).
   *
   * LiveKit emits no telephony metric, so the transport/caller drives this — call
   * it as the call accrues minutes, or once with the final duration. This is
   * per-minute accrual, not live line-cutting: the guard meters the leg, it does
   * not cut the phone line mid-call. Priced from the voice map when `telephony` is
   * set, or the `telephonyUsdPerMinute` override; returns `null` (no-op) when the
   * leg is unconfigured, and fails closed on a vendor the voice map cannot price.
   * Returns the USD accrued, if any.
   */
  meterTelephony(minutes: number): number | null {
    const cost = priceVoiceLeg("telephony", minutes, {
      model: this.telephony,
      override: this.telephonyUsdPerMinute,
    });
    if (cost !== null) this.guard.recordTool("livekit-telephony", cost);
    return cost;
  }

  /**
   * Wrap the `llmNode` stream so a consumer cancel (interrupt) releases this turn's
   * hold — the JS twin of Python's generator `except BaseException`. Chunks pass
   * through untouched; a clean end leaves the hold for the metrics event to settle.
   */
  private guardStream(stream: LlmNodeStream, slot: TurnSlot): LlmNodeStream {
    const release = (): void => {
      // Release only if this turn still owns the open hold — a metrics event may
      // already have settled it, or a later turn early-released it.
      if (this.active === slot) this.earlyReleaseActive();
    };
    return stream.pipeThrough(
      new TransformStream<unknown, unknown>({
        transform(chunk, controller) {
          controller.enqueue(chunk); // passthrough; we only need the lifecycle hooks
        },
        // Clean completion: do NOT release — leave the hold for the metrics event
        // to settle against (mirrors Python's generator finishing normally).
        flush() {},
        // Interrupt / downstream cancel: free the held budget. `cancel` is valid
        // per the Streams spec and supported in Node 18+, but TS's Transformer lib
        // type lags and omits it — cast to keep the type check (as the middleware does).
        cancel() {
          release();
        },
      } as Transformer<unknown, unknown>),
    );
  }

  private onMetrics(ev: MetricsCollectedEventLike): void {
    const m = ev.metrics;
    if (m.type === "llm_metrics") {
      const metrics = m as LlmMetricsLike;
      // Pop the oldest turn slot. Early-released turns contribute 0 so a delayed
      // metrics event meters usage without consuming a later hold. An empty queue
      // (reserve hook bypassed) settles as a plain record().
      let reserved: ReservationHandle = 0;
      const slot = this.slots.shift();
      if (slot !== undefined) {
        if (slot.open) {
          reserved = slot.amount;
          slot.open = false;
        }
        if (this.active === slot) this.clearActive();
      }
      this.guard.settle(this.model, metrics.promptTokens, metrics.completionTokens, { reserved });
    } else if (m.type === "stt_metrics") {
      // audioDurationMs is milliseconds; the STT leg is priced per second.
      const seconds = (m as SttMetricsLike).audioDurationMs / 1000;
      const cost = priceVoiceLeg("stt", seconds, {
        model: this.sttModel,
        override: this.sttUsdPerSecond,
      });
      if (cost !== null) this.guard.recordTool("livekit-stt", cost);
    } else if (m.type === "tts_metrics") {
      const cost = priceVoiceLeg("tts", (m as TtsMetricsLike).charactersCount, {
        model: this.ttsModel,
        override: this.ttsUsdPer1kChars,
      });
      if (cost !== null) this.guard.recordTool("livekit-tts", cost);
    }
    // Other metric kinds (VAD, EOU, realtime, ...) carry no billable leg here.
  }

  private onClose(): void {
    // Session torn down with a turn still reserved — release it. Drop any leftover
    // slots so a post-close metrics event cannot touch a stale hold.
    if (this.pending) this.earlyReleaseActive();
    this.slots.length = 0;
  }

  private clearActive(): void {
    this.active = null;
    this.pending = false;
  }

  /** Drop the open guard hold; leave a zeroed queue slot for delayed metrics. */
  private earlyReleaseActive(): void {
    const slot = this.active;
    if (slot === null || !slot.open) {
      this.clearActive();
      return;
    }
    this.guard.release(slot.amount);
    slot.open = false;
    slot.amount = 0;
    this.clearActive();
  }
}
