/**
 * Retell custom-LLM WebSocket adapter (optional peer: the `ws` server your custom
 * LLM already runs — this adapter never opens or hard-depends on it).
 *
 * Retell's custom LLM runs over a WebSocket (docs.retellai.com/api-references/
 * llm-websocket). Retell sends interaction events to your server; the billable one
 * is **`response_required`** (an auto-incrementing `response_id` plus the running
 * `transcript`), and its twin **`reminder_required`**. Your server streams back
 * **`response`** events — many partials per turn — ending with `content_complete:
 * true`. A NEW `response_required` with a higher `response_id` means the caller
 * barged in: it **interrupts** the turn in flight, which never reaches
 * `content_complete`.
 *
 * Like the LiveKit adapter, a Retell call has no single call site to wrap: turns
 * fire for the life of the socket. The two enforcement points are the arrival of a
 * `response_required` (before the LLM call, so a turn is admitted or blocked before
 * its TTS/telephony spend piles on) and the turn's real token usage (after
 * `content_complete`). {@link RetellBudgetGuard} holds the reservation across both,
 * keyed by `response_id`, and releases it when a newer turn interrupts.
 *
 *     import { BudgetGuard } from "floe-guard";
 *     import { RetellBudgetGuard } from "floe-guard/adapters/retell";
 *
 *     const guard = new BudgetGuard(1.0, {
 *       priceOverrides: {
 *         "gpt-4o-mini": { inputCostPerToken: 0.15e-6, outputCostPerToken: 0.6e-6 },
 *       },
 *     });
 *     const budget = new RetellBudgetGuard(guard, {
 *       model: "gpt-4o-mini",
 *       sttModel: "deepgram-nova-3",
 *       ttsModel: "elevenlabs-flash-v2.5",
 *       telephony: "twilio-us-inbound-local",
 *     });
 *
 *     // In your custom-LLM WS message handler:
 *     ws.on("message", async (raw) => {
 *       const event = JSON.parse(raw);
 *       if (event.interaction_type === "response_required" ||
 *           event.interaction_type === "reminder_required") {
 *         const turn = budget.beginTurn(event);        // reserve before the LLM call
 *         if (!turn.admitted) {                         // budget exhausted
 *           ws.send(JSON.stringify(
 *             budget.response(event.response_id, "I'm out of budget — wrapping up.",
 *                             { complete: true, endCall: true })));
 *           return;
 *         }
 *         const { text, usage } = await callYourLlm(event.transcript); // your LLM
 *         ws.send(JSON.stringify(budget.response(event.response_id, text, { complete: true })));
 *         budget.settleTurn(event.response_id, usage);  // settle real token usage
 *       }
 *     });
 *
 * ## Why a WebSocket adapter, not a request wrapper
 *
 * The custom LLM sees only the **LLM leg** (the tokens behind each `response`). STT,
 * TTS and telephony are billed by Retell/the carrier and never cross this socket, so
 * they are metered **explicitly** — call {@link RetellBudgetGuard.meterStt} /
 * {@link RetellBudgetGuard.meterTts} / {@link RetellBudgetGuard.meterTelephony} from
 * wherever those durations are known (webhook, call-ended event). Each is priced
 * from the bundled voice cost map when you name the vendor, or a per-unit override
 * which wins over the map; a leg with neither is left un-metered (the token-only
 * contract), and a named vendor the map cannot price fails closed
 * ({@link UnpriceableVoiceError}).
 *
 * Scope is strictly **pre-turn / pre-call admission plus per-turn settlement**.
 * There is no mid-call intervention: an admitted turn runs to `content_complete`;
 * nothing here cuts a turn off partway. The only release is the interrupt Retell
 * itself signals (a newer `response_id`) or an explicit {@link RetellBudgetGuard.close}
 * on call teardown.
 */

import { BudgetExceeded } from "../errors.js";
import * as gates from "../gates.js";
import type { BudgetGuard, ReservationHandle } from "../guard.js";
import { priceVoiceLeg } from "../voice-pricing.js";

/** One entry of a Retell `transcript` (loosely typed — we never read into it). */
export interface RetellUtterance {
  readonly role: "agent" | "user";
  readonly content: string;
}

/**
 * The Retell interaction events that require a turn. `response_required` is a real
 * user utterance; `reminder_required` fires after silence — both carry an
 * auto-incrementing `response_id` and expect a `response` stream in reply. Typed
 * structurally so a parsed WS message satisfies it with no Retell SDK dependency.
 */
export interface RetellResponseRequiredEvent {
  readonly interaction_type: "response_required" | "reminder_required";
  /** Auto-incrementing turn id; a higher value than the open turn means an interrupt. */
  readonly response_id: number;
  readonly transcript?: readonly RetellUtterance[];
}

/**
 * A `response` event to stream back to Retell. Emit several partials per turn
 * (`content_complete: false`), then a final one with `content_complete: true`.
 * Build them with {@link RetellBudgetGuard.response}.
 */
export interface RetellResponseEvent {
  readonly response_type: "response";
  readonly response_id: number;
  readonly content: string;
  readonly content_complete: boolean;
  readonly end_call?: boolean;
}

/** Real token usage for a settled turn — the LLM leg the socket sees. */
export interface RetellTokenUsage {
  readonly promptTokens: number;
  readonly completionTokens: number;
}

/**
 * The decision {@link RetellBudgetGuard.beginTurn} returns. `admitted: true` means
 * the turn's reservation is held — run the LLM and settle it. `admitted: false`
 * means the budget is spent: send a wrap-up `response` (with `content_complete`) and
 * do not call the LLM; `error` carries the {@link BudgetExceeded} for logging.
 */
export type RetellTurnDecision =
  | { readonly admitted: true; readonly responseId: number }
  | { readonly admitted: false; readonly responseId: number; readonly error: BudgetExceeded };

export interface RetellBudgetGuardOptions {
  /**
   * Model id to settle LLM cost against (Retell's usage payload names no model we
   * rely on). Must be priceable via the bundled cost map or the guard's
   * `priceOverrides`.
   */
  model: string;
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
 * One turn's reservation, keyed by `response_id`. `open` means the USD amount is
 * still held on the {@link BudgetGuard}. A settle or an interrupt closes it.
 */
interface TurnSlot {
  readonly responseId: number;
  amount: ReservationHandle;
  open: boolean;
}

/**
 * Enforce a {@link BudgetGuard} ceiling on a Retell custom-LLM socket, one turn at
 * a time: reserve when a `response_required` arrives, settle on real token usage
 * after `content_complete`, release when a newer `response_id` interrupts.
 */
export class RetellBudgetGuard {
  private readonly guard: BudgetGuard;
  private readonly model: string;
  private readonly sttModel?: string;
  private readonly ttsModel?: string;
  private readonly telephony?: string;
  private readonly sttUsdPerSecond?: number;
  private readonly ttsUsdPer1kChars?: number;
  private readonly telephonyUsdPerMinute?: number;

  /** Open (and just-settled) turns keyed by `response_id`. */
  private readonly slots = new Map<number, TurnSlot>();
  /** The most recently begun turn — the one a newer `response_required` interrupts. */
  private activeResponseId: number | null = null;

  constructor(guard: BudgetGuard, options: RetellBudgetGuardOptions) {
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
   * Reserve budget for a `response_required` / `reminder_required` turn before its
   * LLM call runs. Returns an admit/block {@link RetellTurnDecision}.
   *
   * A newer `response_id` arriving while a prior turn is still open is Retell's
   * interrupt signal — this releases that prior turn's hold before reserving the new
   * one (its `content_complete` will never arrive). Reserving here is what blocks a
   * turn before its downstream TTS/telephony spend piles on. Calling twice with the
   * same still-open `response_id` is idempotent (no double hold).
   */
  beginTurn(event: RetellResponseRequiredEvent): RetellTurnDecision {
    const responseId = event.response_id;

    // Idempotent: this turn already holds a reservation.
    const existing = this.slots.get(responseId);
    if (existing !== undefined && existing.open) {
      return { admitted: true, responseId };
    }

    // A newer turn interrupts the prior open turn — free its hold before opening this
    // one. Its usage never settles (no content_complete); a stray late settle for it
    // finds no open slot and records actual usage against a zero reservation.
    this.releaseActive(responseId);

    let handle: ReservationHandle;
    try {
      handle = this.guard.reserve(); // throws BudgetExceeded before the turn runs
    } catch (err) {
      if (err instanceof BudgetExceeded) {
        return { admitted: false, responseId, error: err };
      }
      throw err;
    }

    this.slots.set(responseId, { responseId, amount: handle, open: true });
    this.activeResponseId = responseId;
    return { admitted: true, responseId };
  }

  /**
   * Settle a turn's real token usage after its `content_complete`. Consumes the
   * reservation opened by {@link beginTurn} for this `response_id`; a turn with no
   * open slot (already interrupted, or a settle for an id never begun) records the
   * usage against a zero reservation instead of stealing another turn's hold.
   */
  settleTurn(responseId: number, usage: RetellTokenUsage): void {
    let reserved: ReservationHandle = 0;
    const slot = this.slots.get(responseId);
    if (slot !== undefined) {
      if (slot.open) {
        reserved = slot.amount;
        slot.open = false;
      }
      this.slots.delete(responseId);
    }
    if (this.activeResponseId === responseId) this.activeResponseId = null;
    this.guard.settle(this.model, usage.promptTokens, usage.completionTokens, { reserved });
  }

  /**
   * Pre-call admission for Retell's inbound-call webhook, wrapping {@link gates.retell}.
   * On budget exhaustion returns `{ call_inbound: { reject: true } }`; otherwise
   * `{ call_inbound: {...} }` carrying any `admit` overrides (`dynamic_variables`,
   * `metadata`, `override_agent_id`, …).
   *
   * A coarse, non-binding preflight — the binding hard-stop is the per-turn reserve
   * in {@link beginTurn}. Pass `estimatedCallUsd` (e.g. `$/min × expected minutes`)
   * to reject earlier, when the remaining budget can't cover the call.
   */
  admitCall(
    options: { estimatedCallUsd?: number; admit?: Record<string, unknown> } = {},
  ): { call_inbound: Record<string, unknown> } {
    return gates.retell(this.guard, options);
  }

  /**
   * Build a `response` event to stream back to Retell. Send partials with
   * `complete: false`, then a final one with `complete: true`; settle the turn's
   * token usage after that final one. Pass `endCall: true` to hang up (e.g. on a
   * budget wrap-up line). A thin convenience — you may build the shape yourself.
   */
  response(
    responseId: number,
    content: string,
    options: { complete?: boolean; endCall?: boolean } = {},
  ): RetellResponseEvent {
    return {
      response_type: "response",
      response_id: responseId,
      content,
      content_complete: options.complete ?? false,
      ...(options.endCall ? { end_call: true } : {}),
    };
  }

  /**
   * Accrue STT spend for `seconds` of transcribed audio (per-second). The socket
   * never sees the STT leg, so drive this from wherever the duration is known.
   * Priced from the voice map when `sttModel` is set, or the `sttUsdPerSecond`
   * override; returns `null` (no-op) when the leg is unconfigured, and fails closed
   * on a vendor the map cannot price. Returns the USD accrued, if any.
   */
  meterStt(seconds: number): number | null {
    const cost = priceVoiceLeg("stt", seconds, {
      model: this.sttModel,
      override: this.sttUsdPerSecond,
    });
    if (cost !== null) this.guard.recordTool("retell-stt", cost);
    return cost;
  }

  /**
   * Accrue TTS spend for `characters` of synthesized speech (per-1k-chars). Priced
   * from the voice map when `ttsModel` is set, or the `ttsUsdPer1kChars` override;
   * returns `null` when unconfigured, and fails closed on an unpriceable vendor.
   */
  meterTts(characters: number): number | null {
    const cost = priceVoiceLeg("tts", characters, {
      model: this.ttsModel,
      override: this.ttsUsdPer1kChars,
    });
    if (cost !== null) this.guard.recordTool("retell-tts", cost);
    return cost;
  }

  /**
   * Accrue telephony spend for `minutes` of call time (per-minute). Per-minute
   * accrual, not live line-cutting: the guard meters the leg, it does not cut the
   * phone line mid-call. Priced from the voice map when `telephony` is set, or the
   * `telephonyUsdPerMinute` override; returns `null` when unconfigured, and fails
   * closed on an unpriceable vendor. Returns the USD accrued, if any.
   */
  meterTelephony(minutes: number): number | null {
    const cost = priceVoiceLeg("telephony", minutes, {
      model: this.telephony,
      override: this.telephonyUsdPerMinute,
    });
    if (cost !== null) this.guard.recordTool("retell-telephony", cost);
    return cost;
  }

  /**
   * Release any still-open turn's reservation on call teardown (socket close, call
   * ended). A turn that never reached `content_complete` would otherwise leak its
   * hold and shrink `remainingUsd` permanently.
   */
  close(): void {
    for (const slot of this.slots.values()) {
      if (slot.open) this.guard.release(slot.amount);
    }
    this.slots.clear();
    this.activeResponseId = null;
  }

  /** Release the active open turn's hold unless it is `exceptId` (the incoming turn). */
  private releaseActive(exceptId: number): void {
    const id = this.activeResponseId;
    if (id === null || id === exceptId) return;
    const slot = this.slots.get(id);
    if (slot !== undefined && slot.open) {
      this.guard.release(slot.amount);
      slot.open = false;
      this.slots.delete(id);
    }
    this.activeResponseId = null;
  }
}
