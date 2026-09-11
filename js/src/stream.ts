/**
 * Chunk-wise USD enforcement, mirroring Python's StreamGuard.
 * Generated chunks are billed even when they cross the limit: settle first,
 * then throw. Token estimates and source cancellation are provider-dependent.
 */
import { BudgetGuard, type ReservationHandle } from "./guard.js";
import { UnpriceableModelError } from "./errors.js";
import { priceTokens, resolvePrice, type ManualPrice } from "./pricing.js";

export interface StreamGuardOptions {
  /** Known input tokens; final provider usage can override this at finish(). */
  promptTokens?: number;
  /** The original handle from reserve(), including any token hold. */
  reserved?: ReservationHandle;
  price?: ManualPrice;
  label?: string;
  /** Count each text delta; defaults to approxTokens. */
  countTokens?: (delta: string) => number;
}

/** Python's heuristic: four Unicode code points per token, at least one per delta. */
export function approxTokens(text: string): number {
  if (typeof text !== "string") throw new TypeError("stream text must be a string");
  return text ? Math.max(1, Math.floor(Array.from(text).length / 4)) : 0;
}

function tokenCount(value: number): number {
  if (!Number.isFinite(value)) throw new RangeError("token count must be finite");
  return Math.max(0, Math.trunc(value));
}

/**
 * Meter one response; the crossing chunk is settled before BudgetExceeded.
 * Call close() in finally for direct use, or let guardStream own cleanup.
 * Concurrent streams must share the same BudgetGuard within one JS isolate.
 */
export class StreamGuard {
  private tokens = 0;
  private closed = false;
  private readonly priced;
  private key: symbol | undefined;
  private promptTokens: number;
  private readonly reserved: ReservationHandle;
  private readonly price: ManualPrice | undefined;
  private readonly label: string | undefined;
  private readonly countTokens: (delta: string) => number;

  constructor(
    private readonly guard: BudgetGuard,
    private readonly model: string,
    options: StreamGuardOptions = {},
  ) {
    this.reserved = options.reserved ?? 0;
    guard._validateStreamReservation(this.reserved);
    try {
      // Python captures constructor arguments and ManualPrice is immutable.
      // Keep the same configuration for both enforcement and settlement even
      // if the caller later replaces options or mutates its manual price.
      this.price = options.price === undefined ? undefined : { ...options.price };
      this.label = options.label;
      this.countTokens = options.countTokens ?? approxTokens;
      this.promptTokens = tokenCount(options.promptTokens ?? 0);
      this.priced = resolvePrice(
        model,
        this.price === undefined
          ? guard.priceOverrides
          : { ...guard.priceOverrides, [model]: this.price },
      );
      if (this.priced === null && guard.failClosed) {
        console.warn(`Cannot price model '${model}': pass a price override to enforce a streaming budget.`);
        throw new UnpriceableModelError(model);
      }
    } catch (error) {
      guard.release(this.reserved);
      throw error;
    }
  }

  /** Generated completion tokens counted so far (estimated until finish()). */
  get completionTokens(): number {
    return this.tokens;
  }

  /** Meter a text delta with the configured tokenizer. */
  feedText(delta: string): void {
    if (typeof delta !== "string") throw new TypeError("stream text must be a string");
    const count = this.countTokens;
    this.feedTokens(count(delta));
  }

  /** Meter additional completion tokens, settling before a budget interruption. */
  feedTokens(tokens: number): void {
    if (this.closed) throw new Error("stream already settled");
    this.tokens = tokenCount(this.tokens + tokenCount(tokens));
    if (this.priced === null) return;
    const cost = priceTokens(this.priced, this.promptTokens, this.tokens);
    // An unconsumed wrapper owns no registry entry; its reservation still
    // belongs to the caller until iteration starts, as in Python.
    this.key ??= this.guard._registerStream(this.reserved);
    if (this.guard._streamWouldCross(this.key, cost)) {
      this.finish();
      this.guard._blockStream();
    }
  }

  /** Reconcile estimates to provider usage, or settle accumulated estimates. */
  finish(usage: { promptTokens?: number; completionTokens?: number } = {}): number {
    if (this.closed) throw new Error("stream already settled");
    const prompt = tokenCount(usage.promptTokens ?? this.promptTokens);
    const completion = tokenCount(usage.completionTokens ?? this.tokens);
    this.promptTokens = prompt;
    this.tokens = completion;
    this.closed = true;
    try {
      return this.guard.settle(this.model, prompt, completion, {
        reserved: this.reserved,
        price: this.price,
        label: this.label,
      });
    } finally {
      if (this.key !== undefined) this.guard._unregisterStream(this.key);
    }
  }

  /** Idempotent cleanup for a finally block; partial generated usage is billed. */
  close(): void {
    if (!this.closed) this.finish();
  }
}

export interface GuardStreamOptions<C> extends StreamGuardOptions {
  /** Required for structured provider chunks; must return a string. */
  getText?: (chunk: C) => string;
}

/**
 * Meter before yielding; all exits after iteration starts settle partial usage.
 * Model validation is eager. A never-started wrapper leaves its reservation
 * with the caller, who must release it. Early exit closes the source iterator;
 * whether that cancels remote generation depends on the source's implementation.
 */
export function guardStream<C>(
  guard: BudgetGuard, model: string, chunks: AsyncIterable<C>, options?: GuardStreamOptions<C>,
): AsyncIterableIterator<C>;
export function guardStream<C>(
  guard: BudgetGuard, model: string, chunks: Iterable<C>, options?: GuardStreamOptions<C>,
): IterableIterator<C>;
export function guardStream<C>(
  guard: BudgetGuard, model: string, chunks: Iterable<C> | AsyncIterable<C>, options?: GuardStreamOptions<C>,
): IterableIterator<C> | AsyncIterableIterator<C>;
export function guardStream<C>(
  guard: BudgetGuard, model: string, chunks: Iterable<C> | AsyncIterable<C>, options: GuardStreamOptions<C> = {},
): IterableIterator<C> | AsyncIterableIterator<C> {
  const stream = new StreamGuard(guard, model, options);
  const extract = options.getText ?? ((chunk: C): string => {
    if (typeof chunk !== "string") throw new TypeError("guardStream needs getText for non-string chunks");
    return chunk;
  });
  function* run(source: Iterable<C>): IterableIterator<C> {
    try {
      for (const chunk of source) {
        stream.feedText(extract(chunk));
        yield chunk;
      }
    } finally {
      stream.close();
    }
  }
  async function* runAsync(source: AsyncIterable<C>): AsyncIterableIterator<C> {
    try {
      for await (const chunk of source) {
        stream.feedText(extract(chunk));
        yield chunk;
      }
    } finally {
      stream.close();
    }
  }
  return Symbol.asyncIterator in chunks ? runAsync(chunks) : run(chunks);
}
