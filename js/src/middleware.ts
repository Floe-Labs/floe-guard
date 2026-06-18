/**
 * Vercel AI SDK middleware that enforces a {@link BudgetGuard} in the call path.
 *
 * This is the TypeScript counterpart to the Python framework adapters. The AI SDK
 * is TypeScript-only, so it ships as its own npm package.
 *
 * Verified against `ai@4.3.19` (`LanguageModelV1Middleware`):
 *   - `wrapGenerate({ doGenerate, model })` — we `reserve()` (throws to hard-stop)
 *     BEFORE calling `doGenerate()`, hold the reservation across the await, then
 *     `settle()` from `result.usage`.
 *   - `wrapStream({ doStream, model })` — we `reserve()` BEFORE `doStream()`, then
 *     `settle()` from the `finish` part as the stream drains.
 *
 * Reserving before the await is what makes parallel calls (`Promise.all` over
 * several generations) honour the ceiling: each holds its slice instead of all
 * reading the same stale total (issue #18). The reservation is released if the
 * call throws, or if a stream ends without reporting usage.
 *
 * The model id used for pricing comes from `model.modelId`.
 */

import type { LanguageModelV1Middleware } from "ai";

import type { BudgetGuard } from "./guard.js";

// The AI SDK does not re-export `LanguageModelV1StreamPart` from the "ai" entry
// point, so we derive the stream element type from the middleware's own return
// type. This keeps us fully typed without importing a transitive package.
type WrapStreamResult = Awaited<
  ReturnType<NonNullable<LanguageModelV1Middleware["wrapStream"]>>
>;
type StreamPart =
  WrapStreamResult["stream"] extends ReadableStream<infer P> ? P : never;

/**
 * Build a `LanguageModelV1Middleware` that hard-stops the model before a call
 * crosses the guard's USD ceiling, and records priced token usage after.
 *
 * @example
 * import { wrapLanguageModel } from "ai";
 * import { openai } from "@ai-sdk/openai";
 * import { BudgetGuard, budgetGuardMiddleware } from "floe-guard";
 *
 * const guard = new BudgetGuard(5.00);
 * const model = wrapLanguageModel({
 *   model: openai("gpt-4o"),
 *   middleware: budgetGuardMiddleware(guard),
 * });
 */
export function budgetGuardMiddleware(
  guard: BudgetGuard,
): LanguageModelV1Middleware {
  return {
    async wrapGenerate({ doGenerate, model }) {
      const reserved = guard.reserve(); // throws BudgetExceeded before the call runs
      let result: Awaited<ReturnType<typeof doGenerate>>;
      try {
        result = await doGenerate();
      } catch (err) {
        guard.release(reserved); // the call failed before settle() took ownership
        throw err;
      }
      // From here settle() OWNS the reservation: it releases the hold on its own
      // throw (unpriceable / non-finite cost) and consumes it on success. Releasing
      // again would double-subtract and clear a concurrent call's in-flight hold.
      let handled = false;
      try {
        const { promptTokens, completionTokens } = result.usage;
        handled = true;
        guard.settle(model.modelId, promptTokens, completionTokens, { reserved });
        return result;
      } catch (err) {
        if (!handled) guard.release(reserved); // failed reading usage, before settle()
        throw err;
      }
    },

    async wrapStream({ doStream, model }) {
      const reserved = guard.reserve(); // throws BudgetExceeded before the stream starts
      let streamResult: Awaited<ReturnType<typeof doStream>>;
      try {
        streamResult = await doStream();
      } catch (err) {
        guard.release(reserved); // the call failed before settle() took ownership
        throw err;
      }
      const { stream, ...rest } = streamResult;

      // `handled` flips once the reservation is disposed — settled on the finish
      // part, or released exactly once if the stream produced no usage (flush) or
      // ended early via error/cancellation (cancel). settle() owns disposal once
      // called, so nothing else may release after `handled` is set, or we'd
      // double-subtract and clear a concurrent call's hold.
      let handled = false;
      const guarded = stream.pipeThrough(
        new TransformStream<StreamPart, StreamPart>({
          transform(chunk, controller) {
            if (chunk.type === "finish" && !handled) {
              try {
                const { promptTokens, completionTokens } = chunk.usage;
                handled = true;
                guard.settle(model.modelId, promptTokens, completionTokens, { reserved });
              } catch (err) {
                // Release only if we never reached settle() (e.g. usage missing).
                // If settle() itself threw, it already released its own hold.
                if (!handled) {
                  handled = true;
                  guard.release(reserved);
                }
                throw err;
              }
            }
            controller.enqueue(chunk);
          },
          flush() {
            // Clean close with no finish/usage part — free the held budget.
            if (!handled) {
              handled = true;
              guard.release(reserved);
            }
          },
          cancel() {
            // Upstream error or consumer cancellation: flush() does not run here
            // (Web Streams: flush and cancel are mutually exclusive), so release
            // the still-held reservation.
            if (!handled) {
              handled = true;
              guard.release(reserved);
            }
          },
          // `cancel` is valid per the Streams spec and supported in Node 18+, but
          // TS's Transformer lib type lags and omits it — cast to keep the type check.
        } as Transformer<StreamPart, StreamPart>),
      );

      return { stream: guarded, ...rest };
    },
  };
}
