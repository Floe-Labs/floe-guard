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
        guard.release(reserved); // call failed — give the budget back
        throw err;
      }
      guard.settle(
        model.modelId,
        result.usage.promptTokens,
        result.usage.completionTokens,
        { reserved },
      );
      return result;
    },

    async wrapStream({ doStream, model }) {
      const reserved = guard.reserve(); // throws BudgetExceeded before the stream starts
      let started: Awaited<ReturnType<typeof doStream>>;
      try {
        started = await doStream();
      } catch (err) {
        guard.release(reserved);
        throw err;
      }
      const { stream, ...rest } = started;

      let settled = false;
      const guarded = stream.pipeThrough(
        new TransformStream<StreamPart, StreamPart>({
          transform(chunk, controller) {
            if (chunk.type === "finish") {
              guard.settle(
                model.modelId,
                chunk.usage.promptTokens,
                chunk.usage.completionTokens,
                { reserved },
              );
              settled = true;
            }
            controller.enqueue(chunk);
          },
          flush() {
            // Stream ended without a finish/usage part — free the held budget.
            if (!settled) guard.release(reserved);
          },
        }),
      );

      return { stream: guarded, ...rest };
    },
  };
}
