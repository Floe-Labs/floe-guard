/**
 * Vercel AI SDK middleware that enforces a {@link BudgetGuard} in the call path.
 *
 * This is the TypeScript counterpart to the Python framework adapters. The AI SDK
 * is TypeScript-only, so it ships as its own npm package.
 *
 * Verified against `ai@4.3.19` (`LanguageModelV1Middleware`):
 *   - `wrapGenerate({ doGenerate, model })` — we `check()` (throws to hard-stop)
 *     BEFORE calling `doGenerate()`, then `record()` from `result.usage`.
 *   - `wrapStream({ doStream, model })` — we `check()` BEFORE `doStream()`, then
 *     read `usage` from the `finish` part as the stream drains.
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
      guard.check(); // throws BudgetExceeded before the call runs
      const result = await doGenerate();
      guard.record(
        model.modelId,
        result.usage.promptTokens,
        result.usage.completionTokens,
      );
      return result;
    },

    async wrapStream({ doStream, model }) {
      guard.check(); // throws BudgetExceeded before the stream starts
      const { stream, ...rest } = await doStream();

      const guarded = stream.pipeThrough(
        new TransformStream<StreamPart, StreamPart>({
          transform(chunk, controller) {
            if (chunk.type === "finish") {
              guard.record(
                model.modelId,
                chunk.usage.promptTokens,
                chunk.usage.completionTokens,
              );
            }
            controller.enqueue(chunk);
          },
        }),
      );

      return { stream: guarded, ...rest };
    },
  };
}
