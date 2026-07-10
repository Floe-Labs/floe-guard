/**
 * Vercel AI SDK middleware that enforces a {@link BudgetGuard} in the call path.
 *
 * This is the TypeScript counterpart to the Python framework adapters. The AI SDK
 * is TypeScript-only, so it ships as its own npm package.
 *
 * Works with BOTH `ai@4` (`LanguageModelV1Middleware`) and `ai@5`
 * (`LanguageModelV2Middleware`). The two majors renamed the middleware type and
 * the usage fields (`promptTokens`/`completionTokens` → `inputTokens`/
 * `outputTokens`), so this module deliberately imports nothing from `ai`: the
 * middleware is typed structurally against the surface both majors share, and
 * usage is read from whichever field pair the installed SDK reports.
 *
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

import type { BudgetGuard } from "./guard.js";

/**
 * The middleware call surface shared by `ai@4` and `ai@5`. Both majors invoke
 * `wrapGenerate`/`wrapStream` with an options object carrying these members
 * (plus richer `model` fields we don't read). `doGenerate`/`doStream` are
 * declared optional so every 4.x/5.x minor's options type stays assignable —
 * the SDK always provides the one each hook actually calls.
 */
interface MiddlewareCallOptions {
  doGenerate?: () => PromiseLike<any>;
  doStream?: () => PromiseLike<any>;
  model: { modelId: string };
  params?: unknown;
}

/**
 * Structural stand-in for `LanguageModelV1Middleware` (ai@4) and
 * `LanguageModelV2Middleware` (ai@5) — assignable to the `middleware` option of
 * `wrapLanguageModel` on either major.
 */
export interface BudgetGuardMiddleware {
  wrapGenerate: (options: MiddlewareCallOptions) => Promise<any>;
  wrapStream: (options: MiddlewareCallOptions) => Promise<any>;
}

/**
 * Read prompt/completion token counts from an ai@4 usage object
 * (`promptTokens`/`completionTokens`) or an ai@5 one (`inputTokens`/
 * `outputTokens`). Throws when either count is missing or non-numeric: the guard
 * cannot meter spend it cannot see, and treating it as $0 would fail open.
 */
function usageTokens(
  modelId: string,
  usage: unknown,
): { promptTokens: number; completionTokens: number } {
  const u = usage as
    | {
        promptTokens?: unknown;
        completionTokens?: unknown;
        inputTokens?: unknown;
        outputTokens?: unknown;
      }
    | null
    | undefined;
  const promptTokens = u?.promptTokens ?? u?.inputTokens;
  const completionTokens = u?.completionTokens ?? u?.outputTokens;
  if (typeof promptTokens !== "number" || typeof completionTokens !== "number") {
    throw new Error(
      `Model '${modelId}' reported no token usage — the budget guard cannot ` +
        `meter spend it cannot see, so this call is rejected rather than ` +
        `treated as free.`,
    );
  }
  return { promptTokens, completionTokens };
}

/**
 * Build a budget-guard middleware that hard-stops the model before a call
 * crosses the guard's USD ceiling, and records priced token usage after.
 * Compatible with `wrapLanguageModel` from both `ai@4` and `ai@5`.
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
export function budgetGuardMiddleware(guard: BudgetGuard): BudgetGuardMiddleware {
  return {
    async wrapGenerate({ doGenerate, model }) {
      const reserved = guard.reserve(); // throws BudgetExceeded before the call runs
      let result: any;
      try {
        result = await doGenerate!();
      } catch (err) {
        guard.release(reserved); // the call failed before settle() took ownership
        throw err;
      }
      // From here settle() OWNS the reservation: it releases the hold on its own
      // throw (unpriceable / non-finite cost) and consumes it on success. Releasing
      // again would double-subtract and clear a concurrent call's in-flight hold.
      let handled = false;
      try {
        const { promptTokens, completionTokens } = usageTokens(
          model.modelId,
          result?.usage,
        );
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
      let streamResult: any;
      try {
        streamResult = await doStream!();
      } catch (err) {
        guard.release(reserved); // the call failed before settle() took ownership
        throw err;
      }
      const { stream, ...rest } = streamResult as {
        stream: ReadableStream<any>;
      } & Record<string, unknown>;

      // `handled` flips once the reservation is disposed — settled on the finish
      // part, or released exactly once if the stream produced no usage (flush) or
      // ended early via error/cancellation (cancel). settle() owns disposal once
      // called, so nothing else may release after `handled` is set, or we'd
      // double-subtract and clear a concurrent call's hold.
      let handled = false;
      const guarded = stream.pipeThrough(
        new TransformStream<any, any>({
          transform(chunk, controller) {
            if (chunk?.type === "finish" && !handled) {
              try {
                const { promptTokens, completionTokens } = usageTokens(
                  model.modelId,
                  chunk.usage,
                );
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
        } as Transformer<any, any>),
      );

      return { stream: guarded, ...rest };
    },
  };
}
