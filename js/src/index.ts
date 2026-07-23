/**
 * floe-guard — a local budget guardrail for AI agents.
 *
 * Hard-stops your agent before its next LLM call when it would cross a USD spend
 * ceiling. This package is the Vercel AI SDK (TypeScript) adapter; the Python
 * package `floe-guard` (pip) carries the LiteLLM / CrewAI / LangChain adapters.
 */

export {
  BudgetGuard,
  type BudgetGuardOptions,
  type BudgetAdvisory,
  type SpendEvent,
} from "./guard.js";
export {
  LatencyBudget,
  type LatencyBudgetOptions,
  type LatencyAdvisory,
} from "./latency.js";
export {
  FloeGuardError,
  BudgetExceeded,
  DeadlineExceeded,
  UnpriceableModelError,
} from "./errors.js";
export {
  budgetGuardMiddleware,
  type BudgetGuardMiddleware,
} from "./middleware.js";
export {
  type ManualPrice,
  type PricedModel,
  resolvePrice,
  priceTokens,
} from "./pricing.js";
export {
  withBudgetRetry,
  type BudgetRetryOptions,
  type RetryPlan,
} from "./retry.js";

export * as pricing from "./pricing.js";
