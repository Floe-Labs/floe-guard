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
  type BudgetReservation,
  type ReservationHandle,
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
  TokenBudgetExceeded,
  DeadlineExceeded,
  UnpriceableModelError,
  UnpriceableVoiceError,
  LedgerSyncError,
} from "./errors.js";
export { pushLedger } from "./sync.js";
export {
  budgetGuardMiddleware,
  type BudgetGuardMiddleware,
} from "./middleware.js";
export {
  type ManualPrice,
  type PricedModel,
  type TokenCacheUsage,
  resolvePrice,
  priceTokens,
  costMapGeneratedAt,
} from "./pricing.js";
export {
  withBudgetRetry,
  type BudgetRetryOptions,
  type RetryPlan,
} from "./retry.js";

export * as pricing from "./pricing.js";

export {
  type VoiceMode,
  type VoiceRate,
  lookupVoiceRate,
  resolveVoiceRate,
  voiceLegCost,
  priceVoiceLeg,
} from "./voice-pricing.js";

// gates as a namespace, mirroring Python's `floe_guard.gates` — so
// `import { gates } from "floe-guard"` reads like the Python package.
export * as gates from "./gates.js";
