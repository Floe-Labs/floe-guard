# floe-guard (Vercel AI SDK)

**A local budget guardrail for AI agents** — the TypeScript counterpart to the
[Python `floe-guard`](../README.md). It hard-stops your agent *before its next LLM
call* when it would cross a USD spend ceiling. No account, no signup, no network.
Runs in your process.

```bash
npm i floe-guard ai@4 @ai-sdk/openai
```

```ts
import { wrapLanguageModel } from "ai";
import { openai } from "@ai-sdk/openai";
import { BudgetGuard, budgetGuardMiddleware } from "floe-guard";

const guard = new BudgetGuard(5.0); // your ceiling, in USD

const model = wrapLanguageModel({
  model: openai("gpt-4o"),
  middleware: budgetGuardMiddleware(guard),
});
// generateText / streamText with `model` now stop at $5 — the call that would
// cross the ceiling throws `BudgetExceeded` BEFORE it runs.
```

The middleware sits in the call path: it `check()`s before `doGenerate` /
`doStream` (throwing `BudgetExceeded` to halt the run) and `record()`s priced
token usage after — for streaming it reads usage from the `finish` part.

## Pricing

Tokens are priced **offline** from a bundled
[LiteLLM cost map](src/cost_map.json). A model that isn't in the map (and has no
manual price) **fails closed**: `record` throws `UnpriceableModelError` rather
than silently treating spend as free — *you can't cap spend you can't measure.*

```ts
const guard = new BudgetGuard(5.0, {
  priceOverrides: {
    "my-self-hosted-model": { inputCostPerToken: 1e-6, outputCostPerToken: 2e-6 },
  },
  // or failClosed: false to warn-and-skip for models you accept un-metered.
});
```

## Verified against

`ai@4` (`LanguageModelV1Middleware` via `wrapLanguageModel` /
`experimental_wrapLanguageModel`). Declared as a peer dependency.

## Development

```bash
npm install
npm run build
npm test
npm run typecheck
```

## License

MIT — see [../LICENSE](../LICENSE).
