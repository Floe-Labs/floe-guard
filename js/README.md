# floe-guard (Vercel AI SDK)

[![npm version](https://img.shields.io/npm/v/floe-guard.svg)](https://www.npmjs.com/package/floe-guard)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](../LICENSE)

**A local budget guardrail for AI agents** — the TypeScript counterpart to the
[Python `floe-guard`](../README.md). It hard-stops your agent *before its next
LLM or paid tool call* when it would cross a USD spend ceiling — tokens and
tool calls under one local ceiling. No account, no signup, no network. Runs in
your process.

Works with both **AI SDK v4 and v5** (`ai@4` / `ai@5`).

```bash
npm i floe-guard ai @ai-sdk/openai
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

## Context-aware budgeting

`guard.advisory()` returns a soft signal you can act on before a call — `nearLimit`,
`usedBps`, `remainingUsd` — so the agent can taper near the cap instead of being
cut off. The hard-stop (`check`) is still the guarantee. This is the same shape
the Python package exposes and that hosted Floe returns on every proxied call
(`X-Floe-Budget-Advisory`), so the logic ports unchanged to the hosted path.

```ts
const guard = new BudgetGuard(0.1, { nearLimitBps: 7000 }); // flag at 70% used
const adv = guard.advisory();
const model = adv.nearLimit ? openai("gpt-4o-mini") : openai("gpt-4o");
```

## Tool spend under the same ceiling

Paid tool calls (Apollo, Exa, scrapers) burn the same budget as tokens. The
full reserve/settle contract applies — and the price is known *before* the
call, so the pre-call hard-stop is exact:

```ts
const handle = guard.reserveTool(0.02); // throws BudgetExceeded BEFORE the call
const result = await apollo.peopleLookup(...);
guard.settleTool("apollo.people_lookup", 0.02, { reserved: handle });

guard.recordTool("exa.search", 0.004); // post-hoc, for metered APIs
guard.toolCosts; // { "apollo.people_lookup": 0.42, "exa.search": 0.11 }
```

## Per-call spend log

The guard keeps a typed, in-memory ledger of everything it priced: each
`record()` / `settle()` appends one `SpendEvent`, and `recordTool()` lets paid
non-LLM calls spend the same budget and land in the same log. The events sum to
`spentUsd` (unless a `maxLogEvents` ring buffer has evicted old ones).

```ts
const guard = new BudgetGuard(1.0); // { maxLogEvents: N } caps memory
guard.record("gpt-4o", 1_200, 350, { label: "researcher" });
guard.recordTool("serpapi.search", 0.01, { label: "researcher" });

guard.spendLog; // [{ timestamp, kind: "llm", modelOrTool: "gpt-4o", … }, …]
process.stdout.write(guard.exportLog()); // JSONL, one event per line
```

`exportLog()` emits a stable snake_case schema —
`{timestamp, kind: llm|tool, model_or_tool, prompt_tokens, completion_tokens,
cost_usd, label?, reserved?}` — identical to the Python package's
`export_log()`, so every agent produces the same shape regardless of stack.

## Compatibility

`ai` is declared as a peer dependency with the range `>=4.0.0 <6.0.0`:

- **`ai@4`** — `LanguageModelV1Middleware` via `wrapLanguageModel` /
  `experimental_wrapLanguageModel`; usage read from
  `promptTokens`/`completionTokens`.
- **`ai@5`** — `LanguageModelV2Middleware` via `wrapLanguageModel`; usage read
  from `inputTokens`/`outputTokens`.

The middleware imports nothing from `ai` at runtime or in its types, so one
build serves both majors. If a provider reports no usable token counts, the
call is rejected (fail-closed) rather than metered as $0.

## Development

```bash
npm install
npm run build
npm test
npm run typecheck
```

## License

MIT — see [../LICENSE](../LICENSE).
