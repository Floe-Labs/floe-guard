# Changelog

All notable changes to floe-guard are documented here. The repo ships two
packages — `floe-guard` on [PyPI](https://pypi.org/project/floe-guard/) and
`floe-guard` on [npm](https://www.npmjs.com/package/floe-guard) (Vercel AI SDK)
— versioned independently; entries are tagged **py** / **js**.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
both packages adhere to [Semantic Versioning](https://semver.org/).

## Unreleased

### Added (py)

- **Google Gemini adapter** (`pip install floe-guard[gemini]`) —
  `floe_guard.integrations.gemini.guarded_completion` / `guarded_acompletion`
  wrap the `google-genai` SDK's `client.models.generate_content` with the same
  reserve-before / settle-after contract as the OpenAI and Anthropic adapters, so
  a blocked call never reaches Google.
- All five Gemini usage counters are mapped, not just the obvious two:
  `thoughts_token_count` (thinking-model reasoning, billed as output) and
  `tool_use_prompt_token_count` (tool results fed back as input) sit *outside*
  `candidates_token_count` / `prompt_token_count`, so omitting them would
  under-meter thinking and tool-using agents. `cached_content_token_count` is
  carved *out* of the prompt count — Gemini documents it as included there — and
  re-priced at the cache-read rate instead of being charged twice.
- **Vertex AI callers fail closed unless they supply prices.** One SDK serves
  both Google AI Studio and Vertex with identical model ids, and the bundled map
  carries AI Studio rates, so metering a Vertex call against it would under-meter.
  The model id cannot reveal the backend but the client can: the adapter reads
  `client.vertexai` (set by both `vertexai=True` and the newer `enterprise=True`)
  and refuses unless the model has a `price_overrides` entry. Honours
  `fail_closed=False` for callers who accept un-metered spend. A Vertex call
  cleared by an override also settles against that override even when Google
  serves a different snapshot id the bundled map happens to price — otherwise the
  drift would quietly put the call back on AI Studio rates.

### Added (py + js)

- **Google Gemini pricing in the bundled cost map** — 37 Gemini models are now
  priced offline, so `gemini-2.5-flash` and friends are metered instead of
  failing closed. Vendored under the **bare** ids the `google-genai` SDK and
  `@ai-sdk/google` pass; LiteLLM's `gemini/<id>` and the older `models/<id>`
  forms resolve to the same entry through the existing bare-last-segment
  fallback, so no change to `pricing.py` / `pricing.ts` was needed.
- Prices are **Google AI Studio (Gemini Developer API)** rates. Vertex AI serves
  the same ids at its own — sometimes dearer — rates (`gemini-2.0-flash-001`:
  Vertex is 50% higher), and a model id alone cannot say which billing path a
  call used, so Vertex is deliberately not vendored; Vertex agents pass
  `price_overrides`. A `vertex_ai/<id>` caller resolves at AI Studio rates via
  the same fallback that already maps `openrouter/openai/gpt-4o` → `gpt-4o`;
  this is asserted in `tests/test_pricing.py` so it cannot change silently.

### Fixed (py + js)

- **A wrong upstream `mode` can no longer bill a chat model's output for free.**
  Embedding mode zeroes the output rate, and upstream lists `gemini-1.5-flash` — a
  chat/multimodal model — as `mode: "embedding"` with `output_cost_per_token: 0`.
  Vendoring that would meter every `gemini-1.5-flash` completion's output at $0,
  which fail-closed pricing cannot catch because `0` is a finite, valid price. An
  embedding entry's id must now start with a known embedding family
  (`text-embedding-*`, `gemini-embedding-*`), so a single wrong field can't zero a
  price; the same predicate gates both the filter and the writer, and an unknown
  family is dropped with a warning rather than trusted. `gemini-1.5-flash` has no
  correctly-priced variant upstream, so it stays unpriceable and fails closed.
- **Zero-priced models are no longer vendored.** Upstream lists some
  free/experimental tiers at `0`/`0`, and fail-closed pricing cannot catch them
  (`0` is finite, so the model resolves and every call meters at $0 forever).
  They are now dropped and fail closed loudly, on either rate. The one exception
  is an embedding's `0` output rate — that is a real price, not a missing one.
- **Duplicate upstream keys resolve to the dearer rate in each bucket.** Several
  Gemini models are listed both bare and `gemini/`-prefixed; collapsing them onto
  one vendored key previously depended on iteration order. The refresh now takes
  the higher input rate and the higher output rate independently — picking one
  whole entry by total cost still under-meters a prompt/completion mix when one
  duplicate is dearer on input and the other on output. Over-pricing stops an
  agent one call early (safe), under-pricing lets a crossing call through.
- **Realtime/audio models are no longer priced at text rates.** Upstream
  reclassified OpenAI's realtime models from `chat` to `realtime` mode, so they
  drop out of the vendored map. This closes a latent under-meter: they bill audio
  tokens at up to **8×** their text input rate (`gpt-realtime`: $0.000032/token
  audio vs $0.000004 text), which the map had no way to express. They now fail
  closed until given a `price_overrides` entry.

## py 0.8.0 — 2026-07-23

### Added (py)

- **LiveKit Agents adapter** (`floe-guard[livekit]`, issue #39):
  `LiveKitBudgetGuard.attach(session, agent)` wires the reserve-before /
  settle-after contract onto a LiveKit `AgentSession` — reserve in the agent's
  `llm_node`, settle on the session's `metrics_collected` `LLMMetrics`, release
  on `close` or a bypassed turn. Optional per-second / per-1k-char knobs meter
  STT/TTS spend via `record_tool`. `on_budget_exceeded` async callback for a
  graceful spoken wrap-up instead of a hard cut.

## py 0.7.0 / js 0.5.0 — 2026-07-21

### Added (py + js)

- **Tool spend as a first-class primitive** with the full reserve/settle
  contract, sharing the token ceiling: `reserve_tool(estimated_cost)` /
  `reserveTool` holds a tool call's known price in-flight and raises
  `BudgetExceeded` BEFORE the call would cross the cap (stronger than the LLM
  path — the price is exact, not an estimate); `settle_tool(name, cost_usd,
  reserved=…)` / `settleTool` releases the hold and accrues the actual cost;
  `record_tool` / `recordTool` remains the post-hoc form. The caller supplies
  the USD — there is no tool cost-map.
- **`tool_costs` / `toolCosts`**: per-tool-name running totals (e.g.
  `{"apollo.people_lookup": 0.42, "exa.search": 0.11}`), so the token/tool
  split of the one shared ceiling is inspectable. Tool settles land in the
  spend ledger as `kind: "tool"` events with the reservation recorded.
- Example: `examples/tool_budget.py` (no API key) — a prospecting loop whose
  Apollo/Exa spend dies at the ceiling.

### Changed (py + js)

- `record_tool` / `recordTool` (and the new `settle_tool` / `settleTool`) now
  update the next-call estimate, so a plain `check()` + `record_tool` loop
  stops BEFORE the crossing tool call — the same stop-one-early contract as
  tokens. Previously tool costs accrued but did not inform the prediction.
  LLM and tool costs are tracked as separate last-costs and the default
  `check()`/`reserve()` prediction is the max of the two, so a cheap tool call
  can't shrink the estimate ahead of an expensive LLM call (or vice versa).

### Fixed (py + js)

- **Over-release fails loud instead of open**: `settle()`, `settle_tool()`,
  and `release()` previously clamped the in-flight tally to zero when handed a
  `reserved` handle larger than everything currently held — silently freeing
  OTHER callers' reservations and weakening the ceiling under concurrency. A
  handle exceeding the total in-flight sum (which cannot have come from a
  matching `reserve()`) now raises `ValueError` / `RangeError` without
  mutating any state; sub-epsilon float dust still settles cleanly.

## py 0.6.0 — 2026-07-18

### Added (py)

- **LangGraph adapter** (`floe-guard[langgraph]`, issue #33): `guarded_node`
  wraps graph nodes with the reserve-before / settle-after contract, so a
  `StateGraph` fan-out of parallel sub-agents holds one shared ceiling
  atomically; `AdvisoryChannel` / `latest_advisory` expose the typed
  `BudgetAdvisory` in graph state after each metered node, so a router can
  downshift models on `near_limit` before the hard-stop. Ships with a
  no-API-key example (`examples/langgraph_budget_aware.py`).

## py 0.5.0 / js 0.4.0 — 2026-07-16

### Added (py + js)

- **`LatencyBudget`** — BudgetGuard's sibling for time: tracks cumulative
  elapsed time across an agentic tool chain against an end-user SLA.
  `check(expected_ms)` raises the new `DeadlineExceeded` before a call whose
  projected duration would blow the SLA; `remaining_ms` is the readable
  mid-chain signal for router fallback/truncation; `advisory()` returns
  `near_deadline` / `used_bps` / `remaining_ms`, symmetric to the budget
  advisory's `near_limit`. Monotonic clock (`time.monotonic` /
  `performance.now`); cooperative by design — the guard supplies the deadline
  signal, killing a stalled in-flight call remains the framework's job.

## py 0.4.0 — 2026-07-15

### Added (py)

- **Request-sized pre-call estimates**: `BudgetGuard.estimate_call(model,
  prompt_tokens, max_completion_tokens)` prices the actual incoming request
  from the cost map, so `reserve(est)` / `check(est)` block an oversized call
  — including the very first one, which the last-cost prediction is blind to.
  The LiteLLM adapter reserves request-sized automatically (prompt via
  `litellm.token_counter`, cap from `max_tokens`); the LangChain handler sizes
  its pre-call `check()` from the serialized model config and a ~4 chars/token
  prompt heuristic. Anything unpriceable/unsized falls back to the previous
  last-cost behaviour.
- **Mid-stream enforcement**: `StreamGuard` / `guard_stream()` re-price a
  streaming response chunk-by-chunk and raise `BudgetExceeded`
  mid-generation when the running call would cross the ceiling — the partial
  spend is settled (and lands in `spend_log`) instead of the whole overshoot
  being discovered post-mortem. `finish(prompt_tokens=…, completion_tokens=…)`
  reconciles the chunk heuristic to provider-reported usage; unpriceable
  models fail closed before the stream starts; parallel streams count each
  other's in-flight accrual, so unreserved streams share the ceiling instead
  of each spending it in full. Demo: `examples/streaming_guard.py` (no API
  key).

## py 0.3.0 / js 0.3.0 — 2026-07-14

### Added (py + js)

- **Per-call spend ledger**: every priced `record()` / `settle()` appends a
  typed `SpendEvent` (`timestamp`, `kind: llm|tool`, `model_or_tool`,
  `prompt_tokens`, `completion_tokens`, `cost_usd`, optional `label` and
  `reserved`) to `guard.spend_log` (py) / `guard.spendLog` (js), so the ledger
  sums to the running total (unless the ring-buffer cap below has evicted old
  events) — no more rebuilding per-call breakdowns outside the guard. `export_log()` / `exportLog()` serialises it as JSONL with
  an identical snake_case schema in both languages, so heterogeneous agents
  emit one concatenable stream. An optional `max_log_events` / `maxLogEvents`
  ring-buffer cap bounds memory for long-running agents.
- **`record_tool()` / `recordTool()`**: accrue a non-LLM cost (paid tool/API
  call) against the same ceiling and log it as a `kind: "tool"` event, so
  `check()` / `reserve()` enforce the budget across LLM and tool spend
  together.
- `record()` / `settle()` accept an optional `label` to tag events with an
  agent/task name.

### Fixed (py)

- `floe_guard.__version__` now reports the real package version (it had been
  stuck at `0.1.0` since the 0.2.0 release).

## py 0.2.0 / js 0.2.1 — 2026-07-10

Everything the repo grew between the 0.1.0 uploads and this release ships here —
the earlier revision of this changelog misattributed several of these features
to the py 0.1.0 entry; that entry now reflects what the released artifact
actually contained.

### Added (py)

- **Concurrency-safe enforcement**: atomic `reserve()` / `settle()` /
  `release()` with a lock-guarded running total, closing the
  check-then-record race that let parallel callers blow the ceiling
  (issue #18). `check()` / `record()` are unchanged for sequential use.
- **Context-aware budgeting**: `BudgetGuard.advisory()` returning a
  `BudgetAdvisory` (`near_limit`, `used_bps`, `remaining_usd`, totals), with a
  `near_limit_bps` constructor threshold (default 8000 = 80%).
- **Adapters**: LangChain (`budget_guard_callback_handler`), OpenAI
  (`guarded_completion` / `guarded_acompletion`), and Anthropic (same pair,
  with cache-token metering) — each behind an optional extra
  (`[langchain]`, `[openai]`, `[anthropic]`).
- **Hosted budget read**: `hosted_remaining_usd()` (GET
  `/v1/agents/credit-remaining`, opt-in via `FLOE_API_KEY`, host override via
  `FLOE_API_BASE_URL`), `HostedEnforcementError`, and package-root export of
  `hosted_enforcement_available()`. This is the package's only network call
  and never runs unless you set the key.

### Added (py + js)

- **Groq pricing**: curated Groq models vendored in the cost map —
  `llama-3.1-8b-instant`, `llama-3.3-70b-versatile`,
  `meta-llama/llama-4-scout-17b-16e-instruct`, `qwen/qwen3-32b` (new for py;
  these four already shipped in js 0.2.0), plus the current production lineup
  `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `openai/gpt-oss-safeguard-20b`
  (new for both packages). Kept as an explicit allowlist
  (`scripts/update-cost-map.mjs`) so generic multi-provider names stay
  fail-closed instead of under-metering.
- **Smarter model-id resolution** (both packages, identical logic): lookup
  candidates are tried most-specific-first — the raw id, the id with a known
  `openai/` / `anthropic/` / `groq/` first segment stripped (so
  `groq/qwen/qwen3-32b` and ChatGroq's `qwen/qwen3-32b` hit the same entry),
  then the bare last segment; each also with a trailing dated-snapshot suffix
  removed, so an unlisted snapshot like `claude-opus-4-8-<date>` prices at its
  alias entry instead of failing closed. Unknown provider prefixes are
  deliberately not bridged.
- Cost-map refresh adds `claude-sonnet-5` (both packages; the py package also
  gains `claude-fable-5`, which already shipped in js 0.2.0 —
  `claude-opus-4-8` and `claude-sonnet-4-6` shipped in the 0.1.0 maps) and
  warns when a curated Groq model disappears upstream instead of silently
  dropping it.

### Fixed (py)

- **CrewAI / LiteLLM-callback silent footgun**: LiteLLM runs custom-logger
  hooks inside `except Exception`, so the callback's enforcement raise
  (pre-call `BudgetExceeded`, fail-closed `UnpriceableModelError`) could be
  swallowed and a crew kept running unmetered with no visible signal. The
  callback now records the violation on its `tripped` attribute and logs it at
  ERROR level on the `floe_guard` logger, and
  `budget_guarded_llm` returns a `crewai.LLM` subclass that re-raises
  `tripped` and runs `check()` in the call path — outside LiteLLM — so the
  crew hard-stops at the next call. `guard_crew` now returns the registered
  callback (previously `None`) and reuses an existing registration for the
  same guard.

### Changed (py + js)

- Price lookup order is now most-specific-first (raw id before stripped
  forms) for both `price_overrides` and the cost map. Previously the bare name
  was tried before the raw id.

## js 0.2.0 — published 2026-07-10

### Added

- Curated Groq cost-map entries: `llama-3.1-8b-instant`,
  `llama-3.3-70b-versatile`, `meta-llama/llama-4-scout-17b-16e-instruct`,
  `qwen/qwen3-32b`, plus `claude-fable-5`.
- **Vercel AI SDK v5 support.** The middleware now works with both `ai@4`
  (`LanguageModelV1Middleware`, `promptTokens`/`completionTokens`) and `ai@5`
  (`LanguageModelV2Middleware`, `inputTokens`/`outputTokens`) from a single
  build — it no longer imports types from `ai`, and reads whichever usage
  field pair the installed SDK reports. Peer dependency widened to
  `>=4.0.0 <6.0.0`.
- Exported the `BudgetGuardMiddleware` type.

### Changed

- A response or stream `finish` part with no usable token counts is now
  rejected with a clear error (fail-closed) instead of surfacing an internal
  pricing error; the in-flight reservation is released either way.

## py 0.1.0 / js 0.1.0 — 2026-06

Initial public releases (PyPI 2026-06-15, npm 2026-06-16).

- `BudgetGuard` with `check()` / `record()` (sequential; the atomic
  reservation API landed in py 0.2.0).
- Offline pricing from a vendored LiteLLM cost map covering OpenAI and
  Anthropic; unpriceable models fail closed (`UnpriceableModelError`) with
  manual `price_overrides` as the escape hatch.
- Python adapters: CrewAI and LiteLLM, behind optional extras; the core stays
  dependency-free. (The LangChain, OpenAI, and Anthropic adapters landed in
  py 0.2.0.)
- Hosted-Floe hook as a stub (`hosted_enforcement_available()` under
  `floe_guard.hosted`); this release performs no network calls of any kind.
- TypeScript package (`js/`) with Vercel AI SDK middleware
  (`budgetGuardMiddleware`), verified against `ai@4`.
- No runtime telemetry of any kind.
