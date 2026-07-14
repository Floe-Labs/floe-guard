# Changelog

All notable changes to floe-guard are documented here. The repo ships two
packages — `floe-guard` on [PyPI](https://pypi.org/project/floe-guard/) and
`floe-guard` on [npm](https://www.npmjs.com/package/floe-guard) (Vercel AI SDK)
— versioned independently; entries are tagged **py** / **js**.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
both packages adhere to [Semantic Versioning](https://semver.org/).

## Unreleased

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
