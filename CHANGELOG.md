# Changelog

All notable changes to floe-guard are documented here. The repo ships two
packages — `floe-guard` on [PyPI](https://pypi.org/project/floe-guard/) and
`floe-guard` on [npm](https://www.npmjs.com/package/floe-guard) (Vercel AI SDK)
— versioned independently; entries are tagged **py** / **js**.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
both packages adhere to [Semantic Versioning](https://semver.org/).

## Unreleased — py 0.23.5 / js 0.15.5

### Fixed (js)

- **Cache-aware token pricing, wired through the guard.** `priceTokens` /
  `resolvePrice` now consume the cost map's `cache_read_input_token_cost` /
  `cache_creation_input_token_cost` (the JS map already carried them; pricing
  ignored them). `record` / `settle` accept `cacheReadInputTokens` (and
  creation buckets). Middleware reads AI SDK `cachedInputTokens` as a subset of
  prompt tokens; the Vapi adapter reads OpenAI `prompt_tokens_details.cached_tokens`.
  gpt-4o with a 90% cache hit now bills ~0.55× the uncached prompt instead of
  ~1.8× Python.

## Unreleased — py 0.23.2 / js 0.15.3

### Changed (docs)

- **Repositioned around the value, not the mechanism.** The README, package
  one-liners (PyPI/npm), `AGENTS.md`, `SKILL.md`, and `llms.txt` now lead with
  *"Know what every AI call really costs"* — the **live ledger** floe-guard keeps
  locally, and the free **Coverage Score** + **7-day history** you get on connect
  (matching floefinance.com's free tier) — with the budget hard-stop demoted to a
  feature. No API change; the guard behaves exactly as before. Claims stay on the
  right side of the line: the live ledger is local/OSS (`export_log()`), while
  Coverage Score and 7-day history are the free hosted tier the ledger feeds.

## Unreleased — py 0.23.1 / js 0.15.2

### Fixed (py, js)

- **Missing-key sync error now points to key creation.** The `LedgerSyncError`
  raised when `push`/`sync` has no API key said a key was required but not where
  to get one — a dead end on the OSS→hosted handoff. Both packages now point to
  `dev-dashboard.floelabs.xyz/keys` (sign-in is free). Message-only; still raised
  before any network, so the no-network-by-default invariant is unchanged.
- **Agent onboarding verify step uses `floe-guard demo`.** `AGENTS.md`,
  `SKILL.md`, and `llms.txt` told a pip-only install to run
  `python examples/runaway_loop.py`, which 404s because the wheel does not
  ship `examples/`. The packaged command is the same demo README already
  leads with.

## Unreleased — py 0.23.0

### Added (py)

- **`floe-guard estimate` — offline workload cost estimator.** Price a planned
  run (`estimate MODEL --calls N --tokens-in … --tokens-out …`) from the bundled
  cost map — no key, no account, no network — and get a copy-pasteable
  `BudgetGuard(limit_usd=…)` ceiling, rounded **up** so it always covers the run
  it prints. Fail-closed on unpriceable models; oversized inputs exit cleanly.

## Unreleased — py 0.22.0 / js 0.15.1

### Added (py)

- **LiveKit adapter now reconciles avatar and arbitrary-tool legs, not just
  LLM/STT/TTS/telephony.** `LiveKitBudgetGuard.record_tool(tool, cost_usd,
  label=…)` lands any paid leg LiveKit emits no metric for — avatar vendors
  (Tavus, HeyGen, Simli, Beyond Presence) and paid tools/APIs the agent calls —
  on the same ceiling and spend log as the existing voice legs. You supply the
  USD cost (their price shapes vary by vendor, so there is no map to guess
  from), and the leg flows through `guard.export_log()` / `guard.sync()` onto
  Floe's ledger like every other leg, so a self-hosted LiveKit agent can
  reconcile its WHOLE bill (LLM, STT, TTS, telephony, avatars, tools) with the
  local-first / no-network-by-default invariant intact — the ledger leaves the
  process only when you call `sync()`.

## Unreleased — py 0.21.1

### Fixed (py)

- **OpenAI and LiteLLM adapters: cached prompt tokens are priced at the
  cache-read rate, not the full input rate.** Both adapters read only
  `usage.prompt_tokens` / `usage.completion_tokens`, but `prompt_tokens`
  *includes* the share served from the provider's prompt cache
  (`usage.prompt_tokens_details.cached_tokens`) — so every cached token was
  metered at the full input rate. This is the same systematic overcharge
  `turn_cost(prompt_cached_tokens=…)` fixed for the receipt in py 0.20.0, and
  that the Gemini adapter already avoids by carving
  `cached_content_token_count` out of the prompt count. Both adapters now do
  the same and pass the cached share to `settle(…,
  cache_read_input_tokens=…)`, so it prices at the model's published cache-read
  rate. OpenAI enables prompt caching automatically for prompts ≥1024 tokens,
  so any agent loop with a stable system prompt was affected: a 10k-token
  prompt that is 90% cached metered ~1.8x its real cost, hard-stopping the
  agent well before its ceiling was actually reached. A response with no cache
  hit (the block absent, or `cached_tokens` null) is unchanged, and a fully
  cached prompt with no completion now settles its reservation instead of
  taking the usage-less release path.

## Unreleased — py 0.21.0 / js 0.15.1

### Added (py)

- **Vapi and Retell custom-LLM adapters — the in-call voice guards, now in
  Python.** `floe_guard.integrations.vapi.VapiBudgetGuard` guards the
  `/chat/completions` turn (`guard_completion` for JSON, `guard_stream` for SSE
  — both reserve before the upstream call, settle on the real OpenAI `usage`
  after, and release the hold on error/abort; a stream with no `include_usage`
  chunk fails loudly via `VapiUsageMissingError` rather than metering $0),
  answers Vapi's `assistant-request` webhook from the remaining budget
  (`assistant_request`, delegating to `gates.vapi`), and meters the STT/TTS/
  telephony legs the proxy never sees (`meter_stt` / `meter_tts` /
  `meter_telephony`). `floe_guard.integrations.retell.RetellBudgetGuard` does
  the same over Retell's custom-LLM WebSocket: reserve on `response_required`
  (`begin_turn`, idempotent), settle real token usage after `content_complete`
  (`settle_turn`), release the hold when a newer `response_id` interrupts or on
  `close()`, plus `admit_call` (delegating to `gates.retell`), a `response`
  event builder, and the same voice-leg meters. Both adapters are
  **framework-free** — they speak Vapi's OpenAI-format JSON/SSE and Retell's
  plain WS dicts, so a Python voice stack (FastAPI custom-LLM proxy, WS server)
  can enforce per-turn budgets with nothing new to install (`floe-guard[vapi]` /
  `floe-guard[retell]` extras exist for discoverability). They mirror the TS
  adapters (`js/src/adapters/{vapi,retell}.ts`), and ship with no-key, no-network
  examples (`examples/voice_call_cost_vapi.py`,
  `examples/voice_call_cost_retell.py`) that print a pre-call admission decision
  and a per-leg call-cost receipt; the README adapter matrix marks Vapi/Retell
  ✅/✅.

## Unreleased — py 0.20.1 / js 0.15.1

### Added (js)

- **`estimateCall` method on `BudgetGuard`**. Prices the actual incoming request (given model, prompt tokens, and optional max completion tokens) offline using the cost map. Returns `undefined` if the model is unpriceable. Enables TypeScript users to pre-flight check or reserve request-sized budgets to protect against oversized first runs.

### Documentation (py)

- **README Restructure and Examples Index**: Restructured the root README to improve readability, compressed the introductory sections, reordered logical sections (motivation/mechanics earlier), and added a central index table mapping all 16 `examples/*.py` scripts (including the orphaned `langchain_groq_example.py`).

## Unreleased — py 0.20.0 / js 0.15.0

### Added (py)

- **Cache-aware token pricing, driven by per-model cost-map rates.** The bundled
  cost map now carries each model's published `cache_read_input_token_cost` /
  `cache_creation_input_token_cost` (added surgically — existing prices are
  unchanged). `price_tokens` prices cache-read and cache-creation at the model's
  own rate when present, so caching is correct **per provider** (Anthropic reads
  at ~0.1x its input rate, OpenAI at ~0.5x) instead of the old hardcoded
  Anthropic 0.1x for everyone; a model with no published cache rate falls back to
  the conservative multiplier as before. `turn_cost(...)` accepts
  `prompt_cached_tokens` (the subset of `prompt_tokens` served from cache): that
  subset is priced at the cache-read rate rather than the full input rate, fixing
  a systematic overcharge whenever prompt caching was on.

### Changed (js)

- **Bundled cost map now carries per-model cache rates.** The shared cost-map
  snapshot gained `cache_read_input_token_cost` / `cache_creation_input_token_cost`
  per model (byte-identical across the py/js copies). The TS `pricing.ts` does not
  consume them yet — cache-aware TS pricing is a documented parity follow-up; this
  data change is inert for existing js behavior.

## Unreleased — py 0.19.0 / js 0.14.0

### Added (py)

- **`FloeCost` — the per-turn receipt contract, plus `turn_cost()`.** One shape
  for cost whether it's a local estimate (`source="estimate"`, priced from the
  bundled cost map — free, offline, no account) or hosted server-truth
  (`source="hosted"`); `source` is validated to one of those two. `turn_cost(model,
  prompt_tokens, completion_tokens, remaining_usd=?)` returns a `FloeCost` priced
  locally (or `None`, fail-closed, for an unpriceable model) — it never calls the
  network, so the caller passes `remaining_usd` themselves (e.g. the result of
  `hosted_remaining_usd()`, a hosted read) to show budget. `FloeCost.format()`
  renders a one-line receipt (sub-$0.0001 costs get 6 decimals, not a misleading
  `$0.0000`). Intended for adapters to emit once per turn; graduating to hosted is
  a key swap, not a re-integration (identical shape). Adapter wiring and TS parity
  land separately.

### Fixed (js)

- **`wrapStream` now fails closed when a stream ends without a usage-bearing
  finish part.** Previously, `flush()` silently released the reservation and
  allowed the stream to complete as though no tokens were consumed (effective
  $0 spend), violating the package's fail-closed philosophy. The new behavior
  releases the reservation **and** throws a descriptive `Error` — consistent
  with how `wrapGenerate` and `usageTokens()` handle missing token data — so
  the stream is rejected rather than treated as free. Cancellation (`cancel()`)
  behavior is unchanged.

  **AI SDK contract note (verified against `ai@4.3.19` / `@ai-sdk/provider`):**
  `LanguageModelV1StreamPart` defines the `finish` part as carrying `usage`
  (`promptTokens`/`completionTokens`), but the `doStream` contract does **not**
  guarantee that a stream *must* emit a finish part before closing. A stream
  that ends without one is therefore a contract violation that the guard must
  treat as unmeterable spend, not a free call.

### Fixed (py)

- **Retry predicate: block the entire `FloeGuardError` family, not just
  `BudgetExceeded`.** The previous default predicate (`not isinstance(exc,
  BudgetExceeded)`) would retry deterministic guard errors such as
  `UnpriceableModelError`, potentially multiplying LLM spend on errors that
  retrying cannot fix. The new default (`not isinstance(exc, FloeGuardError)`)
  treats every `FloeGuardError` — `BudgetExceeded`, `UnpriceableModelError`,
  `UnpriceableVoiceError`, `HostedEnforcementError`, `DeadlineExceeded`, etc. —
  as terminal. Since `BudgetExceeded` is already a `FloeGuardError` subclass,
  its existing no-retry behavior is preserved with no wiring changes.
- **Step-USD diagnostics: `BudgetExceeded` now carries the step ceiling and
  step committed spend, not the aggregate guard values.** When a per-step USD
  budget (`guard.step(max_usd=…)`) was crossed, the resulting `BudgetExceeded`
  reported the aggregate guard figures (`spent $X of $Y`), even though the
  violated ceiling was the (much smaller) step cap. The blocking cross-check now
  snapshots and returns the step's own `s_committed` and `step.max_usd`, so
  `exc.spent_usd` / `exc.limit_usd` and the message reflect the step boundary.
  Token-step and aggregate-budget behavior are unchanged.

### Internal (py)

- **README anchor link-check in CI** (`tests/test_readme_links.py`) — asserts every
  in-README `](#anchor)` resolves, using GitHub's exact slug algorithm (each space
  becomes its own hyphen, so `stt -> llm -> tts` headings keep their double hyphens).
  Prevents the naive-checker false positive that flagged the live
  `#voice-adapters-stt--llm--tts` / `#latencybudget--deadlines-the-same-way` anchors
  as dead. Current status: **0 dead anchors**.
- **Release maturity: kept at `Development Status :: 4 - Beta`** for 0.17.0. The
  documented enforcement caveats (estimate-based, cold-start concurrency, shared-file
  persistence, approximate stream cut-offs) make `5 - Production/Stable` an overclaim;
  rationale recorded inline in `pyproject.toml`.

### Added (py)

- **Bundled-pricing freshness.** `cost_map_generated_at()` returns the
  `YYYY-MM-DD` the bundled cost map was last generated/verified (from a reserved
  `__meta__` key; `None` for a pre-metadata/invalid snapshot), so you can display
  or gate on how fresh the drift-prone list rates are. `scripts/update-cost-map.mjs`
  stamps it on every refresh; the token resolver excludes `__meta__` like
  `__voice__`. TS parity: `costMapGeneratedAt()`.

- **`floe-guard demo` — the no-key demo, runnable from the installed package.**
  `pip install floe-guard && floe-guard demo` runs the runaway-loop demo (stub
  LLM, no account, no network) straight from the wheel — no repository checkout
  needed. The demo logic moved into the package (`floe_guard.demo.run_demo`);
  `examples/runaway_loop.py` is now a thin wrapper around it, so there's one
  source of truth. `--limit-usd` overrides the $0.10 ceiling.

- **Cost map: Claude 3.5 family.** Added `claude-3-5-sonnet-20241022`,
  `claude-3-5-sonnet-20240620` ($3.00/$15.00 per 1M in/out) and
  `claude-3-5-haiku-20241022` ($0.80/$4.00 per 1M) to the bundled cost map, so
  they price instead of raising `UnpriceableModelError` under the default
  `fail_closed=True`. (Closes #51.)

- **Opt-in ledger sync → Reconcile Mode / Coverage Score.** A new **explicit,
  off-by-default** way to push your local spend ledger to Floe so **Coverage
  Score** can count spend the gateway never routed (BYOK / self-hosted /
  off-path): `guard.enable_sync(api_key=…)` then `guard.sync()` (programmatic), or
  `floe-guard push ledger.jsonl` (CLI). **Zero-telemetry stays the default** —
  nothing leaves the process without the explicit opt-in *and* an explicit send;
  there is no implicit enablement and no background send (asserted in tests:
  `urlopen` is never called for a guard that hasn't opted in). The **request body**
  is exactly the `export_log()` JSONL — priced spend events (timestamp, kind,
  model/tool, tokens, `cost_usd`, optional `label`/`reserved`) — and `push_ledger`
  **validates every line, rejecting any field outside that schema**, so no prompts
  or message content leave even from a hand-supplied ledger. The only
  caller-controlled string transmitted is the optional `label` you set yourself
  (keep identifiers out of it). Your key
  travels in the `Authorization` header; redirects are refused so key + ledger
  can't be re-sent to an unapproved host. `disable_sync()` revokes it.
  Budget, not balance: it reports what you already spent for coverage/attribution;
  it moves no money. New: `push_ledger()`, `LedgerSyncError`, the `floe-guard` CLI
  (`[project.scripts]`). (JS parity + the server ingest endpoint land separately.)

## Unreleased — py 0.16.2

### Fixed (py)

- **LiveKit adapter: migrate off the deprecated session-level `metrics_collected`
  event.** `livekit-agents` 1.5 deprecated `AgentSession.on("metrics_collected", …)`
  (it now logs a warning and points at `session_usage_updated` / `ChatMessage.metrics`).
  `LiveKitBudgetGuard` now settles/meter each turn off the **per-component**
  `metrics_collected` event (the LLM / STT / TTS plugins each emit it and it is
  **not** deprecated) instead of the session facade. The LLM plugin fires per
  LLM turn, so the reserve-before-turn → settle-real-usage contract is preserved
  exactly; `session_usage_updated` was rejected because it delivers a *cumulative*
  `UsageSummary` (itself deprecated), which would force diffing successive
  summaries and lose clean per-turn settlement. Components are resolved from the
  agent first, then the session, mirroring LiveKit's own runtime pick. No public
  API change; the `livekit` extra floor stays `>=1.0` (the per-component event
  predates the deprecation). (The TS `@livekit/agents` does not deprecate the
  event, so the Vercel-AI/js package is unaffected.)
- **LiveKit adapter hardening — parity with the merged JS fixes, plus component
  swaps.** The constructor validates configured voice vendors (a typo'd
  `stt_model` / `tts_model` / `telephony` now raises `UnpriceableVoiceError` at
  construction, not mid-call); `attach()` is single-use (a second call raises
  rather than double-subscribing and double-counting spend); the internal turn
  queue is bounded (`_MAX_SLOTS`) so a long call with realtime / text-only turns
  can't accumulate stale slots. And metric subscriptions are **re-synced on
  component swaps** — `livekit-agents` emits no swap event, so they're reconciled
  on every `agent_state_changed` off `session.current_agent`. Honest limitation
  (documented): `AgentSession.update_agent()` installs a new `Agent` whose
  `llm_node` isn't wrapped, so its turns are still metered (reconcile) but the
  pre-turn hard-stop is bypassed until you re-`attach()`; `update_options` (same
  agent, swapped models) keeps the reserve hook.
- **Windows-safe example output.** Replace non-cp1252 characters (`U+2192 →`,
  `U+2248 ≈`) with ASCII equivalents (`->`, `~=`) in the printed strings of
  `examples/budget_aware.py`, `examples/context_size.py`,
  `examples/retrieval_depth.py`, `examples/streaming_guard.py`, and
  `examples/langgraph_budget_aware.py`. The
  Unicode-only characters caused `UnicodeEncodeError` on Windows consoles using
  the default cp1252 encoding. Characters that appear only in docstrings or
  comments (not printed at runtime) are left unchanged.
- **Deterministic stdout/stderr ordering for redirected demo output.** Added
  `flush=True` to every `print()` status line in
  `examples/runaway_loop.py`, `examples/budget_aware.py`,
  `examples/context_size.py`, `examples/retrieval_depth.py`,
  `examples/tool_budget.py`, `examples/step_budget.py`,
  `examples/streaming_guard.py`, `examples/openai_adapter.py`,
  `examples/anthropic_adapter.py`, `examples/budget_retry.py`, and
  `examples/plan_complexity.py`. When stdout is redirected or piped the OS
  buffers it while stderr (where the library's block banner goes) is
  unbuffered, producing output in the wrong order. Flushing each status print
  ensures the logical sequence is preserved. The library's own stderr behavior
  is unchanged.
- **Subprocess smoke tests for the no-API-key examples.** New
  `tests/test_examples_smoke.py` executes each README-promoted example as a
  real subprocess (no import tricks), strips `FLOE_API_KEY` and
  `OPENAI_API_KEY` from the environment, asserts `returncode == 0`, and
  checks for a meaningful output marker. Covers: `runaway_loop.py`,
  `budget_aware.py`, `context_size.py`, `retrieval_depth.py`,
  `plan_complexity.py`, `tool_budget.py`, `step_budget.py`,
  `streaming_guard.py`, `budget_retry.py`, `openai_adapter.py`, and
  `anthropic_adapter.py`. The ordering test for `runaway_loop.py` merges
  stdout and stderr and asserts that "Starting a runaway loop" precedes the
  "BUDGET EXCEEDED" banner.

## Unreleased — js 0.13.1

### Fixed (js)

- **Retry predicate: block the entire `FloeGuardError` family, not just
  `BudgetExceeded`.** The previous default predicate (`!(error instanceof
  BudgetExceeded)`) would retry deterministic guard errors such as
  `UnpriceableModelError`, potentially multiplying LLM spend on errors that
  retrying cannot fix. The new default (`!(error instanceof FloeGuardError)`)
  treats every `FloeGuardError` as terminal. Since `BudgetExceeded` is already
  a `FloeGuardError` subclass, its existing no-retry behavior is preserved.
- **Step-USD diagnostics: `BudgetExceeded` now carries the step ceiling and
  step committed spend, not the aggregate guard values.** When a per-step USD
  budget (`guard.step({ maxUsd: … })`) was crossed, `BudgetExceeded` reported
  the aggregate guard figures. The blocking cross now captures the step's own
  `sCommitted` and `step.maxUsd` and returns them in a 4-element tuple (matching
  Python's `(dimension, scope, spent, limit)` shape), so `error.spentUsd` /
  `error.limitUsd` and the message reflect the step boundary. Token-step and
  aggregate-budget behavior are unchanged.

### Added (js)

- **`costMapGeneratedAt()`** — parity with the Python
  `cost_map_generated_at()`: the `YYYY-MM-DD` the bundled cost map was last
  generated/verified (from the reserved `__meta__` key), or `undefined` for a
  pre-metadata/invalid snapshot. The token map now also excludes reserved dunder
  keys (`__voice__`, `__meta__`), matching Python.

- **Cost map: Claude 3.5 family.** Added `claude-3-5-sonnet-20241022`,
  `claude-3-5-sonnet-20240620` ($3.00/$15.00 per 1M in/out) and
  `claude-3-5-haiku-20241022` ($0.80/$4.00 per 1M) to the bundled cost map
  (kept byte-identical with the Python copy).

- **Opt-in ledger sync — `BudgetGuard.enableSync()` / `disableSync()` / `sync()`
  + `pushLedger()`** (parity with the Python client). Pushes the guard's
  `exportLog()` JSONL to Floe's Reconcile Mode (`POST /v1/agents/ledger/sync`) so
  BYOK / self-hosted / off-path spend the gateway never routed still lands on the
  ledger and your **Coverage Score** becomes computable. **Budget, not balance:**
  it reports what you already spent for coverage/attribution; it moves no money
  and changes no wallet balance. Re-syncing is safe — the sync endpoint is
  idempotent by design (already-ingested events aren't double-counted).
  - **Zero-telemetry is preserved as the default.** Sync is OFF until you call
    `enableSync(apiKey)`, and even then nothing leaves the process until you call
    `sync()` — there is no implicit enablement and no background send. `sync()` on
    a guard that never opted in throws (no network); `disableSync()` revokes it.
    `sync()` snapshots the opt-in key atomically before any `await`, so a racing
    `disableSync()` can't cause a post-revocation send or an env-key fallback.
  - **The request body is exactly `exportLog()`, validated.** Before any network,
    every ledger line is checked against the `exportLog()` schema and **any
    non-schema / unknown field is rejected** (`LedgerSyncError`) — a hand-supplied
    file can't smuggle prompts, content, or identifiers past the privacy contract.
    The key rides the `Authorization: Bearer` header (`Content-Type:
    application/x-ndjson`).
  - **Fail-closed.** `pushLedger(jsonl, apiKey?, { baseUrl?, timeoutMs? })` (over
    `fetch`) refuses to send without a key (arg or `FLOE_API_KEY`) or over a
    non-https / malformed base URL — the key and ledger are never transmitted in
    those cases — **refuses redirects** (`redirect: "error"`, so a 3xx can't
    re-send the key/ledger to another host), and throws `LedgerSyncError` on a
    non-2xx / network / malformed response or an invalid (non-integer / negative)
    `synced` count. An empty ledger is a no-op (`0`, no network). (The `floe-guard
    push` CLI stays Python-only for now.)

## Unreleased — js 0.12.0

### Added (js)

- **Voice adapters — LiveKit, Vapi, Retell** (the Node voice glue): each wraps its
  platform's model turn with reserve-before / settle-on-real-usage / release-on-
  interrupt, meters STT/TTS/telephony legs from the `__voice__` cost map via
  `priceVoiceLeg` (fail-closed), and answers the platform's inbound admission
  webhook via `gates`. Pre-turn/pre-call admission + per-turn settlement only — no
  mid-call cutoff.
  - `floe-guard/adapters/livekit` → `LiveKitBudgetGuard.attach(session, agent)`:
    reserves in the agent's `llmNode`, settles on the session's `metrics_collected`
    (verified **not** deprecated in `@livekit/agents` 1.6.x, unlike Python 1.5.0),
    releases on stream cancel / close. `@livekit/agents` is an optional peer.
  - `floe-guard/adapters/vapi` → `VapiBudgetGuard`: `guardCompletion` (JSON) /
    `guardStream` (SSE) reserve before the upstream call and settle on the real
    OpenAI `usage`; `assistantRequest` wraps `gates.vapi`. A stream with no `usage`
    (upstream missing `stream_options.include_usage`) throws `VapiUsageMissingError`
    rather than metering a silent $0.
  - `floe-guard/adapters/retell` → `RetellBudgetGuard`: `beginTurn` (returns an
    admit/block decision) / `settleTurn` keyed by `response_id`, a newer id releases
    the prior hold (barge-in); `admitCall` wraps `gates.retell`.
  - Each ships a runnable **stubbed, no-key** demo (`js/examples/*_voice_cost.mjs`)
    that prints a pre-call admission decision and a per-leg call-cost receipt.
    Adapters are structurally typed — no hard runtime SDK dependency.

## py 0.16.1 / js 0.11.0 — 2026-08-14

### Fixed (py)

- **Unblock PyPI publishing** — cap `hatchling<1.32` in `[build-system]`.
  hatchling 1.32.0 bumped the emitted core-metadata to `Metadata-Version: 2.5`,
  which the release action's twine (`pypa/gh-action-pypi-publish@v1.14.0`) rejects
  (`InvalidDistribution: '2.5' is not a valid metadata version`) — that is why
  PyPI froze at 0.13.0 (Aug 6) while the repo moved to 0.16.x. Pinning to the last
  2.4-emitting hatchling restores a wheel both twine and PyPI accept (verified:
  `Metadata-Version: 2.4`, `twine check` passes). Publishing 0.16.1 catches up
  everything since 0.13.0.
- **`gates` / voice-pricing input hardening** (parity with the JS fixes):
  `gates.retell` strips `reject` from `admit` overrides so an errant
  `admit={"reject": True}` on an available-budget call can't flip the response into
  Retell's reject shape; `voice_leg_cost` raises on a non-finite `quantity`
  (`max(0.0, nan)` is `nan`, which would poison the guard's running total).

### Added (py)

- **Voice-native primitives — per-call budgets, $/min burn rate, pre-call gates**
  (P1):
  - `guard.step(max_usd=…)` is now documented as the **per-call budget** primitive
    (per-call cap, `advisory().step_remaining_usd` headroom, `est_calls_remaining`)
    — voice budgets are per-call, not per-day.
  - `advisory().burn_rate_usd_per_min` — the $/min spend rate voice teams watch,
    derived from spend ÷ minutes since the guard was created (one guard per
    call/turn ⇒ a per-call rate).
  - `floe_guard.gates` — **pre-call admission** gates returning each provider's
    exact inbound-webhook shape, so a local user rejects a call on budget
    exhaustion with the same contract the hosted gateway serves: `gates.retell()`
    → `{"call_inbound": {"reject": true}}` (only the boolean rejects; phone/SMS
    inbound, 10s/3-retry); `gates.vapi()` → `{"error": …}` reject vs
    `{"assistant"|"assistantId": …}` admit (~7.5s deadline); `gates.pre_call()` /
    `gates.budget_exhausted()` for Pipecat/custom. **Pre-call admission only** — no
    mid-call intervention. Bland's *Send Call* metadata field name is an open
    verification item and is **not** invented (use `gates.pre_call`). US-only v1.
    Budget, not balance.

- **One line to hosted — `BudgetGuard.from_floe(api_key=…)`**: constructs a guard
  whose local ceiling is read from your **server-side budget headroom** (the min
  of auto-borrow headroom and session spend remaining, via
  `hosted_remaining_usd()`), so the free→hosted upgrade is a one-line constructor
  swap with no other code changes. Budget, not balance: the read is a headroom
  signal and enforcement stays local — hosted Floe remains the source of truth for
  the un-bypassable, cross-vendor cap. Zero-telemetry invariant preserved (no
  network unless a key is set) and fail-closed (a failed read raises
  `HostedEnforcementError`, or degrades to a `fallback_limit_usd` local ceiling
  with a loud warning). The README core pitch now documents the shared near-limit
  signal (`near_limit` + `used_bps`) between local `advisory()` and hosted's
  `X-Floe-Budget-Advisory` header, so tapering logic carries over — the header
  nests it under `tightest` with raw amounts, a light field remap, not an
  identical shape.

- **Voice cost map — meter the whole call by default** (P0): STT/TTS/telephony
  rates ship under the reserved `__voice__` key of `cost_map.json` (units: STT
  $/sec, TTS $/1k-chars, telephony $/min). The Pipecat + LiveKit adapters now
  price the full call — STT + LLM + TTS + telephony — from the map with no
  hand-typed rates (`stt_model` / `tts_model` / `telephony`); per-unit overrides
  still win. A vendor the map can't price (or a wrong unit/mode) fails closed via
  the new `UnpriceableVoiceError` (parity with `UnpriceableModelError`), never a
  silent $0. Seeded US-only v1 — Deepgram, AssemblyAI, ElevenLabs, Cartesia
  (Sonic TTS + Line telephony), Rime, Twilio (Telnyx deferred); rates are a
  drift-prone snapshot. The **js** package carries the same `__voice__` cost-map
  data in lockstep (no JS voice adapter this release).

### Added (js)

- **Voice foundation ported to TypeScript — cost-map pricing, pre-call gates,
  $/min burn rate** (parity with py, no voice adapter yet):
  - **Offline voice cost-map pricing** (`voice-pricing.ts`): `priceVoiceLeg` /
    `resolveVoiceRate` / `lookupVoiceRate` / `voiceLegCost` price a leg from the
    `__voice__` section of `cost_map.json` (STT $/sec, TTS $/1k-chars, telephony
    $/min). A per-unit override wins over the map; a vendor the map can't price (or
    a wrong unit/mode) fails closed via the new `UnpriceableVoiceError` (parity with
    `UnpriceableModelError`), never a silent $0. An unconfigured leg (no vendor, no
    override) returns `null` so the token-only contract is preserved.
  - **`gates` — pre-call admission** (`gates.ts`, exported as the `gates`
    namespace): `gates.retell()` → `{ call_inbound: { reject: true } }` (only the
    boolean rejects; phone/SMS inbound, 10s/3-retry); `gates.vapi()` → `{ error }`
    reject vs `{ assistantId | assistant }` admit (`assistantId` takes precedence;
    ~7.5s deadline); `gates.preCall()` / `gates.budgetExhausted()` for
    Pipecat/custom. A non-finite/negative estimate throws rather than silently
    admitting. **Pre-call admission only** — no mid-call intervention. Bland's
    *Send Call* metadata field name is an open verification item and is **not**
    invented (use `gates.preCall`). US-only v1. Budget, not balance.
  - **`advisory().burnRateUsdPerMin`** — the $/min spend rate voice teams watch,
    spend ÷ minutes since the guard was created (`null` until wall-clock time has
    elapsed; one guard per call/turn ⇒ a per-call rate).

## py 0.13.0 / js 0.9.0 — 2026-08-06

### Added (py)

- **Persistent UTC-day budgets** (issue #47): `BudgetGuard(...,
  window="utc-day", store=SqliteStore(path))` keeps spend and in-flight
  reservations in dependency-free SQLite transactions, so sequential and
  overlapping Python processes share one daily ceiling. With no store/window,
  the existing in-memory behavior is unchanged. (Persistence is USD-only; it
  cannot be combined with `token_limit` / `step()` yet.)

### Added (py + js)

- **Token ceilings + per-step budgets** (issue #46) — a **minimal alternative to
  PR #58** (~1.1k LOC vs 4.1k). Adds a second dimension (tokens) on the existing
  enforcement choke point plus a step scope, not a new reservation system:
  - `BudgetGuard(..., token_limit=N)` / `{ tokenLimit: N }` — an aggregate token
    ceiling. `check` / `reserve` take `estimated_tokens` / `{ estimatedTokens }`
    to pre-emptively block; a cross raises the new `TokenBudgetExceeded`
    (`scope="aggregate"`). Tokens accrue for free from the counts `settle` /
    `record` already receive.
  - `guard.step(max_usd=…, max_tokens=…)` — a per-step cap for a **sequential**
    agent loop. Python is a context manager (`with guard.step(...) as g:`, `g` is
    the guard); TS is a callback (`guard.step({ maxTokens }, (g) => …)`). Both
    yield the SAME guard, so adapters are untouched. A per-step block raises
    `BudgetExceeded` (USD) / `TokenBudgetExceeded` (`scope="step"`) even when the
    aggregate budget has room.
  - `advisory()` gains `token_used_bps` / `remaining_tokens` and per-step
    `step_remaining_usd` / `step_remaining_tokens` (camelCase in TS); `near_limit`
    now also flips on a near token ceiling or step cap.
  - `TokenBudgetExceeded` subclasses `BudgetExceeded`, so budget-aware retry
    treats a token block as terminal with no extra wiring.
  - **Backward compatible:** USD enforcement is unchanged with no `token_limit`
    and no `step()`, and `reserve()` still returns a plain `float` / `number`. A
    `BudgetReservation` handle is returned only when tokens are actually reserved
    or a step is active. `advisory()` additionally gains the token/step fields
    above — additive, and `None` / `null` when their dimension is unused.
- **`advisory()` exposes `expected_cost` + `est_calls_remaining`** (`expectedCost`
  / `estCallsRemaining` in TS, issue #49): the guard's own next-call estimate and
  how many more calls the remaining budget buys, so a planner can see call
  headroom, not just dollars. `expected_cost` / `expectedCost` is `0.0` / `0`
  before the first call is recorded; `est_calls_remaining` / `estCallsRemaining`
  is `None` / `null` until then (unknown, not zero).
  Additive fields — existing advisory consumers are unaffected.

### Added (py)

- **Non-model budget-adaptation examples** (issue #50):
  `examples/retrieval_depth.py`, `examples/context_size.py`, and
  `examples/plan_complexity.py` show `advisory()` driving RAG `top_k`, history and
  `max_tokens` truncation, and optional-sub-task pruning — each with the model
  held fixed, so the savings come from the non-model knob. Examples and docs
  only; no API change.

## py 0.10.0 / js 0.7.0 — 2026-07-23

### Added (py + js)

- **Budget-aware retry helper** (`with_budget_retry` / `withBudgetRetry`,
  issue #45): retry normally when budget is healthy, ask a caller-supplied
  degrade callback for a cheaper retry plan when `advisory().near_limit` /
  `nearLimit` is set, and hard-block an over-budget retry with `check()` before
  it runs. Ships with a no-network Python example.

### Fixed (py)

- **Budget-aware retry**: reject non-integer `max_attempts`, and catch
  `Exception` (not `BaseException`) so `KeyboardInterrupt` /
  `SystemExit` / `CancelledError` propagate instead of being retried.

## py 0.9.0 / js 0.6.0 — 2026-07-23

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
