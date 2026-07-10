# Changelog

All notable changes to floe-guard are documented here. The repo ships two
packages — `floe-guard` on [PyPI](https://pypi.org/project/floe-guard/) and
`floe-guard` on [npm](https://www.npmjs.com/package/floe-guard) (Vercel AI SDK)
— versioned independently; entries are tagged **py** / **js**.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
both packages adhere to [Semantic Versioning](https://semver.org/).

## Unreleased

## js 0.2.0 — 2026-07-08

### Added

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

Initial public release.

- `BudgetGuard` with `check()` / `record()`, concurrency-safe
  `reserve()` / `settle()` / `release()`, and `advisory()` for context-aware
  budgeting (taper before the hard-stop).
- Offline pricing from a vendored LiteLLM cost map; unpriceable models fail
  closed (`UnpriceableModelError`) with manual `price_overrides` as the
  escape hatch.
- Python adapters: CrewAI, LiteLLM, LangChain, OpenAI, Anthropic — each behind
  an optional extra; the core stays dependency-free.
- TypeScript package (`js/`) with Vercel AI SDK middleware
  (`budgetGuardMiddleware`), verified against `ai@4`.
- Optional hosted-Floe budget read (`hosted_remaining_usd()`) via
  `FLOE_API_KEY` — the only network call in the package, opt-in.
- No runtime telemetry of any kind.
