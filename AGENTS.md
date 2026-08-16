# AGENTS.md — floe-guard

Instructions for AI coding agents. Most often you're here to **add floe-guard to
an agent** (below); the last section covers **working on this repo**.

## What floe-guard is

A **local budget guardrail**: it hard-stops an AI agent *before* its next LLM or
tool call — or its next voice turn — would cross a USD spend ceiling. A runaway
loop dies at $0.10 instead of $4,000. Runs **in your process**: no account, no
signup, no network, **no telemetry**. Python (`pip`) + TypeScript (`npm`). MIT.

## Add it to an agent — unattended steps

1. **Install**
   - Python: `pip install floe-guard`
   - TypeScript / Node: `npm i floe-guard`
2. **Wire the hard-stop around each model call** (`check` before, `record` after):
   ```python
   from floe_guard import BudgetGuard

   guard = BudgetGuard(limit_usd=5.00)   # the ceiling
   guard.check()                         # before each LLM call — raises BudgetExceeded if it'd cross
   resp = call_your_llm(...)             # your existing call
   guard.record("gpt-4o", resp.usage.prompt_tokens, resp.usage.completion_tokens)
   ```
   TypeScript is the same shape (`guard.check()` / `guard.record(...)`); see `js/`.
   - **Framework adapters** (no manual check/record): OpenAI, Anthropic, Gemini,
     CrewAI, LangChain, LangGraph, LiteLLM, Vercel AI SDK — README §"Framework adapters".
   - **Voice** (per-turn): Python `floe_guard.integrations.pipecat` / `.livekit`;
     TS `floe-guard/adapters/{livekit,vapi,retell}` — reserve-before-turn.
   - **Paid tools**: `reserve_tool()` / `settle_tool()` block *before* the tool runs.
3. **Verify offline** (no key, no network): `python examples/runaway_loop.py` — the
   guard halts a stubbed loop after a few iterations.

## Rules — get these right

- **The hard-stop must be IN the call path.** `check()` / `reserve()` run *before*
  the call. `record()` alone meters *after* the fact — it can't stop a call already
  made. A passive cost callback is not enforcement.
- **Fail-closed.** An unpriceable model raises `UnpriceableModelError` unless you
  pass a `price_overrides` rate or `fail_closed=False` (explicitly opting into
  un-metered spend for that model).
- **Concurrency**: for parallel calls use `reserve()` / `settle()` (they hold the
  estimate across the await), not `check()` then `record()`.
- **Budget, not balance.** floe-guard caps *spend*; it holds no money and needs no
  account. `advisory()` exposes a near-limit flag + `$/min` burn rate so an agent
  can taper *before* the hard-stop.
- **Optional, opt-in only.** `BudgetGuard.from_floe(api_key=…)` reads your hosted
  budget headroom into the local ceiling; `enable_sync()` / `floe-guard push` sync
  the local ledger to Floe for Coverage Score. **Off by default — zero telemetry
  otherwise.** Nothing leaves the process without an explicit opt-in.

## The full govern-your-spend workflow (skill)

For the complete workflow — spend policies, voice coverage, the hosted upgrade,
Coverage Score — install the **Floe agent skill** (teaches Claude Code / Cursor):

```
npx skills add floe-labs/agent-skills
```

Repo: https://github.com/Floe-Labs/agent-skills · Docs: https://floe-labs.gitbook.io/docs

## Working on THIS repo

- **Python**: `pip install -e ".[dev]"`, then `pytest` and `ruff check .`.
- **TypeScript**: `cd js && npm install`, then `npm run typecheck && npm run build && npm test`.
- The two packages **version independently** (`pyproject.toml` / `js/package.json`);
  a source change needs a version bump or CI's **Version Guard** fails the PR.
- Match the surrounding code's style; keep the zero-telemetry default sacrosanct —
  no network without an explicit opt-in, and tests must prove it.
