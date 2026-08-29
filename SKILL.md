---
name: floe-guard
description: The spend meter and budget gate for AI voice agents — meters STT + TTS + LLM + telephony per call (Pipecat, LiveKit — Python & TypeScript) and hard-stops the next turn before it crosses a USD ceiling. Use when an agent's spend must be capped in-process with no account or telemetry; also guards any LLM agent and paid tool calls.
license: MIT
homepage: https://github.com/Floe-Labs/floe-guard
---

# floe-guard — local budget guardrail

The spend meter and budget gate for AI **voice** agents: it meters STT + TTS + LLM +
telephony **per call** ([Pipecat](https://github.com/Floe-Labs/floe-guard#pipecat-voice),
[LiveKit](https://github.com/Floe-Labs/floe-guard#livekit-voice) — Python & TypeScript)
and hard-stops the next turn *before* it crosses a USD ceiling. In-process, no account,
no telemetry. It also guards **any** LLM/tool call the same way. Budget, not balance —
it caps spend, holds no money, needs no account.

> This is the **library** skill. For the whole govern-your-spend workflow (spend
> policies, voice coverage, hosted upgrade, Coverage Score), add the full Floe skill:
> `npx skills add floe-labs/agent-skills`.

## Install

- Python: `pip install floe-guard`
- TypeScript / Node: `npm i floe-guard`

## Wire the hard-stop

`check()` before each model call (raises if it would cross the ceiling), `record()`
after (prices the tokens offline and accrues them):

```python
from floe_guard import BudgetGuard

guard = BudgetGuard(limit_usd=5.00)
guard.check()                         # raises BudgetExceeded before an over-budget call runs
resp = call_your_llm(...)
guard.record("gpt-4o", resp.usage.prompt_tokens, resp.usage.completion_tokens)
```

Verify offline (no key, no network): `floe-guard demo`.

## Skip the manual wiring — adapters

- **Frameworks**: OpenAI, Anthropic, Gemini, CrewAI, LangChain, LangGraph, LiteLLM,
  Vercel AI SDK.
- **Voice** (per-turn, prices STT + LLM + TTS + telephony): Python
  `floe_guard.integrations.pipecat` / `.livekit`; TS
  `floe-guard/adapters/{livekit,vapi,retell}`.
- **Pre-call admission**: `floe_guard.gates` (`retell` / `vapi` / `pre_call`) reject
  an over-budget call before it connects.
- **Paid tools**: `reserve_tool()` / `settle_tool()` block *before* the tool runs.

## Rules

- Enforcement lives **in the call path**: `check()` / `reserve()` *before* the call.
  `record()` alone meters after the fact — it can't stop a call already made.
- **Fail-closed**: an unpriceable model raises `UnpriceableModelError` unless you pass
  a `price_overrides` rate or `fail_closed=False`.
- Use `reserve()` / `settle()` (not `check()` then `record()`) under concurrency.
- Read `advisory()` (near-limit flag + `$/min` burn rate) to taper before the hard-stop.
- **Opt-in only, off by default**: `BudgetGuard.from_floe(api_key=…)` (hosted headroom)
  and `enable_sync()` / `floe-guard push` (ledger sync for Coverage Score) are the only
  things that touch the network — nothing does without an explicit opt-in.

## References

- Full guide: https://github.com/Floe-Labs/floe-guard#readme
- Integration steps for an agent: https://github.com/Floe-Labs/floe-guard/blob/main/AGENTS.md
- The full Floe skill: https://github.com/Floe-Labs/agent-skills
