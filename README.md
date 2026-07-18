# floe-guard

[![PyPI version](https://img.shields.io/pypi/v/floe-guard.svg)](https://pypi.org/project/floe-guard/)
[![npm version](https://img.shields.io/npm/v/floe-guard.svg)](https://www.npmjs.com/package/floe-guard)
[![Downloads](https://static.pepy.tech/badge/floe-guard/month)](https://pepy.tech/project/floe-guard)
[![Python versions](https://img.shields.io/pypi/pyversions/floe-guard.svg)](https://pypi.org/project/floe-guard/)
[![CI](https://github.com/Floe-Labs/floe-guard/actions/workflows/ci.yml/badge.svg)](https://github.com/Floe-Labs/floe-guard/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**A local budget guardrail for AI agents.** It hard-stops your agent *before its
next LLM call* when it would cross a spend ceiling — so a runaway loop dies at
$0.10 instead of $4,000. No account, no signup, no network, **no telemetry**.
Runs in your process.

Works with [CrewAI](#crewai) · [LiteLLM](#litellm) · [LangChain](#langchain) ·
[OpenAI](#openai) · [Anthropic](#anthropic) ·
[Vercel AI SDK](#vercel-ai-sdk) — or any stack, via plain `check()` / `record()`.

```bash
pip install floe-guard        # Python
npm i floe-guard              # TypeScript (Vercel AI SDK) — see js/
```

```python
from floe_guard import BudgetGuard

guard = BudgetGuard(limit_usd=5.00)   # your ceiling
guard.check()                         # before each LLM call — raises if it'd cross
response = call_your_llm(...)         # your existing call
guard.record("gpt-4o", response.usage.prompt_tokens, response.usage.completion_tokens)
```

When the next call would cross the ceiling, the guard raises `BudgetExceeded` and
prints:

```
BUDGET EXCEEDED — call blocked
  spent so far: $5.001250  |  ceiling: $5.000000
  The next call would cross your budget; floe-guard stopped your agent before it ran.
```

![floe-guard hard-stopping a runaway loop before it crosses a $0.10 ceiling](docs/stop-the-loop.gif)

_Run it yourself: `python examples/runaway_loop.py` — no API key, no account, no network._

## See it stop a loop (no API key needed)

```bash
python examples/runaway_loop.py
```

This rigs a loop against a **stub LLM** — no real API key, no account, no network.
It prices each fake `gpt-4o` call offline and the guard halts the loop after a few
iterations. This is the reproducible "stop the loop" demo.

## Why floe-guard?

You can already *see* what your agent spends — the problem is seeing it too late.
floe-guard is the part that **stops the call**, not the part that reports the damage.

- **`max_tokens` / `max_rpm`** cap size and rate, not **dollars** — a cheap model
  stuck in a loop still drains the budget.
- **Usage logs and provider dashboards** tell you what you spent *after* it's gone.
  floe-guard refuses the call *before* it crosses your ceiling.
- **A cost callback that just logs** is notified after the fact and can't halt the
  run — enforcement has to stand in front of the next call. That's where it lives.
- **A hand-rolled `spent += cost` counter races under parallel agents** (CrewAI
  fan-out, `asyncio`, `Promise.all`): N calls read the same under-limit total and
  all fire. floe-guard reserves atomically (`reserve()`/`settle()`), so the ceiling
  holds under concurrency.

The whole job: a hard stop **before** the next call, that **holds under fan-out** —
no account, no network, no crypto.

## How it works

The guard sits **in the call path**, not on an event bus. A passive listener is
told about spend *after the fact* and can't halt anything — so enforcement has to
be the thing standing in front of the next call:

- **`check()`** runs before each LLM call. It predicts the next call's cost from
  the last one and raises `BudgetExceeded` if that would cross your ceiling — the
  call never runs. (A running-total check also catches an overshoot if an estimate
  came in low.)
- **`record(model, prompt_tokens, completion_tokens)`** runs after each response.
  It prices the tokens **offline** from a bundled
  [LiteLLM cost map](src/floe_guard/cost_map.json) and adds the USD to a running
  total.

### Unpriceable models fail closed

If a model isn't in the cost map and you didn't supply a price, the guard **warns
loudly and refuses** (`UnpriceableModelError`) rather than silently treat it as
free — *you can't cap spend you can't measure.* Give it a price to enforce it:

```python
from floe_guard import BudgetGuard, ManualPrice

guard = BudgetGuard(
    limit_usd=5.00,
    price_overrides={"my-self-hosted-model": ManualPrice(1e-6, 2e-6)},  # USD/token
)
# or, set fail_closed=False to warn-and-skip for models you accept un-metered.
```

### What the bundled map prices

The vendored map deliberately covers **OpenAI, Anthropic, and a curated set of
Groq models** (the allowlist lives in
[`scripts/update-cost-map.mjs`](scripts/update-cost-map.mjs)) — not all of
LiteLLM's upstream list. Generic open-weights names (`qwen3-32b`,
`gpt-oss-120b`) are served by many vendors at very different prices, so
resolving them at one vendor's rate would under-meter a spend guard; they stay
unpriceable unless you scope them (`groq/…`) or pass a manual price.

Model ids resolve flexibly: provider-prefixed forms work
(`openai/gpt-4o`, `groq/qwen/qwen3-32b` and the ChatGroq `qwen/qwen3-32b` both
hit the same entry), and a dated snapshot the map doesn't list yet
(`claude-opus-4-8-<date>`) prices at its alias entry instead of failing closed.
Everything else — Gemini, Mistral, Cohere, Ollama, Bedrock, self-hosted —
needs `price_overrides` (or `fail_closed=False` to accept it un-metered).

## Context-aware budgeting

The hard-stop is the guarantee; `advisory()` is the *upside*. Read it before a
step to let your agent **adapt** as it nears the cap — taper to a cheaper model,
shrink the task, or wrap up — instead of getting cut off mid-run.

```python
guard = BudgetGuard(limit_usd=0.10, near_limit_bps=7000)   # flag at 70% used

adv = guard.advisory()
# BudgetAdvisory(near_limit=False, used_bps=125, remaining_usd=0.0987, ...)
model = "gpt-4o-mini" if adv.near_limit else "gpt-4o"        # downshift near the cap

guard.check()                  # still the hard line — taper or not, this holds
response = call_your_llm(model)
guard.record(model, response.usage.prompt_tokens, response.usage.completion_tokens)
```

`advisory()` returns `near_limit`, `used_bps` (utilization in basis points),
`remaining_usd`, and the budget totals. It's a **soft** signal — the model may
ignore it; `check()` is what enforces the ceiling. See
[`examples/budget_aware.py`](examples/budget_aware.py) for a runnable taper demo
(no API key).

This is the **same advisory shape** hosted Floe returns on every proxied call
(the `X-Floe-Budget-Advisory` header), so the logic you write here ports
unchanged — hosted just answers across *every* vendor and cap with server-truth
balances and rolling-window reset timing, which a single local budget can't know.
The TS package exposes the identical `guard.advisory()`.

## Per-call spend log

The guard keeps a typed, in-memory ledger of everything it priced: each
`record()` / `settle()` appends one `SpendEvent`, and `record_tool()` lets paid
non-LLM calls (search APIs, scrapers) spend the same budget and land in the same
log. The events sum to `spent_usd` (unless a `max_log_events` ring buffer has
evicted old ones) — no more rebuilding per-call breakdowns around the guard.

```python
guard = BudgetGuard(limit_usd=1.00)                      # max_log_events=N caps memory
guard.record("gpt-4o", 1_200, 350, label="researcher")   # label is optional
guard.record_tool("serpapi.search", 0.01, label="researcher")

guard.spend_log      # [SpendEvent(timestamp=…, kind="llm", model_or_tool="gpt-4o",
                     #             prompt_tokens=1200, completion_tokens=350,
                     #             cost_usd=0.0065, label="researcher"), …]
print(guard.export_log(), end="")   # JSONL, one event per line
```

`export_log()` emits a stable snake_case schema —
`{timestamp, kind: llm|tool, model_or_tool, prompt_tokens, completion_tokens,
cost_usd, label?, reserved?}` — identical to the TS package's `exportLog()`, so
every agent produces the same shape regardless of stack and the streams can be
concatenated and analysed together.

## LatencyBudget — deadlines, the same way

Money isn't the only budget an agent burns. `LatencyBudget` is `BudgetGuard`'s
sibling for **time**: it tracks cumulative elapsed time across a tool chain
against an end-user SLA and stops the *next* call before it would blow it.

```python
from floe_guard import LatencyBudget, DeadlineExceeded

deadline = LatencyBudget(sla_ms=5000)          # the user is promised 5s

for step in plan:
    deadline.check(expected_ms=step.est_ms)    # raises DeadlineExceeded when projected over
    model = DEFAULT_MODEL
    if deadline.advisory().near_deadline:      # 80% consumed by default —
        model = FAST_FALLBACK                  # downshift BEFORE the wall
    run(step, model, timeout_ms=deadline.remaining_ms)
```

Same shape in TypeScript: `new LatencyBudget(5000)`, `check(expectedMs)`,
`remainingMs`, `advisory().nearDeadline`.

Honest scope, mirroring the rest of this package:

- **Monotonic clock** (`time.monotonic()` / `performance.now()`) — NTP steps
  and DST can't corrupt the budget.
- **Cooperative, not preemptive.** The guard supplies the deadline *signal*;
  killing an already-running stalled call is your framework's job (asyncio
  cancellation, `AbortSignal`). `check()` prevents the next call from starting.
- **Advisory symmetry.** `near_deadline` / `used_bps` / `remaining_ms` are the
  latency twin of the budget advisory's `near_limit` / `used_bps` /
  `remaining_usd` — taper logic written against one ports to the other.
- **In-process.** One instance per request/run; distributed/server-side latency
  tracking is out of scope.

## Request-sized estimates and mid-stream enforcement

Two gaps in last-cost prediction, closed in 0.4.0 (Python):

**The oversized first call.** `check()`/`reserve()` predict from the *last*
call — blind on call #1, wrong for a call much bigger than the previous one.
`estimate_call()` prices the **actual incoming request** so even a first call
that alone would cross the cap blocks pre-flight:

```python
est = guard.estimate_call("gpt-4o", prompt_tokens=12_000, max_completion_tokens=4_096)
handle = guard.reserve(est)   # raises BudgetExceeded NOW if this call can't fit
```

The LiteLLM adapter does this automatically (prompt tokens via
`litellm.token_counter`, output cap from `max_tokens`), and the LangChain
handler sizes its pre-call `check()` the same way. Unpriceable or unsized
requests fall back to the old last-cost prediction — the wiring only ever
tightens enforcement.

**The stream that runs long.** `record()` meters a *completed* response — too
late for a generation that starts cheap and keeps going. `guard_stream()` (or
the underlying `StreamGuard`) re-prices the call on every chunk and cuts the
stream off **mid-generation**, settling the tokens actually consumed instead of
recording a big overshoot after the fact:

```python
from floe_guard import guard_stream

for chunk in guard_stream(guard, "gpt-4o", stream, prompt_tokens=1_000):
    print(chunk, end="")   # raises BudgetExceeded mid-stream at the ceiling
```

Chunk sizes are estimated at ~4 chars/token (pass `count_tokens=` for a real
tokenizer); the final accrual reconciles to provider-reported usage via
`StreamGuard.finish(...)`. See
[`examples/streaming_guard.py`](examples/streaming_guard.py) for a runnable
demo (no API key).

## Framework adapters (optional extras)

### CrewAI

```bash
pip install floe-guard[crewai]
```

```python
from crewai import Agent, Crew
from floe_guard import BudgetGuard
from floe_guard.integrations.crewai import budget_guarded_llm

guard = BudgetGuard(limit_usd=1.00)
llm = budget_guarded_llm(guard, "gpt-4o")   # meters AND hard-stops
Crew(agents=[Agent(..., llm=llm)], tasks=[...]).kickoff()
```

CrewAI runs on LiteLLM, so one callback meters every agent and task under a
single budget. Use `budget_guarded_llm` (not just `guard_crew`) to get the hard
stop: LiteLLM can swallow exceptions raised inside its callbacks (verified on
litellm 1.91.x), so a callback alone may keep the crew running past a
violation. `budget_guarded_llm` also enforces in the LLM call path — where a
raise reliably reaches CrewAI — re-raising any violation the callback recorded
before the next call runs. `guard_crew(guard)` remains available for metering
existing crews; check the returned callback's `tripped` attribute (and the
`floe_guard` logger's ERROR output) if you use it alone. A recorded violation
latches for the life of the callback — after remediating (say, adding a price
override), call `callback.reset()` or build a fresh guard.

### LiteLLM

```bash
pip install floe-guard[litellm]
```

```python
from floe_guard import BudgetGuard
from floe_guard.integrations.litellm import guarded_completion

guard = BudgetGuard(limit_usd=1.00)
response = guarded_completion(guard, model="gpt-4o", messages=[...])
```

Prefer the LiteLLM-native callback? Register `budget_guard_callback(guard)` on
`litellm.callbacks` — but know its limit: LiteLLM runs callbacks inside
`except Exception`, so the callback's enforcement raise can be swallowed and
your loop keeps going. The callback records any violation on its `tripped`
attribute and logs it at ERROR level; consult `tripped` in your own loop, or
use `guarded_completion` (which enforces at the call site) for the guaranteed
stop. Wrapper enforcement is tested against litellm 1.91.x.

### LangChain

```bash
pip install floe-guard[langchain] langchain-openai   # langchain-openai only for the ChatOpenAI example below
```

```python
from langchain_openai import ChatOpenAI
from floe_guard import BudgetGuard
from floe_guard.integrations.langchain import budget_guard_callback_handler

guard = BudgetGuard(limit_usd=1.00)
llm = ChatOpenAI(model="gpt-4o", callbacks=[budget_guard_callback_handler(guard)])
llm.invoke("hello")            # checks budget before the call, records spend after
```

The handler checks the budget on LLM start (raising `BudgetExceeded` aborts the
call before it runs) and records token usage on LLM end.

### OpenAI

```bash
pip install floe-guard[openai]
```

```python
from openai import OpenAI
from floe_guard import BudgetGuard
from floe_guard.integrations.openai import guarded_completion

guard = BudgetGuard(limit_usd=1.00)
client = OpenAI()
response = guarded_completion(guard, client, model="gpt-4o", messages=[...])
```

`guarded_completion` reserves the budget before the call (raising
`BudgetExceeded` so a blocked call never reaches OpenAI) and records spend after.
Use `guarded_acompletion` with an `AsyncOpenAI` client for async.

### Anthropic

```bash
pip install floe-guard[anthropic]
```

```python
from anthropic import Anthropic
from floe_guard import BudgetGuard
from floe_guard.integrations.anthropic import guarded_completion

guard = BudgetGuard(limit_usd=1.00)
client = Anthropic()
response = guarded_completion(guard, client, model="claude-3-7-sonnet-20250219", max_tokens=1024, messages=[...])
```

Same reserve-before / record-after contract as the OpenAI adapter; Anthropic's
`input_tokens` / `output_tokens` are mapped onto the guard's prompt/completion
pricing. Use `guarded_acompletion` with an `AsyncAnthropic` client for async.

### Vercel AI SDK

The Vercel AI SDK is TypeScript-only, so it ships as a separate npm package that
lives in [`js/`](js/). It works with both **AI SDK v4 and v5**.

```bash
npm i floe-guard ai @ai-sdk/openai
```

```ts
import { wrapLanguageModel } from "ai";
import { openai } from "@ai-sdk/openai";
import { BudgetGuard, budgetGuardMiddleware } from "floe-guard";

const guard = new BudgetGuard(5.0);                   // your ceiling, in USD
const model = wrapLanguageModel({
  model: openai("gpt-4o"),
  middleware: budgetGuardMiddleware(guard),           // throws before crossing
});
```

The middleware `check()`s before each call (throwing `BudgetExceeded` to halt the
run) and `record()`s priced usage after — same semantics as the Python guard. See
[`js/README.md`](js/README.md).

## Honest about what this is

floe-guard is a **local, estimate-based** guardrail. It prices tokens from a
vendored cost map *inside your process*:

- The cost map can drift as vendors change prices — refresh it like any snapshot.
- It only sees the vendors you instrument.
- A determined agent or a bug could route around an in-process check.
- Under heavy or cold-start concurrency it bounds steady-state spend, not the
  first parallel wave. Reservations default to the last call's cost (`0` until
  the first `record()`) — size them to the real request with `estimate_call()`
  (the LiteLLM adapter does this for you), or use hosted Floe for a hard cap
  under arbitrary concurrency.
- Mid-stream enforcement (`guard_stream`) prices chunks by a ~4 chars/token
  heuristic unless you supply a tokenizer, so the cut-off point is approximate;
  the final accrual reconciles to provider-reported usage.

It's genuinely useful on its own, and it's honest about its limits. No inflated
metrics, no "zero defaults" claims — it's a free local stop, not a vault.

## No telemetry

floe-guard does **not** phone home. It sends no usage events, no install pings,
no identifiers — nothing leaves your process at runtime except hosted-budget
reads you explicitly opt into by setting `FLOE_API_KEY` (the
[hosted Floe](#upgrade-to-hosted-floe) path) — never otherwise.

This is a choice, not an oversight. A guardrail's whole value is trust: a
library that silently exfiltrates usage from people's agents is the opposite of
a tool you hand a budget to.

## Upgrade to hosted Floe

When you need the ceiling to be **un-bypassable** and **cross-vendor**, hosted
Floe moves enforcement server-side against a real credit line:

- **Un-bypassable** — enforced at the spend rail, not in your process.
- **Cross-vendor** — one budget over LLM tokens *and* paid (x402) tool calls.
- **Team budgets + analytics** — shared ceilings, per-agent isolation, spend history.

Set `FLOE_API_KEY` (your agent key, `floe_<hex>`) and floe-guard can read your
agent's **server-side remaining budget** from the live Floe endpoint:

```python
from floe_guard import hosted_enforcement_available, hosted_remaining_usd

if hosted_enforcement_available():       # True when FLOE_API_KEY is set
    remaining = hosted_remaining_usd()   # USD left, read from Floe's server
```

`hosted_remaining_usd()` GETs `/v1/agents/credit-remaining` and returns the USD
remaining — the minimum of your auto-borrow headroom and your session spend
remaining. It raises `HostedEnforcementError` on a bad/missing key (401), a
closed or suspended agent (403), an unprovisioned agent (404), or a network
failure.

Env vars:

- `FLOE_API_KEY` — your agent key. Required for the read.
- `FLOE_API_BASE_URL` — override the API host (defaults to
  `https://credit-api.floelabs.xyz`).

Honest scope: this call only **reads** the remaining budget. The un-bypassable,
cross-vendor *enforcement* is the hosted Floe product running server-side — not
this client. Use the number to inform a local ceiling; the server stays the
source of truth.

→ **[dev-dashboard.floelabs.xyz](https://dev-dashboard.floelabs.xyz/?utm_source=floe-guard&utm_medium=readme&utm_campaign=oss)** ·
**[floelabs.xyz](https://floelabs.xyz/?utm_source=floe-guard&utm_medium=readme&utm_campaign=oss)**

Want runnable end-to-end agents on hosted Floe (Vapi voice agents, metered LLM
calls, CrewAI, MCP)? See the
**[Floe Cookbook](https://github.com/Floe-Labs/floe-cookbook)**.

## Built with floe-guard

Using floe-guard in your project? Add the badge so others find it:

[![guarded by floe-guard](https://img.shields.io/badge/guarded%20by-floe--guard-2f81f7.svg)](https://github.com/Floe-Labs/floe-guard)

```markdown
[![guarded by floe-guard](https://img.shields.io/badge/guarded%20by-floe--guard-2f81f7.svg)](https://github.com/Floe-Labs/floe-guard)
```

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

For the TypeScript package, see [`js/README.md`](js/README.md). Contributions
are welcome — start with [CONTRIBUTING.md](CONTRIBUTING.md); releases are
tracked in [CHANGELOG.md](CHANGELOG.md).

## License

MIT — see [LICENSE](LICENSE).
