# Advanced budgeting & limits

Reference for floe-guard's second-dimension budgets and the spend ledger. The dollar hard-stop and the taper/advisory story live in the [README](../README.md); this is the deeper mechanics.

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

## Tool spend under the same ceiling

Tool-heavy agents often spend more on paid APIs (Apollo lookups, Exa searches,
scrapers) than on tokens — and those dollars must count against the same cap,
or the kill-switch guarantee is fiction for them. Tool spend is a first-class
primitive with the full reserve/settle contract; it's actually **stronger**
than the LLM path, because the price is known *before* the call:

```python
# pre-call hard-stop — the crossing call NEVER runs
handle = guard.reserve_tool(0.02)              # raises BudgetExceeded before Apollo
try:
    result = apollo.people_lookup(...)
    guard.settle_tool("apollo.people_lookup", 0.02, reserved=handle)
except Exception:
    guard.release(handle)                      # free the reservation if the call fails
    raise

guard.record_tool("exa.search", 0.004)         # post-hoc, for metered APIs

guard.tool_costs     # {"apollo.people_lookup": 0.42, "exa.search": 0.11}
guard.remaining_usd  # tokens + tools, one ceiling
```

`record_tool` also updates the next-call estimate, so a plain
`check()`/`record_tool` loop stops *before* the crossing call — a runaway tool
loop dies exactly like a runaway LLM loop. (Tool and LLM estimates are tracked
separately; the default prediction is the costlier of the two, so a cheap tool
call never shrinks the hold ahead of an expensive LLM call.) The caller supplies the USD (there
is no tool cost-map); every tool call lands in `spend_log` as a
`kind: "tool"` event. Same API in TS (`reserveTool`/`settleTool`/`recordTool`/
`toolCosts`). See [`examples/tool_budget.py`](../examples/tool_budget.py).

## Token ceilings and per-step budgets

Dollars aren't the only runaway. A token ceiling caps *total recorded token
usage — every bucket the guard counts: prompt, completion, and cache* regardless
of price, and a **per-step** cap keeps one step of a sequential loop from
starving the rest even when the global budget has room.
Both ride on the same reserve/settle machinery — they're a second dimension, not
a second guard:

```python
from floe_guard import BudgetGuard, TokenBudgetExceeded

# aggregate token ceiling alongside the USD ceiling
guard = BudgetGuard(limit_usd=100.0, token_limit=20_000)

guard.check(estimated_tokens=1_200)      # raises TokenBudgetExceeded if it'd cross
guard.record("gpt-4o", 800, 400)         # tokens accrue for free from the counts

# a per-step cap for one step of a sequential loop
with guard.step(max_tokens=5_000) as g:  # g IS guard — adapters pass it through
    g.record("gpt-4o", 3_000, 1_500)
    g.check(estimated_tokens=1_000)       # 4_500 + 1_000 > 5_000 → scope="step"

adv = guard.advisory()
adv.token_used_bps        # aggregate token utilization (None if no token_limit)
adv.remaining_tokens      # tokens left before the ceiling (None if no token_limit)
adv.step_remaining_tokens # active step's headroom (None if no step, or its token cap is unset)
```

`TokenBudgetExceeded` subclasses `BudgetExceeded`, so budget-aware retry treats a
token block as terminal automatically. With no `token_limit` and no `step()`, USD
enforcement is unchanged and `reserve()` still returns a plain `float` — a
`BudgetReservation` handle appears only when tokens are actually reserved or a
step is active. (`advisory()` gains the token/step fields shown above; they're
additive and `None` when their dimension is unused.) In TS the step is a callback
and fields are camelCase:

```ts
const guard = new BudgetGuard(100, { tokenLimit: 20_000 });
guard.check(undefined, { estimatedTokens: 1_200 });
guard.step({ maxTokens: 5_000 }, (g) => {
  g.record("gpt-4o", 3_000, 1_500);
  g.check(undefined, { estimatedTokens: 1_000 }); // throws TokenBudgetExceeded
});
guard.advisory().stepRemainingTokens;
```

See [`examples/step_budget.py`](../examples/step_budget.py) (no network).

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
try:
    response = call_your_llm(model="gpt-4o", ...)
    guard.settle("gpt-4o", response.usage.prompt_tokens, response.usage.completion_tokens, reserved=handle)
except Exception:
    guard.release(handle)      # free the reservation if the call fails
    raise
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
from floe_guard import StreamGuard

# The context manager guarantees the reservation is settled or released:
handle = guard.reserve(guard.estimate_call("gpt-4o", prompt_tokens=1_000, max_completion_tokens=100))
with StreamGuard(guard, "gpt-4o", prompt_tokens=1_000, reserved=handle) as sg:
    for chunk in stream:
        sg.feed_text(chunk.text)                       # raises BudgetExceeded mid-stream
        consume(chunk)
    sg.finish(completion_tokens=reported_usage_tokens) # reconcile to real usage
```

> [!NOTE]
> A generator wrapper `guard_stream(guard, ...)` is also available for automated reservation settling/releasing on stream termination, but does not support post-hoc usage reconciliation via `finish()`.

Chunk sizes are estimated at ~4 chars/token (pass `count_tokens=` for a real
tokenizer). See [`examples/streaming_guard.py`](../examples/streaming_guard.py)
for a runnable demo (no API key).
