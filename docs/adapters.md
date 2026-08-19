# Framework adapters

Optional-extra adapters for floe-guard. Each follows the same reserve-before / record-after contract as the [OpenAI adapter in the README](../README.md#openai) — install the extra, wrap your call, done.

## CrewAI

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

## LiteLLM

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

## LangChain

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

## LangGraph

```bash
pip install floe-guard[langgraph]
```

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict

from floe_guard import BudgetGuard
from floe_guard.integrations.langgraph import AdvisoryChannel, guarded_node

class State(TypedDict):
    results: Annotated[list, operator.add]
    budget: AdvisoryChannel          # typed BudgetAdvisory, refreshed per call

guard = BudgetGuard(limit_usd=0.10)

@guarded_node(guard, estimated_cost=0.01)   # reserve() before, settle()/release() after
def worker(state: State) -> dict:
    response = my_llm_call(state)
    return {"results": [response["text"]], "usage": {
        "model": response["model"],
        "prompt_tokens": response["prompt_tokens"],
        "completion_tokens": response["completion_tokens"],
    }}
```

`guarded_node` gives every branch of a `StateGraph` fan-out its own atomic
slice of the ceiling (reserve-before / settle-after, the same contract the
OpenAI and Anthropic adapters use), so N parallel sub-agents can't race one
shared total. Pass `estimated_cost` to hold a conservative fixed slice on
every call of that node (the `0.01` above); a node that omits it estimates
from the guard's last settled cost instead, which is `0` on a fresh guard, so
seed a cold-start fan-out explicitly. After each settled call it writes the guard's `BudgetAdvisory`
into `state["budget"]`, so a router node can downshift to a cheaper model on
`near_limit` *before* the hard-stop — see
[`examples/langgraph_budget_aware.py`](../examples/langgraph_budget_aware.py) for
the full budget-aware router (no API key needed).

## Anthropic

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
See [`examples/anthropic_adapter.py`](../examples/anthropic_adapter.py) for a
runnable demo of the adapter's native prompt-cache pricing — a cached read
costs a fraction of a fresh one (no API key needed).

## Google Gemini

```bash
pip install 'floe-guard[gemini]'
```

```python
from google import genai
from floe_guard import BudgetGuard
from floe_guard.integrations.gemini import guarded_completion

guard = BudgetGuard(limit_usd=1.00)
client = genai.Client(api_key="...")
response = guarded_completion(guard, client, model="gemini-2.5-flash", contents="hello")
```

Same reserve-before / record-after contract as the OpenAI adapter. Gemini splits
usage across five counters and this adapter maps all of them: thinking tokens
(`thoughts_token_count`) and tool-result tokens (`tool_use_prompt_token_count`)
are billed but sit *outside* the obvious prompt/candidates pair, so omitting them
would under-meter; cached tokens are carved out of the prompt count (Gemini
includes them there) and re-priced at the cheaper cache-read rate rather than
charged twice. Use `guarded_acompletion` for async.

**Vertex AI callers must supply prices.** One SDK serves both Google AI Studio
and Vertex with *identical model ids*, but Vertex bills up to 50% more, and the
bundled map carries AI Studio rates — so metering a Vertex call against it would
under-meter. The model id can't reveal the backend, but the client can: the
adapter reads `client.vertexai` and fails closed unless you pass your own rates.

```python
from floe_guard import ManualPrice

guard = BudgetGuard(limit_usd=1.00, price_overrides={
    "gemini-2.5-flash": ManualPrice(3.0e-7, 2.5e-6),   # your Vertex rates
})
```

Streaming isn't wrapped — `generate_content_stream` only reports usage on its
final chunk (or never, if you stop early), so use
[`guard_stream()`](advanced.md#request-sized-estimates-and-mid-stream-enforcement) to meter
a stream chunk-by-chunk instead.

## Vercel AI SDK

The Vercel AI SDK is TypeScript-only, so it ships as a separate npm package that
lives in [`js/`](../js/). It works with both **AI SDK v4 and v5**.

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
[`js/README.md`](../js/README.md).
