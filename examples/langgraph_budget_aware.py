"""Context-aware budgeting in a LangGraph graph — the router tapers near the cap.

No API key, no account, no network — a stub LLM returns fixed token usage. The
LangGraph port of ``examples/budget_aware.py``: every worker node is wrapped
with ``guarded_node`` (atomic reserve/settle, so this pattern survives a
parallel fan-out unchanged), and each settled call refreshes a typed
``BudgetAdvisory`` in the graph state. The router node reads
``state["budget"].near_limit`` and downshifts to the cheap model, so the run
finishes on budget instead of slamming into the hard-stop mid-task.

The advisory is a *soft* signal you choose to act on; ``reserve()`` is still
the hard guarantee on every guarded node.

Run:  pip install floe-guard[langgraph]
      python examples/langgraph_budget_aware.py
"""

from __future__ import annotations

import operator
from typing import Annotated

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from floe_guard import BudgetGuard
from floe_guard.integrations.langgraph import AdvisoryChannel, guarded_node

FULL = ("gpt-4o", 1000, 1000)  # ~$0.0125 / call
CHEAP = ("gpt-4o-mini", 1000, 1000)  # ~$0.0008 / call


class State(TypedDict):
    steps: Annotated[int, operator.add]
    log: Annotated[list, operator.add]
    budget: AdvisoryChannel


def stub_llm(model: tuple[str, int, int]) -> dict[str, object]:
    """A fake LLM call — no network, no key."""
    name, prompt_tokens, completion_tokens = model
    return {"model": name, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}


def make_worker(model: tuple[str, int, int]):
    def worker(state: State) -> dict:
        response = stub_llm(model)
        return {
            "steps": 1,
            "log": [f"{response['model']}"],
            # Report the call's usage; guarded_node settles it and refreshes
            # state["budget"] with the guard's advisory.
            "usage": response,
        }

    return worker


def route(state: State) -> str:
    """Downshift on the advisory; stop when not even a cheap call fits."""
    adv = state.get("budget")
    if adv is None:
        return "full_step"  # first call — no signal yet
    if adv.remaining_usd < 0.0008:
        return END
    if adv.near_limit:
        return "cheap_step"
    return "full_step"


def main() -> None:
    # Taper at 70% used so there's room to downshift before the ceiling.
    guard = BudgetGuard(limit_usd=0.10, near_limit_bps=7000)
    print(f"Budget ${guard.limit_usd:.2f} · taper at {guard.near_limit_bps / 100:.0f}% used\n")

    graph = StateGraph(State)
    graph.add_node("full_step", guarded_node(guard, make_worker(FULL), estimated_cost=0.0125))
    graph.add_node("cheap_step", guarded_node(guard, make_worker(CHEAP), estimated_cost=0.0008))
    graph.add_conditional_edges(START, route)
    graph.add_conditional_edges("full_step", route)
    graph.add_conditional_edges("cheap_step", route)

    tapered = False
    final = graph.compile().invoke({"steps": 0, "log": []}, {"recursion_limit": 200})

    for step, model in enumerate(final["log"], start=1):
        if model == CHEAP[0] and not tapered:
            tapered = True
            print("  [advisory] near_limit tripped → tapering to", model, "\n")
        print(f"  step {step:>2}: {model}")

    print(
        f"\nFinished at step {final['steps']}. "
        f"Final spend ${guard.spent_usd:.4f} (held under ${guard.limit_usd:.2f}), "
        f"advisory read {final['budget'].used_bps / 100:.0f}% used."
    )


if __name__ == "__main__":
    main()
