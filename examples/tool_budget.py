"""Tool-spend kill-switch demo — runs with NO API key and NO account.

Some agents spend more per run on paid tool calls (Apollo lookups, Exa
searches, scraping APIs) than on LLM tokens. Those costs never touch a token
cost map — the caller knows the price — but they burn the same real dollars.
This demo shows tool spend as a first-class citizen of the SAME ceiling:

1. ``reserve_tool``/``settle_tool`` — the pre-call hard-stop. A tool's price is
   known BEFORE the call, so enforcement is exact: the crossing call never runs.
2. ``check()`` + ``record_tool`` — the sequential loop contract. A runaway
   tool loop dies at the ceiling, exactly like a runaway LLM loop.
3. ``tool_costs`` — per-tool attribution, so you can see where the money went.

Run it::

    python examples/tool_budget.py

The "tools" are stubs — no network, no keys, no real spend.
"""

from __future__ import annotations

from floe_guard import BudgetExceeded, BudgetGuard

APOLLO_COST = 0.02  # $ per people-lookup — known up front
EXA_COST = 0.004  # $ per search


def stub_apollo_lookup(company: str) -> dict[str, str]:
    return {"company": company, "contact": "jane@..."}


def stub_exa_search(query: str) -> list[str]:
    return [f"result for {query!r}"]


def main() -> None:
    guard = BudgetGuard(limit_usd=0.10)
    print(f"Budget: ${guard.limit_usd:.2f} — shared by tokens AND tools\n")

    # ── 1. pre-call hard-stop: reserve the KNOWN price before the call ─────────
    print("Prospecting until the budget says stop...")
    companies = 0
    try:
        while True:
            handle = guard.reserve_tool(APOLLO_COST)  # raises BEFORE the call
            stub_apollo_lookup(f"company-{companies}")
            guard.settle_tool("apollo.people_lookup", APOLLO_COST, reserved=handle)
            for _ in range(2):  # a couple of searches per company
                guard.check()  # sequential contract works for tools too
                stub_exa_search("intent signals")
                guard.record_tool("exa.search", EXA_COST)
            companies += 1
    except BudgetExceeded:
        print(f"  stopped after {companies} companies — the crossing call never ran.\n")

    # ── 2. attribution: where did the money go? ────────────────────────────────
    print(f"spent ${guard.spent_usd:.4f} of ${guard.limit_usd:.2f}, by tool:")
    for tool, total in sorted(guard.tool_costs.items()):
        print(f"  {tool:<22} ${total:.4f}")


if __name__ == "__main__":
    main()
