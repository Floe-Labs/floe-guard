"""Budget-aware planning — optional sub-tasks drop off the plan near the cap.

No API key, no account, no network. Model choice stays FIXED: the lever here is
plan complexity. Each work item expands into sub-tasks (draft, critique,
polish), and every sub-task is a metered LLM call. On a healthy budget the
agent runs the full plan; once ``advisory().near_limit`` trips it keeps only
the core sub-task, finishing every remaining item at reduced depth instead of
finishing half of them at full depth.

The advisory is a *soft* signal you choose to act on; ``check()`` is still the
hard guarantee.

Run:  python examples/budget_aware_planning.py
"""

from __future__ import annotations

from floe_guard import BudgetExceeded, BudgetGuard

MODEL = "gpt-4o"  # never changes — the lever is how many sub-tasks each item gets
FULL_PLAN = ["draft", "critique", "polish"]  # core + optional refinement passes
NEAR_PLAN = ["draft"]  # core only, once near_limit trips


def stub_llm(subtask: str) -> dict[str, int]:
    """A fake sub-task call — refinement passes read more context than drafting."""
    prompt = {"draft": 600, "critique": 900, "polish": 900}[subtask]
    return {"prompt_tokens": prompt, "completion_tokens": 400}


def main() -> None:
    # Taper at 70% used so there's room to simplify the plan before the ceiling.
    guard = BudgetGuard(limit_usd=0.35, near_limit_bps=7000)
    print(f"Budget ${guard.limit_usd:.2f} · taper at {guard.near_limit_bps / 100:.0f}% used\n")

    item = 0
    tapered = False
    try:
        while True:
            item += 1
            adv = guard.advisory()
            plan = NEAR_PLAN if adv.near_limit else FULL_PLAN
            if adv.near_limit and not tapered:
                tapered = True
                print(
                    f"  [advisory] {adv.used_bps / 100:.0f}% used, "
                    f"${adv.remaining_usd:.4f} left → plan "
                    f"{'+'.join(FULL_PLAN)} → {'+'.join(NEAR_PLAN)}\n"
                )

            item_cost = 0.0
            for subtask in plan:
                guard.check()  # the hard guarantee — before every sub-task call
                usage = stub_llm(subtask)
                item_cost += guard.record(MODEL, usage["prompt_tokens"], usage["completion_tokens"])
            print(
                f"  item {item:>2}: {len(plan)} sub-task(s) [{', '.join(plan)}] "
                f"+${item_cost:.4f}  (total ${guard.spent_usd:.4f})"
            )
    except BudgetExceeded:
        print(
            f"\nStopped at item {item}. "
            f"Final spend ${guard.spent_usd:.4f} (held under ${guard.limit_usd:.2f})."
        )


if __name__ == "__main__":
    main()
