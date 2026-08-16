"""Per-step token budgets for a sequential agent loop (issue #46).

Run with:

    python examples/step_budget.py

No API key, no network. A three-step research agent runs under a global token
ceiling, and each step also gets its own per-step token cap. When a step tries
to spend past its cap, the guard hard-stops that call with a
``TokenBudgetExceeded`` (scope="step") — even though the global budget still has
room — so one runaway step can't starve the rest of the loop.
"""

from __future__ import annotations

from floe_guard import BudgetGuard, TokenBudgetExceeded

MODEL = "gpt-4o"


def main() -> None:
    # Global ceiling: $100 (ample) and 20_000 tokens across the whole run.
    guard = BudgetGuard(limit_usd=100.0, token_limit=20_000, near_limit_bps=8000)

    # Step 1 — planning: a tight 2_000-token cap.
    with guard.step(max_tokens=2_000) as g:
        g.check(estimated_tokens=1_200)
        g.record(MODEL, 800, 400)  # 1_200 tokens into the step
        adv = g.advisory()
        print(
            f"step 1 (plan): used {adv.step_remaining_tokens} tokens of headroom left, "
            f"near_limit={adv.near_limit}",
            flush=True,
        )

    # Step 2 — retrieval: a 5_000-token cap, and a call that would blow it.
    try:
        with guard.step(max_tokens=5_000) as g:
            g.record(MODEL, 3_000, 1_500)  # 4_500 into the step
            g.check(estimated_tokens=1_000)  # 4_500 + 1_000 > 5_000 → blocked
            print("step 2: this line should not print", flush=True)
    except TokenBudgetExceeded as exc:
        print(
            f"step 2 (retrieve): hard-stopped at the step cap — {exc} (scope={exc.scope})",
            flush=True,
        )

    # The blocked call consumed no tokens, but Step 2's already-recorded 4,500
    # tokens still count against the global ceiling — the step cap stopped the
    # overshoot, not the run, so the loop keeps going.
    print(f"global tokens spent so far: {guard.spent_tokens} / {guard.token_limit}", flush=True)

    # Step 3 — synthesis: fits comfortably.
    with guard.step(max_tokens=3_000) as g:
        g.record(MODEL, 1_000, 800)
        print(
            f"step 3 (synthesize): done, {g.advisory().step_remaining_tokens} step tokens left",
            flush=True,
        )

    print(f"run complete — {guard.spent_tokens} total tokens across all steps", flush=True)


if __name__ == "__main__":
    main()
