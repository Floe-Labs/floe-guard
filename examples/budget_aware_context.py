"""Budget-aware context size — history window and max_tokens shrink near the cap.

No API key, no account, no network. Model choice stays FIXED: the levers here
are how much conversation history rides along on every turn and how long the
reply is allowed to be. Chat agents resend history each turn, so context is a
compounding cost — the longest turns arrive exactly when the budget is lowest.
Once ``advisory().near_limit`` trips, the agent truncates history to the last
few turns and caps the completion, trading verbosity for staying alive.

The advisory is a *soft* signal you choose to act on; ``check()`` is still the
hard guarantee.

Run:  python examples/budget_aware_context.py
"""

from __future__ import annotations

from floe_guard import BudgetExceeded, BudgetGuard

MODEL = "gpt-4o"  # never changes — the levers are history window and max_tokens
TOKENS_PER_TURN = 400  # each kept history turn adds prompt tokens
FULL_WINDOW = 12  # turns of history on a healthy budget
NEAR_WINDOW = 3  # turns of history once near_limit trips
FULL_MAX_TOKENS = 800
NEAR_MAX_TOKENS = 200


def stub_llm(history_turns: int, max_tokens: int) -> dict[str, int]:
    """A fake chat call — prompt scales with history, completion hits max_tokens."""
    return {
        "prompt_tokens": 300 + TOKENS_PER_TURN * history_turns,
        "completion_tokens": max_tokens,
    }


def main() -> None:
    # Taper at 70% used so there's room to truncate before the ceiling.
    guard = BudgetGuard(limit_usd=0.40, near_limit_bps=7000)
    print(f"Budget ${guard.limit_usd:.2f} · taper at {guard.near_limit_bps / 100:.0f}% used\n")

    history: list[int] = []  # one entry per past turn
    turn = 0
    tapered = False
    while True:
        turn += 1
        adv = guard.advisory()
        window = NEAR_WINDOW if adv.near_limit else FULL_WINDOW
        max_tokens = NEAR_MAX_TOKENS if adv.near_limit else FULL_MAX_TOKENS
        if adv.near_limit and not tapered:
            tapered = True
            print(
                f"  [advisory] {adv.used_bps / 100:.0f}% used, ${adv.remaining_usd:.4f} left "
                f"→ history {FULL_WINDOW} → {NEAR_WINDOW} turns, "
                f"max_tokens {FULL_MAX_TOKENS} → {NEAR_MAX_TOKENS}\n"
            )

        try:
            guard.check()  # the hard guarantee — long or short context, this holds the line
        except BudgetExceeded:
            print(
                f"\nStopped at turn {turn}. "
                f"Final spend ${guard.spent_usd:.4f} against the ${guard.limit_usd:.2f} ceiling."
            )
            break

        kept = min(len(history), window)
        usage = stub_llm(kept, max_tokens)
        cost = guard.record(MODEL, usage["prompt_tokens"], usage["completion_tokens"])
        history.append(turn)
        print(
            f"  turn {turn:>2}: history={kept:>2}/{len(history) - 1:<2} max_tokens={max_tokens:<3} "
            f"+${cost:.4f}  (total ${guard.spent_usd:.4f})"
        )


if __name__ == "__main__":
    main()
