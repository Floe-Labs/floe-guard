"""Context-aware budgeting — the agent tapers as it nears the cap.

No API key, no account, no network. A stub LLM returns fixed token usage; before
each step the agent reads ``BudgetGuard.advisory()`` and, once ``near_limit``
trips, switches to a cheaper model so it keeps making progress and finishes on
budget instead of slamming into the hard-stop mid-task.

The advisory is a *soft* signal you choose to act on; ``check()`` is still the
hard guarantee. (Hosted Floe returns the same advisory shape, but across every
vendor/cap with server-truth balances — your taper logic ports unchanged.)

Run:  python examples/budget_aware.py
"""
from __future__ import annotations

from floe_guard import BudgetExceeded, BudgetGuard

FULL = ("gpt-4o", 1000, 1000)  # ~$0.0125 / call
CHEAP = ("gpt-4o-mini", 1000, 1000)  # ~$0.0008 / call


def stub_llm(model: tuple[str, int, int]) -> dict[str, object]:
    """A fake LLM call — no network, no key."""
    name, pt, ct = model
    return {"model": name, "prompt_tokens": pt, "completion_tokens": ct}


def main() -> None:
    # Taper at 70% used so there's room to downshift before the ceiling.
    guard = BudgetGuard(limit_usd=0.10, near_limit_bps=7000)
    print(f"Budget ${guard.limit_usd:.2f} · taper at {guard.near_limit_bps / 100:.0f}% used\n")

    step = 0
    tapered = False
    while True:
        step += 1
        adv = guard.advisory()
        # Context-aware: downshift to the cheap model once we're near the cap.
        model = CHEAP if adv.near_limit else FULL
        if adv.near_limit and not tapered:
            tapered = True
            print(
                f"  [advisory] {adv.used_bps / 100:.0f}% used, "
                f"${adv.remaining_usd:.4f} left → tapering to {model[0]}\n"
            )

        try:
            guard.check()  # the hard guarantee — taper or not, this holds the line
        except BudgetExceeded:
            print(
                f"\nStopped at step {step}. "
                f"Final spend ${guard.spent_usd:.4f} (held under ${guard.limit_usd:.2f})."
            )
            break

        response = stub_llm(model)
        cost = guard.record(
            str(response["model"]),
            int(response["prompt_tokens"]),  # type: ignore[arg-type]
            int(response["completion_tokens"]),  # type: ignore[arg-type]
        )
        print(f"  step {step:>2}: {model[0]:<12} +${cost:.4f}  (total ${guard.spent_usd:.4f})")


if __name__ == "__main__":
    main()
