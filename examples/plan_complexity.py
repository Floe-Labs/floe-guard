"""Plan complexity adapts to budget — optional sub-tasks drop, the model doesn't.

No API key, no account, no network. A stub sub-task call returns the token usage
a real provider would report; the agent works through the same plan every round
and reads ``BudgetGuard.advisory()`` before choosing how ambitious that round is:
fewer reasoning steps first, then the optional sub-tasks stop running at all.

The model is a constant here on purpose. Dropping work the task does not require
protects the sub-tasks that it does require, so the agent finishes its mandatory
plan on budget instead of being cut off partway through a lavish one.

The advisory is a *soft* signal you choose to act on; ``check()`` is still the
hard guarantee. (Hosted Floe returns the same advisory shape, but across every
vendor/cap with server-truth balances — your taper logic ports unchanged.)

Run:  python examples/plan_complexity.py
"""

from __future__ import annotations

from floe_guard import BudgetAdvisory, BudgetExceeded, BudgetGuard

MODEL = "gpt-4o"  # fixed for the whole run: only the plan adapts

# (sub-task, required, reasoning steps at full complexity)
PLAN = (
    ("gather sources", True, 3),
    ("cross-check claims", False, 3),
    ("draft answer", True, 2),
    ("polish tone", False, 2),
)

TAPER_BPS = 4000  # thin the reasoning at 40% used, before near_limit trips
PROMPT_TOKENS_PER_SUBTASK = 600
COMPLETION_TOKENS_PER_STEP = 400


def plan_round(adv: BudgetAdvisory) -> tuple[bool, bool]:
    """The whole adaptation: how much of the plan this round can afford.

    Returns ``(run_optional, full_reasoning)``. Reasoning depth is cut first
    because it degrades quality gradually; the optional sub-tasks are dropped
    only once ``near_limit`` trips, to protect the required ones.
    """
    if adv.near_limit:
        return False, False
    if adv.used_bps >= TAPER_BPS:
        return True, False
    return True, True


def reasoning_steps_for(steps: int, full_reasoning: bool) -> int:
    """Calculate the number of reasoning steps to execute."""
    # One pass fewer, never zero — a sub-task that runs still has to produce an answer.
    return steps if full_reasoning else max(1, steps - 1)


def stub_subtask_call(reasoning_steps: int) -> dict[str, object]:
    """A fake sub-task call — no network, no key."""
    return {
        "model": MODEL,
        "prompt_tokens": PROMPT_TOKENS_PER_SUBTASK,
        "completion_tokens": reasoning_steps * COMPLETION_TOKENS_PER_STEP,
    }


def main() -> None:
    """Run the plan complexity adaptation example."""
    # Reasoning thins at 40% used (TAPER_BPS), optional tasks drop at 70% (near_limit).
    guard = BudgetGuard(limit_usd=0.10, near_limit_bps=7000)
    required = sum(1 for _, is_required, _ in PLAN if is_required)
    print(
        f"Budget ${guard.limit_usd:.2f} · model fixed at {MODEL} · plan complexity adapts",
        flush=True,
    )
    print(
        f"  {len(PLAN)} sub-tasks ({required} required) · thinner reasoning at "
        f"{TAPER_BPS / 100:.0f}% used · optional dropped at "
        f"{guard.near_limit_bps / 100:.0f}%\n",
        flush=True,
    )

    round_no = 0
    completed_required = 0
    stopped = False
    while not stopped:
        round_no += 1
        adv = guard.advisory()
        run_optional, full_reasoning = plan_round(adv)
        planned = len(PLAN) if run_optional else required
        print(
            f"  round {round_no}: {planned} of {len(PLAN)} sub-tasks · "
            f"{'full' if full_reasoning else 'reduced'} reasoning  "
            f"[{adv.used_bps / 100:.0f}% used, ${adv.remaining_usd:.4f} left]",
            flush=True,
        )

        for name, is_required, steps in PLAN:
            if not is_required and not run_optional:
                print(f"    {name:<19} [skipped — optional]", flush=True)
                continue

            reasoning_steps = reasoning_steps_for(steps, full_reasoning)
            est = guard.estimate_call(
                MODEL, PROMPT_TOKENS_PER_SUBTASK, reasoning_steps * COMPLETION_TOKENS_PER_STEP
            )
            try:
                guard.check(est)  # the hard guarantee — simplified or not, this holds the line
            except BudgetExceeded:
                print(f"    {name:<19} [blocked — would cross the ceiling]", flush=True)
                stopped = True
                break

            response = stub_subtask_call(reasoning_steps)
            cost = guard.record(
                str(response["model"]),
                int(response["prompt_tokens"]),  # type: ignore[arg-type]
                int(response["completion_tokens"]),  # type: ignore[arg-type]
            )
            if is_required:
                completed_required += 1
            plural = "s" if reasoning_steps > 1 else ""
            steps_label = f"{reasoning_steps} reasoning step{plural}"
            print(
                f"    {name:<19} {steps_label:<18} +${cost:.4f}  (total ${guard.spent_usd:.4f})",
                flush=True,
            )

    print(
        f"\nStopped in round {round_no} after {completed_required} required sub-tasks. "
        f"Final spend ${guard.spent_usd:.4f} (held under ${guard.limit_usd:.2f}).",
        flush=True,
    )


if __name__ == "__main__":
    main()
