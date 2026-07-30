"""Context size adapts to budget — history and ``max_tokens`` shrink, the model doesn't.

No API key, no account, no network. A stub chat call returns the token usage a
real provider would report; the conversation history grows every turn, and once
``BudgetGuard.advisory()`` reports ``near_limit`` the agent stops resending the
whole transcript and caps its replies shorter.

The model is a constant here on purpose. A long-running conversation gets more
expensive per turn even on a fixed model, because the prompt carries every turn
before it — so trimming the context is the lever, not downgrading the model.

The advisory is a *soft* signal you choose to act on; ``check()`` is still the
hard guarantee. (Hosted Floe returns the same advisory shape, but across every
vendor/cap with server-truth balances — your taper logic ports unchanged.)

Run:  python examples/context_size.py
"""

from __future__ import annotations

from floe_guard import BudgetAdvisory, BudgetExceeded, BudgetGuard

MODEL = "gpt-4o"  # fixed for the whole run: only the context size adapts

SYSTEM_PROMPT_TOKENS = 400
TOKENS_PER_TURN = 400
MAX_TOKENS_FULL = 800  # full replies while there's headroom
MAX_TOKENS_TAPER = 250  # terser replies near the cap
KEEP_RECENT_TURNS = 2  # how much transcript survives the trim


def plan_context(adv: BudgetAdvisory, turns_held: int) -> tuple[int, int]:
    """The whole adaptation: how much history to resend, and how long a reply.

    Returns ``(turns_to_send, max_tokens)``. Near the cap the agent keeps only
    the most recent turns instead of the full transcript.
    """
    if adv.near_limit:
        return min(turns_held, KEEP_RECENT_TURNS), MAX_TOKENS_TAPER
    return turns_held, MAX_TOKENS_FULL


def prompt_tokens_for(turns_sent: int) -> int:
    """The resent transcript is the part of the prompt the agent controls."""
    return SYSTEM_PROMPT_TOKENS + turns_sent * TOKENS_PER_TURN


def stub_chat_call(turns_sent: int, max_tokens: int) -> dict[str, object]:
    """A fake chat completion — no network, no key."""
    return {
        "model": MODEL,
        "prompt_tokens": prompt_tokens_for(turns_sent),
        "completion_tokens": max_tokens,
    }


def main() -> None:
    # Taper at 70% used so there's room to trim before the ceiling.
    guard = BudgetGuard(limit_usd=0.10, near_limit_bps=7000)
    print(f"Budget ${guard.limit_usd:.2f} · model fixed at {MODEL} · context size adapts")
    print(
        f"  full transcript + max_tokens {MAX_TOKENS_FULL} → last {KEEP_RECENT_TURNS} turns "
        f"+ max_tokens {MAX_TOKENS_TAPER} at {guard.near_limit_bps / 100:.0f}% used\n"
    )

    history: list[str] = []
    turn = 0
    trimmed = False
    while True:
        turn += 1
        history.append(f"turn {turn}")
        adv = guard.advisory()
        turns_sent, max_tokens = plan_context(adv, len(history))
        if turns_sent < len(history) and not trimmed:
            trimmed = True
            print(
                f"  [advisory] {adv.used_bps / 100:.0f}% used, "
                f"${adv.remaining_usd:.4f} left → trimming to the last "
                f"{turns_sent} turns, max_tokens {max_tokens}\n"
            )

        # Request-sized: the prompt grows with the transcript, so the last call's
        # cost under-predicts the next one. estimate_call() prices THIS request.
        est = guard.estimate_call(MODEL, prompt_tokens_for(turns_sent), max_tokens)
        try:
            guard.check(est)  # the hard guarantee — trimmed or not, this holds the line
        except BudgetExceeded:
            print(
                f"\nStopped at turn {turn}. "
                f"Final spend ${guard.spent_usd:.4f} (held under ${guard.limit_usd:.2f})."
            )
            break

        response = stub_chat_call(turns_sent, max_tokens)
        cost = guard.record(
            str(response["model"]),
            int(response["prompt_tokens"]),  # type: ignore[arg-type]
            int(response["completion_tokens"]),  # type: ignore[arg-type]
        )
        print(
            f"  turn {turn:>2}: {turns_sent:>2} of {len(history):>2} turns sent · "
            f"max_tokens {max_tokens:>3}  +${cost:.4f}  (total ${guard.spent_usd:.4f})"
        )


if __name__ == "__main__":
    main()
