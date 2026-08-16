"""Retrieval depth adapts to budget — ``top_k`` shrinks, the model never changes.

No API key, no account, no network. A stub retrieve-then-generate step returns
the token usage a real provider would report; before each step the agent reads
``BudgetGuard.advisory()`` and pulls fewer chunks into the context window as the
remaining budget drains.

The model is a constant here on purpose. Model choice is only one adaptation
axis — in a RAG agent most of the bill is prompt tokens the retriever decided to
send, so ``top_k`` is the cheaper lever to pull first.

The advisory is a *soft* signal you choose to act on; ``check()`` is still the
hard guarantee. (Hosted Floe returns the same advisory shape, but across every
vendor/cap with server-truth balances — your taper logic ports unchanged.)

Run:  python examples/retrieval_depth.py
"""

from __future__ import annotations

from floe_guard import BudgetAdvisory, BudgetExceeded, BudgetGuard

MODEL = "gpt-4o"  # fixed for the whole run: only the retrieval depth adapts

TOP_K_FULL = 20  # ~$0.0130 / step
TOP_K_TAPER = 12  # ~$0.0098 / step
TOP_K_FLOOR = 5  # ~$0.0070 / step

TAPER_BPS = 4000  # first downshift at 40% used, before near_limit trips
BASE_PROMPT_TOKENS = 400  # system prompt + question, independent of top_k
TOKENS_PER_CHUNK = 160
MAX_COMPLETION_TOKENS = 400


def prompt_tokens_for(top_k: int) -> int:
    """Retrieved chunks are the part of the prompt the agent controls."""
    return BASE_PROMPT_TOKENS + top_k * TOKENS_PER_CHUNK


def stub_rag_call(top_k: int) -> dict[str, object]:
    """A fake retrieve-then-generate call — no network, no key."""
    return {
        "model": MODEL,
        "prompt_tokens": prompt_tokens_for(top_k),
        "completion_tokens": MAX_COMPLETION_TOKENS,
    }


def choose_top_k(adv: BudgetAdvisory) -> int:
    """The whole adaptation: retrieve shallower as utilization climbs."""
    if adv.near_limit:
        return TOP_K_FLOOR
    if adv.used_bps >= TAPER_BPS:
        return TOP_K_TAPER
    return TOP_K_FULL


def main() -> None:
    """Run the retrieval depth adaptation example."""
    # First downshift at 40% used (TAPER_BPS), final floor at 70% (near_limit).
    guard = BudgetGuard(limit_usd=0.10, near_limit_bps=7000)
    print(
        f"Budget ${guard.limit_usd:.2f} · model fixed at {MODEL} · retrieval depth adapts",
        flush=True,
    )
    print(
        f"  {TOP_K_FULL} chunks -> {TOP_K_TAPER} at {TAPER_BPS / 100:.0f}% used "
        f"-> {TOP_K_FLOOR} at {guard.near_limit_bps / 100:.0f}%\n",
        flush=True,
    )

    step = 0
    depth = TOP_K_FULL
    while True:
        step += 1
        adv = guard.advisory()
        top_k = choose_top_k(adv)
        if top_k != depth:
            print(
                f"  [advisory] {adv.used_bps / 100:.0f}% used, "
                f"${adv.remaining_usd:.4f} left -> retrieving {top_k} chunks, not {depth}\n",
                flush=True,
            )
            depth = top_k

        # Request-sized: the prompt changes with top_k every step, so the last
        # call's cost is the wrong prediction. estimate_call() prices THIS one.
        est = guard.estimate_call(MODEL, prompt_tokens_for(top_k), MAX_COMPLETION_TOKENS)
        try:
            guard.check(est)  # the hard guarantee — taper or not, this holds the line
        except BudgetExceeded:
            print(
                f"\nStopped at step {step}. "
                f"Final spend ${guard.spent_usd:.4f} (held under ${guard.limit_usd:.2f}).",
                flush=True,
            )
            break

        response = stub_rag_call(top_k)
        cost = guard.record(
            str(response["model"]),
            int(response["prompt_tokens"]),  # type: ignore[arg-type]
            int(response["completion_tokens"]),  # type: ignore[arg-type]
        )
        print(
            f"  step {step:>2}: {MODEL:<8} {top_k:>2} chunks  "
            f"+${cost:.4f}  (total ${guard.spent_usd:.4f})",
            flush=True,
        )


if __name__ == "__main__":
    main()
