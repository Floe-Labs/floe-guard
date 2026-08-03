"""Budget-aware retrieval depth — RAG top_k shrinks as the budget drains.

No API key, no account, no network. Model choice stays FIXED throughout: the
adaptation lever here is how many retrieved chunks get stuffed into the prompt.
Every chunk costs prompt tokens, so retrieval depth is often a bigger spend
lever than the model picker. Once ``advisory().near_limit`` trips, the agent
drops from 20 chunks to 5 and keeps answering questions instead of slamming
into the hard-stop mid-run.

The advisory is a *soft* signal you choose to act on; ``check()`` is still the
hard guarantee.

Run:  python examples/budget_aware_retrieval.py
"""

from __future__ import annotations

from floe_guard import BudgetExceeded, BudgetGuard

MODEL = "gpt-4o"  # never changes — the lever is top_k, not the model
FULL_K = 20  # chunks per query on a healthy budget
NEAR_K = 5  # chunks per query once near_limit trips
TOKENS_PER_CHUNK = 150  # each retrieved chunk adds prompt tokens
COMPLETION_TOKENS = 300


def stub_retriever(query_id: int, top_k: int) -> list[str]:
    """A fake vector store — returns top_k 'chunks' for the query."""
    return [f"chunk-{query_id}-{i}" for i in range(top_k)]


def stub_llm(chunks: list[str]) -> dict[str, int]:
    """A fake RAG call — prompt size scales with retrieval depth."""
    return {
        "prompt_tokens": 200 + TOKENS_PER_CHUNK * len(chunks),
        "completion_tokens": COMPLETION_TOKENS,
    }


def main() -> None:
    # Taper at 70% used so there's room to shrink retrieval before the ceiling.
    guard = BudgetGuard(limit_usd=0.30, near_limit_bps=7000)
    print(f"Budget ${guard.limit_usd:.2f} · taper at {guard.near_limit_bps / 100:.0f}% used\n")

    query = 0
    tapered = False
    while True:
        query += 1
        adv = guard.advisory()
        top_k = NEAR_K if adv.near_limit else FULL_K
        if adv.near_limit and not tapered:
            tapered = True
            print(
                f"  [advisory] {adv.used_bps / 100:.0f}% used, "
                f"${adv.remaining_usd:.4f} left → top_k {FULL_K} → {NEAR_K}\n"
            )

        try:
            guard.check()  # the hard guarantee — deep or shallow, this holds the line
        except BudgetExceeded:
            print(
                f"\nStopped at query {query}. "
                f"Final spend ${guard.spent_usd:.4f} against the ${guard.limit_usd:.2f} ceiling."
            )
            break

        chunks = stub_retriever(query, top_k)
        usage = stub_llm(chunks)
        cost = guard.record(MODEL, usage["prompt_tokens"], usage["completion_tokens"])
        print(
            f"  query {query:>2}: top_k={top_k:<2}  {usage['prompt_tokens']:>4} prompt tok  "
            f"+${cost:.4f}  (total ${guard.spent_usd:.4f})"
        )


if __name__ == "__main__":
    main()
