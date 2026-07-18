"""Mid-stream kill-switch demo — runs with NO API key and NO account.

Two enforcement gaps this demo closes (both new in 0.4.0):

1. **The oversized first call.** ``check()``/``reserve()`` predict the next
   call from the LAST one — on call #1 there is no baseline, so a huge request
   sails through and the overshoot is only discovered after the money is spent.
   ``estimate_call()`` prices the ACTUAL request (real prompt size + output
   cap) so the very first call blocks pre-flight if it alone would cross.

2. **The stream that runs long.** ``record()`` meters a COMPLETED response —
   too late to stop a generation that starts cheap and keeps going.
   ``guard_stream()`` re-prices the call on every chunk and cuts the stream
   off mid-generation, settling only the tokens actually consumed.

Run it::

    python examples/streaming_guard.py

The "LLM" is a stub generator — no network, no key. Costs are computed offline
from the bundled cost map, exactly as for a real ``gpt-4o`` call.
"""

from __future__ import annotations

from collections.abc import Iterator

from floe_guard import BudgetExceeded, BudgetGuard, guard_stream

MODEL = "gpt-4o"


def stub_llm_stream() -> Iterator[str]:
    """A fake streaming LLM that never stops talking."""
    while True:
        yield "and another thing... "  # ~5 tokens per chunk by the heuristic


def main() -> None:
    guard = BudgetGuard(limit_usd=0.01)  # ≙ 1_000 gpt-4o output tokens

    # ── 1. the oversized FIRST call is blocked pre-flight ──────────────────────
    print(f"Budget: ${guard.limit_usd:.2f}\n")
    print("1) First call asks for 100k output tokens (≈ $1.00):")
    estimate = guard.estimate_call(MODEL, prompt_tokens=1_000, max_completion_tokens=100_000)
    try:
        guard.reserve(estimate)  # request-sized, so call #1 has no free pass
    except BudgetExceeded:
        print("   blocked BEFORE the request was sent — $0.00 spent.\n")

    # ── 2. a stream that starts cheap but runs long is cut off mid-flight ──────
    print("2) Streaming a response that never wants to stop:")
    chunks = 0
    try:
        for _ in guard_stream(guard, MODEL, stub_llm_stream()):
            chunks += 1
    except BudgetExceeded:
        print(f"   stream cut off mid-generation after {chunks} chunks.")
        print(f"   spent ${guard.spent_usd:.4f} of the ${guard.limit_usd:.2f} ceiling — "
              "the partial spend is settled, not lost:")
        for event in guard.spend_log:
            print(f"     ledger: {event.kind} {event.model_or_tool} "
                  f"{event.completion_tokens} tokens → ${event.cost_usd:.4f}")


if __name__ == "__main__":
    main()
