"""Stop-the-loop demo — runs with NO API key and NO account.

A naive agent loop that calls an LLM forever. The only thing standing between you
and a five-figure overnight bill is ``floe-guard``: it hard-stops the loop before
the call that would cross your $0.10 ceiling.

Run it::

    python examples/runaway_loop.py

The "LLM" here is a stub that returns fixed token usage — no network, no key, no
crewai/litellm needed. The cost is computed offline from the bundled cost map,
exactly as it would be for a real ``gpt-4o`` call.
"""

from __future__ import annotations

from floe_guard import BudgetExceeded, BudgetGuard

MODEL = "gpt-4o"


def stub_llm(prompt: str) -> dict[str, object]:
    """A fake LLM call. No network, no API key — returns fixed token usage."""
    return {
        "model": MODEL,
        "text": "...thinking... let me call myself again...",
        "prompt_tokens": 1_000,
        "completion_tokens": 1_000,
    }


def main() -> None:
    # $0.10 ceiling. gpt-4o at 1k in + 1k out ≈ $0.0125/call, so the guard should
    # stop the loop after a handful of iterations — well before any real damage.
    guard = BudgetGuard(limit_usd=0.10)

    print(f"Starting a runaway loop with a ${guard.limit_usd:.2f} budget...\n")
    call = 0
    while True:  # a real runaway loop never decides to stop on its own
        call += 1
        try:
            guard.check()  # <-- the kill-switch: raises before the crossing call
        except BudgetExceeded:
            print(f"\nLoop stopped at call #{call}. The agent never got to spend past the budget.")
            break

        response = stub_llm("keep going")
        cost = guard.record(
            str(response["model"]),
            int(response["prompt_tokens"]),  # type: ignore[arg-type]
            int(response["completion_tokens"]),  # type: ignore[arg-type]
        )
        print(f"  call #{call}: +${cost:.4f}  (running total ${guard.spent_usd:.4f})")


if __name__ == "__main__":
    main()
