"""floe-guard + LangChain + Groq example.

Budget enforcement with ``ChatGroq`` — the guard hard-stops before the next
call once spending would cross the ceiling.

Run::

    pip install floe-guard[langchain] langchain-groq
    export GROQ_API_KEY=gsk_...
    python examples/langchain_groq_example.py

ChatGroq surfaces token counts via ``usage_metadata`` (``input_tokens`` /
``output_tokens``) rather than the ``token_usage`` block OpenAI uses — the
adapter's existing fallback handles this with no changes needed.
"""

from __future__ import annotations

import os
import sys

from langchain_groq import ChatGroq

from floe_guard import BudgetExceeded, BudgetGuard
from floe_guard.integrations.langchain import budget_guard_callback_handler


def main() -> None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print(
            "Set GROQ_API_KEY before running this example.\n"
            "  export GROQ_API_KEY=gsk_...",
            file=sys.stderr,
        )
        sys.exit(1)

    # A tight ceiling so the second call is blocked without spending much.
    guard = BudgetGuard(limit_usd=0.001)
    handler = budget_guard_callback_handler(guard)

    llm = ChatGroq(
        model="llama-3.1-8b-instant",   # fast, cheap — good for demos
        api_key=api_key,
        callbacks=[handler],
    )

    print("Call 1 — under budget, should go through...")
    response = llm.invoke("Reply with one word: hello")
    print(f"  response : {response.content!r}")
    print(f"  spent so far: ${guard.spent_usd:.6f}\n")

    print("Call 2 — projected cost would cross the ceiling, should be blocked...")
    try:
        llm.invoke("Reply with one word: world")
        print("  ERROR: expected BudgetExceeded but no exception was raised")
    except BudgetExceeded as exc:
        print(f"  BudgetExceeded raised as expected: {exc}")

    print(f"\nFinal spend: ${guard.spent_usd:.6f}  (ceiling: ${guard.limit_usd:.6f})")


if __name__ == "__main__":
    main()
