"""The no-key "stop a loop" demo, runnable straight from the installed package.

``floe-guard demo`` (or ``from floe_guard.demo import run_demo``) runs a naive
agent loop against a **stub LLM** — no API key, no account, no network. The guard
hard-stops the loop before the call that would cross a ``$0.10`` ceiling; cost is
priced offline from the bundled cost map, exactly as it would be for a real
``gpt-4o`` call.

This is the same demo as ``examples/runaway_loop.py`` (which is a thin wrapper
around :func:`run_demo`), but it ships inside the wheel — so ``pip install
floe-guard`` is enough to see the guard work, no repository checkout needed.
"""

from __future__ import annotations

from .errors import BudgetExceeded
from .guard import BudgetGuard

MODEL = "gpt-4o"


def _stub_llm() -> dict[str, object]:
    """A fake LLM call — no network, no API key. Returns fixed token usage."""
    return {"model": MODEL, "prompt_tokens": 1_000, "completion_tokens": 1_000}


def run_demo(limit_usd: float = 0.10) -> None:
    """Run the runaway-loop demo: a loop a ``limit_usd`` guard hard-stops.

    A naive agent loop calls a stub LLM forever; ``floe-guard`` stops it before
    the call that would cross ``limit_usd``. Prints one line per call plus the
    stop line — identical to ``examples/runaway_loop.py``.
    """
    guard = BudgetGuard(limit_usd=limit_usd)

    print(f"Starting a runaway loop with a ${guard.limit_usd:.2f} budget...\n", flush=True)
    call = 0
    while True:  # a real runaway loop never decides to stop on its own
        call += 1
        try:
            # Size the check to the KNOWN request so the guard hard-stops before
            # the crossing call even on call #1 — a bare check() is blind until the
            # first record() (it estimates from the last call's cost, which is $0
            # up front). estimate_call() prices the stub payload we're about to send.
            est = guard.estimate_call(MODEL, 1_000, max_completion_tokens=1_000)
            guard.check(estimated_next_cost=est)  # the kill-switch: raises before the crossing call
        except BudgetExceeded:
            print(
                f"\nLoop stopped at call #{call}. The agent never got to spend past the budget.",
                flush=True,
            )
            # Write the recorded spend ledger to a LOCAL file — no network, no
            # account (the demo's promise holds). ``push`` below is the explicit
            # opt-in step that actually sends it.
            ledger_path = "./floe-ledger.jsonl"
            with open(ledger_path, "w", encoding="utf-8") as fh:
                fh.write(guard.export_log())
            print(f"\nWrote spend ledger to {ledger_path}", flush=True)
            print(
                f"Opt-in next step (priced spend events only — no prompts, no content):\n"
                f"  floe-guard push {ledger_path}",
                flush=True,
            )
            break

        response = _stub_llm()
        cost = guard.record(
            str(response["model"]),
            int(response["prompt_tokens"]),  # type: ignore[arg-type]
            int(response["completion_tokens"]),  # type: ignore[arg-type]
        )
        print(f"  call #{call}: +${cost:.4f}  (running total ${guard.spent_usd:.4f})", flush=True)
