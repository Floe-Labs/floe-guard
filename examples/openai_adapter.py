"""floe-guard + the OpenAI adapter — runs with NO API key, NO account, NO network.

``examples/runaway_loop.py`` drives the guard directly (``check()`` /
``record()``). This demo instead goes through
``floe_guard.integrations.openai.guarded_completion`` — the real
reserve-before / settle-after wrapper you'd point at a live ``openai.OpenAI``
client — so it exercises the actual hard-stop contract: a blocked call raises
``BudgetExceeded`` *before* ``client.chat.completions.create`` ever runs.

The client below is a duck-typed stub (same shape the adapter's own tests use)
that returns a fixed ``usage`` block instead of calling OpenAI, so this needs
no ``openai`` install and no ``OPENAI_API_KEY``.

Run:  python examples/openai_adapter.py
"""

from __future__ import annotations

from dataclasses import dataclass

from floe_guard import BudgetExceeded, BudgetGuard
from floe_guard.integrations.openai import guarded_completion

MODEL = "gpt-4o"


@dataclass
class _Usage:
    prompt_tokens: int
    completion_tokens: int


@dataclass
class _Response:
    model: str
    usage: _Usage


class _Completions:
    """Stub of ``client.chat.completions`` — no network, fixed token usage."""

    def __init__(self) -> None:
        self.call_count = 0

    def create(self, **kwargs: object) -> _Response:
        self.call_count += 1
        # A real call reached here — this is what the hard-stop must prevent
        # once the budget is exhausted.
        return _Response(model=MODEL, usage=_Usage(prompt_tokens=1_000, completion_tokens=1_000))


class _Chat:
    def __init__(self, completions: _Completions) -> None:
        self.completions = completions


class _Client:
    """Duck-typed stand-in for ``openai.OpenAI`` — same shape, no network."""

    def __init__(self) -> None:
        self.chat = _Chat(_Completions())


def main() -> None:
    # gpt-4o at 1k in + 1k out ~= $0.0125/call, so a $0.05 ceiling clears
    # exactly 4 calls before the 5th is blocked pre-flight.
    guard = BudgetGuard(limit_usd=0.05)
    client = _Client()

    print(f"Starting with a ${guard.limit_usd:.2f} budget against a stub OpenAI client...\n")
    call = 0
    while True:
        call += 1
        messages = [{"role": "user", "content": "go"}]
        try:
            response = guarded_completion(guard, client, model=MODEL, messages=messages)
        except BudgetExceeded:
            calls_made = client.chat.completions.call_count
            print(f"\nCall #{call} blocked before reaching the client.")
            print(f"client.chat.completions.create was invoked {calls_made} times (not {call}).")
            break
        print(f"  call #{call}: served by {response.model}  (running total ${guard.spent_usd:.4f})")


if __name__ == "__main__":
    main()
