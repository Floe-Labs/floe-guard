"""floe-guard + the Anthropic adapter — runs with NO API key, NO account, NO network.

Same reserve-before / settle-after contract as ``examples/openai_adapter.py``,
but this adapter has something the OpenAI one doesn't: native prompt-cache
pricing (PR #27). A cached read of the same tokens costs a fraction of a fresh
read — ``cache_read_input_tokens`` prices at 0.1x the input rate, vs. 1.25x
(5m TTL) or 2.0x (1h TTL) to *write* the cache in the first place. See
``_CACHE_READ_MULTIPLIER`` / ``_CACHE_CREATION_MULTIPLIER`` in
``floe_guard/pricing.py``.

This demo simulates a multi-turn conversation that re-sends a large shared
context (e.g. a codebase excerpt pasted as a system prompt): turn 1 writes it
to the cache, turns 2+ read it back near-free. A probe call measures what
that same turn would have cost had it re-sent the context uncached, so the
comparison is priced by the real engine — not a hardcoded number that could
drift from the bundled cost map.

The client below is a duck-typed stub (same shape the adapter's own tests
use), so this needs no ``anthropic`` install and no ``ANTHROPIC_API_KEY``.

Run:  python examples/anthropic_adapter.py
"""

from __future__ import annotations

from dataclasses import dataclass, field

from floe_guard import BudgetGuard
from floe_guard.integrations.anthropic import guarded_completion

MODEL = "claude-3-7-sonnet-20250219"

# A large shared context (e.g. a codebase excerpt) re-sent on every turn, plus
# a small new question and a short reply.
CONTEXT_TOKENS = 6_000
QUESTION_TOKENS = 40
OUTPUT_TOKENS = 150


@dataclass
class _CacheCreation:
    ephemeral_5m_input_tokens: int = 0
    ephemeral_1h_input_tokens: int = 0


@dataclass
class _Usage:
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation: _CacheCreation = field(default_factory=_CacheCreation)


@dataclass
class _Response:
    model: str
    usage: _Usage


class _Messages:
    """Stub of ``client.messages`` — no network, fixed token usage per call."""

    def __init__(self, usage: _Usage) -> None:
        self._usage = usage

    def create(self, **kwargs: object) -> _Response:
        return _Response(model=MODEL, usage=self._usage)


class _Client:
    """Duck-typed stand-in for ``anthropic.Anthropic`` — same shape, no network."""

    def __init__(self, usage: _Usage) -> None:
        self.messages = _Messages(usage)


def _call(guard: BudgetGuard, usage: _Usage) -> float:
    """Run one guarded call against a stub returning ``usage`` and return its cost."""
    before = guard.spent_usd
    guarded_completion(guard, _Client(usage), model=MODEL, max_tokens=1024, messages=[])
    return guard.spent_usd - before


def main() -> None:
    guard = BudgetGuard(limit_usd=1.00)

    print(f"Simulating a conversation that re-sends a {CONTEXT_TOKENS}-token context...\n")

    # Turn 1: nothing cached yet — write the context to the 5m-TTL cache.
    cold_usage = _Usage(
        input_tokens=QUESTION_TOKENS,
        output_tokens=OUTPUT_TOKENS,
        cache_creation_input_tokens=CONTEXT_TOKENS,
        cache_creation=_CacheCreation(ephemeral_5m_input_tokens=CONTEXT_TOKENS),
    )
    cold_cost = _call(guard, cold_usage)
    print(f"  turn 1 (cache write): ${cold_cost:.5f}")

    # Turns 2-3: same context, now served from cache.
    warm_usage = _Usage(
        input_tokens=QUESTION_TOKENS,
        output_tokens=OUTPUT_TOKENS,
        cache_read_input_tokens=CONTEXT_TOKENS,
    )
    warm_cost = _call(guard, warm_usage)
    print(f"  turn 2 (cache read):  ${warm_cost:.5f}")
    warm_cost_2 = _call(guard, warm_usage)
    print(f"  turn 3 (cache read):  ${warm_cost_2:.5f}")

    # Probe: what would turn 2 have cost if it re-sent the context uncached
    # (no cache_read_input_tokens — just a plain, full-price input)? Measured
    # via a throwaway guard through the same pricing engine, so this can't
    # drift from whatever the bundled cost map says today.
    probe_guard = BudgetGuard(limit_usd=1.00)
    uncached_usage = _Usage(
        input_tokens=CONTEXT_TOKENS + QUESTION_TOKENS, output_tokens=OUTPUT_TOKENS
    )
    uncached_cost = _call(probe_guard, uncached_usage)

    savings = uncached_cost - warm_cost
    multiple = uncached_cost / warm_cost if warm_cost else float("inf")
    print(f"\nSame turn, uncached: ${uncached_cost:.5f}")
    print(f"Cached read was ${savings:.5f} cheaper ({multiple:.1f}x less) than resending fresh.")
    print(f"\nTotal spent over 3 turns: ${guard.spent_usd:.5f}")


if __name__ == "__main__":
    main()
