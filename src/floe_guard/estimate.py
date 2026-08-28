"""Offline workload cost estimator — ``floe-guard estimate``.

Prices an agent workload (model, calls, tokens per call) from the same bundled
cost map the guard enforces against — no API key, no account, no network. The
output answers the two questions a builder has before a run: *what will this
cost?* and *what ceiling should I set?* — then shows the three lines that put a
``BudgetGuard`` on exactly that workload.
"""

from __future__ import annotations

import math

from .pricing import cost_map_generated_at, price_tokens, resolve_price


def run_estimate(
    model: str,
    calls: int = 1,
    tokens_in: int = 1_000,
    tokens_out: int = 1_000,
) -> None:
    """Print the priced estimate for a workload. Raises ``ValueError`` on bad input.

    Fail-closed like the guard itself: a model the bundled map cannot price is a
    clean error, never a $0.00 guess.
    """
    if not model.strip():
        raise ValueError("model id must not be empty")
    if calls < 1:
        raise ValueError(f"--calls must be >= 1 (got {calls})")
    if tokens_in < 0 or tokens_out < 0:
        raise ValueError(f"token counts must be >= 0 (got in={tokens_in}, out={tokens_out})")

    priced = resolve_price(model)
    if priced is None:
        raise ValueError(
            f"cannot price {model!r} from the bundled cost map "
            "(unknown model or no published rates) — "
            "pass BudgetGuard(price_overrides=...) for models the map doesn't list"
        )

    per_call = price_tokens(priced, tokens_in, tokens_out)
    total = per_call * calls
    if not math.isfinite(total):
        raise ValueError("non-finite total — workload is too large to estimate")

    print(f"Estimating {calls:,} call(s) to {model} ({tokens_in:,} in / {tokens_out:,} out per call)\n")
    print(f"  per call:     ${per_call:.6f}")
    print(f"  total:        ${total:.6f}")
    source_line = f"  price source: {priced.source}"
    generated = cost_map_generated_at()
    if generated is not None:
        source_line += f" (snapshot {generated})"
    source_line += " — offline, no network"
    print(source_line + "\n")

    print(f"Guard this exact workload with a ceiling at the run total:\n")
    print("  from floe_guard import BudgetGuard")
    print(f"  guard = BudgetGuard(limit_usd={total:.6f})   # covers {calls:,} call(s)")
    print("  # guard.check() before each call; guard.record(model, in, out) after")
    print("\nAdd headroom for retries — the guard hard-stops AT the ceiling, not near it.")
