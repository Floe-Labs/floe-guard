"""Offline token pricing from a vendored LiteLLM cost map.

This mirrors the fail-closed logic of Floe's metered proxy
(``floe-monorepo/apps/api/src/services/llm-pricing.ts``): BOTH the input and
output per-token prices must be finite numbers, otherwise the model is treated
as unpriceable. A half-valid entry would silently undercharge — so we refuse it.

No network. The cost map (``cost_map.json``) is a snapshot of LiteLLM's
``model_prices_and_context_window.json``; refresh it on a schedule, exactly like
the proxy does, or estimates drift as vendors change prices.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from importlib import resources
from typing import Any


@dataclass(frozen=True)
class ManualPrice:
    """A user-supplied per-token price, in USD, for a model the map cannot price."""

    input_cost_per_token: float
    output_cost_per_token: float


@dataclass(frozen=True)
class PricedModel:
    """A resolved per-token price plus where it came from."""

    input_cost_per_token: float
    output_cost_per_token: float
    source: str  # "override" | "cost_map"


def _load_cost_map() -> dict[str, Any]:
    with resources.files("floe_guard").joinpath("cost_map.json").open("r", encoding="utf-8") as fh:
        return json.load(fh)


_COST_MAP: dict[str, Any] = _load_cost_map()


# The one "<provider>/" prefix that is safe to strip: the remainder of a
# "groq/…" id is the ChatGroq id the map vendors (e.g. "groq/qwen/qwen3-32b" →
# "qwen/qwen3-32b"). "openai/" and "anthropic/" are deliberately excluded:
# their own model ids never contain slashes (single-segment remainders are
# already covered by the bare-last-segment fallback), so a multi-segment
# remainder under those prefixes is some OTHER vendor's model behind an
# OpenAI-compatible endpoint (vLLM, OpenRouter, …) — bridging it into a
# Groq-priced key would under-meter. Unknown prefixes fail closed the same way.
_PROVIDER_PREFIXES = frozenset({"groq"})

# A trailing dated-snapshot suffix: Anthropic's "-20250929" or OpenAI's
# "-2024-08-06". Vendors resolve alias ids to dated snapshots in responses, so a
# snapshot the map doesn't list yet prices at its alias entry (same model, same
# rate) instead of failing closed. re.ASCII: TS "\d" is ASCII-only — a Unicode
# digit suffix must not strip in one package and not the other.
_DATE_SUFFIX = re.compile(r"-(?:\d{8}|\d{4}-\d{2}-\d{2})$", re.ASCII)


def _candidate_groups(model: str) -> tuple[list[str], list[str]]:
    """Lookup keys for a model id in two specificity groups, deduplicated.

    Group 1 (exact): the raw id, the id with a known ``provider/`` first
    segment stripped, the bare last segment. Group 2 (date-stripped): the same
    forms with a trailing dated-snapshot suffix removed. Kept separate so a
    less-specific date-stripped key (in overrides OR the map) can never shadow
    an exact dated entry — e.g. an alias override must not absorb a snapshot
    the map prices differently.
    """
    m = model.strip()
    base = [m]
    first, _, rest = m.partition("/")
    if rest and first in _PROVIDER_PREFIXES:
        base.append(rest)
    slash = m.rfind("/")
    if slash != -1:
        base.append(m[slash + 1 :])
    exact: list[str] = []
    for cand in base:
        if cand and cand not in exact:
            exact.append(cand)
    stripped: list[str] = []
    for cand in exact:
        c = _DATE_SUFFIX.sub("", cand)
        if c and c not in exact and c not in stripped:
            stripped.append(c)
    return exact, stripped


def resolve_price(
    model: str,
    overrides: dict[str, ManualPrice] | None = None,
) -> PricedModel | None:
    """Resolve a model to its per-token price, or ``None`` if it cannot be priced.

    Per specificity group (exact forms first, date-stripped fallbacks second):
    overrides win, then the bundled cost map. Fail-closed: the first matching
    entry must have finite prices, else ``None``.
    """
    for candidates in _candidate_groups(model):
        if overrides:
            for cand in candidates:
                ov = overrides.get(cand)
                if ov is not None:
                    if _both_finite(ov.input_cost_per_token, ov.output_cost_per_token):
                        return PricedModel(
                            input_cost_per_token=ov.input_cost_per_token,
                            output_cost_per_token=ov.output_cost_per_token,
                            source="override",
                        )
                    return None

        for cand in candidates:
            entry = _COST_MAP.get(cand)
            if not entry:
                continue
            input_cost = entry.get("input_cost_per_token")
            output_cost = entry.get("output_cost_per_token")
            if not _both_finite(input_cost, output_cost):
                return None
            return PricedModel(
                input_cost_per_token=float(input_cost),
                output_cost_per_token=float(output_cost),
                source="cost_map",
            )
    return None


# Anthropic prompt-cache pricing multipliers: creation (5m) is 1.25x base input,
# creation (1h) is 2.0x base input, read is 0.1x base input.
_CACHE_CREATION_MULTIPLIER = 1.25
_CACHE_CREATION_1H_MULTIPLIER = 2.00
_CACHE_READ_MULTIPLIER = 0.10


def _both_finite(a: Any, b: Any) -> bool:
    return (
        isinstance(a, (int, float))
        and isinstance(b, (int, float))
        and math.isfinite(a)
        and math.isfinite(b)
    )


def price_tokens(
    priced: PricedModel,
    prompt_tokens: int,
    completion_tokens: int,
    *,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens_1h: int = 0,
) -> float:
    """USD cost for token usage. Negative counts are clamped to zero."""
    p = max(0, prompt_tokens)
    c = max(0, completion_tokens)
    cc = max(0, cache_creation_input_tokens)
    cc_1h = max(0, cache_creation_input_tokens_1h)
    cr = max(0, cache_read_input_tokens)

    cache_creation_cost = cc * priced.input_cost_per_token * _CACHE_CREATION_MULTIPLIER
    cache_creation_1h_cost = cc_1h * priced.input_cost_per_token * _CACHE_CREATION_1H_MULTIPLIER
    cache_read_cost = cr * priced.input_cost_per_token * _CACHE_READ_MULTIPLIER

    cost = (
        (p * priced.input_cost_per_token)
        + (c * priced.output_cost_per_token)
        + cache_creation_cost
        + cache_creation_1h_cost
        + cache_read_cost
    )
    if not math.isfinite(cost):
        # Defense-in-depth: resolve_price already guarantees finite rates.
        raise ValueError("Non-finite LLM cost — pricing entry is invalid")
    return max(0.0, cost)
