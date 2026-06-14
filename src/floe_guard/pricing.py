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


def _bare_model(model: str) -> str:
    """Strip an optional ``provider/`` prefix (LiteLLM convention, e.g. ``openai/gpt-4o``)."""
    m = model.strip()
    slash = m.rfind("/")
    return m if slash == -1 else m[slash + 1 :]


def resolve_price(
    model: str,
    overrides: dict[str, ManualPrice] | None = None,
) -> PricedModel | None:
    """Resolve a model to its per-token price, or ``None`` if it cannot be priced.

    Overrides win, then the bundled cost map (looked up by bare name, then the
    raw field). Fail-closed: both prices must be finite, else ``None``.
    """
    bare = _bare_model(model)

    if overrides:
        ov = overrides.get(bare) or overrides.get(model.strip())
        if ov is not None:
            if _both_finite(ov.input_cost_per_token, ov.output_cost_per_token):
                return PricedModel(
                    input_cost_per_token=ov.input_cost_per_token,
                    output_cost_per_token=ov.output_cost_per_token,
                    source="override",
                )
            return None

    entry = _COST_MAP.get(bare) or _COST_MAP.get(model.strip())
    if not entry:
        return None
    input_cost = entry.get("input_cost_per_token")
    output_cost = entry.get("output_cost_per_token")
    if not _both_finite(input_cost, output_cost):
        return None
    return PricedModel(
        input_cost_per_token=float(input_cost),
        output_cost_per_token=float(output_cost),
        source="cost_map",
    )


def _both_finite(a: Any, b: Any) -> bool:
    return (
        isinstance(a, (int, float))
        and isinstance(b, (int, float))
        and math.isfinite(a)
        and math.isfinite(b)
    )


def price_tokens(priced: PricedModel, prompt_tokens: int, completion_tokens: int) -> float:
    """USD cost for token usage. Negative counts are clamped to zero."""
    p = max(0, prompt_tokens)
    c = max(0, completion_tokens)
    cost = p * priced.input_cost_per_token + c * priced.output_cost_per_token
    if not math.isfinite(cost):
        # Defense-in-depth: resolve_price already guarantees finite rates.
        raise ValueError("Non-finite LLM cost — pricing entry is invalid")
    return max(0.0, cost)
