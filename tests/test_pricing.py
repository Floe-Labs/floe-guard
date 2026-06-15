"""Tests for offline pricing — fail-closed resolution and token math."""

from __future__ import annotations

import pytest

from floe_guard.pricing import (
    ManualPrice,
    PricedModel,
    price_tokens,
    resolve_price,
)


def test_resolves_known_model_from_cost_map() -> None:
    priced = resolve_price("gpt-4o")
    assert priced is not None
    assert priced.source == "cost_map"
    assert priced.input_cost_per_token == pytest.approx(2.5e-6)
    assert priced.output_cost_per_token == pytest.approx(1e-5)


def test_strips_provider_prefix() -> None:
    bare = resolve_price("gpt-4o")
    prefixed = resolve_price("openai/gpt-4o")
    assert prefixed is not None and bare is not None
    assert prefixed.input_cost_per_token == bare.input_cost_per_token


def test_unknown_model_is_unpriceable() -> None:
    assert resolve_price("no-such-model-anywhere") is None


def test_override_wins_over_cost_map() -> None:
    priced = resolve_price("gpt-4o", {"gpt-4o": ManualPrice(1e-9, 2e-9)})
    assert priced is not None
    assert priced.source == "override"
    assert priced.input_cost_per_token == 1e-9


def test_override_with_non_finite_price_is_unpriceable() -> None:
    # Fail closed: a malformed override (NaN/inf) must not be used.
    assert resolve_price("x", {"x": ManualPrice(float("nan"), 1e-6)}) is None
    assert resolve_price("y", {"y": ManualPrice(1e-6, float("inf"))}) is None


def test_price_tokens_math() -> None:
    priced = PricedModel(input_cost_per_token=1e-6, output_cost_per_token=2e-6, source="cost_map")
    assert price_tokens(priced, 1_000, 500) == pytest.approx(1e-3 + 1e-3)


def test_price_tokens_clamps_negative_counts() -> None:
    priced = PricedModel(input_cost_per_token=1e-6, output_cost_per_token=2e-6, source="cost_map")
    assert price_tokens(priced, -50, -50) == 0.0
