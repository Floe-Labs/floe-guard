"""The per-turn receipt contract (``FloeCost`` / ``turn_cost``)."""

from __future__ import annotations

import pytest

from floe_guard import FloeCost, turn_cost
from floe_guard.pricing import price_tokens, resolve_price


def test_turn_cost_prices_known_model() -> None:
    cost = turn_cost("gpt-4o", 1000, 500)
    assert cost is not None
    assert cost.usd == pytest.approx(0.0075)  # 1000*2.5e-6 + 500*1e-5
    assert cost.source == "estimate"
    assert cost.model == "gpt-4o"
    assert cost.remaining_usd is None


def test_turn_cost_fails_closed_on_unpriceable() -> None:
    assert turn_cost("no-such-model-anywhere", 1000, 500) is None


def test_turn_cost_carries_remaining_budget() -> None:
    cost = turn_cost("gpt-4o", 1000, 500, remaining_usd=12.34)
    assert cost is not None
    assert cost.remaining_usd == 12.34


def test_format_estimate_only() -> None:
    line = FloeCost(usd=0.0075, source="estimate", model="gpt-4o").format()
    assert line == "floe · gpt-4o · $0.0075 est"


def test_format_with_budget() -> None:
    line = FloeCost(usd=0.0075, source="hosted", model="gpt-4o", remaining_usd=12.34).format()
    assert line == "floe · gpt-4o · $0.0075 floe · left $12.34"


def test_format_micro_cost_does_not_round_to_zero() -> None:
    # A real gpt-4o-mini short turn (~$0.000008) must not render as a misleading $0.0000.
    line = FloeCost(usd=8e-6, source="estimate", model="gpt-4o-mini").format()
    assert line == "floe · gpt-4o-mini · $0.000008 est"


def test_format_micro_cost_below_threshold_keeps_precision() -> None:
    # Just under $0.0001 must keep 6 decimals, not round up to $0.0001.
    assert FloeCost(usd=0.000099, source="estimate", model="m").format().endswith("$0.000099 est")


def test_format_at_threshold_uses_four_decimals() -> None:
    # At $0.0001 exactly, drop to the compact 4-decimal form.
    assert FloeCost(usd=0.0001, source="estimate", model="m").format().endswith("$0.0001 est")


def test_source_must_be_estimate_or_hosted() -> None:
    with pytest.raises(ValueError, match="honesty contract"):
        FloeCost(usd=0.01, source="other", model="m")


def test_turn_cost_prices_cached_subset_at_cache_rate() -> None:
    # gpt-4o: input 2.5e-6, cache_read 1.25e-6 (0.5x). 1000 total prompt tokens,
    # 400 of them cached. The cached subset is priced at the cache-read rate, so
    # the total must be LESS than pricing all 1000 at the full input rate.
    model = "gpt-4o"
    full = turn_cost(model, 1000, 500)
    cached = turn_cost(model, 1000, 500, prompt_cached_tokens=400)
    assert full is not None and cached is not None
    assert cached.usd < full.usd

    # And it matches price_tokens with the explicit non-cached / cached split.
    priced = resolve_price(model)
    assert priced is not None
    expected = price_tokens(priced, 600, 500, cache_read_input_tokens=400)
    assert cached.usd == pytest.approx(expected)


def test_turn_cost_clamps_cached_to_prompt_tokens() -> None:
    # prompt_cached_tokens above prompt_tokens is clamped: every input token is
    # cached, none billed at the full rate.
    model = "gpt-4o"
    priced = resolve_price(model)
    assert priced is not None
    over = turn_cost(model, 1000, 500, prompt_cached_tokens=99_999)
    assert over is not None
    expected = price_tokens(priced, 0, 500, cache_read_input_tokens=1000)
    assert over.usd == pytest.approx(expected)


def test_turn_cost_cached_zero_is_unchanged() -> None:
    # The default (0 cached) prices exactly as before the cache-aware change.
    model = "gpt-4o"
    default = turn_cost(model, 1000, 500)
    explicit = turn_cost(model, 1000, 500, prompt_cached_tokens=0)
    assert default is not None and explicit is not None
    assert explicit.usd == pytest.approx(default.usd)
    assert default.usd == pytest.approx(0.0075)  # 1000*2.5e-6 + 500*1e-5


def test_turn_cost_returns_the_public_floecost_schema() -> None:
    cost = turn_cost("gpt-4o", 1000, 500)
    assert isinstance(cost, FloeCost)
    # The byte-for-byte contract both local and hosted must speak.
    assert tuple(cost.__dataclass_fields__) == ("usd", "source", "model", "remaining_usd")
