"""The per-turn receipt contract (``FloeCost`` / ``turn_cost``)."""

from __future__ import annotations

import pytest

from floe_guard import FloeCost, turn_cost


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


def test_floecost_shape_is_identical_local_and_hosted() -> None:
    # The contract's whole point: same fields whether estimated or hosted.
    est = FloeCost(usd=0.01, source="estimate", model="m", remaining_usd=None)
    hosted = FloeCost(usd=0.01, source="hosted", model="m", remaining_usd=5.0)
    assert est.__dataclass_fields__.keys() == hosted.__dataclass_fields__.keys()
