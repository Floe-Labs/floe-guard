"""A cost-map-shaped dict passed to ``price=`` should fail at the call site.

`cost_map.json` entries carry exactly the key names `ManualPrice` uses, so
handing one straight to `settle(price=...)` is the obvious move. It used to
surface as ``AttributeError: 'dict' object has no attribute
'input_cost_per_token'`` from three frames inside `resolve_price`.
"""

from __future__ import annotations

import pytest

from floe_guard import BudgetGuard, ManualPrice

COST_MAP_ENTRY = {
    "input_cost_per_token": 5e-06,
    "output_cost_per_token": 1.5e-05,
    "litellm_provider": "openai",
    "mode": "chat",
}


def test_dict_to_price_raises_typeerror_naming_manualprice():
    guard = BudgetGuard(limit_usd=1.0)
    with pytest.raises(TypeError, match="must be a ManualPrice"):
        guard.settle("gpt-4o", 100, 100, price=COST_MAP_ENTRY)


def test_the_message_shows_how_to_convert():
    guard = BudgetGuard(limit_usd=1.0)
    with pytest.raises(TypeError) as exc:
        guard.settle("gpt-4o", 100, 100, price=COST_MAP_ENTRY)
    msg = str(exc.value)
    assert "ManualPrice(input_cost_per_token=5e-06" in msg
    assert "output_cost_per_token=1.5e-05" in msg


def test_mapping_without_the_required_keys_says_so():
    guard = BudgetGuard(limit_usd=1.0)
    with pytest.raises(TypeError, match="needs both"):
        guard.settle("gpt-4o", 100, 100, price={"mode": "chat"})


def test_price_overrides_is_checked_too():
    guard = BudgetGuard(limit_usd=1.0, price_overrides={"gpt-4o": COST_MAP_ENTRY})
    with pytest.raises(TypeError, match=r"price_overrides\['gpt-4o'\]"):
        guard.settle("gpt-4o", 100, 100)


def test_manual_price_still_works():
    guard = BudgetGuard(limit_usd=1.0)
    cost = guard.settle("gpt-4o", 100, 100, price=ManualPrice(5e-06, 1.5e-05))
    assert cost == pytest.approx(100 * 5e-06 + 100 * 1.5e-05)
