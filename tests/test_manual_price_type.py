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


def test_price_overrides_is_checked_at_construction():
    """Checked once in __init__, not per call.

    Validating overrides inside _resolve costs O(len(overrides)) on every
    reserve/settle. Measured on this machine, that took a 500-entry override map
    from 3.5us to 45us per turn, which is a lot for a guard that sits in the hot
    path of a voice turn.
    """
    with pytest.raises(TypeError, match=r"price_overrides\['gpt-4o'\]"):
        BudgetGuard(limit_usd=1.0, price_overrides={"gpt-4o": COST_MAP_ENTRY})


def test_overrides_are_validated_once_at_construction_not_per_call():
    """Guards the regression deterministically — count the validation calls
    instead of timing them (a wall-clock bound is flaky on shared CI runners and
    would also measure the per-call ``{**overrides, model: price}`` copy, which is
    unrelated to the validation this fix moved to construction). The check runs
    once per override at construction and never again in the settle() hot path,
    so per-call cost is O(1) in the override-map size, not O(len(overrides))."""
    from unittest import mock

    import floe_guard.guard as guard_mod

    overrides = {f"m-{i}": ManualPrice(1e-6, 2e-6) for i in range(500)}
    with mock.patch(
        "floe_guard.guard._require_manual_price",
        wraps=guard_mod._require_manual_price,
    ) as spy:
        guard = BudgetGuard(limit_usd=1e9, price_overrides=overrides)
        # Every override is validated exactly once — at construction.
        assert spy.call_count == len(overrides)

        # A per-call settle with no per-call price re-validates nothing: the
        # stored overrides are already trusted.
        spy.reset_mock()
        guard.settle("gpt-4o", 800, 120)
        assert spy.call_count == 0

        # A per-call price is validated once (that price only), never
        # once-per-stored-override — so the hot path stays independent of map size.
        guard.settle("gpt-4o", 800, 120, price=ManualPrice(5e-06, 1.5e-05))
        assert spy.call_count == 1


def test_manual_price_still_works():
    guard = BudgetGuard(limit_usd=1.0)
    cost = guard.settle("gpt-4o", 100, 100, price=ManualPrice(5e-06, 1.5e-05))
    assert cost == pytest.approx(100 * 5e-06 + 100 * 1.5e-05)
