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


def test_large_override_map_does_not_slow_the_hot_path():
    """Guards the regression above: per-call cost must not scale with overrides."""
    import time

    def median_ns(n_overrides: int) -> float:
        overrides = {f"m-{i}": ManualPrice(1e-6, 2e-6) for i in range(n_overrides)}
        guard = BudgetGuard(limit_usd=1e9, price_overrides=overrides or None)
        samples = []
        for _ in range(2000):
            t0 = time.perf_counter_ns()
            guard.settle("gpt-4o", 800, 120, price=ManualPrice(5e-06, 1.5e-05))
            samples.append(time.perf_counter_ns() - t0)
        samples.sort()
        return samples[len(samples) // 2]

    # generous bound: a per-call scan of 500 overrides was ~13x, not ~1x
    assert median_ns(500) < median_ns(0) * 5


def test_manual_price_still_works():
    guard = BudgetGuard(limit_usd=1.0)
    cost = guard.settle("gpt-4o", 100, 100, price=ManualPrice(5e-06, 1.5e-05))
    assert cost == pytest.approx(100 * 5e-06 + 100 * 1.5e-05)
