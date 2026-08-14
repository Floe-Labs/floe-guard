"""Pre-call admission gates return the exact provider inbound-webhook shapes.

Reject on budget exhaustion, admit otherwise — same contract a free/local user
serves as the hosted gateway. Pre-call only; no mid-call behaviour is exercised.
"""

from __future__ import annotations

import pytest

from floe_guard import BudgetGuard, gates


def _spent(limit: float, spend: float) -> BudgetGuard:
    guard = BudgetGuard(limit_usd=limit)
    if spend > 0:
        guard.record_tool("api", spend)
    return guard


# ── generic decision ──────────────────────────────────────────────────────────


def test_budget_exhausted_and_pre_call() -> None:
    assert gates.budget_exhausted(_spent(1.00, 1.00)) is True
    assert gates.budget_exhausted(_spent(1.00, 0.50)) is False
    assert gates.pre_call(_spent(1.00, 0.50)) is True
    assert gates.pre_call(_spent(1.00, 1.00)) is False


def test_estimated_call_usd_reserves_headroom() -> None:
    # $0.30 left, but a call is estimated to cost $0.50 → reject at admission.
    guard = _spent(1.00, 0.70)
    assert gates.pre_call(guard, estimated_call_usd=0.50) is False
    assert gates.pre_call(guard, estimated_call_usd=0.20) is True


def test_estimate_exactly_equal_to_remaining_admits() -> None:
    # Exact fit admits — inclusive ceiling, matching BudgetGuard.reserve().
    guard = _spent(1.00, 0.50)  # $0.50 remaining
    assert gates.budget_exhausted(guard, estimated_call_usd=0.50) is False
    assert gates.budget_exhausted(guard, estimated_call_usd=0.5001) is True


def test_fully_spent_is_exhausted_regardless_of_estimate() -> None:
    guard = _spent(1.00, 1.00)  # $0 remaining
    assert gates.budget_exhausted(guard) is True
    assert gates.budget_exhausted(guard, estimated_call_usd=0.0) is True


@pytest.mark.parametrize("bad", [-1.0, -0.01, float("nan"), float("inf")])
def test_bad_estimate_raises_not_admits(bad: float) -> None:
    # A negative/NaN/inf estimate must not silently admit an exhausted guard.
    guard = _spent(1.00, 1.00)  # exhausted
    with pytest.raises(ValueError, match="finite, non-negative"):
        gates.budget_exhausted(guard, estimated_call_usd=bad)


# ── Retell ────────────────────────────────────────────────────────────────────


def test_retell_rejects_on_exhaustion() -> None:
    assert gates.retell(_spent(1.00, 1.00)) == {"call_inbound": {"reject": True}}


def test_retell_admits_with_overrides_and_no_reject_key() -> None:
    resp = gates.retell(_spent(1.00, 0.10), admit={"dynamic_variables": {"name": "Ada"}})
    assert resp == {"call_inbound": {"dynamic_variables": {"name": "Ada"}}}
    assert "reject" not in resp["call_inbound"]


def test_retell_reject_is_the_boolean_true_not_a_string() -> None:
    # Retell ignores any non-boolean-true value; assert we emit real `True`.
    reject = gates.retell(_spent(1.00, 1.00))["call_inbound"]["reject"]
    assert reject is True


# ── Vapi ──────────────────────────────────────────────────────────────────────


def test_vapi_rejects_with_error_shape() -> None:
    resp = gates.vapi(_spent(1.00, 1.00), assistant_id="asst_1", error_message="No budget.")
    assert resp == {"error": "No budget."}


def test_vapi_admits_with_assistant_id() -> None:
    assert gates.vapi(_spent(1.00, 0.10), assistant_id="asst_1") == {"assistantId": "asst_1"}


def test_vapi_admits_with_inline_assistant() -> None:
    assistant = {"model": {"provider": "openai", "model": "gpt-4o"}}
    assert gates.vapi(_spent(1.00, 0.10), assistant=assistant) == {"assistant": assistant}


def test_vapi_admit_without_target_raises() -> None:
    with pytest.raises(ValueError, match="assistant"):
        gates.vapi(_spent(1.00, 0.10))


def test_retell_admit_cannot_inject_reject() -> None:
    # An errant admit override must not flip an available-budget call into a reject.
    resp = gates.retell(
        _spent(1.00, 0.10), admit={"reject": True, "dynamic_variables": {"name": "Ada"}}
    )
    assert resp == {"call_inbound": {"dynamic_variables": {"name": "Ada"}}}
