"""Budget-aware retry helper tests."""

from __future__ import annotations

import pytest

from floe_guard import (
    BudgetExceeded,
    BudgetGuard,
    RetryPlan,
    async_with_budget_retry,
    with_budget_retry,
)


class RetryableError(RuntimeError):
    pass


def test_ample_budget_retries_same_call() -> None:
    guard = BudgetGuard(limit_usd=1.00)
    calls = {"primary": 0}

    def primary() -> str:
        calls["primary"] += 1
        if calls["primary"] == 1:
            raise RetryableError("temporary failure")
        return "primary-ok"

    assert with_budget_retry(guard, primary, estimated_cost=0.05, max_attempts=2) == "primary-ok"
    assert calls == {"primary": 2}


def test_near_limit_failure_retries_with_degraded_plan() -> None:
    guard = BudgetGuard(limit_usd=1.00, near_limit_bps=8000)
    guard.record_tool("seed", 0.85)
    calls = {"primary": 0, "cheap": 0}

    def primary() -> str:
        calls["primary"] += 1
        raise RetryableError("temporary failure")

    def cheap() -> str:
        calls["cheap"] += 1
        return "cheap-ok"

    def degrade(exc: BaseException, _advisory) -> RetryPlan[str]:
        assert isinstance(exc, RetryableError)
        return RetryPlan(call=cheap, estimated_cost=0.01)

    result = with_budget_retry(
        guard,
        primary,
        estimated_cost=0.20,
        max_attempts=2,
        on_degrade=degrade,
    )

    assert result == "cheap-ok"
    assert calls == {"primary": 1, "cheap": 1}


def test_over_budget_retry_aborts_before_second_call() -> None:
    guard = BudgetGuard(limit_usd=1.00, on_block=lambda *_: None)
    guard.record_tool("seed", 0.95)
    calls = {"primary": 0}

    def primary() -> str:
        calls["primary"] += 1
        raise RetryableError("temporary failure")

    with pytest.raises(BudgetExceeded):
        with_budget_retry(guard, primary, estimated_cost=0.10, max_attempts=2)

    assert calls == {"primary": 1}


def test_non_retryable_failure_is_raised_without_retry() -> None:
    guard = BudgetGuard(limit_usd=1.00)
    calls = {"primary": 0}

    def primary() -> str:
        calls["primary"] += 1
        raise ValueError("bad request")

    with pytest.raises(ValueError, match="bad request"):
        with_budget_retry(
            guard,
            primary,
            estimated_cost=0.01,
            retry_if=lambda exc: not isinstance(exc, ValueError),
        )

    assert calls == {"primary": 1}


def test_invalid_max_attempts_rejected() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        with_budget_retry(BudgetGuard(limit_usd=1.00), lambda: "ok", max_attempts=0)


@pytest.mark.asyncio
async def test_async_helper_degrades_near_limit() -> None:
    guard = BudgetGuard(limit_usd=1.00, near_limit_bps=8000)
    guard.record_tool("seed", 0.85)
    calls = {"primary": 0, "cheap": 0}

    async def primary() -> str:
        calls["primary"] += 1
        raise RetryableError("temporary failure")

    async def cheap() -> str:
        calls["cheap"] += 1
        return "cheap-ok"

    async def degrade(_exc: BaseException, _advisory):
        return RetryPlan(call=cheap, estimated_cost=0.01)

    result = await async_with_budget_retry(
        guard,
        primary,
        estimated_cost=0.20,
        max_attempts=2,
        on_degrade=degrade,
    )

    assert result == "cheap-ok"
    assert calls == {"primary": 1, "cheap": 1}
