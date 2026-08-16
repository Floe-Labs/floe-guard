"""Budget-aware retry helper tests."""

from __future__ import annotations

import asyncio

import pytest

from floe_guard import (
    BudgetExceeded,
    BudgetGuard,
    RetryPlan,
    async_with_budget_retry,
    with_budget_retry,
)
from floe_guard.errors import (
    DeadlineExceeded,
    FloeGuardError,
    HostedEnforcementError,
    UnpriceableModelError,
    UnpriceableVoiceError,
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


@pytest.mark.parametrize("bad", [1.5, True, "2"])
def test_non_integer_max_attempts_rejected(bad: object) -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        with_budget_retry(BudgetGuard(limit_usd=1.00), lambda: "ok", max_attempts=bad)  # type: ignore[arg-type]


def test_keyboard_interrupt_is_not_retried() -> None:
    guard = BudgetGuard(limit_usd=1.00)
    calls = {"primary": 0}

    def primary() -> str:
        calls["primary"] += 1
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        with_budget_retry(guard, primary, estimated_cost=0.01, max_attempts=3)

    assert calls == {"primary": 1}


@pytest.mark.asyncio
async def test_async_cancelled_error_is_not_retried() -> None:
    guard = BudgetGuard(limit_usd=1.00)
    calls = {"primary": 0}

    async def primary() -> str:
        calls["primary"] += 1
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await async_with_budget_retry(guard, primary, estimated_cost=0.01, max_attempts=3)

    assert calls == {"primary": 1}


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


# ── FloeGuardError family is non-retryable ────────────────────────────────────


def test_unpriceable_model_error_is_not_retried() -> None:
    """UnpriceableModelError must NOT be retried — it is deterministic."""
    guard = BudgetGuard(limit_usd=1.00)
    calls = {"count": 0}

    def call() -> str:
        calls["count"] += 1
        raise UnpriceableModelError("some-mystery-model")

    with pytest.raises(UnpriceableModelError):
        with_budget_retry(guard, call, estimated_cost=0.01, max_attempts=3)

    assert calls["count"] == 1


def test_plain_value_error_is_retried_by_default() -> None:
    """Ordinary non-FloeGuard exceptions should still retry."""
    guard = BudgetGuard(limit_usd=1.00)
    calls = {"count": 0}

    def call() -> str:
        calls["count"] += 1
        if calls["count"] < 3:
            raise ValueError("transient")
        return "ok"

    result = with_budget_retry(guard, call, estimated_cost=0.01, max_attempts=3)
    assert result == "ok"
    assert calls["count"] == 3


def test_budget_exceeded_is_not_retried() -> None:
    """BudgetExceeded (a FloeGuardError subclass) must remain non-retryable."""
    guard = BudgetGuard(limit_usd=1.00, on_block=lambda *_: None)
    guard.record_tool("seed", 0.99)
    calls = {"count": 0}

    def call() -> str:
        calls["count"] += 1
        raise RetryableError("fail")

    with pytest.raises(BudgetExceeded):
        with_budget_retry(guard, call, estimated_cost=0.02, max_attempts=3)

    # The first attempt raised RetryableError, guard.check() then raised
    # BudgetExceeded — call itself ran exactly once.
    assert calls["count"] == 1


@pytest.mark.parametrize(
    "exc",
    [
        UnpriceableVoiceError("elevenlabs", "tts"),
        HostedEnforcementError("network timeout"),
        DeadlineExceeded(500.0, 300.0),
    ],
)
def test_floe_guard_error_subclasses_are_not_retried(exc: FloeGuardError) -> None:
    """Every FloeGuardError subclass must be treated as terminal."""
    guard = BudgetGuard(limit_usd=1.00)
    calls = {"count": 0}

    def call() -> str:
        calls["count"] += 1
        raise exc

    with pytest.raises(type(exc)):
        with_budget_retry(guard, call, estimated_cost=0.01, max_attempts=3)

    assert calls["count"] == 1
