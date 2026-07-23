"""Budget-aware retry / graceful degradation demo.

Run with:

    python examples/budget_retry.py

No API key, no network. The first call fails near the cap, so the retry helper
asks the caller for a cheaper path and checks that the cheaper retry fits before
running it.
"""

from __future__ import annotations

from floe_guard import BudgetGuard, RetryPlan, with_budget_retry


class TransientProviderError(RuntimeError):
    pass


def main() -> None:
    guard = BudgetGuard(limit_usd=1.00, near_limit_bps=8000)
    guard.record_tool("previous-work", 0.85)
    calls = {"premium": 0, "mini": 0}

    def premium_model() -> str:
        calls["premium"] += 1
        raise TransientProviderError("temporary upstream timeout")

    def mini_model() -> str:
        calls["mini"] += 1
        return "retried on cheaper model"

    def degrade(_exc: BaseException, _advisory) -> RetryPlan[str]:
        return RetryPlan(call=mini_model, estimated_cost=0.01)

    result = with_budget_retry(
        guard,
        premium_model,
        estimated_cost=0.20,
        max_attempts=2,
        on_degrade=degrade,
    )

    print(result)
    print(calls)


if __name__ == "__main__":
    main()
