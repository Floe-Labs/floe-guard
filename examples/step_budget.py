"""Per-step token budget demo — no API key, account, or network.

Run with:

    python examples/step_budget.py

The router sees a step at 80% token utilization, downshifts to a cheaper model,
then the hard guard blocks a request that still cannot fit in the step.
"""

from __future__ import annotations

from floe_guard import BudgetGuard, ManualPrice, TokenBudgetExceeded

PRICE = ManualPrice(input_cost_per_token=1e-6, output_cost_per_token=2e-6)


def main() -> None:
    guard = BudgetGuard(limit_usd=1.00, token_limit=1_000)

    with guard.step(max_usd=0.10, max_tokens=100) as step:
        # Work already completed in this agent step.
        step.record("demo-model", 60, 20, price=PRICE)
        advisory = step.advisory()
        model = "flash-model" if advisory.near_limit else "premium-model"
        print(
            f"step tokens: {advisory.step_spent_tokens}/"
            f"{advisory.step_token_limit}; router chose {model}"
        )

        try:
            step.check(estimated_next_cost=0.001, estimated_next_tokens=25)
        except TokenBudgetExceeded as exc:
            print(exc)
            print("provider was not called")

    print(f"aggregate tokens remain recorded: {guard.spent_tokens}")


if __name__ == "__main__":
    main()
