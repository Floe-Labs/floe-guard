"""Tests for token ceilings and per-step budgets (issue #46).

The feature is a second dimension (tokens) on the existing enforcement choke
point plus a step scope — so these tests also pin the backward-compat contract:
an aggregate-only USD guard behaves byte-for-byte as before, including
``reserve()`` still returning a plain float.
"""

from __future__ import annotations

import pytest

from floe_guard import (
    BudgetExceeded,
    BudgetGuard,
    BudgetReservation,
    TokenBudgetExceeded,
)

MODEL = "gpt-4o"  # $2.5e-6/input token, $1e-5/output token


# ── aggregate token ceiling ─────────────────────────────────────────────────────


def test_aggregate_token_limit_hard_blocks_when_crossing() -> None:
    guard = BudgetGuard(limit_usd=100.0, token_limit=1_000, on_block=lambda *_: None)
    guard.record(MODEL, 400, 200)  # 600 tokens accrued
    # 600 already spent; a 500-token call would cross the 1_000 ceiling.
    with pytest.raises(TokenBudgetExceeded) as exc:
        guard.check(estimated_tokens=500)
    assert exc.value.scope == "aggregate"
    assert exc.value.limit_tokens == 1_000


def test_aggregate_token_reserve_holds_and_blocks() -> None:
    guard = BudgetGuard(limit_usd=100.0, token_limit=1_000, on_block=lambda *_: None)
    handle = guard.reserve(0.01, estimated_tokens=900)
    assert isinstance(handle, BudgetReservation)
    assert handle.tokens == 900
    # 900 held in-flight; another 200 would cross → blocked without reserving.
    with pytest.raises(TokenBudgetExceeded):
        guard.reserve(0.01, estimated_tokens=200)


def test_tokens_accrue_from_all_billed_buckets() -> None:
    guard = BudgetGuard(limit_usd=100.0, token_limit=10_000)
    guard.record(
        MODEL,
        100,
        50,
        cache_creation_input_tokens=10,
        cache_read_input_tokens=5,
    )
    assert guard.spent_tokens == 100 + 50 + 10 + 5


def test_token_limit_validation_rejects_bool_and_negative() -> None:
    with pytest.raises(ValueError):
        BudgetGuard(limit_usd=1.0, token_limit=True)  # bool is not a valid int here
    with pytest.raises(ValueError):
        BudgetGuard(limit_usd=1.0, token_limit=-1)


# ── per-step caps ────────────────────────────────────────────────────────────────


def test_step_token_cap_hard_blocks() -> None:
    guard = BudgetGuard(limit_usd=100.0, on_block=lambda *_: None)  # ample aggregate
    with guard.step(max_tokens=500) as g:
        assert g is guard  # yields the SAME guard — no adapter needs to know
        g.record(MODEL, 300, 100)  # 400 tokens into the step
        with pytest.raises(TokenBudgetExceeded) as exc:
            g.check(estimated_tokens=200)  # 400 + 200 > 500
    assert exc.value.scope == "step"
    assert exc.value.limit_tokens == 500


def test_step_usd_cap_hard_blocks() -> None:
    guard = BudgetGuard(limit_usd=100.0, on_block=lambda *_: None)
    with guard.step(max_usd=0.01) as g:
        with pytest.raises(BudgetExceeded) as exc:
            g.reserve(0.02)  # one call alone exceeds the step's $0.01
    # A step USD block is a plain BudgetExceeded (scope lives on the token error).
    assert not isinstance(exc.value, TokenBudgetExceeded)


def test_step_cap_frees_on_exit() -> None:
    guard = BudgetGuard(limit_usd=100.0, on_block=lambda *_: None)
    with guard.step(max_tokens=100) as g:
        with pytest.raises(TokenBudgetExceeded):
            g.check(estimated_tokens=200)
    # Outside the step, only the (absent) aggregate token ceiling applies.
    guard.check(estimated_tokens=1_000_000)  # does not raise


def test_step_reservation_returns_budget_reservation() -> None:
    guard = BudgetGuard(limit_usd=100.0)
    with guard.step(max_usd=1.0) as g:
        handle = g.reserve(0.02)  # USD-only, but a step is active
        assert isinstance(handle, BudgetReservation)
        g.settle(MODEL, 100, 100, reserved=handle)


def test_nested_steps_innermost_blocks_first() -> None:
    guard = BudgetGuard(limit_usd=100.0, on_block=lambda *_: None)
    with guard.step(max_tokens=10_000):
        with guard.step(max_tokens=200) as inner:
            with pytest.raises(TokenBudgetExceeded) as exc:
                inner.check(estimated_tokens=500)
    assert exc.value.limit_tokens == 200  # the inner cap, not the outer


# ── advisory ─────────────────────────────────────────────────────────────────────


def test_advisory_reports_token_utilization() -> None:
    guard = BudgetGuard(limit_usd=100.0, token_limit=1_000)
    guard.record(MODEL, 500, 100)  # 600 / 1000 = 60%
    adv = guard.advisory()
    assert adv.token_used_bps == 6000
    assert adv.remaining_tokens == 400


def test_advisory_step_near_limit_fires_before_hard_block() -> None:
    # near_limit_bps=8000 → advisory flags near at 80% of the step cap, BEFORE
    # the hard block at 100%, so a router can downshift.
    guard = BudgetGuard(limit_usd=100.0, near_limit_bps=8000)
    with guard.step(max_tokens=1_000) as g:
        g.record(MODEL, 500, 300)  # 800 / 1000 = 80%
        adv = g.advisory()
        assert adv.near_limit is True
        assert adv.step_remaining_tokens == 200
        # Still under the cap — a fitting call is NOT blocked yet.
        g.check(estimated_tokens=100)


def test_advisory_no_token_fields_when_dimension_unused() -> None:
    guard = BudgetGuard(limit_usd=1.0)
    adv = guard.advisory()
    assert adv.token_used_bps is None
    assert adv.remaining_tokens is None
    assert adv.step_remaining_usd is None
    assert adv.step_remaining_tokens is None


# ── backward-compat regression: aggregate-only USD is unchanged ──────────────────


def test_usd_only_reserve_still_returns_plain_float() -> None:
    # THE regression guard: with no token_limit and no step, reserve() returns a
    # plain float exactly as before — old callers are byte-for-byte unchanged.
    guard = BudgetGuard(limit_usd=1.0)
    handle = guard.reserve(0.0125)
    assert type(handle) is float
    assert handle == pytest.approx(0.0125)
    guard.settle(MODEL, 1_000, 1_000, reserved=handle)
    assert guard.spent_usd == pytest.approx(0.0125)


def test_usd_only_default_reserve_returns_float_zero() -> None:
    guard = BudgetGuard(limit_usd=1.0, on_block=lambda *_: None)
    assert guard.reserve() == 0.0  # first-call default, plain float


def test_usd_only_advisory_shape_unchanged() -> None:
    guard = BudgetGuard(limit_usd=1.0)
    guard.record(MODEL, 100, 100)
    adv = guard.advisory()
    # The pre-existing fields still populate; the new ones stay None.
    assert adv.spent_usd == pytest.approx(guard.spent_usd)
    assert adv.token_used_bps is None
    assert adv.step_remaining_usd is None


def test_token_block_is_terminal_like_budget_exceeded() -> None:
    # Retry logic treats a block as terminal via isinstance(exc, BudgetExceeded);
    # TokenBudgetExceeded subclasses it, so a token block is terminal too.
    assert issubclass(TokenBudgetExceeded, BudgetExceeded)
    err = TokenBudgetExceeded(600, 500, "step")
    assert isinstance(err, BudgetExceeded)


# ── reservation handle lifecycle ────────────────────────────────────────────────


def test_reservation_settled_after_owning_step_exits() -> None:
    # A BudgetReservation may outlive the step() that created it. Settling it once
    # the step has exited must drain the aggregate token hold (there's no active
    # step to drain) and accrue the actual counts — not raise.
    guard = BudgetGuard(limit_usd=100.0, token_limit=20_000)
    with guard.step(max_tokens=5_000) as g:
        handle = g.reserve(estimated_tokens=1_000)
        assert isinstance(handle, BudgetReservation)
    # Step popped; the aggregate hold is still outstanding until we settle it.
    guard.settle(MODEL, 400, 300, reserved=handle)
    assert guard._reserved_tokens == 0  # hold fully drained, no phantom left behind
    assert guard.spent_tokens == 700


def test_stream_guard_accepts_token_reservation() -> None:
    # Regression: a token-aware BudgetReservation (from a token/step-aware
    # reserve()) must be accepted by StreamGuard — it used to crash on a
    # float-only isfinite() check before the stream even started.
    from floe_guard import StreamGuard

    guard = BudgetGuard(limit_usd=100.0, token_limit=20_000)
    handle = guard.reserve(0.01, estimated_tokens=500)
    assert isinstance(handle, BudgetReservation)
    with StreamGuard(guard, MODEL, prompt_tokens=100, reserved=handle) as sg:
        sg.feed_tokens(50)
        sg.finish(completion_tokens=50)
    # settle() drained the token hold and accrued prompt+completion (100 + 50).
    assert guard._reserved_tokens == 0
    assert guard.spent_tokens == 150


def test_budget_reservation_rejects_bad_fields() -> None:
    # Public, re-exported value object: a hand-rolled reservation with a NaN usd
    # or negative/bool tokens must be refused at construction, not silently
    # corrupt _reserved / _reserved_tokens when passed to settle()/release().
    with pytest.raises(ValueError):
        BudgetReservation(usd=float("nan"), tokens=0)
    with pytest.raises(ValueError):
        BudgetReservation(usd=-1.0, tokens=0)
    with pytest.raises(ValueError):
        BudgetReservation(usd=0.0, tokens=-5)
    with pytest.raises(ValueError):
        BudgetReservation(usd=0.0, tokens=True)  # bool is not a valid token count
