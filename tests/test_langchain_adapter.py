"""LangChain adapter tests.

The parsing helpers (``_model_from_result``/``_usage_from_result``/
``_record_result``) are duck-typed over an ``LLMResult`` and need no langchain
install, so they run in CI without the optional extra. The two handler tests
call the factory, which hard-imports ``langchain_core`` — they are skipped
unless that extra is installed. When available they exercise the real callback
and prove the hard-stop: a call under budget is allowed and accrued, and a call
that would cross the ceiling raises ``BudgetExceeded`` in ``on_llm_start`` —
before the call runs.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import pytest

from floe_guard import (
    BudgetExceeded,
    BudgetGuard,
    TokenBudgetExceeded,
    UnpriceableModelError,
    UnpriceableModelWarning,
)
from floe_guard.integrations.langchain import (
    _estimate_start_tokens,
    _model_from_result,
    _record_result,
    _usage_from_result,
    budget_guard_callback_handler,
)


@dataclass
class _Msg:
    usage_metadata: dict


@dataclass
class _Gen:
    message: _Msg


@dataclass
class _Result:
    """Stand-in for a LangChain ``LLMResult``."""

    llm_output: dict | None = None
    generations: list = field(default_factory=list)


def _openai_result(prompt: int, completion: int, model: str = "gpt-4o") -> _Result:
    return _Result(
        llm_output={
            "model_name": model,
            "token_usage": {"prompt_tokens": prompt, "completion_tokens": completion},
        }
    )


def test_start_token_estimate_keeps_known_prompt_without_model_kwargs() -> None:
    assert _estimate_start_tokens({}, ["abcdefgh"]) == 2
    assert _estimate_start_tokens(None, ["abcdefgh"]) == 2
    assert _estimate_start_tokens({"kwargs": "invalid"}, ["abcdefgh"]) == 2


def test_start_token_estimate_without_known_tokens_remains_unknown() -> None:
    assert _estimate_start_tokens({}, []) is None
    assert _estimate_start_tokens({}, [object()]) is None  # type: ignore[list-item]


def test_start_token_estimate_adds_valid_output_cap() -> None:
    serialized = {"kwargs": {"max_completion_tokens": 7}}
    assert _estimate_start_tokens(serialized, ["abcdefgh"]) == 9


def test_model_from_result_reads_llm_output() -> None:
    assert _model_from_result(_openai_result(1, 1)) == "gpt-4o"


def test_model_from_result_missing_is_empty() -> None:
    assert _model_from_result(_Result(llm_output=None)) == ""


def test_usage_from_token_usage_block() -> None:
    assert _usage_from_result(_openai_result(5, 7)) == (5, 7)


def test_usage_from_usage_metadata_fallback() -> None:
    # No token_usage in llm_output — fall back to per-message usage_metadata.
    result = _Result(
        llm_output={"model_name": "gpt-4o"},
        generations=[[_Gen(_Msg({"input_tokens": 5, "output_tokens": 7}))]],
    )
    assert _usage_from_result(result) == (5, 7)


def test_record_result_accrues() -> None:
    guard = BudgetGuard(limit_usd=1.0)
    _record_result(guard, _openai_result(1_000, 1_000))
    assert guard.spent_usd == pytest.approx(0.0125)


def test_usage_present_but_model_missing_fails_closed() -> None:
    # Tokens were spent but the model id is missing. This MUST go through record()
    # (fail-closed → raise), not be silently skipped unmetered.
    guard = BudgetGuard(limit_usd=1.0)  # fail_closed defaults to True
    result = _Result(
        llm_output={"token_usage": {"prompt_tokens": 1_000, "completion_tokens": 1_000}}
    )
    with pytest.warns(UnpriceableModelWarning):
        with pytest.raises(UnpriceableModelError):
            _record_result(guard, result)
    assert guard.spent_usd == 0.0


def test_usage_present_but_model_missing_fail_open_warns_and_skips() -> None:
    guard = BudgetGuard(limit_usd=1.0, fail_closed=False)
    result = _Result(
        llm_output={"token_usage": {"prompt_tokens": 1_000, "completion_tokens": 1_000}}
    )
    with pytest.warns(UnpriceableModelWarning):
        _record_result(guard, result)
    assert guard.spent_usd == 0.0


def test_no_usage_response_is_a_noop() -> None:
    guard = BudgetGuard(limit_usd=1.0)
    _record_result(guard, _Result(llm_output={"model_name": "gpt-4o"}))
    _record_result(guard, _openai_result(0, 0))
    assert guard.spent_usd == 0.0


def test_handler_allows_under_budget_and_records() -> None:
    pytest.importorskip("langchain_core")
    guard = BudgetGuard(limit_usd=1.0)
    handler = budget_guard_callback_handler(guard)
    run_id = uuid4()

    handler.on_llm_start({}, ["hello"], run_id=run_id)  # under budget — no raise
    handler.on_llm_end(_openai_result(1_000, 1_000), run_id=run_id)
    assert guard.spent_usd == pytest.approx(0.0125)


def test_handler_blocks_before_crossing() -> None:
    pytest.importorskip("langchain_core")
    # First call costs 0.0125 and primes _last_cost; the next call's projection
    # (0.025) crosses the 0.02 ceiling, so on_llm_start raises BEFORE it runs.
    guard = BudgetGuard(limit_usd=0.02)
    handler = budget_guard_callback_handler(guard)

    first_run = uuid4()
    handler.on_llm_start({}, ["hello"], run_id=first_run)
    handler.on_llm_end(_openai_result(1_000, 1_000), run_id=first_run)
    assert guard.spent_usd == pytest.approx(0.0125)

    with pytest.raises(BudgetExceeded):
        handler.on_llm_start({}, ["hello"], run_id=uuid4())


def test_parallel_handler_starts_reserve_token_ceiling_atomically() -> None:
    pytest.importorskip("langchain_core")
    guard = BudgetGuard(
        limit_usd=1.0,
        token_limit=100,
        on_token_block=lambda *_: None,
    )
    handler = budget_guard_callback_handler(guard)
    serialized = {"kwargs": {"model_name": "gpt-4o", "max_tokens": 60}}
    run_ids = [uuid4(), uuid4()]

    def start(run_id: UUID) -> tuple[UUID, str]:
        try:
            handler.on_llm_start(serialized, [""], run_id=run_id)
        except TokenBudgetExceeded:
            return run_id, "blocked"
        return run_id, "held"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(start, run_ids))

    assert sorted(result for _, result in results) == ["blocked", "held"]
    held_run = next(run_id for run_id, result in results if result == "held")
    handler.on_llm_error(RuntimeError("provider failed"), run_id=held_run)
    assert guard.remaining_tokens == 100


def test_parallel_handler_starts_reserve_usd_ceiling_atomically() -> None:
    pytest.importorskip("langchain_core")
    guard = BudgetGuard(limit_usd=0.001, on_block=lambda *_: None)
    handler = budget_guard_callback_handler(guard)
    serialized = {"kwargs": {"model_name": "gpt-4o", "max_tokens": 60}}
    run_ids = [uuid4(), uuid4()]

    def start(run_id: UUID) -> tuple[UUID, str]:
        try:
            handler.on_llm_start(serialized, [""], run_id=run_id)
        except BudgetExceeded:
            return run_id, "blocked"
        return run_id, "held"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(start, run_ids))

    assert sorted(result for _, result in results) == ["blocked", "held"]
    held_run = next(run_id for run_id, result in results if result == "held")
    handler.on_llm_error(RuntimeError("provider failed"), run_id=held_run)
    assert guard.remaining_usd == pytest.approx(guard.limit_usd)


def test_parallel_handler_runs_settle_their_own_reservations() -> None:
    pytest.importorskip("langchain_core")
    guard = BudgetGuard(limit_usd=1.0, token_limit=200)
    handler = budget_guard_callback_handler(guard)
    serialized = {"kwargs": {"model_name": "gpt-4o", "max_tokens": 60}}
    runs = [(uuid4(), _openai_result(20, 10)), (uuid4(), _openai_result(25, 15))]

    for run_id, _ in runs:
        handler.on_llm_start(serialized, [""], run_id=run_id)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(
            pool.map(
                lambda item: handler.on_llm_end(item[1], run_id=item[0]),
                runs,
            )
        )

    assert guard.spent_tokens == 70
    assert guard.remaining_tokens == 130


def test_handler_releases_usage_less_and_failed_runs() -> None:
    pytest.importorskip("langchain_core")
    guard = BudgetGuard(limit_usd=1.0, token_limit=100)
    handler = budget_guard_callback_handler(guard)
    serialized = {"kwargs": {"model_name": "gpt-4o", "max_tokens": 40}}

    usage_less_run = uuid4()
    handler.on_llm_start(serialized, [""], run_id=usage_less_run)
    handler.on_llm_end(_Result(llm_output={"model_name": "gpt-4o"}), run_id=usage_less_run)
    assert guard.remaining_tokens == 100

    failed_run = uuid4()
    handler.on_llm_start(serialized, [""], run_id=failed_run)
    handler.on_llm_error(RuntimeError("provider failed"), run_id=failed_run)
    assert guard.remaining_tokens == 100


def test_handler_releases_reservation_when_usage_is_malformed() -> None:
    pytest.importorskip("langchain_core")
    guard = BudgetGuard(limit_usd=1.0, token_limit=100)
    handler = budget_guard_callback_handler(guard)
    serialized = {"kwargs": {"model_name": "gpt-4o", "max_tokens": 40}}
    run_id = uuid4()
    malformed = _Result(
        llm_output={
            "model_name": "gpt-4o",
            "token_usage": {"prompt_tokens": "bad", "completion_tokens": 1},
        }
    )

    handler.on_llm_start(serialized, [""], run_id=run_id)
    with pytest.raises(ValueError):
        handler.on_llm_end(malformed, run_id=run_id)
    assert guard.remaining_tokens == 100


def test_handler_rejects_duplicate_active_run_without_leaking() -> None:
    pytest.importorskip("langchain_core")
    guard = BudgetGuard(limit_usd=1.0, token_limit=100)
    handler = budget_guard_callback_handler(guard)
    serialized = {"kwargs": {"model_name": "gpt-4o", "max_tokens": 40}}
    run_id = uuid4()

    handler.on_llm_start(serialized, [""], run_id=run_id)
    with pytest.raises(RuntimeError, match="duplicate LangChain run_id"):
        handler.on_llm_start(serialized, [""], run_id=run_id)
    assert guard.remaining_tokens == 60

    handler.on_llm_error(RuntimeError("provider failed"), run_id=run_id)
    assert guard.remaining_tokens == 100


def test_handler_enforces_scoped_step_ceiling() -> None:
    pytest.importorskip("langchain_core")
    guard = BudgetGuard(limit_usd=1.0, token_limit=1_000, on_token_block=lambda *_: None)
    serialized = {"kwargs": {"model_name": "gpt-4o", "max_tokens": 51}}

    with guard.step(max_tokens=50) as step:
        handler = budget_guard_callback_handler(step)
        with pytest.raises(TokenBudgetExceeded, match=r"STEP TOKEN BUDGET EXCEEDED"):
            handler.on_llm_start(serialized, [""], run_id=uuid4())
        assert step.advisory().step_remaining_tokens == 50


def test_handler_end_or_error_without_start_uses_unreserved_path() -> None:
    pytest.importorskip("langchain_core")
    guard = BudgetGuard(limit_usd=1.0, token_limit=100)
    handler = budget_guard_callback_handler(guard)

    handler.on_llm_end(_openai_result(5, 7), run_id=uuid4())
    handler.on_llm_error(RuntimeError("provider failed"), run_id=uuid4())

    assert guard.spent_tokens == 12
    assert guard.remaining_tokens == 88
