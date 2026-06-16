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

from dataclasses import dataclass, field

import pytest

from floe_guard import BudgetExceeded, BudgetGuard, UnpriceableModelError, UnpriceableModelWarning
from floe_guard.integrations.langchain import (
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

    handler.on_llm_start({}, ["hello"])  # under budget — no raise
    handler.on_llm_end(_openai_result(1_000, 1_000))
    assert guard.spent_usd == pytest.approx(0.0125)


def test_handler_blocks_before_crossing() -> None:
    pytest.importorskip("langchain_core")
    # First call costs 0.0125 and primes _last_cost; the next call's projection
    # (0.025) crosses the 0.02 ceiling, so on_llm_start raises BEFORE it runs.
    guard = BudgetGuard(limit_usd=0.02)
    handler = budget_guard_callback_handler(guard)

    handler.on_llm_start({}, ["hello"])
    handler.on_llm_end(_openai_result(1_000, 1_000))
    assert guard.spent_usd == pytest.approx(0.0125)

    with pytest.raises(BudgetExceeded):
        handler.on_llm_start({}, ["hello"])
