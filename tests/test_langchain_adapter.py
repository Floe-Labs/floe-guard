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

import sys
import threading
import types
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


def test_handler_reserves_atomically_for_parallel_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    # Exercise the callback without requiring the optional LangChain dependency.
    core = types.ModuleType("langchain_core")
    callbacks = types.ModuleType("langchain_core.callbacks")
    callbacks.BaseCallbackHandler = type("BaseCallbackHandler", (), {})
    core.callbacks = callbacks
    monkeypatch.setitem(sys.modules, "langchain_core", core)
    monkeypatch.setitem(sys.modules, "langchain_core.callbacks", callbacks)

    limit = 0.01875
    guard = BudgetGuard(limit_usd=limit)
    handler = budget_guard_callback_handler(guard)
    serialized = {"kwargs": {"model": "gpt-4o", "max_tokens": 1_000}}
    prompts = ["x" * 4_000]  # ~1,000 prompt tokens; estimate is $0.0125.
    response = _openai_result(1_000, 1_000)

    start_barrier = threading.Barrier(2)
    release_providers = threading.Event()
    both_admissions_finished = threading.Event()
    lock = threading.Lock()
    provider_calls: list[str] = []
    blocked_runs: list[str] = []
    unexpected_errors: list[BaseException] = []
    admissions_finished = 0

    def run(run_id: str) -> None:
        nonlocal admissions_finished
        start_barrier.wait()
        admitted = False
        try:
            handler.on_llm_start(serialized, prompts, run_id=run_id)
            admitted = True
            with lock:
                provider_calls.append(run_id)
        except BudgetExceeded:
            with lock:
                blocked_runs.append(run_id)
        except BaseException as error:  # surface thread failures in the main test
            with lock:
                unexpected_errors.append(error)
        finally:
            with lock:
                admissions_finished += 1
                if admissions_finished == 2:
                    both_admissions_finished.set()

        if admitted:
            if not release_providers.wait(timeout=5):
                with lock:
                    unexpected_errors.append(AssertionError("provider gate timed out"))
                return
            try:
                handler.on_llm_end(response, run_id=run_id)
            except BaseException as error:
                with lock:
                    unexpected_errors.append(error)

    workers = [threading.Thread(target=run, args=(f"run-{index}",)) for index in range(2)]
    for worker in workers:
        worker.start()

    assert both_admissions_finished.wait(timeout=5)
    release_providers.set()
    for worker in workers:
        worker.join(timeout=5)

    assert all(not worker.is_alive() for worker in workers)
    assert unexpected_errors == []
    assert len(provider_calls) == 1
    assert len(blocked_runs) == 1
    assert set(provider_calls).isdisjoint(blocked_runs)
    assert len(set(provider_calls + blocked_runs)) == 2
    assert guard.spent_usd == pytest.approx(0.0125)
    assert guard.remaining_usd == pytest.approx(limit - 0.0125)


def test_handler_error_settles_or_releases_only_its_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = types.ModuleType("langchain_core")
    callbacks = types.ModuleType("langchain_core.callbacks")
    callbacks.BaseCallbackHandler = type("BaseCallbackHandler", (), {})
    core.callbacks = callbacks
    monkeypatch.setitem(sys.modules, "langchain_core", core)
    monkeypatch.setitem(sys.modules, "langchain_core.callbacks", callbacks)

    guard = BudgetGuard(limit_usd=0.04)
    handler = budget_guard_callback_handler(guard)
    serialized = {"kwargs": {"model": "gpt-4o", "max_tokens": 1_000}}
    prompts = ["x" * 4_000]

    for run_id in ("run-a", "run-b"):
        handler.on_llm_start(serialized, prompts, run_id=run_id)
    assert guard.remaining_usd == pytest.approx(0.015)

    # A failed call with no usage releases A's hold, leaving B's hold untouched.
    handler.on_llm_error(RuntimeError("provider failed"), run_id="run-a")
    assert guard.spent_usd == 0.0
    assert guard.remaining_usd == pytest.approx(0.0275)

    # If an error includes billable partial usage, settle that run instead.
    handler.on_llm_error(
        RuntimeError("stream failed after usage"),
        response=_openai_result(1_000, 1_000),
        run_id="run-b",
    )
    assert guard.spent_usd == pytest.approx(0.0125)
    assert guard.remaining_usd == pytest.approx(0.0275)
    assert len(guard.spend_log) == 1

    for run_id in ("run-c", "run-d"):
        handler.on_llm_start(serialized, prompts, run_id=run_id)
    malformed_usage = _Result(
        llm_output={
            "model_name": "gpt-4o",
            "token_usage": {"prompt_tokens": "not-an-integer", "completion_tokens": 1},
        }
    )
    with pytest.raises(ValueError):
        handler.on_llm_end(malformed_usage, run_id="run-c")
    assert guard.remaining_usd == pytest.approx(0.015)

    # Parse failure releases C, while D remains held until its own completion.
    handler.on_llm_end(_openai_result(1_000, 1_000), run_id="run-d")
    assert guard.spent_usd == pytest.approx(0.025)
    assert guard.remaining_usd == pytest.approx(0.015)

    # A duplicate terminal callback cannot settle or release the same run again.
    handler.on_llm_end(_openai_result(1_000, 1_000), run_id="run-b")
    handler.on_llm_error(RuntimeError("duplicate"), run_id="run-b")
    assert guard.spent_usd == pytest.approx(0.025)
    assert guard.remaining_usd == pytest.approx(0.015)
    assert len(guard.spend_log) == 2
