"""LangChain adapter tests for ChatGroq's token-usage shape.

ChatGroq surfaces token counts via ``usage_metadata`` on the message
(``input_tokens`` / ``output_tokens``) rather than the ``token_usage`` block
OpenAI uses. These tests confirm the adapter's existing fallback handles it
and verify the full allow → record → block lifecycle.

Parsing helper tests run without a real API key or langchain install.
Handler tests require ``langchain_core`` and are skipped without it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from floe_guard import BudgetExceeded, BudgetGuard
from floe_guard.integrations.langchain import (
    _model_from_result,
    _record_result,
    _usage_from_result,
    budget_guard_callback_handler,
)

# Minimal stubs that mirror ChatGroq's LLMResult shape

@dataclass
class _Msg:
    usage_metadata: dict


@dataclass
class _Gen:
    message: _Msg


@dataclass
class _Result:
    llm_output: dict | None = None
    generations: list = field(default_factory=list)


def _groq_result(
    input_tokens: int, output_tokens: int, model: str = "llama-3.1-8b-instant"
) -> _Result:
    """Build a stub LLMResult in the shape ChatGroq emits.

    ChatGroq sets ``model`` (not ``model_name``) in ``llm_output`` and surfaces
    token counts via ``usage_metadata`` on the message rather than via a
    ``token_usage`` block — the provider-agnostic shape.
    """
    return _Result(
        llm_output={"model": model},
        generations=[
            [_Gen(_Msg({"input_tokens": input_tokens, "output_tokens": output_tokens}))]
        ],
    )


# Parsing helpers — no langchain install required


def test_model_from_groq_result() -> None:
    # ChatGroq uses the key "model", not "model_name".
    assert _model_from_result(_groq_result(5, 7)) == "llama-3.1-8b-instant"


def test_usage_from_groq_result_via_usage_metadata() -> None:
    # Token counts come from usage_metadata, not token_usage.
    assert _usage_from_result(_groq_result(5, 7)) == (5, 7)


def test_record_groq_result_accrues() -> None:
    guard = BudgetGuard(limit_usd=1.0)
    # llama-3.1-8b-instant: $0.05/M prompt, $0.08/M completion
    _record_result(guard, _groq_result(1_000, 1_000))
    assert guard.spent_usd > 0.0


def test_zero_usage_groq_result_is_noop() -> None:
    guard = BudgetGuard(limit_usd=1.0)
    _record_result(guard, _groq_result(0, 0))
    assert guard.spent_usd == 0.0



# Full handler lifecycle — requires langchain_core


def test_handler_allows_groq_call_under_budget_and_records() -> None:
    pytest.importorskip("langchain_core")
    guard = BudgetGuard(limit_usd=1.0)
    handler = budget_guard_callback_handler(guard)

    handler.on_chat_model_start({}, [[]])  # under budget — no raise
    handler.on_llm_end(_groq_result(1_000, 1_000))
    assert guard.spent_usd > 0.0


def test_handler_blocks_groq_call_before_crossing() -> None:
    pytest.importorskip("langchain_core")
    guard = BudgetGuard(limit_usd=0.0002)
    handler = budget_guard_callback_handler(guard)

    # First call goes through and primes _last_cost.
    handler.on_chat_model_start({}, [[]])
    handler.on_llm_end(_groq_result(1_000, 1_000))
    assert guard.spent_usd > 0.0

    # Second call's projected cost would cross the ceiling.
    with pytest.raises(BudgetExceeded):
        handler.on_chat_model_start({}, [[]])
