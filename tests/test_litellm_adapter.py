"""Adapter-internal tests that need no litellm install.

These exercise the response-parsing helpers directly (the parts that decide
whether a call gets accrued), so the HIGH-severity dict-response path is covered
even in CI without the optional extra.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from floe_guard import BudgetGuard
from floe_guard.integrations.litellm import _model_from, _record_response, _usage_from


@dataclass
class _Usage:
    prompt_tokens: int
    completion_tokens: int


@dataclass
class _ObjResponse:
    model: str
    usage: _Usage


def test_model_from_object_response_without_kwargs_model() -> None:
    resp = _ObjResponse(model="gpt-4o", usage=_Usage(1, 1))
    assert _model_from({}, resp) == "gpt-4o"


def test_model_from_dict_response_without_kwargs_model() -> None:
    # The regression: a dict response with no kwargs["model"] must still resolve.
    resp = {"model": "gpt-4o", "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    assert _model_from({}, resp) == "gpt-4o"


def test_model_from_prefers_kwargs() -> None:
    resp = {"model": "gpt-4o"}
    assert _model_from({"model": "claude-opus-4-5"}, resp) == "claude-opus-4-5"


def test_usage_from_dict_and_object() -> None:
    assert _usage_from({"usage": {"prompt_tokens": 5, "completion_tokens": 7}}) == (5, 7)
    assert _usage_from(_ObjResponse("gpt-4o", _Usage(5, 7))) == (5, 7)


def test_record_response_accrues_dict_response() -> None:
    # End-to-end of the fix: a dict LiteLLM response with no kwargs model is
    # priced and accrued (previously it was silently skipped → unenforced).
    guard = BudgetGuard(limit_usd=1.0)
    resp = {"model": "gpt-4o", "usage": {"prompt_tokens": 1_000, "completion_tokens": 1_000}}
    _record_response(guard, {}, resp)
    assert guard.spent_usd == pytest.approx(0.0125)
