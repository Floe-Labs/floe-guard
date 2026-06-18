"""Adapter-internal tests that need no litellm install.

These exercise the response-parsing helpers directly (the parts that decide
whether a call gets accrued), so the HIGH-severity dict-response path is covered
even in CI without the optional extra.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from floe_guard import BudgetGuard, UnpriceableModelError, UnpriceableModelWarning
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


def test_usage_present_but_model_missing_fails_closed() -> None:
    # The Major fix: tokens were spent but the model id is missing. This MUST go
    # through record() (fail-closed → raise), not be silently skipped unmetered.
    guard = BudgetGuard(limit_usd=1.0)  # fail_closed defaults to True
    resp = {"usage": {"prompt_tokens": 1_000, "completion_tokens": 1_000}}
    with pytest.warns(UnpriceableModelWarning):
        with pytest.raises(UnpriceableModelError):
            _record_response(guard, {}, resp)
    assert guard.spent_usd == 0.0


def test_usage_present_but_model_missing_fail_open_warns_and_skips() -> None:
    guard = BudgetGuard(limit_usd=1.0, fail_closed=False)
    resp = {"usage": {"prompt_tokens": 1_000, "completion_tokens": 1_000}}
    with pytest.warns(UnpriceableModelWarning):
        _record_response(guard, {}, resp)
    assert guard.spent_usd == 0.0


def test_no_usage_response_is_a_noop() -> None:
    # A genuinely empty (no-usage) response: nothing spent, so no record/raise.
    guard = BudgetGuard(limit_usd=1.0)
    _record_response(guard, {}, {})  # no model, no usage
    _record_response(guard, {}, {"usage": {"prompt_tokens": 0, "completion_tokens": 0}})
    assert guard.spent_usd == 0.0


def test_usageless_response_releases_the_reservation() -> None:
    # A usage-less response must free the in-flight reservation, or the callback
    # path leaks _reserved and remaining_usd shrinks permanently.
    guard = BudgetGuard(limit_usd=1.0)
    base = guard.remaining_usd
    reserved = guard.reserve(0.01)  # explicit estimate (fresh guard has no last cost)
    assert guard.remaining_usd < base  # hold counted against the ceiling
    _record_response(guard, {}, {}, reserved=reserved)  # no usage -> release
    assert guard.spent_usd == 0.0
    assert guard.remaining_usd == pytest.approx(base, abs=1e-9)
    assert guard._reserved == pytest.approx(0.0, abs=1e-9)


def test_record_response_tolerates_non_dict_kwargs() -> None:
    # LiteLLM hooks pass kwargs as Any; a None/non-dict must not crash the
    # metering callback on .get(). The model resolves from the response instead.
    guard = BudgetGuard(limit_usd=1.0)
    resp = {"model": "gpt-4o", "usage": {"prompt_tokens": 1_000, "completion_tokens": 1_000}}
    _record_response(guard, None, resp)  # type: ignore[arg-type]
    assert guard.spent_usd > 0
