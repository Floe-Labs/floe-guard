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
class _PromptTokensDetails:
    cached_tokens: int | None = None


@dataclass
class _Usage:
    prompt_tokens: int
    completion_tokens: int
    prompt_tokens_details: _PromptTokensDetails | None = None


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
    assert _usage_from({"usage": {"prompt_tokens": 5, "completion_tokens": 7}}) == (5, 7, 0)
    assert _usage_from(_ObjResponse("gpt-4o", _Usage(5, 7))) == (5, 7, 0)


def test_cached_tokens_are_carved_out_of_the_prompt_not_added() -> None:
    # prompt_tokens INCLUDES the cached share, so it is subtracted and re-priced
    # at the cheaper cache-read rate. Charging both would bill it twice.
    usage = _Usage(1_000, 40, _PromptTokensDetails(cached_tokens=800))
    assert _usage_from(_ObjResponse("gpt-4o", usage)) == (200, 40, 800)
    assert _usage_from(
        {
            "usage": {
                "prompt_tokens": 1_000,
                "completion_tokens": 40,
                "prompt_tokens_details": {"cached_tokens": 800},
            }
        }
    ) == (200, 40, 800)


def test_missing_or_null_cached_tokens_read_as_zero() -> None:
    # A provider with no cache hit omits the block, or carries it with no count.
    assert _usage_from(_ObjResponse("gpt-4o", _Usage(10, 5, None))) == (10, 5, 0)
    assert _usage_from(_ObjResponse("gpt-4o", _Usage(10, 5, _PromptTokensDetails()))) == (10, 5, 0)


def test_cached_tokens_capped_at_prompt_tokens() -> None:
    # cached_tokens exceeding prompt_tokens must be capped at prompt, so we never
    # price more input than the provider reported (zero fresh, cached == prompt).
    usage = _Usage(1_000, 5, _PromptTokensDetails(cached_tokens=1_200))
    assert _usage_from(_ObjResponse("gpt-4o", usage)) == (0, 5, 1_000)
    assert _usage_from(
        {
            "usage": {
                "prompt_tokens": 1_000,
                "completion_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 1_200},
            }
        }
    ) == (0, 5, 1_000)


def test_cached_tokens_are_priced_at_the_cache_read_rate() -> None:
    guard = BudgetGuard(limit_usd=1.0)
    resp = {
        "model": "gpt-4o",
        "usage": {
            "prompt_tokens": 1_000,
            "completion_tokens": 0,
            "prompt_tokens_details": {"cached_tokens": 800},
        },
    }
    _record_response(guard, {}, resp)
    # 200 fresh input at 2.5e-6, plus 800 cached at gpt-4o's published 1.25e-6
    # cache-read rate — NOT 1000 at the full input rate.
    assert guard.spent_usd == pytest.approx(200 * 2.5e-06 + 800 * 1.25e-06)


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
