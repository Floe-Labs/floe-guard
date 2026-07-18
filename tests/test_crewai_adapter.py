"""CrewAI adapter tests — the callback-swallowing footgun must stay closed.

LiteLLM runs custom-logger hooks inside ``except Exception`` (verified on
litellm 1.91.x), so an enforcement error raised inside the callback can be
swallowed and a crew keeps running unmetered. These tests simulate exactly that
swallowing and assert the ``budget_guarded_llm`` call path still hard-stops.

Needs litellm (the callback is a real ``CustomLogger``); crewai itself is
stubbed so the suite doesn't depend on the heavy extra.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

litellm = pytest.importorskip("litellm")

from floe_guard import (  # noqa: E402
    BudgetExceeded,
    BudgetGuard,
    UnpriceableModelError,
    UnpriceableModelWarning,
)
from floe_guard.integrations.crewai import budget_guarded_llm, guard_crew  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_litellm_callbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts with no global callbacks and leaks none."""
    monkeypatch.setattr(litellm, "callbacks", [])


@pytest.fixture(autouse=True)
def _stub_crewai(monkeypatch: pytest.MonkeyPatch) -> None:
    """A minimal crewai module: just the LLM class the adapter subclasses.

    Mirrors CrewAI 0.x (no ``__new__`` factory); the 1.x factory behavior has
    its own stub in the factory test below.
    """

    class LLM:
        def __init__(self, model: str, **kwargs: Any) -> None:
            self.model = model
            self.kwargs = kwargs

        def call(self, *args: Any, **kwargs: Any) -> str:
            return "stub-response"

        async def acall(self, *args: Any, **kwargs: Any) -> str:
            return "stub-async-response"

    stub = types.ModuleType("crewai")
    stub.LLM = LLM  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "crewai", stub)


def _swallow(fn: Any, *args: Any) -> None:
    """Invoke a callback hook the way litellm's handler does: eat Exceptions."""
    try:
        fn(*args)
    except Exception:
        pass


def _response(model: str) -> dict[str, Any]:
    return {"model": model, "usage": {"prompt_tokens": 1_000, "completion_tokens": 1_000}}


def test_guard_crew_registers_once_and_returns_the_callback() -> None:
    guard = BudgetGuard(limit_usd=1.0)
    cb1 = guard_crew(guard)
    cb2 = guard_crew(guard)
    assert cb1 is cb2
    assert [cb for cb in litellm.callbacks if getattr(cb, "guard", None) is guard] == [cb1]


def test_unpriceable_model_hard_stops_next_call_despite_swallowed_settle(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The released-0.1.0 footgun: an unpriceable model's fail-closed raise dies
    # inside litellm, spend stays $0, and the crew keeps running. The wrapper
    # must now stop the NEXT call, outside litellm's callback machinery.
    guard = BudgetGuard(limit_usd=1.0)
    llm = budget_guarded_llm(guard, "groq/brand-new-model-2099")
    (cb,) = [cb for cb in litellm.callbacks if getattr(cb, "guard", None) is guard]

    kwargs = {"litellm_call_id": "c1", "model": "groq/brand-new-model-2099"}
    with pytest.warns(UnpriceableModelWarning):
        _swallow(cb.log_success_event, kwargs, _response("groq/brand-new-model-2099"), None, None)

    assert guard.spent_usd == 0.0  # the completed call itself went unmetered
    assert isinstance(cb.tripped, UnpriceableModelError)
    assert any(
        r.name == "floe_guard" and r.levelname == "ERROR" for r in caplog.records
    ), "the swallowed violation must be loud on the logging channel"
    with pytest.raises(UnpriceableModelError):
        llm.call("next step")


def test_ceiling_hard_stops_next_call_despite_swallowed_pre_call_block() -> None:
    # litellm also swallows the pre-call BudgetExceeded, so even a PRICED model
    # never hard-stopped through the callback alone. The wrapper's call-path
    # check must block once accrued spend crosses the ceiling.
    guard = BudgetGuard(limit_usd=0.001)
    llm = budget_guarded_llm(guard, "gpt-4o")
    (cb,) = [cb for cb in litellm.callbacks if getattr(cb, "guard", None) is guard]

    # One completed call accrues past the ceiling (settle never blocks; check does).
    _swallow(cb.log_success_event, {"litellm_call_id": "c1", "model": "gpt-4o"},
             _response("gpt-4o"), None, None)
    assert guard.spent_usd > guard.limit_usd

    # The swallowed pre-call block of the runaway's next attempt.
    _swallow(cb.log_pre_api_call, "gpt-4o", [], {"litellm_call_id": "c2", "model": "gpt-4o"})
    assert isinstance(cb.tripped, BudgetExceeded)

    with pytest.raises(BudgetExceeded):
        llm.call("next step")


def test_wrapper_passes_through_while_under_budget() -> None:
    guard = BudgetGuard(limit_usd=1.0)
    llm = budget_guarded_llm(guard, "gpt-4o")
    assert llm.call("hello") == "stub-response"
    assert llm.model == "gpt-4o"


def test_acall_enforces_like_call() -> None:
    # CrewAI 1.x async crews await acall(); it must not bypass enforcement.
    import asyncio

    guard = BudgetGuard(limit_usd=1.0)
    llm = budget_guarded_llm(guard, "gpt-4o")
    assert asyncio.run(llm.acall("hello")) == "stub-async-response"

    (cb,) = [cb for cb in litellm.callbacks if getattr(cb, "guard", None) is guard]
    cb.tripped = UnpriceableModelError("gpt-4o")
    with pytest.raises(UnpriceableModelError):
        asyncio.run(llm.acall("next step"))


def test_tripped_reraise_does_not_accumulate_traceback() -> None:
    guard = BudgetGuard(limit_usd=1.0)
    llm = budget_guarded_llm(guard, "gpt-4o")
    (cb,) = [cb for cb in litellm.callbacks if getattr(cb, "guard", None) is guard]
    cb.tripped = UnpriceableModelError("gpt-4o")

    def frames() -> int:
        with pytest.raises(UnpriceableModelError) as excinfo:
            llm.call("hi")
        n, tb = 0, excinfo.value.__traceback__
        while tb is not None:
            n, tb = n + 1, tb.tb_next
        return n

    # Re-raising the latched instance must not grow its traceback per call.
    assert frames() == frames()


def test_callback_reset_clears_the_latch() -> None:
    guard = BudgetGuard(limit_usd=1.0)
    llm = budget_guarded_llm(guard, "gpt-4o")
    (cb,) = [cb for cb in litellm.callbacks if getattr(cb, "guard", None) is guard]
    cb.tripped = UnpriceableModelError("gpt-4o")
    with pytest.raises(UnpriceableModelError):
        llm.call("hi")
    cb.reset()
    assert cb.tripped is None
    assert llm.call("hi") == "stub-response"


def test_crewai_1x_new_factory_gets_forced_litellm_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # CrewAI >= 1.0: LLM.__new__ is a factory that can return a native
    # provider-SDK instance, discarding the subclass (and bypassing LiteLLM
    # entirely). The adapter must construct with is_litellm=True so the
    # subclass — and the metering callback — survive.
    class NativeLLM:
        def __init__(self, model: str | None = None, **kwargs: Any) -> None:
            self.model = model

        def call(self, *args: Any, **kwargs: Any) -> str:
            return "native-unguarded"

    class LLM:
        def __new__(cls, *args: Any, **kwargs: Any):  # the 1.x factory
            if not kwargs.get("is_litellm"):
                return NativeLLM(**kwargs)
            return super().__new__(cls)

        def __init__(self, model: str | None = None, **kwargs: Any) -> None:
            self.model = model
            self.kwargs = kwargs

        def call(self, *args: Any, **kwargs: Any) -> str:
            return "stub-response"

        async def acall(self, *args: Any, **kwargs: Any) -> str:
            return "stub-async-response"

    stub = types.ModuleType("crewai")
    stub.LLM = LLM  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "crewai", stub)

    guard = BudgetGuard(limit_usd=0.0)  # $0 ceiling: first call must block
    llm = budget_guarded_llm(guard, "gpt-4o")
    assert type(llm).__name__ == "BudgetGuardedLLM"  # factory did not swap it out
    with pytest.raises(BudgetExceeded):
        llm.call("hi")
