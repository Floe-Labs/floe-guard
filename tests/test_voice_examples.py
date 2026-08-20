"""The voice-call-cost examples must run with no API key and no network.

AC1: LiveKit, Pipecat, Vapi, and Retell demos each emit a per-leg breakdown
(STT / LLM / TTS / telephony) summing to a total call cost, priced entirely from
the bundled cost map with no manual rates.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _load(module_name: str):
    sys.path.insert(0, str(EXAMPLES))
    try:
        return importlib.import_module(module_name)
    finally:
        sys.path.remove(str(EXAMPLES))


def _assert_four_legs_sum_to_total(guard, expected_tools: set[str]) -> None:
    legs = {e.model_or_tool for e in guard.spend_log}
    assert legs == expected_tools, legs
    total = sum(e.cost_usd for e in guard.spend_log)
    assert total == pytest.approx(guard.advisory().spent_usd)
    assert total > 0


@pytest.mark.asyncio
async def test_livekit_voice_call_cost_example(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("livekit.agents")
    monkeypatch.delenv("FLOE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    demo = _load("voice_call_cost_livekit")

    guard = await demo.run()

    _assert_four_legs_sum_to_total(
        guard, {"gpt-4o", "livekit-stt", "livekit-tts", "livekit-telephony"}
    )
    assert "OPENAI_API_KEY" not in os.environ


@pytest.mark.asyncio
async def test_pipecat_voice_call_cost_example(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("pipecat")
    monkeypatch.delenv("FLOE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    demo = _load("voice_call_cost_pipecat")

    guard = await demo.run()

    _assert_four_legs_sum_to_total(
        guard, {"gpt-4o", "pipecat-stt", "pipecat-tts", "pipecat-telephony"}
    )
    assert "OPENAI_API_KEY" not in os.environ


@pytest.mark.asyncio
async def test_vapi_voice_call_cost_example(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FLOE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    demo = _load("voice_call_cost_vapi")

    guard = await demo.run()

    _assert_four_legs_sum_to_total(
        guard, {"gpt-4o", "vapi-stt", "vapi-tts", "vapi-telephony"}
    )
    assert "OPENAI_API_KEY" not in os.environ


@pytest.mark.asyncio
async def test_retell_voice_call_cost_example(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FLOE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    demo = _load("voice_call_cost_retell")

    guard = await demo.run()

    _assert_four_legs_sum_to_total(
        guard, {"gpt-4o", "retell-stt", "retell-tts", "retell-telephony"}
    )
    assert "OPENAI_API_KEY" not in os.environ
