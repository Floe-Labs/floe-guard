"""What did this voice call cost? — LiveKit, priced entirely from the cost map.

No API key, no account, no network. A minimal fake ``AgentSession`` + agent (all
``LiveKitBudgetGuard.attach`` touches) drives one turn and emits the STT / LLM /
TTS metrics a real session would, plus a telephony leg. **No manual prices** —
every leg is priced from the bundled cost map by naming its vendor, so floe-guard
answers "what did this call cost" at the $0 tier out of the box.

Run:
    python examples/voice_call_cost_livekit.py
"""

from __future__ import annotations

import asyncio

from livekit.agents.metrics import LLMMetrics, STTMetrics, TTSMetrics

from floe_guard import BudgetGuard
from floe_guard.integrations.livekit import LiveKitBudgetGuard


class _FakeEmitter:
    """A LiveKit component (LLM/STT/TTS plugin) — an event emitter that emits its
    own ``metrics_collected``, which is where the adapter meters usage."""

    def __init__(self) -> None:
        self._handlers: dict = {}

    def on(self, event, cb):
        self._handlers[event] = cb

    def emit(self, event, arg):
        if event in self._handlers:
            self._handlers[event](arg)


class _FakeSession:
    """The tiny event-emitter surface attach() wires onto. The LLM/STT/TTS
    plugins each carry their own ``metrics_collected`` (the session-level event
    is deprecated in livekit-agents 1.5+); ``close`` fires on the session."""

    def __init__(self) -> None:
        self._handlers: dict = {}
        self.llm = _FakeEmitter()
        self.stt = _FakeEmitter()
        self.tts = _FakeEmitter()

    def on(self, event, cb):
        self._handlers[event] = cb

    def emit(self, event, arg):
        self._handlers[event](arg)


class _FakeAgent:
    async def llm_node(self, chat_ctx, tools, model_settings):
        for chunk in ("Sure,", " one moment."):
            yield chunk


def _llm_metrics(prompt_tokens, completion_tokens):
    return LLMMetrics(
        label="llm",
        request_id="r",
        timestamp=0.0,
        duration=0.0,
        ttft=0.0,
        cancelled=False,
        completion_tokens=completion_tokens,
        prompt_tokens=prompt_tokens,
        prompt_cached_tokens=0,
        total_tokens=prompt_tokens + completion_tokens,
        tokens_per_second=0.0,
    )


def _stt_metrics(audio_duration):
    return STTMetrics(
        label="stt",
        request_id="r",
        timestamp=0.0,
        duration=0.0,
        audio_duration=audio_duration,
        streamed=True,
    )


def _tts_metrics(characters_count):
    return TTSMetrics(
        label="tts",
        request_id="r",
        timestamp=0.0,
        ttfb=0.0,
        duration=0.0,
        audio_duration=0.0,
        cancelled=False,
        characters_count=characters_count,
        streamed=True,
    )


async def run() -> BudgetGuard:
    guard = BudgetGuard(limit_usd=1.00)
    budget = LiveKitBudgetGuard(
        guard,
        model="gpt-4o",  # LLM leg — priced from the token cost map
        stt_model="deepgram-nova-3",  # $/sec from the voice map
        tts_model="elevenlabs-flash-v2.5",  # $/1k-chars from the voice map
        telephony="twilio-us-inbound-local",  # $/min from the voice map
    )

    agent, session = _FakeAgent(), _FakeSession()
    budget.attach(session, agent)

    # One turn: caller speaks (STT) → model answers (LLM) → bot speaks (TTS).
    async for _ in agent.llm_node(None, None, None):
        pass
    session.stt.emit("metrics_collected", _stt_metrics(8.0))  # 8s of caller audio
    session.llm.emit("metrics_collected", _llm_metrics(600, 220))
    session.tts.emit("metrics_collected", _tts_metrics(180))  # 180 chars spoken

    # Telephony is per-minute accrual driven by the transport — a 1.5-minute call.
    budget.meter_telephony(1.5)
    return guard


def _print_breakdown(guard: BudgetGuard) -> None:
    print("Per-leg call cost (all priced from the bundled cost map, no manual rates):")
    for event in guard.spend_log:
        print(f"  {event.model_or_tool:<20} ${event.cost_usd:.6f}")
    print(f"  {'TOTAL':<20} ${guard.advisory().spent_usd:.6f}")


async def main() -> None:
    guard = await run()
    _print_breakdown(guard)


if __name__ == "__main__":
    asyncio.run(main())
