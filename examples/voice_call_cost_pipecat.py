"""What did this voice call cost? — Pipecat, priced entirely from the cost map.

No API key, no account, no network. Runs a real (single-processor) Pipecat
Pipeline via PipelineTask/PipelineRunner — exactly like
examples/voice_turn_budget.py — so the guard processor gets a fully initialized
lifecycle. One turn's LLM + TTS usage metrics flow through as frames; STT and
telephony (which Pipecat emits no usage frame for) are metered explicitly.

**No manual prices** — every leg is priced from the bundled cost map by naming
its vendor, so floe-guard answers "what did this call cost" at the $0 tier out
of the box.

Run:
    python examples/voice_call_cost_pipecat.py
"""

from __future__ import annotations

import asyncio

from pipecat.frames.frames import EndFrame, LLMFullResponseStartFrame, MetricsFrame
from pipecat.metrics.metrics import (
    LLMTokenUsage,
    LLMUsageMetricsData,
    TTSUsageMetricsData,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask

from floe_guard import BudgetGuard
from floe_guard.integrations.pipecat import FloeBudgetGuardProcessor


def _llm_frame(prompt_tokens, completion_tokens):
    usage = LLMTokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return MetricsFrame(data=[LLMUsageMetricsData(processor="llm", model="gpt-4o", value=usage)])


def _tts_frame(characters):
    return MetricsFrame(data=[TTSUsageMetricsData(processor="tts", value=characters)])


async def run() -> BudgetGuard:
    guard = BudgetGuard(limit_usd=1.00)
    processor = FloeBudgetGuardProcessor(
        guard,
        model="gpt-4o",  # LLM leg — priced from the token cost map
        stt_model="deepgram-nova-3",  # $/sec from the voice map
        tts_model="elevenlabs-flash-v2.5",  # $/1k-chars from the voice map
        telephony="twilio-us-inbound-local",  # $/min from the voice map
    )

    pipeline = Pipeline([processor])
    task = PipelineTask(
        pipeline, params=PipelineParams(enable_metrics=True, enable_usage_metrics=True)
    )
    runner = PipelineRunner()

    async def drive():
        await task.queue_frame(LLMFullResponseStartFrame())
        await task.queue_frame(_llm_frame(600, 220))  # LLM leg (auto)
        await task.queue_frame(_tts_frame(180))  # TTS leg (auto, from usage frame)
        await asyncio.sleep(0.05)
        # STT + telephony have no native usage frame — meter them explicitly.
        processor.meter_stt(8.0)  # 8s of caller audio
        processor.meter_telephony(1.5)  # 1.5-minute call
        if not task.has_finished():
            await task.queue_frame(EndFrame())

    await asyncio.gather(runner.run(task), drive())
    return guard


def _print_breakdown(guard: BudgetGuard) -> None:
    print("Per-leg call cost (all priced from the bundled cost map, no manual rates):")
    for event in guard.spend_log:
        print(f"  {event.model_or_tool:<22} ${event.cost_usd:.6f}")
    print(f"  {'TOTAL':<22} ${guard.advisory().spent_usd:.6f}")


async def main() -> None:
    guard = await run()
    _print_breakdown(guard)


if __name__ == "__main__":
    asyncio.run(main())
