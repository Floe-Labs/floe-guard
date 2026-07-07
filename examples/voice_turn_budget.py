"""Reproducible demo: floe-guard stopping a runaway voice conversation.

No API key, no account, no network -- runs a real (single-processor)
Pipecat Pipeline via PipelineTask/PipelineRunner, so FloeBudgetGuardProcessor
gets a fully initialized lifecycle (TaskManager, clock, etc.) exactly like it
would in a real bot. A second coroutine queues frames in to simulate a
multi-turn conversation, mirroring the style of examples/runaway_loop.py.

Uses a fixed ManualPrice rather than the bundled gpt-4o cost-map entry, so
the demo deterministically crosses its ceiling regardless of what the live
cost map says gpt-4o costs on any given day.

Run:
    python examples/voice_turn_budget.py
"""

import asyncio

from pipecat.frames.frames import EndFrame, LLMFullResponseStartFrame, MetricsFrame
from pipecat.metrics.metrics import LLMTokenUsage, LLMUsageMetricsData
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask

from floe_guard import BudgetGuard, ManualPrice
from floe_guard.integrations.pipecat import FloeBudgetGuardProcessor

MODEL = "demo-voice-model"
TURN_COUNT = 10


def _metrics_frame(prompt_tokens: int, completion_tokens: int) -> MetricsFrame:
    usage = LLMTokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return MetricsFrame(data=[LLMUsageMetricsData(processor="fake-llm", model=MODEL, value=usage)])


async def main():
    guard = BudgetGuard(
        limit_usd=0.02,
        price_overrides={MODEL: ManualPrice(1e-5, 3e-5)},  # USD/token -- fixed, demo-only price
    )

    async def on_budget_exceeded(exc):
        # In a real bot you'd push one final TTSSpeakFrame here ("wrapping up
        # now...") before ending the call, instead of just ending it outright.
        print(f"\nBUDGET EXCEEDED: {exc}")
        await task.queue_frame(EndFrame())

    processor = FloeBudgetGuardProcessor(guard, model=MODEL, on_budget_exceeded=on_budget_exceeded)

    pipeline = Pipeline([processor])
    task = PipelineTask(
        pipeline, params=PipelineParams(enable_metrics=True, enable_usage_metrics=True)
    )
    runner = PipelineRunner()

    async def drive_conversation():
        for turn in range(1, TURN_COUNT + 1):
            if task.has_finished():
                break
            print(f"\n--- Turn {turn} ---")
            await task.queue_frame(LLMFullResponseStartFrame())
            await task.queue_frame(_metrics_frame(prompt_tokens=300, completion_tokens=120))
            await asyncio.sleep(0.05)  # let this turn finish processing before the next
        else:
            print("\nConversation finished without hitting the ceiling.")
        if not task.has_finished():
            await task.queue_frame(EndFrame())

    await asyncio.gather(runner.run(task), drive_conversation())


if __name__ == "__main__":
    asyncio.run(main())
