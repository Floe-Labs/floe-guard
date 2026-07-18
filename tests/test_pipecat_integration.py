"""Tests for the Pipecat integration.

Drives frames through a real (single-processor) Pipeline + PipelineTask +
PipelineRunner, exactly like examples/voice_turn_budget.py -- this is the
harness verified to work end-to-end. An earlier version of this file used
pipecat.tests.utils.run_test, but MetricsFrame is a SystemFrame and
run_test's harness did not appear to settle its effects (guard.settle())
before returning control to the test, causing false failures even though
the underlying adapter logic was correct (confirmed by the demo). If you
find a cleaner way to get run_test to wait on SystemFrame delivery (e.g. an
observers= or expected_down_frames= combination), that would be a nice
simplification -- but this harness is the one known to work.

Style follows tests/test_openai_adapter.py -- adjust fixtures/naming to
match exactly once you've opened that file, for consistency with the rest
of the test suite.
"""

from __future__ import annotations

import asyncio

import pytest

# The adapter under test hard-imports pipecat at module load, so skip the whole
# module (rather than erroring collection) when the optional extra is absent.
pytest.importorskip("pipecat")

from pipecat.frames.frames import (  # noqa: E402
    EndFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    MetricsFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage, LLMUsageMetricsData  # noqa: E402
from pipecat.pipeline.pipeline import Pipeline  # noqa: E402
from pipecat.pipeline.runner import PipelineRunner  # noqa: E402
from pipecat.pipeline.task import PipelineParams, PipelineTask  # noqa: E402

from floe_guard import BudgetGuard  # noqa: E402
from floe_guard.errors import BudgetExceeded  # noqa: E402
from floe_guard.integrations.pipecat import FloeBudgetGuardProcessor  # noqa: E402


def _metrics_frame(prompt_tokens, completion_tokens, model="gpt-4o") -> MetricsFrame:
    usage = LLMTokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return MetricsFrame(data=[LLMUsageMetricsData(processor="test-llm", model=model, value=usage)])


async def _run_frames(
    processor: FloeBudgetGuardProcessor, frames: list[Frame], delay: float = 0.05
):
    """Drive `frames` through a real Pipeline containing just `processor`.

    Needed because FrameProcessor requires a fully initialized lifecycle
    (TaskManager, clock, etc.) that only a running PipelineRunner sets up.
    """
    pipeline = Pipeline([processor])
    task = PipelineTask(
        pipeline, params=PipelineParams(enable_metrics=True, enable_usage_metrics=True)
    )
    runner = PipelineRunner()

    async def drive():
        for frame in frames:
            await task.queue_frame(frame)
            await asyncio.sleep(delay)
        if not task.has_finished():
            await task.queue_frame(EndFrame())

    await asyncio.gather(runner.run(task), drive())


@pytest.mark.asyncio
async def test_settles_usage_reported_after_a_turn():
    guard = BudgetGuard(limit_usd=100.00)
    processor = FloeBudgetGuardProcessor(guard, model="gpt-4o")

    await _run_frames(processor, [LLMFullResponseStartFrame(), _metrics_frame(100, 50)])

    assert guard.advisory().spent_usd > 0
    assert not processor._pending


@pytest.mark.asyncio
async def test_on_budget_exceeded_callback_used_instead_of_raising():
    guard = BudgetGuard(limit_usd=0.0001)
    called = {}

    async def handle_exceeded(exc):
        called["exc"] = exc

    processor = FloeBudgetGuardProcessor(guard, model="gpt-4o", on_budget_exceeded=handle_exceeded)

    await _run_frames(
        processor,
        [
            LLMFullResponseStartFrame(),
            _metrics_frame(1000, 500),
            LLMFullResponseStartFrame(),  # second turn should now exceed the tiny ceiling
        ],
    )

    assert "exc" in called
    assert isinstance(called["exc"], BudgetExceeded)


@pytest.mark.asyncio
async def test_interrupted_turn_releases_reservation_instead_of_leaking():
    guard = BudgetGuard(limit_usd=1.00)
    processor = FloeBudgetGuardProcessor(guard, model="gpt-4o")

    await _run_frames(processor, [LLMFullResponseStartFrame(), LLMFullResponseEndFrame()])

    assert not processor._pending
    assert processor._reserved == 0.0


@pytest.mark.asyncio
async def test_ignores_metrics_frames_without_usage_data():
    """A MetricsFrame carrying only e.g. TTFB data (no LLMUsageMetricsData)
    should not settle or release the pending reservation.

    Checked mid-flight, before the EndFrame teardown -- which now *does* release
    a dangling reservation, so ``_run_frames`` (it queues an EndFrame) can't be
    used here or the assertion would race the teardown.
    """
    guard = BudgetGuard(limit_usd=1.00)
    processor = FloeBudgetGuardProcessor(guard, model="gpt-4o")

    pipeline = Pipeline([processor])
    task = PipelineTask(
        pipeline, params=PipelineParams(enable_metrics=True, enable_usage_metrics=True)
    )
    runner = PipelineRunner()

    async def drive():
        await task.queue_frame(LLMFullResponseStartFrame())
        await asyncio.sleep(0.05)
        await task.queue_frame(MetricsFrame(data=[]))
        await asyncio.sleep(0.05)
        # Usage-less metrics neither settled nor released -- still holding the turn.
        assert processor._pending
        if not task.has_finished():
            await task.queue_frame(EndFrame())

    await asyncio.gather(runner.run(task), drive())

    # ...and the EndFrame teardown released the dangling reservation (no leak).
    assert not processor._pending
    assert processor._reserved == 0.0


@pytest.mark.asyncio
async def test_pushes_fatal_error_when_ceiling_crossed_without_callback():
    """Without an on_budget_exceeded callback, the processor pushes a fatal
    ErrorFrame rather than raising.

    This matters because Pipecat's FrameProcessor catches any exception
    raised inside process_frame() and downgrades it to a *non-fatal*,
    merely-logged ErrorFrame (confirmed empirically: raising BudgetExceeded
    directly resulted in "ErrorFrame(..., fatal: False)" and the pipeline
    continuing to run). A fatal ErrorFrame is what actually terminates a
    Pipecat pipeline, so it's the only way to get a genuine hard stop as
    the default (no-callback) behavior.
    """
    guard = BudgetGuard(limit_usd=0.0001)
    processor = FloeBudgetGuardProcessor(guard, model="gpt-4o")

    pipeline = Pipeline([processor])
    task = PipelineTask(
        pipeline, params=PipelineParams(enable_metrics=True, enable_usage_metrics=True)
    )
    runner = PipelineRunner()

    captured_errors = []

    @task.event_handler("on_pipeline_error")
    async def _on_pipeline_error(task, frame):
        captured_errors.append(frame)

    async def drive():
        await task.queue_frame(LLMFullResponseStartFrame())
        await asyncio.sleep(0.05)
        await task.queue_frame(_metrics_frame(1000, 500))
        await asyncio.sleep(0.05)
        await task.queue_frame(LLMFullResponseStartFrame())  # should now exceed the ceiling
        await asyncio.sleep(0.05)
        if not task.has_finished():
            await task.queue_frame(EndFrame())

    await asyncio.gather(runner.run(task), drive())

    assert len(captured_errors) == 1
    assert captured_errors[0].fatal is True


@pytest.mark.asyncio
async def test_unbalanced_start_frames_do_not_leak_reservation():
    """Two LLMFullResponseStartFrames without a settle/end between them must not
    leak: the first turn's reservation is released before the second reserves, so
    the guard never holds more than the processor is tracking."""
    guard = BudgetGuard(limit_usd=100.00)
    processor = FloeBudgetGuardProcessor(guard, model="gpt-4o")

    pipeline = Pipeline([processor])
    task = PipelineTask(
        pipeline, params=PipelineParams(enable_metrics=True, enable_usage_metrics=True)
    )
    runner = PipelineRunner()

    async def drive():
        # A full turn first, so _last_cost > 0 and later reservations are non-zero
        # (a fresh guard reserves the last call's cost, which starts at $0).
        await task.queue_frame(LLMFullResponseStartFrame())
        await asyncio.sleep(0.05)
        await task.queue_frame(_metrics_frame(100, 50))
        await asyncio.sleep(0.05)
        await task.queue_frame(LLMFullResponseStartFrame())  # turn B reserves
        await asyncio.sleep(0.05)
        await task.queue_frame(LLMFullResponseStartFrame())  # turn C, unbalanced
        await asyncio.sleep(0.05)
        assert processor._reserved > 0.0
        # Exactly one live reservation -- not two. On a leak the guard would hold
        # turn B's hold on top of turn C's, i.e. twice what the processor tracks.
        assert guard._reserved == pytest.approx(processor._reserved)
        if not task.has_finished():
            await task.queue_frame(EndFrame())

    await asyncio.gather(runner.run(task), drive())

    assert guard._reserved == pytest.approx(0.0)  # teardown released the last hold


@pytest.mark.asyncio
async def test_settles_usage_even_without_a_start_frame():
    """Usage is metered even if the LLMFullResponseStartFrame was missed -- the
    MetricsFrame settle path is not gated on a live reservation."""
    guard = BudgetGuard(limit_usd=100.00)
    processor = FloeBudgetGuardProcessor(guard, model="gpt-4o")

    await _run_frames(processor, [_metrics_frame(100, 50)])

    assert guard.advisory().spent_usd > 0
    assert not processor._pending
    assert processor._reserved == 0.0


@pytest.mark.asyncio
async def test_unpriceable_usage_pushes_fatal_error():
    """If settle() can't price the turn, the guard can't measure spend -- the
    processor hard-stops with a fatal ErrorFrame rather than letting Pipecat
    downgrade the failure to a non-fatal log."""
    guard = BudgetGuard(limit_usd=100.00)  # fail_closed defaults to True
    # Both the reported model and the fallback are unpriceable, so _settle_model
    # can't fall back to a priceable id and settle() fail-closes.
    processor = FloeBudgetGuardProcessor(guard, model="totally-made-up-model")

    pipeline = Pipeline([processor])
    task = PipelineTask(
        pipeline, params=PipelineParams(enable_metrics=True, enable_usage_metrics=True)
    )
    runner = PipelineRunner()

    captured_errors = []

    @task.event_handler("on_pipeline_error")
    async def _on_pipeline_error(task, frame):
        captured_errors.append(frame)

    async def drive():
        await task.queue_frame(LLMFullResponseStartFrame())
        await asyncio.sleep(0.05)
        await task.queue_frame(_metrics_frame(100, 50, model="also-unpriceable"))
        await asyncio.sleep(0.05)
        if not task.has_finished():
            await task.queue_frame(EndFrame())

    await asyncio.gather(runner.run(task), drive())

    assert len(captured_errors) == 1
    assert captured_errors[0].fatal is True
    assert guard.advisory().spent_usd == 0.0  # nothing was metered
