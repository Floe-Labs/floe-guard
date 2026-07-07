"""Pipecat voice-agent pipeline adapter (optional extra: ``pip install floe-guard[pipecat]``).

Unlike request/response frameworks (OpenAI, Anthropic, LangChain), a Pipecat
pipeline has no single call site to wrap: the LLM sits inside a running
``Pipeline`` of ``FrameProcessor``s (STT -> context aggregator -> LLM -> TTS)
and turns fire continuously for the life of a call. So the enforcement
surface here is a ``FrameProcessor`` placed directly after the LLM service,
not a function wrapper around a single call.

The contract matches the OpenAI/Anthropic adapters: ``reserve()`` when a turn
starts (before TTS/audio spend piles on top of a call that would already
cross the ceiling), ``settle(model, prompt_tokens, completion_tokens,
reserved=...)`` once real usage is reported, and ``release(reserved)`` if a
turn ends without ever reporting usage (e.g. an interrupted turn) so the
reservation doesn't leak against the ceiling forever.

    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.task import PipelineTask, PipelineParams
    from floe_guard import BudgetGuard
    from floe_guard.integrations.pipecat import FloeBudgetGuardProcessor

    guard = BudgetGuard(limit_usd=1.00)

    pipeline = Pipeline([
        transport.input(),
        stt,
        context_aggregator.user(),
        llm,
        FloeBudgetGuardProcessor(guard, model="gpt-4o"),
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])
    # enable_usage_metrics=True is required -- without it Pipecat never emits
    # the LLMUsageMetricsData this adapter settles from.
    task = PipelineTask(
        pipeline, params=PipelineParams(enable_metrics=True, enable_usage_metrics=True)
    )
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    MetricsFrame,
)
from pipecat.metrics.metrics import LLMUsageMetricsData
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from ..errors import BudgetExceeded
from ..guard import BudgetGuard
from ..pricing import resolve_price

logger = logging.getLogger(__name__)


def _usage_from(frame: MetricsFrame) -> tuple[str | None, int, int] | None:
    """Pull (model, prompt_tokens, completion_tokens) from a MetricsFrame.

    ``frame.data`` is a list of MetricsData entries (TTFB, processing time,
    token usage, etc. can all show up on the same frame) -- only the
    LLMUsageMetricsData entry carries token usage, so find that one and
    ignore the rest.
    """
    for entry in frame.data:
        if isinstance(entry, LLMUsageMetricsData):
            usage = entry.value
            return entry.model, usage.prompt_tokens, usage.completion_tokens
    return None


class FloeBudgetGuardProcessor(FrameProcessor):
    """Enforce a BudgetGuard ceiling on a Pipecat pipeline, one turn at a time.

    Place directly after the LLM service. Calls ``guard.reserve()`` on
    ``LLMFullResponseStartFrame`` (raising ``BudgetExceeded`` before the
    turn's TTS/audio spend accrues on top of a call that would already cross
    the ceiling), ``guard.settle()`` once a ``MetricsFrame`` reports real
    token usage, and ``guard.release()`` on ``LLMFullResponseEndFrame`` if no
    usage was ever reported (e.g. the turn was interrupted) so the
    reservation doesn't leak.

    Requires the pipeline's ``PipelineTask`` to be created with
    ``PipelineParams(enable_metrics=True, enable_usage_metrics=True)`` --
    Pipecat only emits ``LLMUsageMetricsData`` when usage metrics are on.

    Args:
        guard: the BudgetGuard instance to enforce.
        model: fallback model name to settle cost against, used only if the
            LLM service's own MetricsFrame doesn't report one. If the
            reported model can't be priced but this fallback can, the
            fallback is used instead (mirrors openai.py's served-vs-requested
            model handling) -- so a provider snapshot newer than the bundled
            cost map doesn't fail-close a call that would otherwise price
            cleanly.
        on_budget_exceeded: optional async callback invoked with the
            ``BudgetExceeded`` exception when a turn is blocked, so the
            caller can push a graceful "wrapping up" TTS frame before the
            pipeline ends. If omitted, a fatal ``ErrorFrame`` is pushed
            instead, which terminates the pipeline -- this is the "hard
            stop" default that matches every other floe-guard adapter.
            (A bare raise here would *not* achieve that: Pipecat's
            FrameProcessor catches exceptions raised inside process_frame()
            and downgrades them to a non-fatal, merely-logged ErrorFrame.)
    """

    def __init__(
        self,
        guard: BudgetGuard,
        model: str,
        on_budget_exceeded: Callable[[BudgetExceeded], Awaitable[None]] | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._guard = guard
        self._model = model
        self._on_budget_exceeded = on_budget_exceeded
        self._reserved: float = 0.0
        self._pending = False

    def _settle_model(self, reported_model: str | None) -> str:
        if (
            reported_model
            and reported_model != self._model
            and resolve_price(reported_model, self._guard.price_overrides) is None
            and resolve_price(self._model, self._guard.price_overrides) is not None
        ):
            return self._model
        return reported_model or self._model

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            try:
                self._reserved = self._guard.reserve()
                self._pending = True
            except BudgetExceeded as exc:
                logger.warning("floe-guard blocked a turn: %s", exc)
                self._pending = False
                if self._on_budget_exceeded is not None:
                    await self._on_budget_exceeded(exc)
                else:
                    # Pipecat's FrameProcessor catches any exception raised
                    # inside process_frame() and downgrades it to a
                    # *non-fatal* ErrorFrame rather than propagating it or
                    # stopping the pipeline (confirmed empirically -- see
                    # push_error_frame in frame_processor.py). Simply raising
                    # here would just log a warning and let the pipeline
                    # keep running, silently defeating floe-guard's whole
                    # point of being a *hard* stop. Push a fatal ErrorFrame
                    # ourselves instead, which does actually terminate the
                    # pipeline, so the default (no-callback) behavior is a
                    # real hard stop rather than a swallowed log line.
                    await self.push_error(str(exc), exception=exc, fatal=True)
                return  # don't forward the frame for a call that was blocked

        elif isinstance(frame, MetricsFrame) and self._pending:
            usage = _usage_from(frame)
            if usage is not None:
                reported_model, prompt_tokens, completion_tokens = usage
                self._guard.settle(
                    self._settle_model(reported_model),
                    prompt_tokens,
                    completion_tokens,
                    reserved=self._reserved,
                )
                self._reserved = 0.0
                self._pending = False

        elif isinstance(frame, LLMFullResponseEndFrame) and self._pending:
            # Turn ended without ever reporting usage (e.g. interrupted
            # mid-response) -- release the held reservation instead of
            # leaking it against the ceiling forever.
            self._guard.release(self._reserved)
            self._reserved = 0.0
            self._pending = False

        # Always forward every frame -- a FrameProcessor that swallows frames
        # breaks the pipeline for everything downstream.
        await self.push_frame(frame, direction)


__all__ = ["FloeBudgetGuardProcessor"]
