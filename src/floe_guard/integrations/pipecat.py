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
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    MetricsFrame,
)
from pipecat.metrics.metrics import LLMUsageMetricsData, TTSUsageMetricsData
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from ..errors import BudgetExceeded
from ..guard import BudgetGuard
from ..pricing import resolve_price
from ..voice_pricing import price_voice_leg

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


def _tts_chars_from(frame: MetricsFrame) -> int | None:
    """Pull the TTS character count from a MetricsFrame, or None if absent.

    Pipecat's TTS service emits ``TTSUsageMetricsData`` whose ``value`` is the
    number of characters synthesised on that turn -- the unit ElevenLabs /
    Cartesia / Rime bill in. Other entries on the same frame are ignored.
    """
    for entry in frame.data:
        if isinstance(entry, TTSUsageMetricsData):
            return entry.value
    return None


class FloeBudgetGuardProcessor(FrameProcessor):
    """Enforce a BudgetGuard ceiling on a Pipecat pipeline, one turn at a time.

    Place directly after the LLM service. Calls ``guard.reserve()`` on
    ``LLMFullResponseStartFrame`` (raising ``BudgetExceeded`` before the
    turn's TTS/audio spend accrues on top of a call that would already cross
    the ceiling), ``guard.settle()`` once a ``MetricsFrame`` reports real
    token usage, and ``guard.release()`` whenever a turn ends with its
    reservation still unsettled — an ``LLMFullResponseEndFrame``, a new turn
    starting before the last one settled, or the pipeline tearing down
    (``EndFrame``/``CancelFrame``/``ErrorFrame``) — so the reservation never
    leaks against the ceiling.

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
        stt_model: voice-map vendor key for the STT leg (e.g.
            ``"deepgram-nova-3"``). Pipecat emits no STT-usage metric, so meter
            STT via :meth:`meter_stt` with the audio-seconds of the turn.
        tts_model: voice-map vendor key for the TTS leg (e.g.
            ``"elevenlabs-flash-v2.5"``). When set, the ``TTSUsageMetricsData``
            Pipecat emits (character count) is metered automatically via
            ``record_tool`` at the bundled per-1k-chars rate -- no hand-typed
            rate needed. A vendor the voice map cannot price fails closed with a
            fatal ``ErrorFrame`` (same hard-stop as an unpriceable LLM turn).
        telephony: voice-map vendor key for the telephony leg (e.g.
            ``"twilio-us-inbound-local"``), metered per minute via
            :meth:`meter_telephony`. US-only in v1.
        stt_usd_per_second: per-second STT override -- wins over ``stt_model``.
        tts_usd_per_1k_chars: per-1k-chars TTS override -- wins over ``tts_model``.
        telephony_usd_per_minute: per-minute telephony override -- wins over
            ``telephony``. A leg with neither a vendor nor an override is left
            un-metered (the token-only contract).
    """

    def __init__(
        self,
        guard: BudgetGuard,
        model: str,
        on_budget_exceeded: Callable[[BudgetExceeded], Awaitable[None]] | None = None,
        *,
        stt_model: str | None = None,
        tts_model: str | None = None,
        telephony: str | None = None,
        stt_usd_per_second: float | None = None,
        tts_usd_per_1k_chars: float | None = None,
        telephony_usd_per_minute: float | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._guard = guard
        self._model = model
        self._on_budget_exceeded = on_budget_exceeded
        self._stt_model = stt_model
        self._tts_model = tts_model
        self._telephony = telephony
        self._stt_usd_per_second = stt_usd_per_second
        self._tts_usd_per_1k_chars = tts_usd_per_1k_chars
        self._telephony_usd_per_minute = telephony_usd_per_minute
        self._reserved: float = 0.0
        self._pending = False

    def meter_stt(self, seconds: float) -> float | None:
        """Accrue STT spend for ``seconds`` of audio (per-second).

        Pipecat emits no STT-usage metric, so the caller drives this with the
        turn's audio duration. Priced from the voice map when ``stt_model`` is
        set, or the ``stt_usd_per_second`` override; returns ``None`` (no-op)
        when the leg is unconfigured, and raises
        :class:`~floe_guard.errors.UnpriceableVoiceError` on a vendor the voice
        map cannot price. Returns the USD accrued, if any.
        """
        cost = price_voice_leg(
            "stt", seconds, model=self._stt_model, override=self._stt_usd_per_second
        )
        if cost is not None:
            self._guard.record_tool("pipecat-stt", cost)
        return cost

    def meter_telephony(self, minutes: float) -> float | None:
        """Accrue telephony spend for ``minutes`` of call time (per-minute).

        **Per-minute accrual, not live line-cutting**: the guard meters the leg,
        it does not cut the phone line mid-call. Priced from the voice map when
        ``telephony`` is set, or the ``telephony_usd_per_minute`` override;
        returns ``None`` (no-op) when unconfigured, and fails closed on a vendor
        the voice map cannot price. Returns the USD accrued, if any.
        """
        cost = price_voice_leg(
            "telephony", minutes, model=self._telephony, override=self._telephony_usd_per_minute
        )
        if cost is not None:
            self._guard.record_tool("pipecat-telephony", cost)
        return cost

    def _settle_model(self, reported_model: str | None) -> str:
        if (
            reported_model
            and reported_model != self._model
            and resolve_price(reported_model, self._guard.price_overrides) is None
            and resolve_price(self._model, self._guard.price_overrides) is not None
        ):
            return self._model
        return reported_model or self._model

    def _release_pending(self) -> None:
        """Drop a still-held turn reservation -- an interrupted turn, a new turn
        starting before the last one settled, or pipeline teardown -- so it
        never leaks against the ceiling."""
        self._guard.release(self._reserved)
        self._reserved = 0.0
        self._pending = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            # If the previous turn never settled its usage (interrupted, or no
            # MetricsFrame ever arrived), its reservation is still held -- release
            # it before opening the new turn's, or back-to-back start frames would
            # overwrite the handle and leak the earlier hold against the ceiling.
            if self._pending:
                self._release_pending()
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

        elif isinstance(frame, MetricsFrame):
            # Settle whenever a frame carries real token usage -- NOT gated on
            # _pending, so usage is still metered if the start frame was missed
            # or a prior frame already cleared the reservation. With no live
            # reservation ``self._reserved`` is 0.0, i.e. a plain record().
            usage = _usage_from(frame)
            if usage is not None:
                reported_model, prompt_tokens, completion_tokens = usage
                reserved = self._reserved
                self._reserved = 0.0
                self._pending = False
                try:
                    self._guard.settle(
                        self._settle_model(reported_model),
                        prompt_tokens,
                        completion_tokens,
                        reserved=reserved,
                    )
                except Exception as exc:
                    # settle() fail-closes on a model it can't price (and
                    # re-raises a non-finite cost). A guard that can't measure
                    # spend must hard-stop the pipeline, not let the exception be
                    # downgraded to a non-fatal log by Pipecat (same reasoning as
                    # the BudgetExceeded branch above). settle() already released
                    # the reservation on its raise path, so don't release again.
                    logger.warning("floe-guard could not settle a turn: %s", exc)
                    await self.push_error(str(exc), exception=exc, fatal=True)
                    return

            # Meter the TTS leg from the same frame family (a separate
            # TTSUsageMetricsData entry, emitted by the TTS service). Priced from
            # the voice map when tts_model is set, or the override; None (neither
            # configured) leaves TTS un-metered. A vendor the map can't price
            # fails closed with a fatal ErrorFrame, same hard-stop as an
            # unpriceable LLM turn -- a guard that can't measure spend must not
            # let Pipecat downgrade the failure to a non-fatal log.
            tts_chars = _tts_chars_from(frame)
            if tts_chars is not None:
                try:
                    tts_cost = price_voice_leg(
                        "tts",
                        tts_chars,
                        model=self._tts_model,
                        override=self._tts_usd_per_1k_chars,
                    )
                except Exception as exc:
                    logger.warning("floe-guard could not price a TTS leg: %s", exc)
                    await self.push_error(str(exc), exception=exc, fatal=True)
                    return
                if tts_cost is not None:
                    self._guard.record_tool("pipecat-tts", tts_cost)

        elif isinstance(frame, (LLMFullResponseEndFrame, EndFrame, CancelFrame, ErrorFrame)):
            # A turn that ended (LLMFullResponseEndFrame) or the pipeline tearing
            # down (End/Cancel/Error) while a reservation is still held and no
            # usage was ever reported -- release it instead of leaking. Fall
            # through so the terminating frame still propagates downstream.
            if self._pending:
                self._release_pending()

        # Always forward every frame -- a FrameProcessor that swallows frames
        # breaks the pipeline for everything downstream.
        await self.push_frame(frame, direction)


__all__ = ["FloeBudgetGuardProcessor"]
