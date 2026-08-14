"""LiveKit Agents adapter (optional extra: ``pip install floe-guard[livekit]``).

Like Pipecat, a LiveKit ``AgentSession`` (STT -> LLM -> TTS) has no single call
site to wrap: turns fire for the life of a call. The two enforcement points are
the agent's ``llm_node`` (before the LLM call, so a turn is blocked before its
TTS/audio spend piles on) and each component's ``metrics_collected`` event —
the LLM/STT/TTS plugins each emit their real usage right after doing their work.
(The session-level ``metrics_collected`` event was deprecated in
``livekit-agents`` 1.5; the per-component event carries the same per-turn metric
object and is not deprecated, so we settle each turn straight off the LLM plugin
— cleanly per-turn, unlike the cumulative ``session_usage_updated`` summary.)
``LiveKitBudgetGuard`` holds the reservation state across both.

    from floe_guard import BudgetGuard, ManualPrice
    from floe_guard.integrations.livekit import LiveKitBudgetGuard

    guard = BudgetGuard(
        limit_usd=1.00,
        price_overrides={"gemini-2.0-flash": ManualPrice(0.30e-6, 2.50e-6)},
    )
    budget = LiveKitBudgetGuard(guard, model="gemini-2.0-flash")

    session = AgentSession(...)
    budget.attach(session, agent)      # wire reserve / settle / release
    await session.start(agent=agent, room=ctx.room)

LiveKit's ``LLMMetrics`` does not report the served model, so cost is settled
against the ``model`` passed here (unlike the Pipecat adapter, which reads it
off the frame). STT/TTS spend — often a voice agent's larger bill — is metered
whenever you name the vendor: pass ``stt_model`` / ``tts_model`` and the rate
comes from the bundled voice cost map (no hand-typed rate needed), or pass a
per-unit override (``stt_usd_per_second`` / ``tts_usd_per_1k_chars``) which wins
over the map. Telephony (per-minute) is metered via :meth:`meter_telephony`,
since LiveKit emits no telephony metric. A leg with neither a vendor nor an
override is left un-metered (the token-only contract); a vendor the voice map
cannot price fails closed (:class:`~floe_guard.errors.UnpriceableVoiceError`).
"""

from __future__ import annotations

import inspect
import logging
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from livekit.agents.metrics import LLMMetrics, STTMetrics, TTSMetrics

from ..errors import BudgetExceeded
from ..guard import BudgetGuard
from ..voice_pricing import price_voice_leg

logger = logging.getLogger(__name__)


@dataclass
class _TurnSlot:
    """One llm_node invocation awaiting (or already past) its LLMMetrics event.

    ``open`` means the USD amount is still held on the BudgetGuard. Early
    release (next turn, exception, close) clears the guard hold but leaves the
    slot in the FIFO queue so a delayed metrics event settles against this
    turn's slot (with ``amount`` 0) instead of stealing a later turn's hold.
    """

    amount: float
    open: bool = True


class LiveKitBudgetGuard:
    """Enforce a BudgetGuard ceiling on a LiveKit ``AgentSession``, one turn at a time.

    Args:
        guard: the BudgetGuard to enforce.
        model: model id to settle LLM cost against (LiveKit's LLMMetrics carries
            no model name). Must be priceable via the cost map or the guard's
            ``price_overrides``.
        on_budget_exceeded: optional async callback invoked with the
            ``BudgetExceeded`` when a turn is blocked, so a bot can speak a
            graceful "wrapping up" line before the turn ends silently. If
            omitted, the ``BudgetExceeded`` propagates out of ``llm_node``.
        stt_model: voice-map vendor key for the STT leg (e.g.
            ``"deepgram-nova-3"``). When set, ``STTMetrics.audio_duration`` is
            metered via ``record_tool`` at the bundled per-second rate — no
            hand-typed rate needed. A key the voice map cannot price fails closed
            (:class:`~floe_guard.errors.UnpriceableVoiceError`).
        tts_model: voice-map vendor key for the TTS leg (e.g.
            ``"elevenlabs-flash-v2.5"``). When set, ``TTSMetrics.characters_count``
            is metered via ``record_tool`` at the bundled per-1k-chars rate.
        telephony: voice-map vendor key for the telephony leg (e.g.
            ``"twilio-us-inbound-local"``), metered per minute via
            :meth:`meter_telephony`. US-only in v1.
        stt_usd_per_second: per-second STT override — wins over ``stt_model``.
            Omit both to keep the token-only contract (STT un-metered).
        tts_usd_per_1k_chars: per-1k-chars TTS override — wins over ``tts_model``.
            Omit both to keep the token-only contract (TTS un-metered).
        telephony_usd_per_minute: per-minute telephony override — wins over
            ``telephony``.
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
    ):
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
        self._active: _TurnSlot | None = None
        # FIFO of turns that may still emit LLMMetrics (including early-released).
        self._slots: deque[_TurnSlot] = deque()

    def attach(self, session, agent) -> None:
        """Wire reserve (agent.llm_node), settle/meter (each component's
        ``metrics_collected``) and release (session close) onto a session + agent
        pair."""
        orig_llm_node = agent.llm_node

        # forward LiveKit's (chat_ctx, tools, model_settings) unchanged, so an
        # upstream signature change can't break the wrapper.
        async def _guarded_llm_node(*args, **kwargs):
            # A previous turn that never settled still holds its reservation —
            # release the guard hold before opening this one, but keep a queue
            # slot so delayed metrics cannot steal this turn's hold.
            if self._pending:
                self._early_release_active()
            try:
                amount = self._guard.reserve()
            except BudgetExceeded as exc:
                self._clear_active()
                logger.warning("floe-guard blocked a turn: %s", exc)
                if self._on_budget_exceeded is None:
                    raise
                await self._on_budget_exceeded(exc)
                return
            slot = _TurnSlot(amount=amount)
            self._slots.append(slot)
            self._active = slot
            self._reserved = amount
            self._pending = True
            try:
                stream = orig_llm_node(*args, **kwargs)
                if inspect.isawaitable(stream):
                    stream = await stream
                async for chunk in stream:
                    yield chunk
            except BaseException:
                # Release only if this invocation still owns the open hold — a
                # later turn may already have early-released ours and reserved
                # its own, making its slot (not ours) the active one.
                if self._active is slot:
                    self._early_release_active()
                raise

        agent.llm_node = _guarded_llm_node
        # Settle/meter off each component's own metrics_collected — the LLM plugin
        # fires per LLM turn (so settlement lands right after the turn we reserved),
        # STT/TTS per utterance. The session-level metrics_collected is deprecated
        # in livekit-agents 1.5+, so we no longer listen on the session for it.
        for component in self._metric_sources(session, agent):
            component.on("metrics_collected", self._on_metric)
        session.on("close", self._on_close)

    @staticmethod
    def _metric_sources(session, agent) -> list:
        """Resolve the LLM/STT/TTS plugins to meter, mirroring LiveKit's runtime
        pick (an agent component overrides the session's). Deduped by identity so
        a unified model exposed under two slots is subscribed once."""
        sources: list = []
        seen: set[int] = set()
        for name in ("llm", "stt", "tts"):
            component = getattr(agent, name, None)
            if component is None or not hasattr(component, "on"):
                component = getattr(session, name, None)
            if component is None or not hasattr(component, "on"):
                continue
            if id(component) in seen:
                continue
            seen.add(id(component))
            sources.append(component)
        return sources

    def _clear_active(self) -> None:
        self._active = None
        self._reserved = 0.0
        self._pending = False

    def _early_release_active(self) -> None:
        """Drop the open guard hold; leave a zeroed queue slot for delayed metrics."""
        slot = self._active
        if slot is None or not slot.open:
            self._clear_active()
            return
        self._guard.release(slot.amount)
        slot.open = False
        slot.amount = 0.0
        self._clear_active()

    def _on_metric(self, m) -> None:
        # Component-level metrics_collected hands us the raw metric object
        # (LLMMetrics / STTMetrics / TTSMetrics) directly, not wrapped in a
        # session MetricsCollectedEvent.
        if isinstance(m, LLMMetrics):
            # Pop the oldest turn slot. Early-released turns contribute 0 so a
            # delayed metrics event meters usage without consuming a later hold.
            # An empty queue (reserve hook bypassed) settles as a plain record().
            if self._slots:
                slot = self._slots.popleft()
                reserved = slot.amount if slot.open else 0.0
                if slot.open:
                    slot.open = False
                if self._active is slot:
                    self._clear_active()
            else:
                reserved = 0.0
            self._guard.settle(self._model, m.prompt_tokens, m.completion_tokens, reserved=reserved)
        elif isinstance(m, STTMetrics):
            # Priced from the voice map when stt_model is set, or the override;
            # None (neither configured) leaves STT un-metered, an unpriceable
            # vendor fails closed. Quantity is audio-seconds.
            cost = price_voice_leg(
                "stt", m.audio_duration, model=self._stt_model, override=self._stt_usd_per_second
            )
            if cost is not None:
                self._guard.record_tool("livekit-stt", cost)
        elif isinstance(m, TTSMetrics):
            cost = price_voice_leg(
                "tts",
                m.characters_count,
                model=self._tts_model,
                override=self._tts_usd_per_1k_chars,
            )
            if cost is not None:
                self._guard.record_tool("livekit-tts", cost)

    def meter_telephony(self, minutes: float) -> float | None:
        """Accrue telephony spend for ``minutes`` of call time (per-minute).

        LiveKit emits no telephony metric, so the transport/caller drives this —
        call it as the call accrues minutes, or once with the final duration.
        This is **per-minute accrual, not live line-cutting**: the guard meters
        the leg, it does not cut the phone line mid-call. Priced from the voice
        map when ``telephony`` is set, or the ``telephony_usd_per_minute``
        override; returns ``None`` (no-op) when the leg is unconfigured, and
        fails closed on a vendor the voice map cannot price. Returns the USD
        accrued, if any.
        """
        cost = price_voice_leg(
            "telephony", minutes, model=self._telephony, override=self._telephony_usd_per_minute
        )
        if cost is not None:
            self._guard.record_tool("livekit-telephony", cost)
        return cost

    def _on_close(self, ev) -> None:
        # Session torn down with a turn still reserved — release it. Drop any
        # leftover slots so a post-close metrics event cannot touch a stale hold.
        if self._pending:
            self._early_release_active()
        self._slots.clear()


__all__ = ["LiveKitBudgetGuard"]
