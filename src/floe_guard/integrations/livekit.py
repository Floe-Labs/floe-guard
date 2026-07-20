"""LiveKit Agents adapter (optional extra: ``pip install floe-guard[livekit]``).

Like Pipecat, a LiveKit ``AgentSession`` (STT -> LLM -> TTS) has no single call
site to wrap: turns fire for the life of a call. The two enforcement points are
the agent's ``llm_node`` (before the LLM call, so a turn is blocked before its
TTS/audio spend piles on) and the session's ``metrics_collected`` event (real
usage, after). ``LiveKitBudgetGuard`` holds the reservation state across both.

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
only if per-unit prices are supplied, since the bundled cost map is LLM-only.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable

from livekit.agents.metrics import LLMMetrics, STTMetrics, TTSMetrics

from ..errors import BudgetExceeded
from ..guard import BudgetGuard

logger = logging.getLogger(__name__)


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
        stt_usd_per_second: if set, meter ``STTMetrics.audio_duration`` via
            ``record_tool`` (per-second). Omit to keep the token-only contract.
        tts_usd_per_1k_chars: if set, meter ``TTSMetrics.characters_count`` via
            ``record_tool`` (per 1k chars). Omit to keep the token-only contract.
    """

    def __init__(
        self,
        guard: BudgetGuard,
        model: str,
        on_budget_exceeded: Callable[[BudgetExceeded], Awaitable[None]] | None = None,
        *,
        stt_usd_per_second: float | None = None,
        tts_usd_per_1k_chars: float | None = None,
    ):
        self._guard = guard
        self._model = model
        self._on_budget_exceeded = on_budget_exceeded
        self._stt_usd_per_second = stt_usd_per_second
        self._tts_usd_per_1k_chars = tts_usd_per_1k_chars
        self._reserved: float = 0.0
        self._pending = False

    def attach(self, session, agent) -> None:
        """Wire reserve (agent.llm_node), settle/meter (metrics_collected) and
        release (close) onto a session + agent pair."""
        orig_llm_node = agent.llm_node

        async def _guarded_llm_node(chat_ctx, tools, model_settings):
            # A previous turn that never settled (no metrics emitted) still holds
            # its reservation — release before opening this one, or it leaks.
            if self._pending:
                self._release_pending()
            try:
                self._reserved = self._guard.reserve()
                self._pending = True
            except BudgetExceeded as exc:
                self._pending = False
                logger.warning("floe-guard blocked a turn: %s", exc)
                if self._on_budget_exceeded is None:
                    raise
                await self._on_budget_exceeded(exc)
                return
            stream = orig_llm_node(chat_ctx, tools, model_settings)
            if inspect.isawaitable(stream):
                stream = await stream
            async for chunk in stream:
                yield chunk

        agent.llm_node = _guarded_llm_node
        session.on("metrics_collected", self._on_metrics)
        session.on("close", self._on_close)

    def _release_pending(self) -> None:
        self._guard.release(self._reserved)
        self._reserved = 0.0
        self._pending = False

    def _on_metrics(self, ev) -> None:
        m = ev.metrics
        if isinstance(m, LLMMetrics):
            # Settle whenever usage arrives, even if the reserve hook was bypassed
            # (then _reserved is 0.0, i.e. a plain record()). A cancelled turn
            # still reports its partial tokens and releases the reservation here.
            reserved = self._reserved
            self._reserved = 0.0
            self._pending = False
            self._guard.settle(self._model, m.prompt_tokens, m.completion_tokens, reserved=reserved)
        elif self._stt_usd_per_second and isinstance(m, STTMetrics):
            self._guard.record_tool("livekit-stt", m.audio_duration * self._stt_usd_per_second)
        elif self._tts_usd_per_1k_chars and isinstance(m, TTSMetrics):
            self._guard.record_tool(
                "livekit-tts", m.characters_count / 1000 * self._tts_usd_per_1k_chars
            )

    def _on_close(self, ev) -> None:
        # Session torn down with a turn still reserved and no usage ever reported
        # — release it so the reservation doesn't leak against the ceiling.
        if self._pending:
            self._release_pending()


__all__ = ["LiveKitBudgetGuard"]
