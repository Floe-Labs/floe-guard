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
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from livekit.agents.metrics import LLMMetrics, STTMetrics, TTSMetrics

from ..errors import BudgetExceeded
from ..guard import BudgetGuard

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
        self._active: _TurnSlot | None = None
        # FIFO of turns that may still emit LLMMetrics (including early-released).
        self._slots: deque[_TurnSlot] = deque()

    def attach(self, session, agent) -> None:
        """Wire reserve (agent.llm_node), settle/meter (metrics_collected) and
        release (close) onto a session + agent pair."""
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
        session.on("metrics_collected", self._on_metrics)
        session.on("close", self._on_close)

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

    def _on_metrics(self, ev) -> None:
        m = ev.metrics
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
        elif self._stt_usd_per_second is not None and isinstance(m, STTMetrics):
            self._guard.record_tool("livekit-stt", m.audio_duration * self._stt_usd_per_second)
        elif self._tts_usd_per_1k_chars is not None and isinstance(m, TTSMetrics):
            self._guard.record_tool(
                "livekit-tts", m.characters_count / 1000 * self._tts_usd_per_1k_chars
            )

    def _on_close(self, ev) -> None:
        # Session torn down with a turn still reserved — release it. Drop any
        # leftover slots so a post-close metrics event cannot touch a stale hold.
        if self._pending:
            self._early_release_active()
        self._slots.clear()


__all__ = ["LiveKitBudgetGuard"]
