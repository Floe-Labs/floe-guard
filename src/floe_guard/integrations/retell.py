"""Retell custom-LLM WebSocket adapter (framework-free — no Retell SDK, no ``ws``).

Retell's custom LLM runs over a WebSocket (docs.retellai.com/api-references/
llm-websocket). Retell sends interaction events to your server; the billable one
is **``response_required``** (an auto-incrementing ``response_id`` plus the
running ``transcript``), and its twin **``reminder_required``**. Your server
streams back **``response``** events — many partials per turn — ending with
``content_complete: True``. A NEW ``response_required`` with a higher
``response_id`` means the caller barged in: it **interrupts** the turn in
flight, which never reaches ``content_complete``.

Like the LiveKit/Pipecat adapters, a Retell call has no single call site to
wrap: turns fire for the life of the socket. The two enforcement points are the
arrival of a ``response_required`` (before the LLM call, so a turn is admitted
or blocked before its TTS/telephony spend piles on) and the turn's real token
usage (after ``content_complete``). :class:`RetellBudgetGuard` holds the
reservation across both, keyed by ``response_id``, and releases it when a newer
turn interrupts.

    from floe_guard import BudgetGuard
    from floe_guard.integrations.retell import RetellBudgetGuard

    guard = BudgetGuard(
        limit_usd=1.00,
        price_overrides={"gpt-4o-mini": ManualPrice(0.15e-6, 0.6e-6)},
    )
    budget = RetellBudgetGuard(
        guard,
        model="gpt-4o-mini",
        stt_model="deepgram-nova-3",
        tts_model="elevenlabs-flash-v2.5",
        telephony="twilio-us-inbound-local",
    )

    # In your custom-LLM WS message handler:
    async def on_message(raw: bytes | str) -> None:
        event = json.loads(raw)
        if event["interaction_type"] in ("response_required", "reminder_required"):
            turn = budget.begin_turn(event)        # reserve before the LLM call
            if not turn.admitted:                  # budget exhausted
                await ws.send(json.dumps(budget.response(
                    event["response_id"], "I'm out of budget — wrapping up.",
                    complete=True, end_call=True)))
                return
            text, usage = await call_your_llm(event["transcript"])  # your LLM
            await ws.send(json.dumps(
                budget.response(event["response_id"], text, complete=True)))
            budget.settle_turn(event["response_id"], usage)  # settle real usage

## Why a WebSocket adapter, not a request wrapper

The custom LLM sees only the **LLM leg** (the tokens behind each ``response``).
STT, TTS and telephony are billed by Retell/the carrier and never cross this
socket, so they are metered **explicitly** — call :meth:`RetellBudgetGuard.meter_stt`
/ :meth:`RetellBudgetGuard.meter_tts` / :meth:`RetellBudgetGuard.meter_telephony`
from wherever those durations are known (webhook, call-ended event). Each is
priced from the bundled voice cost map when you name the vendor, or a per-unit
override which wins over the map; a leg with neither is left un-metered (the
token-only contract), and a named vendor the map cannot price fails closed
(:class:`~floe_guard.errors.UnpriceableVoiceError`).

Scope is strictly **pre-turn / pre-call admission plus per-turn settlement**.
There is no mid-call intervention: an admitted turn runs to
``content_complete``; nothing here cuts a turn off partway. The only release is
the interrupt Retell itself signals (a newer ``response_id``) or an explicit
:meth:`RetellBudgetGuard.close` on call teardown.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..errors import BudgetExceeded
from ..gates import retell as retell_gate
from ..guard import BudgetGuard, ReservationHandle
from ..voice_pricing import price_voice_leg

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetellTurnDecision:
    """The decision :meth:`RetellBudgetGuard.begin_turn` returns.

    ``admitted=True`` means the turn's reservation is held — run the LLM and
    settle it. ``admitted=False`` means the budget is spent: send a wrap-up
    ``response`` (with ``content_complete``) and do not call the LLM; ``error``
    carries the :class:`~floe_guard.errors.BudgetExceeded` for logging.
    """

    admitted: bool
    response_id: int
    error: BudgetExceeded | None = None


@dataclass
class _TurnSlot:
    """One turn's reservation, keyed by ``response_id``.

    ``open`` means the USD amount is still held on the BudgetGuard. A settle or
    an interrupt closes it.
    """

    response_id: int
    amount: ReservationHandle
    open: bool = True


class RetellBudgetGuard:
    """Enforce a BudgetGuard ceiling on a Retell custom-LLM socket, one turn at a
    time: reserve when a ``response_required`` arrives, settle on real token
    usage after ``content_complete``, release when a newer ``response_id``
    interrupts.

    Args:
        guard: the BudgetGuard to enforce.
        model: model id to settle LLM cost against (Retell's usage payload names
            no model we rely on). Must be priceable via the bundled cost map or
            the guard's ``price_overrides``.
        stt_model: voice-map vendor key for the STT leg (e.g.
            ``"deepgram-nova-3"``), metered per second via :meth:`meter_stt`.
        tts_model: voice-map vendor key for the TTS leg (e.g.
            ``"elevenlabs-flash-v2.5"``), metered per 1k chars via
            :meth:`meter_tts`.
        telephony: voice-map vendor key for the telephony leg (e.g.
            ``"twilio-us-inbound-local"``), metered per minute via
            :meth:`meter_telephony`. US-only in v1.
        stt_usd_per_second: per-second STT override — wins over ``stt_model``.
            Omit both to leave STT un-metered.
        tts_usd_per_1k_chars: per-1k-chars TTS override — wins over
            ``tts_model``. Omit both to leave TTS un-metered.
        telephony_usd_per_minute: per-minute telephony override — wins over
            ``telephony``.
    """

    def __init__(
        self,
        guard: BudgetGuard,
        model: str,
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
        self._stt_model = stt_model
        self._tts_model = tts_model
        self._telephony = telephony
        self._stt_usd_per_second = stt_usd_per_second
        self._tts_usd_per_1k_chars = tts_usd_per_1k_chars
        self._telephony_usd_per_minute = telephony_usd_per_minute
        # Open (and just-settled) turns keyed by response_id.
        self._slots: dict[int, _TurnSlot] = {}
        # The most recently begun turn — the one a newer response_required
        # interrupts.
        self._active_response_id: int | None = None

    def begin_turn(self, event: Mapping[str, Any]) -> RetellTurnDecision:
        """Reserve budget for a ``response_required`` / ``reminder_required``
        turn before its LLM call runs. Returns an admit/block
        :class:`RetellTurnDecision`.

        A newer ``response_id`` arriving while a prior turn is still open is
        Retell's interrupt signal — this releases that prior turn's hold before
        reserving the new one (its ``content_complete`` will never arrive).
        Reserving here is what blocks a turn before its downstream TTS/telephony
        spend piles on. Calling twice with the same still-open ``response_id``
        is idempotent (no double hold).
        """
        response_id = event["response_id"]

        # Idempotent: this turn already holds a reservation.
        existing = self._slots.get(response_id)
        if existing is not None and existing.open:
            return RetellTurnDecision(True, response_id)

        # A newer turn interrupts the prior open turn — free its hold before
        # opening this one. Its usage never settles (no content_complete); a
        # stray late settle for it finds no open slot and records actual usage
        # against a zero reservation.
        self._release_active(response_id)

        try:
            handle = self._guard.reserve()  # raises BudgetExceeded before the turn runs
        except BudgetExceeded as exc:
            return RetellTurnDecision(False, response_id, exc)

        self._slots[response_id] = _TurnSlot(response_id, handle)
        self._active_response_id = response_id
        return RetellTurnDecision(True, response_id)

    def settle_turn(self, response_id: int, usage: Mapping[str, Any]) -> None:
        """Settle a turn's real token usage after its ``content_complete``.

        Consumes the reservation opened by :meth:`begin_turn` for this
        ``response_id``; a turn with no open slot (already interrupted, or a
        settle for an id never begun) records the usage against a zero
        reservation instead of stealing another turn's hold. ``usage`` carries
        Retell's camelCase fields (``promptTokens`` / ``completionTokens``).
        """
        reserved: ReservationHandle = 0.0
        slot = self._slots.pop(response_id, None)
        if slot is not None:
            if slot.open:
                reserved = slot.amount
                slot.open = False
        if self._active_response_id == response_id:
            self._active_response_id = None
        self._guard.settle(
            self._model,
            usage["promptTokens"],
            usage["completionTokens"],
            reserved=reserved,
        )

    def admit_call(
        self,
        *,
        estimated_call_usd: float = 0.0,
        admit: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Pre-call admission for Retell's inbound-call webhook, wrapping
        :func:`floe_guard.gates.retell`. On budget exhaustion returns
        ``{"call_inbound": {"reject": True}}``; otherwise ``{"call_inbound": {...}}``
        carrying any ``admit`` overrides (``dynamic_variables``, ``metadata``,
        ``override_agent_id``, …).

        A coarse, non-binding preflight — the binding hard-stop is the per-turn
        reserve in :meth:`begin_turn`. Pass ``estimated_call_usd`` (e.g.
        ``$/min × expected minutes``) to reject earlier, when the remaining
        budget can't cover the call.
        """
        return retell_gate(self._guard, estimated_call_usd=estimated_call_usd, admit=admit)

    def response(
        self,
        response_id: int,
        content: str,
        *,
        complete: bool = False,
        end_call: bool = False,
    ) -> dict[str, Any]:
        """Build a ``response`` event to stream back to Retell.

        Send partials with ``complete=False``, then a final one with
        ``complete=True``; settle the turn's token usage after that final one.
        Pass ``end_call=True`` to hang up (e.g. on a budget wrap-up line). A
        thin convenience — you may build the shape yourself.
        """
        event: dict[str, Any] = {
            "response_type": "response",
            "response_id": response_id,
            "content": content,
            "content_complete": complete,
        }
        if end_call:
            event["end_call"] = True
        return event

    def meter_stt(self, seconds: float) -> float | None:
        """Accrue STT spend for ``seconds`` of transcribed audio (per-second).

        The socket never sees the STT leg, so drive this from wherever the
        duration is known. Priced from the voice map when ``stt_model`` is set,
        or the ``stt_usd_per_second`` override; returns ``None`` (no-op) when
        the leg is unconfigured, and fails closed on a vendor the voice map
        cannot price. Returns the USD accrued, if any.
        """
        cost = price_voice_leg(
            "stt", seconds, model=self._stt_model, override=self._stt_usd_per_second
        )
        if cost is not None:
            self._guard.record_tool("retell-stt", cost)
        return cost

    def meter_tts(self, characters: float) -> float | None:
        """Accrue TTS spend for ``characters`` of synthesized speech
        (per-1k-chars). Priced from the voice map when ``tts_model`` is set, or
        the ``tts_usd_per_1k_chars`` override; returns ``None`` when
        unconfigured, and fails closed on an unpriceable vendor.
        """
        cost = price_voice_leg(
            "tts", characters, model=self._tts_model, override=self._tts_usd_per_1k_chars
        )
        if cost is not None:
            self._guard.record_tool("retell-tts", cost)
        return cost

    def meter_telephony(self, minutes: float) -> float | None:
        """Accrue telephony spend for ``minutes`` of call time (per-minute).

        **Per-minute accrual, not live line-cutting**: the guard meters the leg,
        it does not cut the phone line mid-call. Priced from the voice map when
        ``telephony`` is set, or the ``telephony_usd_per_minute`` override;
        returns ``None`` when unconfigured, and fails closed on an unpriceable
        vendor. Returns the USD accrued, if any.
        """
        cost = price_voice_leg(
            "telephony", minutes, model=self._telephony, override=self._telephony_usd_per_minute
        )
        if cost is not None:
            self._guard.record_tool("retell-telephony", cost)
        return cost

    def close(self) -> None:
        """Release any still-open turn's reservation on call teardown (socket
        close, call ended). A turn that never reached ``content_complete`` would
        otherwise leak its hold and shrink ``remaining_usd`` permanently."""
        for slot in self._slots.values():
            if slot.open:
                self._guard.release(slot.amount)
        self._slots.clear()
        self._active_response_id = None

    def _release_active(self, except_id: int) -> None:
        """Release the active open turn's hold unless it is ``except_id`` (the
        incoming turn)."""
        active_id = self._active_response_id
        if active_id is None or active_id == except_id:
            return
        slot = self._slots.get(active_id)
        if slot is not None and slot.open:
            self._guard.release(slot.amount)
            slot.open = False
            self._slots.pop(active_id, None)
        self._active_response_id = None


__all__ = ["RetellBudgetGuard", "RetellTurnDecision"]
