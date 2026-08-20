"""Vapi custom-LLM adapter (framework-free — no Vapi or OpenAI SDK dependency).

Vapi's **custom-LLM** feature points Vapi's model leg at YOUR OpenAI-compatible
``POST /chat/completions`` endpoint (docs.vapi.ai/customization/custom-llm/using-your-server).
Vapi sends an OpenAI-format request (``{ model, messages, temperature, tools?,
stream }``); your endpoint proxies it to an upstream LLM and returns an
OpenAI-compatible completion — a single JSON object, or an SSE stream of chunks.

Like the LiveKit/Pipecat adapters, a Vapi call has no single wrap point that
sees every cost: the custom-LLM proxy only sees the **model leg**. So this
adapter has three jobs, and only the first is automatic:

1. **Guard the model turn** — :meth:`VapiBudgetGuard.guard_completion` (JSON)
   and :meth:`VapiBudgetGuard.guard_stream` (SSE) reserve the estimated cost
   BEFORE the upstream call, meter the **real** ``usage`` after, and release the
   hold on error/abort. Reserving first is what refuses a turn before its spend
   lands: an over-budget turn raises :class:`~floe_guard.errors.BudgetExceeded`
   instead of being proxied.
2. **Admit the call** — :meth:`VapiBudgetGuard.assistant_request` answers
   Vapi's ``assistant-request`` webhook from the remaining budget (exhausted →
   a spoken error; else hands back ``assistant``/``assistantId``), delegating to
   :func:`floe_guard.gates.vapi`.
3. **Meter the other legs** — the proxy never sees STT/TTS/telephony, so
   :meth:`VapiBudgetGuard.meter_stt` / :meth:`VapiBudgetGuard.meter_tts` /
   :meth:`VapiBudgetGuard.meter_telephony` accrue them explicitly, priced from
   the bundled voice cost map or a per-unit override.

    from floe_guard import BudgetGuard
    from floe_guard.errors import BudgetExceeded
    from floe_guard.integrations.vapi import VapiBudgetGuard

    guard = BudgetGuard(limit_usd=1.00)
    budget = VapiBudgetGuard(
        guard,
        stt_model="deepgram-nova-3",
        tts_model="elevenlabs-flash-v2.5",
        telephony="twilio-us-inbound-local",
    )

    # POST /chat/completions — the custom-LLM endpoint Vapi calls each turn.
    @app.post("/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        model, messages, tools = body["model"], body["messages"], body.get("tools")
        try:
            if body.get("stream"):
                # Upstream MUST set stream_options:{"include_usage": True} — below.
                stream = budget.guard_stream(
                    lambda: openai_client.chat.completions.create(
                        model=model, messages=messages, tools=tools,
                        stream=True, stream_options={"include_usage": True},
                    ),
                    model=model,
                )
                return sse_response(stream)  # pipe chunks straight through
            completion = await budget.guard_completion(
                lambda: openai_client.chat.completions.create(
                    model=model, messages=messages, tools=tools,
                ),
                model=model,
            )
            return JSONResponse(content=completion)
        except BudgetExceeded as exc:
            return JSONResponse(status_code=402, content={"error": str(exc)})

## The streaming usage requirement (read this)

OpenAI-style SSE omits ``usage`` from every chunk **unless** the caller sets
``stream_options: { "include_usage": True }`` on the upstream request — with it,
a final chunk (empty ``choices``) carries the token ``usage``. This adapter
meters the model turn from that real ``usage``; if a stream ends with no usage
anywhere, :meth:`VapiBudgetGuard.guard_stream` **fails loudly**
(:class:`VapiUsageMissingError`) and releases the hold rather than silently
metering the turn at $0. Set ``include_usage: True`` on your upstream streaming
call. Aborting a stream early (an ``aclose()``, or the task cancellation a
framework raises on client disconnect) runs the generator's ``finally``, which
releases the hold; nothing is metered for an aborted turn.

Scope is strictly **pre-call admission plus per-turn settlement**. There is no
mid-call intervention: an admitted turn runs to completion; nothing here cuts a
turn — or a stream — off partway.
"""

from __future__ import annotations

import inspect
import logging
import math
from collections.abc import AsyncGenerator, AsyncIterable, Awaitable, Callable
from typing import Any, TypeVar

from ..errors import BudgetExceeded, FloeGuardError
from ..gates import vapi as vapi_gate
from ..guard import BudgetGuard, ReservationHandle
from ..voice_pricing import price_voice_leg

logger = logging.getLogger(__name__)

T = TypeVar("T")
C = TypeVar("C")


class VapiUsageMissingError(FloeGuardError):
    """Raised when a guarded stream (or usage-less completion) has nothing to settle.

    Almost always the fix is ``stream_options: { "include_usage": True }`` on the
    upstream streaming request (OpenAI omits ``usage`` from SSE without it). We
    refuse rather than meter the turn at $0 — "we cannot cap what we cannot
    measure". Extends :class:`~floe_guard.errors.FloeGuardError` so it is caught
    by the same family as the priced errors; it is adapter-local (not part of the
    shared ``errors.py`` cross-language family, mirroring ``js/src/adapters/vapi.ts``).
    """

    def __init__(self, model: str) -> None:
        self.model = model
        super().__init__(
            f"Vapi custom-LLM result for model {model!r} contained no token usage "
            f"to settle against. For streaming OpenAI-style SSE, set "
            f"stream_options:{{'include_usage': True}}; for non-streaming calls, "
            f"ensure the completion includes its usage block. The guard refuses "
            f"rather than accruing a silent $0."
        )


def _finite_number(value: Any) -> bool:
    """A finite, non-negative integer token count (``True`` counts as invalid)."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
        and (isinstance(value, int) or float(value).is_integer())
    )


def read_usage(usage: Any) -> tuple[int, int] | None:
    """Read an OpenAI ``usage`` block into settled prompt/completion counts.

    ``usage`` is the OpenAI wire format Vapi speaks (``prompt_tokens`` /
    ``completion_tokens``), accepted as a dict or any object exposing both
    attributes. Returns ``None`` when absent/malformed — both fields must be
    finite numbers, so a partial or non-numeric usage is treated as no usage
    (fail-closed), never coerced to 0.
    """
    if isinstance(usage, dict):
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
    else:
        prompt = getattr(usage, "prompt_tokens", None)
        completion = getattr(usage, "completion_tokens", None)
    if not _finite_number(prompt) or not _finite_number(completion):
        return None
    return int(prompt), int(completion)


def _completion_usage(completion: Any) -> Any:
    """The ``usage`` on a completion-like — dict key or attribute, either way."""
    if isinstance(completion, dict):
        return completion.get("usage")
    return getattr(completion, "usage", None)


class _GuardedStream(AsyncGenerator[Any, None]):
    """Async-iterator wrapper that owns a guarded generator's reservation.

    A generator's ``finally`` cannot run before the generator is started —
    ``aclose()`` on an unstarted generator raises ``GeneratorExit`` at the
    function's first byte, not at the ``try`` (so an unconsumed guarded stream
    would silently leak its reservation). The wrapper closes that gap: closing
    it before any iteration releases the hold eagerly and deterministically.
    After the generator starts, behaviour is unchanged — the generator's own
    ``finally`` settles or releases exactly once on completion, error, or abort.
    """

    __slots__ = ("_gen", "_release", "_started")

    def __init__(
        self,
        gen: AsyncGenerator[Any, None],
        release: Callable[[], None],
    ) -> None:
        self._gen: AsyncGenerator[Any, None] | None = gen
        self._release = release
        self._started = False

    def __aiter__(self) -> _GuardedStream:
        return self

    async def __anext__(self) -> Any:
        gen = self._gen
        if gen is None:
            raise StopAsyncIteration
        self._started = True
        try:
            item = await gen.__anext__()
        except StopAsyncIteration:
            self._gen = None
            raise
        except BaseException:
            self._gen = None
            raise
        return item

    async def asend(self, value: Any) -> Any:
        gen = self._gen
        if gen is None:
            raise StopAsyncIteration
        self._started = True
        try:
            return await gen.asend(value)
        except StopAsyncIteration:
            self._gen = None
            raise
        except BaseException:
            self._gen = None
            raise

    async def athrow(self, *args: Any, **kwargs: Any) -> Any:
        gen = self._gen
        if gen is None:
            raise StopAsyncIteration
        self._started = True
        try:
            return await gen.athrow(*args, **kwargs)
        except StopAsyncIteration:
            self._gen = None
            raise
        except BaseException:
            self._gen = None
            raise

    async def aclose(self) -> None:
        gen, self._gen = self._gen, None
        if gen is None:
            return
        if not self._started:
            self._release()
        await gen.aclose()

    def __del__(self) -> None:
        try:
            started = getattr(self, "_started", True)
            gen = getattr(self, "_gen", None)
            if not started and gen is not None:
                release = getattr(self, "_release", None)
                if release is not None:
                    release()
        except Exception:
            pass


class VapiBudgetGuard:
    """Enforce a BudgetGuard ceiling on a Vapi custom-LLM endpoint: reserve
    before the model turn, settle on the real OpenAI ``usage``, release on
    error/abort, and meter the STT/TTS/telephony legs the proxy never sees.

    Args:
        guard: the BudgetGuard to enforce.
        model: default model id to settle LLM cost against when a per-call
            ``model`` is not passed to :meth:`guard_completion` /
            :meth:`guard_stream`. Vapi's request carries the model, so passing it
            per-call is usual; this is the fallback. Must be priceable via the
            bundled cost map or the guard's ``price_overrides``.
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
        *,
        model: str | None = None,
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

    def assistant_request(
        self,
        *,
        assistant: dict[str, Any] | None = None,
        assistant_id: str | None = None,
        error_message: str = "Sorry, this agent is out of budget right now.",
        estimated_call_usd: float = 0.0,
    ) -> dict[str, Any]:
        """Answer Vapi's ``assistant-request`` webhook from the remaining budget.

        Budget exhausted → ``{"error": error_message}`` (Vapi speaks it, then
        ends the call); otherwise admits with ``{"assistantId": assistant_id}``
        (precedence) or ``{"assistant": assistant}``. A thin wrapper over
        :func:`floe_guard.gates.vapi` — the same webhook contract the hosted
        gateway serves, so the paid upgrade is a URL swap, not a rewrite.
        Respond within ~7.5 s.

        This is coarse, non-binding pre-call admission (a budget read, no
        reservation); the binding hard-stop is the per-turn reserve in
        :meth:`guard_completion` / :meth:`guard_stream`. Pass
        ``estimated_call_usd`` (e.g. ``$/min × expected minutes``) to reject
        earlier, when the remaining budget can't cover the call.

        Raises:
            ValueError: admitted but neither ``assistant`` nor ``assistant_id``
                was given — there'd be nothing to hand Vapi (surfaced by
                :func:`floe_guard.gates.vapi`).
        """
        return vapi_gate(
            self._guard,
            assistant=assistant,
            assistant_id=assistant_id,
            error_message=error_message,
            estimated_call_usd=estimated_call_usd,
        )

    async def guard_completion(
        self,
        run: Callable[[], T | Awaitable[T]],
        *,
        model: str | None = None,
        estimated_cost: float | None = None,
    ) -> T:
        """Guard a **non-streaming** model turn: reserve, run the upstream
        completion, settle on its real ``usage``, release the hold on error.

        Reserving first raises :class:`~floe_guard.errors.BudgetExceeded`
        BEFORE ``run`` is called when the turn would cross the ceiling, so an
        over-budget turn never reaches the upstream LLM. The completion is
        returned untouched for the handler to forward to Vapi. A completion with
        no ``usage`` fails loudly (:class:`VapiUsageMissingError`) and releases
        the hold rather than metering $0.
        """
        resolved = self._resolve_model(model)
        reserved = self._guard.reserve(estimated_cost)
        try:
            result = run()
            if inspect.isawaitable(result):
                result = await result
        except BaseException:
            self._guard.release(reserved)
            raise
        usage = read_usage(_completion_usage(result))
        if usage is None:
            self._guard.release(reserved)
            raise VapiUsageMissingError(resolved)
        self._guard.settle(resolved, usage[0], usage[1], reserved=reserved)
        return result

    def guard_stream(
        self,
        run: Callable[[], AsyncIterable[Any] | Awaitable[AsyncIterable[Any]]],
        *,
        model: str | None = None,
        estimated_cost: float | None = None,
    ) -> _GuardedStream:
        """Guard a **streaming** model turn: reserve, then wrap the SSE chunk
        stream so it settles on the final chunk's ``usage``, releasing the hold
        on error or early abort.

        Reserving first raises
        :class:`~floe_guard.errors.BudgetExceeded` BEFORE the stream is opened
        when the turn would cross the ceiling (this method raises synchronously —
        the handler learns immediately, before piping anything to Vapi). Chunks
        pass through untouched; the returned async iterable is what you forward
        to the SSE response.

        **Usage requirement:** OpenAI SSE only carries ``usage`` when the
        upstream request set ``stream_options: { "include_usage": True }``. A
        stream that ends with no usage anywhere fails loudly
        (:class:`VapiUsageMissingError`) and releases the hold — it is not
        metered at $0.

        **Aborting a stream early** (client disconnect, interrupt) releases
        the hold from the generator's ``finally``, and nothing is metered for
        an aborted turn. The unwinding is triggered by ``aclose()``: an
        explicit ``await stream.aclose()``, a task cancellation raised into
        the consuming generator (what HTTP/WS frameworks do on disconnect),
        or garbage collection of the wrapper. A bare ``break`` in an ``async
        for`` drops the reference and finalizes on a later loop iteration —
        close it explicitly when the hold must clear immediately.

        The returned iterable holds the reservation until it is consumed to
        completion (settled), or closed/aborted (released). Closing it before
        it is ever consumed also releases the hold — a generator's ``finally``
        cannot run before its first pull, so the returned wrapper (not the raw
        generator) owns that path — pipe it straight to the response.
        """
        resolved = self._resolve_model(model)
        # Eager reserve (outside the generator) so a block raises synchronously
        # here, not lazily on first pull — the handler refuses the turn before
        # streaming.
        reserved = self._guard.reserve(estimated_cost)
        return _GuardedStream(
            self._iterate_stream(run, resolved, reserved),
            release=lambda: self._guard.release(reserved),
        )

    async def _iterate_stream(
        self,
        run: Callable[[], AsyncIterable[Any] | Awaitable[AsyncIterable[Any]]],
        model: str,
        reserved: ReservationHandle,
    ) -> AsyncGenerator[Any, None]:
        usage: tuple[int, int] | None = None
        settled = False
        try:
            stream = run()
            if inspect.isawaitable(stream):
                stream = await stream
            async for chunk in stream:
                # Last non-null usage wins — the final (empty-choices) chunk
                # carries it.
                current = read_usage(_completion_usage(chunk))
                if current is not None:
                    usage = current
                yield chunk
            # Clean end: settle on real usage, or fail loudly if none was seen.
            if usage is None:
                raise VapiUsageMissingError(model)
            # Mark settled BEFORE the call: guard.settle owns the reservation on
            # every exit — it releases the hold on its own failure paths
            # (unpriceable model, price_tokens error) before raising — so the
            # finally must not release it a second time (a double release drives
            # `reserved` negative and weakens the ceiling for other in-flight
            # turns).
            settled = True
            self._guard.settle(model, usage[0], usage[1], reserved=reserved)
        finally:
            # Any exit before settle — error mid-stream, missing usage, or an
            # early consumer abort (async-for break → generator aclose()) —
            # frees the hold.
            if not settled:
                self._guard.release(reserved)

    def meter_stt(self, seconds: float) -> float | None:
        """Accrue STT spend for ``seconds`` of transcribed audio (per second).

        The custom-LLM proxy sees only the model leg, so the caller drives this
        (and :meth:`meter_tts` / :meth:`meter_telephony`) explicitly — from
        Vapi's ``end-of-call-report`` webhook, or as the call accrues. Priced
        from the voice map when ``stt_model`` is set, or the
        ``stt_usd_per_second`` override; returns ``None`` (no-op) when the leg
        is unconfigured, and fails closed
        (:class:`~floe_guard.errors.UnpriceableVoiceError`) on a vendor the
        voice map cannot price. Returns the USD accrued, if any.
        """
        cost = price_voice_leg(
            "stt", seconds, model=self._stt_model, override=self._stt_usd_per_second
        )
        if cost is not None:
            self._guard.record_tool("vapi-stt", cost)
        return cost

    def meter_tts(self, characters: float) -> float | None:
        """Accrue TTS spend for ``characters`` of synthesized speech (per 1k chars).

        Priced from the voice map when ``tts_model`` is set, or the
        ``tts_usd_per_1k_chars`` override; returns ``None`` when unconfigured,
        and fails closed on an unpriceable vendor. See :meth:`meter_telephony`.
        """
        cost = price_voice_leg(
            "tts", characters, model=self._tts_model, override=self._tts_usd_per_1k_chars
        )
        if cost is not None:
            self._guard.record_tool("vapi-tts", cost)
        return cost

    def meter_telephony(self, minutes: float) -> float | None:
        """Accrue telephony spend for ``minutes`` of call time (per minute).

        **Per-minute accrual, not live line-cutting**: the guard meters the leg,
        it does not cut the call mid-turn. Priced from the voice map when
        ``telephony`` is set, or the ``telephony_usd_per_minute`` override;
        returns ``None`` (no-op) when the leg is unconfigured, and fails closed
        on a vendor the voice map cannot price. Returns the USD accrued, if any.
        """
        cost = price_voice_leg(
            "telephony", minutes, model=self._telephony, override=self._telephony_usd_per_minute
        )
        if cost is not None:
            self._guard.record_tool("vapi-telephony", cost)
        return cost

    def _resolve_model(self, override: str | None) -> str:
        """The model to settle against: the per-call override (Vapi's request
        model) or the constructor default. Fails loudly if neither is set — the
        guard cannot price a turn it cannot name."""
        model = override or self._model
        if not model:
            raise ValueError(
                "no model to settle the Vapi turn against: pass model= to "
                "guard_completion/guard_stream (Vapi's request model), or set "
                "`model` on the VapiBudgetGuard constructor."
            )
        return model


# BudgetExceeded re-exported so a handler can `except BudgetExceeded` using the
# adapter import path alone, matching the TS `export { BudgetExceeded }`.
__all__ = ["VapiBudgetGuard", "VapiUsageMissingError", "BudgetExceeded"]
