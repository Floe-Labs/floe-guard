"""Pre-call admission gates for voice orchestrators.

These decide **at the call boundary** whether to *admit* or *reject* an incoming
call, from the guard's remaining budget. They return the exact JSON shape each
provider's inbound webhook expects, so a free/local user can reject a call on
budget exhaustion with the **same contract** the hosted gateway serves — the paid
upgrade is a URL swap, not a rewrite.

Scope — strictly pre-call. A gate answers one question: *given the budget left,
do we let this call start?* It does **not** intervene mid-call. Once a call is
admitted it runs to completion; nothing here cuts a call off partway. (The
``guard_stream`` helper can stop a single LLM *generation*, which is not
call-level intervention and must not be described as one.)

Budget, not balance: admission reads the guard's local *remaining budget*, a
ceiling signal — not an account balance. US-only telephony, v1.
"""

from __future__ import annotations

from typing import Any

from .guard import BudgetGuard

__all__ = ["budget_exhausted", "pre_call", "retell", "vapi"]


def budget_exhausted(guard: BudgetGuard, *, estimated_call_usd: float = 0.0) -> bool:
    """True when there isn't budget left to admit one more call.

    With ``estimated_call_usd`` (a per-minute × expected-length estimate, say), the
    call is rejected when the remaining budget can't cover it; with the default
    ``0.0`` it's rejected only once the budget is fully spent (``remaining_usd`` at
    or below zero). Reads the guard's ``remaining_usd`` (net of in-flight
    reservations) — a local budget signal, not a balance.
    """
    return guard.remaining_usd <= estimated_call_usd


def pre_call(guard: BudgetGuard, *, estimated_call_usd: float = 0.0) -> bool:
    """Generic pre-call admission decision: ``True`` = admit, ``False`` = reject.

    The provider-agnostic gate for Pipecat, custom telephony, or any inbound hook
    without a first-class helper below. Wire the ``False`` case into your
    provider's reject path.

    Bland: Bland's *Send Call* metadata field name is an OPEN VERIFICATION ITEM
    (pending confirmation) — this package does not invent it. Use ``pre_call`` to
    get the admit/reject decision and map ``False`` onto Bland's Pathway Webhook
    node once the field name is confirmed.
    """
    return not budget_exhausted(guard, estimated_call_usd=estimated_call_usd)


def retell(
    guard: BudgetGuard,
    *,
    estimated_call_usd: float = 0.0,
    admit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Response for Retell's inbound-call webhook — reject or admit the call.

    On budget exhaustion returns ``{"call_inbound": {"reject": True}}``; otherwise
    ``{"call_inbound": {...}}`` carrying any ``admit`` overrides you pass
    (``dynamic_variables``, ``metadata``, ``override_agent_id``, …).

    Retell contract (verified against docs.retellai.com/features/inbound-call-webhook):
    **only the boolean ``true`` rejects** — any other value is ignored. The webhook
    fires for inbound **phone and SMS** calls (not dial-to-sip); its behaviour for
    web calls is not documented, so treat this as phone/SMS-only. 10s timeout, up to
    3 retries.
    """
    if budget_exhausted(guard, estimated_call_usd=estimated_call_usd):
        return {"call_inbound": {"reject": True}}
    return {"call_inbound": dict(admit or {})}


def vapi(
    guard: BudgetGuard,
    *,
    assistant: dict[str, Any] | None = None,
    assistant_id: str | None = None,
    error_message: str = "Sorry, this agent is out of budget right now.",
    estimated_call_usd: float = 0.0,
) -> dict[str, Any]:
    """Response for Vapi's ``assistant-request`` webhook — reject or admit the call.

    On budget exhaustion returns ``{"error": error_message}`` (Vapi speaks it, then
    ends the call). Otherwise admits with ``{"assistantId": assistant_id}`` if given,
    else ``{"assistant": assistant}``.

    Vapi has **no** ``reject: true`` boolean — rejection *is* returning an error, and
    admission *is* returning an assistant. Respond within **~7.5 s** or the call may
    fail (verified against docs.vapi.ai/server-url/spam-call-rejection).

    Raises:
        ValueError: admitted (budget available) but neither ``assistant`` nor
            ``assistant_id`` was provided — there'd be nothing to hand Vapi.
    """
    if budget_exhausted(guard, estimated_call_usd=estimated_call_usd):
        return {"error": error_message}
    if assistant_id is not None:
        return {"assistantId": assistant_id}
    if assistant is not None:
        return {"assistant": assistant}
    raise ValueError(
        "vapi() admitted the call but has nothing to return: pass assistant= or "
        "assistant_id= for the admit path."
    )
