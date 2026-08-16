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

import math
from typing import Any

from .guard import BudgetGuard

__all__ = ["budget_exhausted", "pre_call", "retell", "vapi"]


def budget_exhausted(guard: BudgetGuard, *, estimated_call_usd: float = 0.0) -> bool:
    """True when there isn't budget left to admit one more call.

    A **non-binding preflight read**: it checks the guard's ``remaining_usd`` but
    does *not* reserve anything, so under concurrent inbound calls it can admit
    more than the budget strictly covers (two calls may see the same headroom and
    both pass). It is coarse admission control — the binding, atomic hard-stop is
    the in-call guard (``check`` / ``reserve`` while the call runs). This mirrors
    the hosted pre-dial gate, which is likewise check-only.

    With the default ``estimated_call_usd=0.0`` a call is rejected only once the
    budget is fully spent (``remaining_usd`` at or below zero). Pass an estimate
    (e.g. ``$/min × expected minutes``) to reject earlier, when the remaining
    budget can't cover it; an estimate that *exactly* equals the remaining budget
    still admits (inclusive ceiling, matching ``BudgetGuard.reserve``). Budget
    signal, not a balance.

    Raises:
        ValueError: ``estimated_call_usd`` is negative or non-finite — a bad
            estimate must not silently admit an exhausted guard (NaN/negative
            comparisons are always false, the same trap ``BudgetGuard`` rejects for
            ``limit_usd``).
    """
    if not math.isfinite(estimated_call_usd) or estimated_call_usd < 0.0:
        raise ValueError(
            "estimated_call_usd must be a finite, non-negative number, "
            f"got {estimated_call_usd!r}"
        )
    remaining_usd = guard.remaining_usd
    return remaining_usd <= 0.0 or remaining_usd < estimated_call_usd


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
    (``dynamic_variables``, ``metadata``, ``override_agent_id``, …). A top-level
    ``reject`` key in ``admit`` is **ignored** — the gate's budget decision, not the
    caller's overrides, controls admission.

    Retell contract (verified against docs.retellai.com/features/inbound-call-webhook):
    **only the boolean ``true`` rejects** — any other value is ignored. The webhook
    fires for inbound **phone and SMS** calls (not dial-to-sip); its behaviour for
    web calls is not documented, so treat this as phone/SMS-only. 10s timeout, up to
    3 retries.
    """
    if budget_exhausted(guard, estimated_call_usd=estimated_call_usd):
        return {"call_inbound": {"reject": True}}
    # The gate's budget decision — not the caller's overrides — controls admission.
    # Strip ``reject`` from admit overrides so an errant ``admit={"reject": True}``
    # on an available-budget call can't flip the response into Retell's reject shape.
    overrides = dict(admit or {})
    overrides.pop("reject", None)
    return {"call_inbound": overrides}


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
    ends the call) — e.g. ``{"error": "No budget."}``. Otherwise admits with
    ``{"assistantId": assistant_id}`` if given (``assistant_id`` takes precedence),
    else ``{"assistant": assistant}`` — e.g.
    ``{"assistant": {"model": {"provider": "openai", "model": "gpt-4o"}}}``.

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
