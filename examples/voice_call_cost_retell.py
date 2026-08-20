"""What did this voice call cost? — Retell custom-LLM WS, priced from the map.

No API key, no account, no network. Fake ``response_required`` interaction
events drive two turns through ``RetellBudgetGuard`` (reserve on arrival,
settle on real token usage after ``content_complete``), plus the STT / TTS /
telephony legs the socket never sees. **No manual prices** — every leg is priced
from the bundled cost map by naming its vendor, so floe-guard answers "what did
this call cost" at the $0 tier out of the box.

Run:
    python examples/voice_call_cost_retell.py
"""

from __future__ import annotations

import asyncio

from floe_guard import BudgetGuard
from floe_guard.integrations.retell import RetellBudgetGuard


def _response_required(response_id: int) -> dict:
    """A Retell interaction event as it arrives over the WS."""
    return {
        "interaction_type": "response_required",
        "response_id": response_id,
        "transcript": [{"role": "user", "content": "hi"}],
    }


async def run() -> BudgetGuard:
    guard = BudgetGuard(limit_usd=1.00)
    budget = RetellBudgetGuard(
        guard,
        model="gpt-4o",  # LLM leg — priced from the token cost map
        stt_model="deepgram-nova-3",  # $/sec from the voice map
        tts_model="elevenlabs-flash-v2.5",  # $/1k-chars from the voice map
        telephony="twilio-us-inbound-local",  # $/min from the voice map
    )

    # Pre-call admission: Retell's inbound-call webhook, from the budget left.
    admit = budget.admit_call(admit={"dynamic_variables": {"plan": "free"}})
    print(f"inbound-call admission: {admit}")

    # Two turns over the WS: admit, stream a response, settle real usage.
    for response_id in (1, 2):
        turn = budget.begin_turn(_response_required(response_id))
        assert turn.admitted  # reserves before the LLM call
        event = budget.response(response_id, "Sure, one moment!", complete=True)
        assert event["content_complete"] is True
        # Real token usage from your LLM, settled after content_complete.
        budget.settle_turn(response_id, {"promptTokens": 300, "completionTokens": 110})

    # The socket never sees STT/TTS/telephony — meter them from the call record.
    budget.meter_stt(8.0)  # 8s of caller audio
    budget.meter_tts(180)  # 180 chars spoken
    budget.meter_telephony(1.5)  # a 1.5-minute call

    budget.close()  # release any still-open hold on call teardown
    return guard


def _print_breakdown(guard: BudgetGuard) -> None:
    print("Per-leg call cost (all priced from the bundled cost map, no manual rates):")
    for event in guard.spend_log:
        print(f"  {event.model_or_tool:<20} ${event.cost_usd:.6f}")
    print(f"  {'TOTAL':<20} ${guard.advisory().spent_usd:.6f}")


async def main() -> None:
    guard = await run()
    _print_breakdown(guard)


if __name__ == "__main__":
    asyncio.run(main())
