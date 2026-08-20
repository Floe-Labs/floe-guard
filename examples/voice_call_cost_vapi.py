"""What did this voice call cost? — Vapi custom-LLM, priced from the cost map.

No API key, no account, no network. A fake OpenAI-shaped completion drives one
turn through ``VapiBudgetGuard.guard_completion`` (reserve → run → settle on the
real ``usage``), plus the STT / TTS / telephony legs metered from Vapi's
``end-of-call-report``. **No manual prices** — every leg is priced from the
bundled cost map by naming its vendor, so floe-guard answers "what did this call
cost" at the $0 tier out of the box.

Run:
    python examples/voice_call_cost_vapi.py
"""

from __future__ import annotations

import asyncio

from floe_guard import BudgetGuard
from floe_guard.integrations.vapi import VapiBudgetGuard


def _fake_completion(prompt_tokens: int, completion_tokens: int) -> dict:
    """An OpenAI-format completion as Vapi's custom-LLM endpoint returns it."""
    return {
        "id": "chatcmpl-fake",
        "choices": [{"message": {"role": "assistant", "content": "Sure, one moment!"}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


async def run() -> BudgetGuard:
    guard = BudgetGuard(limit_usd=1.00)
    budget = VapiBudgetGuard(
        guard,
        stt_model="deepgram-nova-3",  # $/sec from the voice map
        tts_model="elevenlabs-flash-v2.5",  # $/1k-chars from the voice map
        telephony="twilio-us-inbound-local",  # $/min from the voice map
    )

    # Pre-call admission: Vapi's assistant-request webhook, from the budget left.
    print(f"assistant-request admission: {budget.assistant_request(assistant_id='asst_1')}")

    # One turn through the custom-LLM proxy: reserve → run → settle on real usage.
    completion = await budget.guard_completion(
        lambda: _fake_completion(prompt_tokens=600, completion_tokens=220),
        model="gpt-4o",  # LLM leg — priced from the token cost map
    )
    assert completion["usage"]["completion_tokens"] == 220

    # The proxy never sees STT/TTS/telephony — meter them from end-of-call-report.
    budget.meter_stt(8.0)  # 8s of caller audio
    budget.meter_tts(180)  # 180 chars spoken
    budget.meter_telephony(1.5)  # a 1.5-minute call
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
