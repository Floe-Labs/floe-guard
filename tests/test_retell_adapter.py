"""RetellBudgetGuard — reserve when a `response_required` arrives, settle on real
token usage after `content_complete`, release when a newer `response_id`
interrupts, and price STT/TTS/telephony legs via the voice cost map.

Fake Retell interaction events drive the adapter — no `ws` server, no Retell SDK,
no network, no keys. Everything is a plain parsed-message dict.
"""

from __future__ import annotations

import pytest

from floe_guard import BudgetGuard, ManualPrice, gates
from floe_guard.errors import BudgetExceeded, UnpriceableVoiceError
from floe_guard.integrations.retell import RetellBudgetGuard
from floe_guard.voice_pricing import price_voice_leg

PRICE = ManualPrice(1e-6, 2e-6)


def response_required(response_id: int) -> dict:
    """A `response_required` interaction event, as Retell delivers it over the WS."""
    return {
        "interaction_type": "response_required",
        "response_id": response_id,
        "transcript": [{"role": "user", "content": "hi"}],
    }


def _guard(**overrides) -> BudgetGuard:
    return BudgetGuard(limit_usd=1.00, price_overrides={"m": PRICE, **overrides})


# ── reserve before the LLM turn ────────────────────────────────────────────────


def test_blocks_over_budget_turn() -> None:
    guard = _guard()
    guard.record_tool("prior", 1.0)  # spend the ceiling
    budget = RetellBudgetGuard(guard, model="m")

    decision = budget.begin_turn(response_required(1))
    assert decision.admitted is False
    assert decision.response_id == 1
    assert isinstance(decision.error, BudgetExceeded)


def test_admits_under_budget_turn() -> None:
    guard = _guard()
    budget = RetellBudgetGuard(guard, model="m")

    decision = budget.begin_turn(response_required(1))
    assert decision.admitted is True
    assert decision.response_id == 1
    assert decision.error is None


def test_begin_turn_is_idempotent_for_a_still_open_turn() -> None:
    # Calling twice with the same still-open response_id must not double-hold.
    guard = _guard()
    budget = RetellBudgetGuard(guard, model="m")

    assert budget.begin_turn(response_required(7)).admitted is True
    assert budget.begin_turn(response_required(7)).admitted is True
    assert len(budget._slots) == 1
    # Only one reservation held: the default (zero) estimate for the first turn.
    assert guard._reserved == pytest.approx(0.0)


# ── settle on real usage ───────────────────────────────────────────────────────


def test_settles_priced_cost_and_frees_the_hold() -> None:
    guard = _guard()
    budget = RetellBudgetGuard(guard, model="m")

    budget.begin_turn(response_required(1))
    budget.settle_turn(1, {"promptTokens": 1000, "completionTokens": 500})

    # 1000 * 1e-6 + 500 * 2e-6 = 0.002
    assert guard.advisory().spent_usd == pytest.approx(0.002)
    # Reservation consumed (not leaked): remaining == limit - spent, no hold slice.
    assert guard.remaining_usd == pytest.approx(1.0 - 0.002)
    assert len(guard.spend_log) == 1
    assert guard.spend_log[0].model_or_tool == "m"


def test_settles_a_turn_never_begun_as_plain_record() -> None:
    guard = _guard()
    budget = RetellBudgetGuard(guard, model="m")

    budget.settle_turn(99, {"promptTokens": 1000, "completionTokens": 0})
    assert guard.advisory().spent_usd == pytest.approx(0.001)


# ── release on interrupt ───────────────────────────────────────────────────────


def test_newer_response_id_releases_prior_open_turn() -> None:
    guard = _guard()
    budget = RetellBudgetGuard(guard, model="m")

    # Turn 1: settle so the guard's next-call estimate becomes non-zero (0.002),
    # making the following turn's reservation observable in remaining_usd.
    budget.begin_turn(response_required(1))
    budget.settle_turn(1, {"promptTokens": 1000, "completionTokens": 500})
    after_turn1 = guard.remaining_usd  # 1.0 - 0.002

    # Turn 2: reserves the 0.002 estimate — remaining drops.
    budget.begin_turn(response_required(2))
    assert guard.remaining_usd == pytest.approx(after_turn1 - 0.002)

    # Turn 3 (higher response_id) arrives before turn 2 settles — turn 2 is
    # interrupted and its hold released. Turn 3 then holds its own 0.002.
    budget.begin_turn(response_required(3))
    assert guard.remaining_usd == pytest.approx(after_turn1 - 0.002)  # only turn 3 held
    assert guard.advisory().spent_usd == pytest.approx(0.002)  # interrupt accrued nothing


def test_close_releases_still_open_turn() -> None:
    guard = _guard()
    budget = RetellBudgetGuard(guard, model="m")

    budget.begin_turn(response_required(1))
    budget.settle_turn(1, {"promptTokens": 1000, "completionTokens": 500})
    after_turn1 = guard.remaining_usd

    budget.begin_turn(response_required(2))  # holds 0.002, never settles
    assert guard.remaining_usd == pytest.approx(after_turn1 - 0.002)

    budget.close()
    assert guard.remaining_usd == pytest.approx(after_turn1)


def test_interrupted_turn_settled_late_accrues_without_stealing_a_hold() -> None:
    # Turn 1 is interrupted by turn 2; turn 1's real usage then arrives late.
    # It must record actual usage against a zero reservation (not turn 2's hold).
    guard = _guard()
    budget = RetellBudgetGuard(guard, model="m")

    budget.begin_turn(response_required(1))
    budget.begin_turn(response_required(2))  # interrupts turn 1, releases its hold

    assert guard.remaining_usd == pytest.approx(1.0)  # both holds released

    # Turn 2 settles normally — consumes its own hold.
    budget.settle_turn(2, {"promptTokens": 0, "completionTokens": 250})
    # Turn 1's late usage settles as a plain record (no open slot).
    budget.settle_turn(1, {"promptTokens": 1000, "completionTokens": 500})
    assert guard.advisory().spent_usd == pytest.approx(0.002 + 250 * 2e-6)
    assert guard.remaining_usd == pytest.approx(1.0 - (0.002 + 250 * 2e-6))


# ── pre-call admission via gates.retell ────────────────────────────────────────


def test_admit_call_matches_gates_retell() -> None:
    guard = BudgetGuard(limit_usd=1.00)
    budget = RetellBudgetGuard(guard, model="m")

    admit = budget.admit_call(admit={"metadata": {"plan": "free"}})
    assert admit == gates.retell(guard, admit={"metadata": {"plan": "free"}})
    assert "reject" not in admit["call_inbound"]

    guard.record_tool("prior", 1.0)  # spend the ceiling
    assert budget.admit_call() == {"call_inbound": {"reject": True}}


# ── voice legs price via the cost map ──────────────────────────────────────────


def test_meters_stt_tts_telephony_from_named_vendors() -> None:
    guard = BudgetGuard(limit_usd=10.00, price_overrides={"m": PRICE})
    budget = RetellBudgetGuard(
        guard,
        model="m",
        stt_model="deepgram-nova-3",
        tts_model="elevenlabs-flash-v2.5",
        telephony="twilio-us-inbound-local",
    )

    stt = budget.meter_stt(30)  # 30s
    tts = budget.meter_tts(1_200)  # 1.2k chars
    tel = budget.meter_telephony(2.5)  # 2.5 min

    assert stt == pytest.approx(price_voice_leg("stt", 30, model="deepgram-nova-3"))
    assert tts == pytest.approx(price_voice_leg("tts", 1_200, model="elevenlabs-flash-v2.5"))
    assert tel == pytest.approx(price_voice_leg("telephony", 2.5, model="twilio-us-inbound-local"))
    assert guard.tool_costs["retell-stt"] == pytest.approx(stt)
    assert guard.tool_costs["retell-tts"] == pytest.approx(tts)
    assert guard.tool_costs["retell-telephony"] == pytest.approx(tel)


def test_unconfigured_leg_is_unmetered() -> None:
    guard = BudgetGuard(limit_usd=10.00, price_overrides={"m": PRICE})
    budget = RetellBudgetGuard(guard, model="m")

    assert budget.meter_stt(30) is None
    assert budget.meter_tts(30) is None
    assert budget.meter_telephony(30) is None
    assert "retell-stt" not in guard.tool_costs
    assert guard.advisory().spent_usd == 0.0


@pytest.mark.parametrize(
    "leg,mode",
    [
        ("stt_model", "stt"),
        ("tts_model", "tts"),
        ("telephony", "telephony"),
    ],
)
def test_fails_closed_on_unpriceable_vendor(leg: str, mode: str) -> None:
    guard = BudgetGuard(limit_usd=10.00, price_overrides={"m": PRICE})
    budget = RetellBudgetGuard(guard, model="m", **{leg: "no-such-vendor"})

    with pytest.raises(UnpriceableVoiceError):
        getattr(budget, f"meter_{mode}")(1)
    assert guard.advisory().spent_usd == 0.0


# ── response() event shape ─────────────────────────────────────────────────────


def test_response_event_shape() -> None:
    guard = BudgetGuard(limit_usd=1.00)
    budget = RetellBudgetGuard(guard, model="m")

    partial = budget.response(3, "Sure,", complete=False)
    assert partial == {
        "response_type": "response",
        "response_id": 3,
        "content": "Sure,",
        "content_complete": False,
    }

    final = budget.response(3, " one moment!", complete=True, end_call=True)
    assert final["content_complete"] is True
    assert final["end_call"] is True


# -- fail-closed payloads -------------------------------------------------------


def test_begin_turn_requires_an_integer_response_id() -> None:
    # A malformed event (missing or odd response_id) is refused fail-closed
    # rather than mis-keying a turn or releasing a live one.
    guard = _guard()
    budget = RetellBudgetGuard(guard, model="m")

    with pytest.raises(ValueError, match="response_id"):
        budget.begin_turn({"interaction_type": "response_required"})  # no response_id
    with pytest.raises(ValueError, match="response_id"):
        budget.begin_turn(
            {"interaction_type": "response_required", "response_id": "1"}
        )  # not a number
    assert guard.remaining_usd == pytest.approx(1.0)  # nothing reserved or released


def test_settle_turn_refuses_malformed_usage_payload() -> None:
    guard = _guard()
    budget = RetellBudgetGuard(guard, model="m")

    with pytest.raises(ValueError, match="promptTokens"):
        budget.settle_turn(1, {"completionTokens": 5})  # missing promptTokens
    with pytest.raises(ValueError, match="completionTokens"):
        budget.settle_turn(1, {"promptTokens": 10, "completionTokens": None})
    with pytest.raises(ValueError, match="promptTokens"):
        budget.settle_turn(1, {"promptTokens": "lots", "completionTokens": 5})
    with pytest.raises(ValueError, match="promptTokens"):
        budget.settle_turn(1, {"promptTokens": -10, "completionTokens": 5})
    with pytest.raises(ValueError, match="completionTokens"):
        budget.settle_turn(1, {"promptTokens": 10, "completionTokens": 5.5})
    assert guard.advisory().spent_usd == 0.0  # nothing settled at a guessed cost


def test_settle_turn_accepts_attribute_style_usage() -> None:
    from types import SimpleNamespace

    guard = _guard()
    budget = RetellBudgetGuard(guard, model="m")
    budget.begin_turn(response_required(1))

    budget.settle_turn(1, SimpleNamespace(promptTokens=1000, completionTokens=500))
    assert guard.advisory().spent_usd == pytest.approx(0.002)
    assert guard.remaining_usd == pytest.approx(1.0 - 0.002)
