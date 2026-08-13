"""Tests for offline voice pricing — fail-closed resolution and per-unit math.

The voice twin of tests/test_pricing.py: STT is billed per second, TTS per 1k
chars, telephony per minute, and every schema/vendor mismatch fails closed
(``UnpriceableVoiceError``) rather than metering a leg at a silent $0.
"""

from __future__ import annotations

import pytest

from floe_guard.errors import UnpriceableVoiceError
from floe_guard.pricing import _VOICE_MAP
from floe_guard.voice_pricing import (
    _UNIT_FOR_MODE,
    VoiceRate,
    lookup_voice_rate,
    price_voice_leg,
    resolve_voice_rate,
    voice_leg_cost,
)


def test_resolves_known_stt_vendor_from_cost_map() -> None:
    # $0.0077/min mono ÷ 60 = $0.0001283333/sec.
    rate = lookup_voice_rate("deepgram-nova-3", "stt")
    assert rate == pytest.approx(0.0077 / 60, rel=1e-4)


def test_resolves_known_tts_vendor_from_cost_map() -> None:
    assert lookup_voice_rate("elevenlabs-multilingual-v2", "tts") == pytest.approx(0.10)
    assert lookup_voice_rate("elevenlabs-flash-v2.5", "tts") == pytest.approx(0.05)


def test_resolves_known_telephony_vendor_from_cost_map() -> None:
    assert lookup_voice_rate("twilio-us-inbound-local", "telephony") == pytest.approx(0.0085)


def test_unknown_vendor_is_unpriceable() -> None:
    assert lookup_voice_rate("no-such-vendor-anywhere", "stt") is None
    assert lookup_voice_rate(None, "stt") is None


def test_mode_mismatch_fails_closed() -> None:
    # AC3: an STT entry asked to price a TTS leg is a schema mismatch — refuse it
    # rather than mis-bill a per-second rate as if it were per-1k-chars.
    assert lookup_voice_rate("deepgram-nova-3", "tts") is None
    assert lookup_voice_rate("elevenlabs-flash-v2.5", "stt") is None
    assert lookup_voice_rate("twilio-us-inbound-local", "stt") is None


def test_unit_mismatch_fails_closed() -> None:
    # AC3: an entry whose unit disagrees with its mode's canonical unit is
    # rejected even if the mode matches — a corrupted/edited entry can't slip a
    # wrong-unit rate through as a valid-looking price.
    from floe_guard import voice_pricing

    poisoned = dict(voice_pricing._VOICE_MAP)
    poisoned["bad-stt"] = {
        "mode": "stt",
        "unit": "usd_per_minute",  # wrong: stt must be usd_per_second
        "rate": 0.0077,
        "provider": "test",
    }
    original = voice_pricing._VOICE_MAP
    try:
        voice_pricing._VOICE_MAP = poisoned
        assert lookup_voice_rate("bad-stt", "stt") is None
    finally:
        voice_pricing._VOICE_MAP = original


def test_non_finite_or_negative_rate_is_unpriceable() -> None:
    from floe_guard import voice_pricing

    poisoned = dict(voice_pricing._VOICE_MAP)
    poisoned["nan-stt"] = {
        "mode": "stt",
        "unit": "usd_per_second",
        "rate": float("inf"),
        "provider": "test",
    }
    poisoned["neg-stt"] = {
        "mode": "stt",
        "unit": "usd_per_second",
        "rate": -0.01,
        "provider": "test",
    }
    original = voice_pricing._VOICE_MAP
    try:
        voice_pricing._VOICE_MAP = poisoned
        assert lookup_voice_rate("nan-stt", "stt") is None
        assert lookup_voice_rate("neg-stt", "stt") is None
    finally:
        voice_pricing._VOICE_MAP = original


def test_resolve_voice_rate_raises_fail_closed_for_unknown_vendor() -> None:
    # AC2: a vendor absent from the voice map with no override raises the
    # fail-closed error — never a silent $0.
    with pytest.raises(UnpriceableVoiceError) as exc:
        resolve_voice_rate("mystery-tts", "tts")
    assert exc.value.vendor == "mystery-tts"
    assert exc.value.mode == "tts"


def test_override_wins_over_cost_map() -> None:
    resolved = resolve_voice_rate("deepgram-nova-3", "stt", override=0.0002)
    assert isinstance(resolved, VoiceRate)
    assert resolved.source == "override"
    assert resolved.rate == 0.0002
    assert resolved.unit == _UNIT_FOR_MODE["stt"]


def test_override_prices_a_vendor_the_map_cannot() -> None:
    resolved = resolve_voice_rate("some-brand-new-tts", "tts", override=0.07)
    assert resolved.source == "override"
    assert resolved.rate == 0.07


def test_override_rejects_non_finite() -> None:
    with pytest.raises(ValueError):
        resolve_voice_rate("x", "stt", override=float("nan"))
    with pytest.raises(ValueError):
        resolve_voice_rate("x", "stt", override=-1.0)


def test_voice_leg_cost_units() -> None:
    # STT: seconds * $/sec
    assert voice_leg_cost("stt", 10.0, 0.0001) == pytest.approx(0.001)
    # TTS: chars / 1000 * $/1k-chars
    assert voice_leg_cost("tts", 2000, 0.05) == pytest.approx(0.10)
    # telephony: minutes * $/min
    assert voice_leg_cost("telephony", 3.0, 0.0085) == pytest.approx(0.0255)


def test_voice_leg_cost_clamps_negative_quantity() -> None:
    assert voice_leg_cost("stt", -5.0, 0.01) == 0.0


def test_price_voice_leg_skips_when_unconfigured() -> None:
    # Neither a vendor nor an override — the leg is un-metered (token-only
    # contract preserved), NOT a fail-closed raise.
    assert price_voice_leg("stt", 10.0) is None


def test_price_voice_leg_fails_closed_when_configured_but_unpriceable() -> None:
    with pytest.raises(UnpriceableVoiceError):
        price_voice_leg("tts", 1000, model="ghost-vendor")


def test_cost_map_has_every_required_vendor() -> None:
    # AC4: ≥1 entry each for Deepgram, AssemblyAI, ElevenLabs, Cartesia, Rime,
    # Twilio. Telnyx is deferred (verified list rate pending) — see the
    # `# TODO(telnyx)` note in voice_pricing.py and scripts/update-cost-map.mjs.
    providers = {entry["provider"] for entry in _VOICE_MAP.values()}
    for required in ("deepgram", "assemblyai", "elevenlabs", "cartesia", "rime", "twilio"):
        assert required in providers, f"voice map missing a {required} entry"
    assert "telnyx" not in providers  # deferred, on purpose


def test_every_voice_entry_matches_its_declared_unit() -> None:
    # Invariant that survives a hand-edit or a script refresh: every entry's unit
    # is the canonical one for its mode, so no leg can ever be mis-billed.
    for vendor, entry in _VOICE_MAP.items():
        mode = entry.get("mode")
        assert mode in _UNIT_FOR_MODE, f"{vendor} has unknown mode {mode!r}"
        assert entry.get("unit") == _UNIT_FOR_MODE[mode], f"{vendor} unit/mode mismatch"
        assert isinstance(entry.get("rate"), (int, float)) and entry["rate"] > 0, (
            f"{vendor} has a non-positive rate"
        )
        assert entry.get("provider"), f"{vendor} missing provider"
