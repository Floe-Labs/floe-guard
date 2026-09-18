"""Tests for offline voice pricing — fail-closed resolution and per-unit math.

The voice twin of tests/test_pricing.py: STT is billed per second, TTS per 1k
chars, telephony per minute, and every schema/vendor mismatch fails closed
(``UnpriceableVoiceError``) rather than metering a leg at a silent $0.
"""

from __future__ import annotations

import re

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
    # ≥1 entry each for the voice vendors, plus the P1.11 modalities.
    providers = {entry["provider"] for entry in _VOICE_MAP.values()}
    for required in (
        "deepgram",
        "assemblyai",
        "elevenlabs",
        "cartesia",
        "rime",
        "twilio",
        "telnyx",  # SMS (P1.11)
        "modal",  # GPU-second
        "gcp",  # OCR
        "aws",  # OCR
        "tavus",  # avatar
    ):
        assert required in providers, f"leg map missing a {required} entry"


def test_telnyx_ships_sms_but_still_no_telephony_rate() -> None:
    """The deferral was always about the TELEPHONY rate, not the vendor.

    P1.11 added sourced Telnyx SMS rates, so a blanket "telnyx not in providers"
    assertion would now fail for the wrong reason. What must stay true is the
    narrow thing: no Telnyx per-minute telephony rate was ever verified, so none
    ships. See the `# TODO(telnyx)` note in scripts/update-cost-map.mjs.
    """
    telnyx_modes = {e["mode"] for e in _VOICE_MAP.values() if e.get("provider") == "telnyx"}
    assert telnyx_modes == {"sms"}


def test_runpod_is_not_vendored() -> None:
    """Runpod quotes per hour across two tiers and the map cannot know which one
    a caller is on, so vendoring either would be a guess about someone else's
    bill. Runpod users add it through their own rate card."""
    assert not [k for k in _VOICE_MAP if k.startswith("runpod")]


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


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_voice_leg_cost_rejects_non_finite_quantity(bad: float) -> None:
    # A NaN/inf quantity would poison the guard's running total — fail closed.
    with pytest.raises(ValueError, match="finite"):
        voice_leg_cost("stt", bad, 0.0001)
    with pytest.raises(ValueError, match="finite"):
        price_voice_leg("stt", bad, model="deepgram-nova-3")


# ── P1.11: SMS / OCR / GPU / avatar ──────────────────────────────────────────


def test_every_mode_has_a_unit() -> None:
    """A mode with no unit fails closed on EVERY call, which is worse than not
    having the mode at all. This is the bar for adding one."""
    assert set(_UNIT_FOR_MODE) == {"stt", "tts", "telephony", "sms", "ocr", "gpu", "avatar"}


def test_new_modes_bill_in_their_own_units() -> None:
    # segments x $/segment, pages x $/page, gpu-seconds x $/gpu-second,
    # minutes x $/minute. Each is linear; only tts divides (per 1k chars).
    assert voice_leg_cost("sms", 3, 0.0083) == pytest.approx(0.0249)
    assert voice_leg_cost("ocr", 200, 0.0015) == pytest.approx(0.30)
    assert voice_leg_cost("gpu", 90, 0.001097) == pytest.approx(0.09873)
    assert voice_leg_cost("avatar", 2.5, 0.37) == pytest.approx(0.925)


def test_resolves_a_vendor_in_each_new_modality() -> None:
    assert lookup_voice_rate("twilio-sms-us-outbound", "sms") == pytest.approx(0.0083)
    assert lookup_voice_rate("telnyx-sms-us-outbound", "sms") == pytest.approx(0.004)
    assert lookup_voice_rate("modal-h100-sxm5", "gpu") == pytest.approx(0.001097)
    assert lookup_voice_rate("aws-textract-detect-document-text", "ocr") == pytest.approx(0.0015)
    assert lookup_voice_rate("tavus-cvi-starter", "avatar") == pytest.approx(0.37)


def test_gpu_rate_is_per_gpu_second_not_per_hour() -> None:
    """Modal publishes $/sec natively, so no conversion is applied. A per-hour
    figure stored here unconverted would over-bill by 3600x — the same class of
    error the unit check exists to catch."""
    assert lookup_voice_rate("modal-h100-sxm5", "gpu") == pytest.approx(0.001097)
    assert _VOICE_MAP["modal-h100-sxm5"]["unit"] == "usd_per_gpu_second"


def test_ocr_rate_is_per_page_not_per_thousand_pages() -> None:
    """Vendors quote per 1,000 pages; the map stores per PAGE. Getting this wrong
    is a 1000x mis-bill, so it is pinned rather than trusted."""
    # Google lists $1.50/1k pages -> $0.0015/page.
    gcp = "gcp-vision-document-text-detection"
    assert lookup_voice_rate(gcp, "ocr") == pytest.approx(0.0015)
    assert price_voice_leg("ocr", 1000, model=gcp) == pytest.approx(1.50)


def test_volume_tiers_ship_as_separate_keys() -> None:
    """Which tier you are on is a fact about YOU, so it is a choice of key rather
    than a curation guess baked into one number."""
    assert lookup_voice_rate("gcp-vision-document-text-detection", "ocr") == pytest.approx(0.0015)
    assert lookup_voice_rate(
        "gcp-vision-document-text-detection-high-volume", "ocr"
    ) == pytest.approx(0.0006)
    # Tavus: every plan is its own key, and no bare "tavus-cvi" implies a default.
    assert lookup_voice_rate("tavus-cvi-growth", "avatar") == pytest.approx(0.32)
    assert lookup_voice_rate("tavus-cvi-business", "avatar") == pytest.approx(0.26)
    assert "tavus-cvi" not in _VOICE_MAP


def test_p111_entries_cite_a_source_and_a_retrieval_date() -> None:
    """No entry curated in P1.11 ships without a public citation."""
    for key, entry in _VOICE_MAP.items():
        if entry.get("mode") not in ("sms", "ocr", "gpu", "avatar"):
            continue
        assert entry.get("source_url", "").startswith("https://"), f"{key} has no source_url"
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", entry.get("retrieved_at", "")), (
            f"{key} has no retrieval date"
        )


def test_bundled_rates_resolve_as_unconfirmed() -> None:
    """A list price is not a cost. It resolves with its provenance and
    confirmed=False so a console can ask the user to confirm or correct it,
    rather than reporting a guess as though it were their bill."""
    resolved = resolve_voice_rate("tavus-cvi-starter", "avatar")
    assert resolved.source == "cost_map"
    assert resolved.confirmed is False
    assert resolved.provider == "tavus"
    assert resolved.source_url == "https://www.tavus.io/pricing"
    assert resolved.retrieved_at == "2026-09-18"


def test_legacy_voice_rates_have_no_invented_provenance() -> None:
    """The 13 pre-P1.11 rates have no verified URL on record. Back-filling a
    plausible-looking one would launder a guess into a citation, so absent
    provenance is reported honestly as absent."""
    resolved = resolve_voice_rate("deepgram-nova-3", "stt")
    assert resolved.source == "cost_map"
    assert resolved.source_url is None
    assert resolved.confirmed is False
