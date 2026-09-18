"""User-supplied rate cards — the rates you actually pay beat the bundled list.

The bundled map holds PUBLIC LIST prices. Almost nobody pays list, so the load-
bearing behaviour here is that a declared rate wins, that a declared rate for a
vendor the map has never heard of works at all, and that a BROKEN declaration
fails closed instead of silently reverting to a list price the user never chose.
"""

from __future__ import annotations

import json

import pytest

from floe_guard import rate_card
from floe_guard.errors import UnpriceableVoiceError
from floe_guard.voice_pricing import lookup_voice_rate, price_voice_leg, resolve_voice_rate


@pytest.fixture(autouse=True)
def _clear_rate_card():
    """Never leak a card between tests — it is process-global state."""
    rate_card.set_rate_card({})
    yield
    rate_card.set_rate_card({})


def test_no_rate_card_is_the_zero_config_default() -> None:
    assert rate_card.current_rate_card() == {}
    # The bundled list price still resolves.
    assert lookup_voice_rate("gcp-vision-document-text-detection", "ocr") == pytest.approx(0.0015)


def test_declared_rate_beats_the_bundled_list_price() -> None:
    # The whole point: you negotiated $0.0004/page, the list is $0.0015.
    rate_card.set_rate_card(
        {
            "gcp-vision-document-text-detection": {
                "mode": "ocr",
                "unit": "usd_per_page",
                "rate": 0.0004,
            }
        }
    )
    resolved = resolve_voice_rate("gcp-vision-document-text-detection", "ocr")
    assert resolved.rate == pytest.approx(0.0004)
    assert resolved.source == "rate_card"
    assert resolved.confirmed is True  # you typed it


def test_rate_card_can_price_a_vendor_the_map_has_never_heard_of() -> None:
    # This is how Mistral OCR / Azure DI / HeyGen / Simli get priced at all: they
    # publish no usable public figure, so nothing ships for them by design.
    assert lookup_voice_rate("mistral-ocr", "ocr") is None
    rate_card.set_rate_card({"mistral-ocr": {"mode": "ocr", "unit": "usd_per_page", "rate": 0.001}})
    assert price_voice_leg("ocr", 250, model="mistral-ocr") == pytest.approx(0.25)


def test_a_broken_declaration_fails_closed_rather_than_using_list_price() -> None:
    """The dangerous direction, pinned.

    The user declared what this leg costs them. If that declaration is unusable,
    quietly substituting Google's list price would produce a confident, wrong
    number — so it raises instead.
    """
    rate_card.set_rate_card(
        {
            "gcp-vision-document-text-detection": {
                "mode": "ocr",
                "unit": "usd_per_1k_pages",  # wrong: ocr must be usd_per_page
                "rate": 0.4,
            }
        }
    )
    with pytest.raises(UnpriceableVoiceError):
        resolve_voice_rate("gcp-vision-document-text-detection", "ocr")


def test_a_card_entry_for_another_mode_does_not_shadow_this_leg() -> None:
    # Declaring an avatar rate for a key says nothing about its OCR leg, so the
    # bundled price must still resolve.
    rate_card.set_rate_card(
        {
            "gcp-vision-document-text-detection": {
                "mode": "avatar",
                "unit": "usd_per_minute",
                "rate": 9.0,
            }
        }
    )
    assert lookup_voice_rate("gcp-vision-document-text-detection", "ocr") == pytest.approx(0.0015)


def test_override_still_beats_the_rate_card() -> None:
    rate_card.set_rate_card({"acme": {"mode": "sms", "unit": "usd_per_segment", "rate": 0.002}})
    resolved = resolve_voice_rate("acme", "sms", override=0.001)
    assert resolved.source == "override"
    assert resolved.rate == pytest.approx(0.001)


def test_confirmed_false_survives_the_round_trip() -> None:
    # An imported or draft card can mark entries unconfirmed so a console flags
    # them for review before anyone trusts the margin they produce.
    rate_card.set_rate_card(
        {"acme": {"mode": "sms", "unit": "usd_per_segment", "rate": 0.002, "confirmed": False}}
    )
    assert resolve_voice_rate("acme", "sms").confirmed is False


def test_loads_from_a_file_and_from_inline_json(tmp_path) -> None:
    entry = {"acme": {"mode": "sms", "unit": "usd_per_segment", "rate": 0.002}}
    path = tmp_path / "rates.json"
    path.write_text(json.dumps(entry), encoding="utf-8")
    assert rate_card.load_rate_card(str(path)) == entry
    assert rate_card.load_rate_card(json.dumps(entry)) == entry


def test_env_var_is_honoured(monkeypatch, tmp_path) -> None:
    path = tmp_path / "rates.json"
    path.write_text(
        json.dumps({"acme": {"mode": "ocr", "unit": "usd_per_page", "rate": 0.5}}), encoding="utf-8"
    )
    monkeypatch.setenv(rate_card.RATE_CARD_ENV, str(path))
    rate_card.set_rate_card()  # no argument -> read the env var
    assert lookup_voice_rate("acme", "ocr") == pytest.approx(0.5)


@pytest.mark.parametrize(
    "bad",
    [
        {"acme": {"mode": "ocr", "unit": "usd_per_page"}},  # no rate
        {"acme": {"mode": "ocr", "unit": "usd_per_page", "rate": -1}},  # negative
        {"acme": {"mode": "ocr", "unit": "usd_per_page", "rate": float("inf")}},  # non-finite
        {"acme": {"mode": "ocr", "unit": "usd_per_page", "rate": 1, "prive": "x"}},  # typo'd key
        {"acme": {"mode": "", "unit": "usd_per_page", "rate": 1}},  # empty mode
        {"acme": {"mode": "ocr", "unit": "usd_per_page", "rate": 1, "confirmed": 1}},  # not bool
        {"acme": "0.002"},  # not an object
    ],
)
def test_a_malformed_card_raises_rather_than_being_skipped(bad: dict) -> None:
    """Loud, not lenient. A skipped rate card means your declared costs vanish
    and list prices quietly take their place."""
    with pytest.raises(ValueError):
        rate_card.load_rate_card(bad)


def test_unreadable_path_raises() -> None:
    with pytest.raises(ValueError, match="Cannot read rate card"):
        rate_card.load_rate_card("/nonexistent/definitely/not/here.json")
