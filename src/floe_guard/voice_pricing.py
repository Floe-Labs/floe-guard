"""Offline pricing for voice legs (STT / TTS / telephony) from the vendored map.

The voice twin of :mod:`floe_guard.pricing`. The token cost map is per-token; a
voice pipeline's bill is per-second (STT), per-1k-chars (TTS), and per-minute
(telephony) instead, so the voice rates live under the reserved ``"__voice__"``
key of ``cost_map.json`` with their own schema::

    "deepgram-nova-3": {
        "mode": "stt",              # discriminator: "stt" | "tts" | "telephony"
        "unit": "usd_per_second",   # explicit unit — a mismatch fails closed
        "rate": 0.0001283333,       # USD in that unit
        "provider": "deepgram"
    }

Same fail-closed contract as token pricing: a vendor absent from the map (or an
entry whose ``unit``/``mode`` does not match the leg it is asked to price) is
unpriceable — :func:`lookup_voice_rate` returns ``None`` and the adapter raises
:class:`~floe_guard.errors.UnpriceableVoiceError` rather than silently metering
the leg at $0. Pass a per-unit override to enforce a leg the map cannot price.

No network. The rates are a **drift-prone snapshot** of each vendor's public
list price — refresh them (``scripts/update-cost-map.mjs``) like the token map,
or estimates drift as vendors change prices. Telephony is **US-only in v1**.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from .errors import UnpriceableVoiceError
from .pricing import _VOICE_MAP

VoiceMode = Literal["stt", "tts", "telephony"]

# The one unit each leg is billed in. An entry whose ``unit`` disagrees with its
# leg is a schema mismatch and fails closed — a Deepgram $/min figure stored
# without the ÷60 conversion would over-bill 60x if it were silently accepted.
_UNIT_FOR_MODE: dict[str, str] = {
    "stt": "usd_per_second",
    "tts": "usd_per_1k_chars",
    "telephony": "usd_per_minute",
}


@dataclass(frozen=True)
class VoiceRate:
    """A resolved per-unit voice rate plus where it came from."""

    mode: str
    unit: str
    rate: float
    source: str  # "override" | "cost_map"


def _finite_non_negative(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def lookup_voice_rate(model: str | None, mode: VoiceMode) -> float | None:
    """Resolve a voice vendor to its per-unit rate, or ``None`` if unpriceable.

    The pure resolver (the voice twin of :func:`floe_guard.pricing.resolve_price`):
    it never raises. Fail-closed on every mismatch — a missing vendor, an entry
    whose ``mode`` is not ``mode``, an entry whose ``unit`` is not the canonical
    unit for ``mode``, or a non-finite/negative rate — so a schema slip can never
    surface as a silent, valid-looking price.
    """
    if model is None:
        return None
    entry = _VOICE_MAP.get(model)
    if not entry:
        return None
    if entry.get("mode") != mode:
        return None
    if entry.get("unit") != _UNIT_FOR_MODE[mode]:
        return None
    rate = entry.get("rate")
    if not _finite_non_negative(rate):
        return None
    return float(rate)


def resolve_voice_rate(
    model: str | None, mode: VoiceMode, override: float | None = None
) -> VoiceRate:
    """Resolve a leg's per-unit rate, fail-closed. An ``override`` wins over the map.

    Raises :class:`~floe_guard.errors.UnpriceableVoiceError` when neither an
    override nor the bundled map can price the leg — the guard refuses to meter
    spend it cannot measure.
    """
    if override is not None:
        if not _finite_non_negative(override):
            raise ValueError(
                f"voice {mode} override must be a finite, non-negative number, got {override!r}"
            )
        return VoiceRate(mode, _UNIT_FOR_MODE[mode], float(override), "override")
    rate = lookup_voice_rate(model, mode)
    if rate is None:
        raise UnpriceableVoiceError(model, mode)
    return VoiceRate(mode, _UNIT_FOR_MODE[mode], rate, "cost_map")


def voice_leg_cost(mode: VoiceMode, quantity: float, rate: float) -> float:
    """USD for one leg. ``quantity`` is seconds (stt), characters (tts), or
    minutes (telephony); negative quantities clamp to zero.

    Raises:
        ValueError: ``quantity`` is non-finite (NaN/inf) — a non-finite cost would
            poison the guard's running total, so it fails closed.
    """
    # Reject non-finite quantities: max(0.0, nan) is nan, so a NaN/inf quantity
    # would return a non-finite USD amount that then poisons the guard's running
    # total (NaN disables every ceiling comparison). Negatives still clamp.
    if not math.isfinite(quantity):
        raise ValueError(f"voice {mode} quantity must be a finite number, got {quantity!r}")
    q = max(0.0, quantity)
    if mode == "stt":
        return q * rate  # seconds * $/second
    if mode == "tts":
        return q / 1000.0 * rate  # chars / 1000 * $/1k-chars
    if mode == "telephony":
        return q * rate  # minutes * $/minute
    raise ValueError(f"unknown voice mode {mode!r}")  # pragma: no cover


def price_voice_leg(
    mode: VoiceMode,
    quantity: float,
    *,
    model: str | None = None,
    override: float | None = None,
) -> float | None:
    """USD for one voice leg, or ``None`` when the leg is unconfigured (skip it).

    The single entry point the adapters use. A leg with neither a vendor
    (``model``) nor an ``override`` is unconfigured — return ``None`` so the
    token-only contract is preserved and nothing is metered. A leg that *is*
    configured but cannot be priced raises
    :class:`~floe_guard.errors.UnpriceableVoiceError` (via
    :func:`resolve_voice_rate`) rather than accruing a silent $0.
    """
    if model is None and override is None:
        return None
    resolved = resolve_voice_rate(model, mode, override)
    return voice_leg_cost(mode, quantity, resolved.rate)


__all__ = [
    "VoiceMode",
    "VoiceRate",
    "lookup_voice_rate",
    "resolve_voice_rate",
    "voice_leg_cost",
    "price_voice_leg",
]
