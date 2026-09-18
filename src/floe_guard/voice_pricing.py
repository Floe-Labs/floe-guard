"""Offline pricing for per-unit legs from the vendored map.

The per-unit twin of :mod:`floe_guard.pricing`. The token cost map is per-token;
these legs are billed in anything but tokens — per second (STT), per 1k chars
(TTS), per minute (telephony, avatar), per segment (SMS), per page (OCR) and per
GPU-second (GPU) — so their rates live under the reserved ``"__legs__"`` key of
``cost_map.json`` with their own schema::

    "deepgram-nova-3": {
        "mode": "stt",              # discriminator: see LegMode
        "unit": "usd_per_second",   # explicit unit — a mismatch fails closed
        "rate": 0.0001283333,       # USD in that unit
        "provider": "deepgram"
    }

The section was called ``"__voice__"`` until P1.11. The mechanism was never
voice-specific, and once it grew SMS, OCR, GPU and avatar rates the old name was
actively misleading — so it is now ``"__legs__"``. :mod:`floe_guard.pricing`
reads the new name first and falls back to the old one, which is what makes the
rename non-breaking for a ``cost_map.json`` generated before it.

The line between this map and the token map is the BILLING UNIT, not the
modality: per-token spend lives in the flat model map, per-anything-else lives
here. LLM inference is therefore NOT a leg, while a rented GPU-second is.

Same fail-closed contract as token pricing: a vendor absent from the map (or an
entry whose ``unit``/``mode`` does not match the leg it is asked to price) is
unpriceable — :func:`lookup_voice_rate` returns ``None`` and the adapter raises
:class:`~floe_guard.errors.UnpriceableVoiceError` rather than silently metering
the leg at $0. Pass a per-unit override to enforce a leg the map cannot price.

No network. The rates are a **drift-prone snapshot** of each vendor's public
list price — refresh them (``scripts/update-cost-map.mjs``, which names the
source URL and retrieval date for every entry) like the token map, or estimates
drift as vendors change prices. Telephony and SMS are **US-only in v1**, and
several rates are volume- or region-tiered: the map vendors the DEAREST public
tier, because over-pricing a spend guard stops one call early while
under-pricing lets a crossing call through.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from . import rate_card
from .errors import UnpriceableLegError
from .pricing import _VOICE_MAP

#: The legs the bundled map can price per unit. Named for the leg rather than
#: for voice — the mechanism is not voice-specific.
#:
#: P1.11 is the pricing change this was held open for: ``sms``, ``ocr``, ``gpu``
#: and ``avatar`` each arrived WITH a unit in :data:`_UNIT_FOR_MODE`, an arm in
#: :func:`voice_leg_cost`, and sourced entries in the cost map. That is the bar
#: for adding a member — a mode with no unit and no priced entry is a mode that
#: fails closed on every call, which is worse than not having it.
LegMode = Literal["stt", "tts", "telephony", "sms", "ocr", "gpu", "avatar"]

#: Deprecated alias for :data:`LegMode`. Prefer the leg-shaped name. It now spans
#: legs that are not voice at all; the name is kept only for compatibility.
VoiceMode = LegMode

# The one unit each leg is billed in. An entry whose ``unit`` disagrees with its
# leg is a schema mismatch and fails closed — a Deepgram $/min figure stored
# without the ÷60 conversion would over-bill 60x if it were silently accepted.
# The same trap applies to the P1.11 legs: OCR vendors quote per 1,000 pages and
# GPU vendors quote per hour, so both are converted at curation time and stored
# in the canonical per-unit form.
_UNIT_FOR_MODE: dict[str, str] = {
    "stt": "usd_per_second",
    "tts": "usd_per_1k_chars",
    "telephony": "usd_per_minute",
    "sms": "usd_per_segment",
    "ocr": "usd_per_page",
    "gpu": "usd_per_gpu_second",
    "avatar": "usd_per_minute",
}


@dataclass(frozen=True)
class VoiceRate:
    """A resolved per-unit leg rate plus where it came from.

    The provenance fields are the difference between a number and an auditable
    number. A COGS figure built on an unverified list price should be presentable
    as exactly that — "$1.50/1k pages, Google's list page, read 2026-09-18, not
    yet confirmed by you" — so a console can ask for confirmation instead of
    quietly reporting a guess as a cost.
    """

    mode: str
    unit: str
    rate: float
    #: Where the number came from, cheapest-trust first:
    #: ``"cost_map"`` (public list price) < ``"rate_card"`` (you declared it) <
    #: ``"override"`` (passed at the call site). A later vendor-invoice read
    #: supersedes all of them at reconcile time, and the gap between the estimate
    #: and that actual is the variance signal.
    source: str  # "override" | "rate_card" | "cost_map"
    provider: str | None = None
    #: Public price-list URL and the date it was read. Present only on rates
    #: curated with a citation; ``None`` means UNVERIFIED and should be shown as
    #: such rather than rendered as a blank.
    source_url: str | None = None
    retrieved_at: str | None = None
    #: Has a human accepted this number as what they actually pay? Bundled list
    #: prices are ``False`` until confirmed; a rate you declared is ``True``.
    confirmed: bool = False


def _finite_non_negative(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def _entry_is_usable(entry: Any, mode: VoiceMode) -> bool:
    """Fail-closed on every mismatch: wrong mode, wrong unit, or a bad rate."""
    if not isinstance(entry, dict):
        return False
    if entry.get("mode") != mode:
        return False
    if entry.get("unit") != _UNIT_FOR_MODE[mode]:
        return False
    return _finite_non_negative(entry.get("rate"))


def _lookup_leg_entry(model: str | None, mode: VoiceMode) -> tuple[dict[str, Any], str] | None:
    """The entry that prices this leg and which source it came from.

    Rate card first, bundled map second — your negotiated rate beats a list
    price. Returns ``None`` when neither can price the leg.

    One deliberate asymmetry: if the rate card declares this exact vendor AND
    this exact leg but the declaration is broken (wrong unit, bad rate), this
    returns ``None`` instead of falling through to the bundled price. You said
    what this leg costs you; silently substituting a list price you never chose
    would be a confident wrong number, which is the failure this whole module
    exists to prevent. A card entry for a DIFFERENT mode is not a declaration
    about this leg, so that falls through normally.
    """
    if model is None:
        return None
    declared = rate_card.current_rate_card().get(model)
    if isinstance(declared, dict) and declared.get("mode") == mode:
        return (declared, "rate_card") if _entry_is_usable(declared, mode) else None
    entry = _VOICE_MAP.get(model)
    if _entry_is_usable(entry, mode):
        return entry, "cost_map"
    return None


def lookup_voice_rate(model: str | None, mode: VoiceMode) -> float | None:
    """Resolve a vendor to its per-unit rate, or ``None`` if unpriceable.

    The pure resolver (the twin of :func:`floe_guard.pricing.resolve_price`): it
    never raises. Consults the user's rate card before the bundled map. Use
    :func:`resolve_voice_rate` when you need to know WHERE the number came from.
    """
    found = _lookup_leg_entry(model, mode)
    return None if found is None else float(found[0]["rate"])


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
        # Passed at the call site by a human who knows what this leg costs, so it
        # is confirmed by construction.
        return VoiceRate(
            mode, _UNIT_FOR_MODE[mode], float(override), "override", confirmed=True
        )
    found = _lookup_leg_entry(model, mode)
    if found is None:
        raise UnpriceableLegError(model, mode)
    entry, source = found
    return VoiceRate(
        mode,
        _UNIT_FOR_MODE[mode],
        float(entry["rate"]),
        source,
        provider=entry.get("provider"),
        source_url=entry.get("source_url"),
        retrieved_at=entry.get("retrieved_at"),
        # A declared rate is confirmed unless the card says otherwise (an
        # imported or draft card can mark entries unconfirmed); a bundled list
        # price is UNCONFIRMED until a human accepts it.
        confirmed=bool(entry.get("confirmed", source == "rate_card")),
    )


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
    # Every mode except tts is a plain quantity x rate; the unit is what differs,
    # and the unit was already checked when the rate was resolved. tts is the one
    # arm with arithmetic, because its rate is quoted per 1000 chars.
    if mode in ("stt", "telephony", "sms", "ocr", "gpu", "avatar"):
        # seconds x $/sec, minutes x $/min, segments x $/segment,
        # pages x $/page, gpu-seconds x $/gpu-second, minutes x $/min
        return q * rate
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
    "LegMode",
    # Deprecated alias for LegMode — kept exported so existing annotations and
    # `from floe_guard.voice_pricing import VoiceMode` keep working.
    "VoiceMode",
    "VoiceRate",
    "lookup_voice_rate",
    "resolve_voice_rate",
    "voice_leg_cost",
    "price_voice_leg",
]
