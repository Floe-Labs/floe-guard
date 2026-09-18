"""The per-unit leg vocabulary: which legs exist, and the one unit each bills in.

Deliberately dependency-free. This module imports nothing from the rest of the
package, which is the whole point: both the PRICING path
(:mod:`floe_guard.voice_pricing`) and the VALIDATION path
(:mod:`floe_guard.rate_card`) need this table, and rate_card cannot import
voice_pricing because voice_pricing imports rate_card.

That cycle is why the rate-card loader originally checked only that ``mode`` was
a non-empty string — and that gap recreated the exact failure the rate card
exists to prevent. A typo'd ``"orc"`` passed validation, then resolved as a
different leg from ``"ocr"``, so a lookup for a key that ALSO exists in the
bundled map silently fell through to the public list price. The user declared a
rate, the guard used someone else's number, and nothing said a word.

One table, imported by both sides, cannot drift from itself.
"""

from __future__ import annotations

from typing import Literal

#: The legs the bundled map can price per unit. Named for the leg rather than for
#: voice — the mechanism is not voice-specific.
#:
#: The bar for adding a member: it arrives WITH a unit below, an arm in
#: :func:`floe_guard.voice_pricing.voice_leg_cost`, and sourced entries in the
#: cost map. A mode with no unit fails closed on every call, which is worse than
#: not having the mode at all.
LegMode = Literal["stt", "tts", "telephony", "sms", "ocr", "gpu", "avatar"]

#: The one unit each leg is billed in. An entry whose ``unit`` disagrees with its
#: leg is a schema mismatch and fails closed — a Deepgram $/min figure stored
#: without the ÷60 conversion would over-bill 60x if it were silently accepted.
#: The same trap applies to the P1.11 legs: OCR vendors quote per 1,000 pages and
#: GPU vendors quote per hour, so both are converted at curation time and stored
#: in the canonical per-unit form.
UNIT_FOR_MODE: dict[str, str] = {
    "stt": "usd_per_second",
    "tts": "usd_per_1k_chars",
    "telephony": "usd_per_minute",
    "sms": "usd_per_segment",
    "ocr": "usd_per_page",
    "gpu": "usd_per_gpu_second",
    "avatar": "usd_per_minute",
}

#: Every legal mode, for validation and error messages.
LEG_MODES: tuple[str, ...] = tuple(UNIT_FOR_MODE)

__all__ = ["LegMode", "UNIT_FOR_MODE", "LEG_MODES"]
