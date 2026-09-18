"""User-supplied rate card — the rates YOU actually pay.

The bundled cost map is a snapshot of each vendor's PUBLIC LIST price. Almost
nobody pays list: volume tiers, negotiated contracts, committed-spend discounts,
carrier passthrough fees and plan minimums all move the real number. A COGS tool
that reports list prices as if they were your costs is guessing, so the map is a
starting point the user confirms or replaces — never an assertion about a bill.

Resolution order (see :mod:`floe_guard.voice_pricing`)::

    per-call override  ->  rate card  ->  bundled cost map  ->  UnpriceableLegError

A rate card can both OVERRIDE a bundled vendor and ADD one the map has never
heard of, which is how vendors that publish no publishable figure (Mistral OCR,
Azure Document Intelligence, HeyGen, Simli, Beyond Presence) get priced at all.

Entries use the same schema as the bundled ``__legs__`` section::

    {
      "acme-ocr": {
        "mode": "ocr",              # the leg this prices
        "unit": "usd_per_page",     # must be the canonical unit for that mode
        "rate": 0.0009,             # USD in that unit
        "provider": "acme",         # optional
        "confirmed": true           # optional, defaults true — see below
      }
    }

``confirmed`` is the difference between a number someone checked and a number we
guessed. Bundled rates resolve as ``confirmed=False`` (list price, unverified);
rate-card rates default to ``confirmed=True`` because you typed them. Set it to
``false`` on an imported or draft card to have it resolve as unconfirmed too, so
a console can flag it for review before anyone trusts the margin it produces.

Loading is fail-loud. A malformed rate card raises instead of being skipped: if
your declared rates cannot be read, silently falling back to list prices would
produce confident, wrong COGS — the exact failure this module exists to stop.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

#: Environment variable holding a path to a rate-card JSON file, or the JSON
#: itself (anything starting with ``{`` is parsed inline).
RATE_CARD_ENV = "FLOE_RATE_CARD"

# The only keys an entry may carry. An unknown key is rejected rather than
# ignored: a typo'd "rates" or "price" would otherwise leave the real rate
# undefined and silently fall through to a list price.
_ALLOWED_ENTRY_KEYS = frozenset(
    {"mode", "unit", "rate", "provider", "source_url", "retrieved_at", "confirmed"}
)
_REQUIRED_ENTRY_KEYS = ("mode", "unit", "rate")

_RATE_CARD: dict[str, Any] = {}


def _validate_entry(key: str, entry: Any) -> None:
    """Reject a malformed entry loudly. Shape only.

    The mode/unit CONSISTENCY check (``ocr`` must be ``usd_per_page`` etc.) lives
    in :mod:`floe_guard.voice_pricing`, which owns that table — checking it here
    would make this module import the thing that imports it.
    """
    if not isinstance(entry, dict):
        raise ValueError(f"Rate card entry {key!r} must be an object, got {type(entry).__name__}.")
    extra = sorted(set(entry) - _ALLOWED_ENTRY_KEYS)
    if extra:
        raise ValueError(
            f"Rate card entry {key!r} has unknown field(s): {extra}. "
            f"Allowed: {sorted(_ALLOWED_ENTRY_KEYS)}."
        )
    missing = [k for k in _REQUIRED_ENTRY_KEYS if k not in entry]
    if missing:
        raise ValueError(f"Rate card entry {key!r} is missing required field(s): {missing}.")
    for field in ("mode", "unit"):
        if not isinstance(entry[field], str) or not entry[field]:
            raise ValueError(f"Rate card entry {key!r}: {field} must be a non-empty string.")
    rate = entry["rate"]
    if (
        isinstance(rate, bool)
        or not isinstance(rate, (int, float))
        or not math.isfinite(rate)
        or rate < 0
    ):
        raise ValueError(
            f"Rate card entry {key!r}: rate must be a finite, non-negative number, got {rate!r}."
        )
    if "confirmed" in entry and not isinstance(entry["confirmed"], bool):
        raise ValueError(f"Rate card entry {key!r}: confirmed must be true or false.")
    for field in ("provider", "source_url", "retrieved_at"):
        if field in entry and not isinstance(entry[field], str):
            raise ValueError(f"Rate card entry {key!r}: {field} must be a string.")


def load_rate_card(source: Any = None) -> dict[str, Any]:
    """Read and validate a rate card.

    ``source`` may be a mapping (used directly), a path to a JSON file, or the
    JSON text itself. When omitted, ``$FLOE_RATE_CARD`` is consulted; if that is
    unset the result is an empty card, which is the zero-config default.

    Raises:
        ValueError: the card is unreadable, is not a JSON object, or any entry is
            malformed. Deliberately loud — see the module docstring.
    """
    if source is None:
        source = os.environ.get(RATE_CARD_ENV)
        if not source:
            return {}
    if isinstance(source, dict):
        raw: Any = source
    else:
        text = str(source)
        if text.lstrip().startswith("{"):
            try:
                raw = json.loads(text)
            except ValueError as exc:
                raise ValueError(f"Rate card is not valid JSON: {exc}") from exc
        else:
            try:
                with open(text, encoding="utf-8") as fh:
                    raw = json.load(fh)
            except OSError as exc:
                raise ValueError(f"Cannot read rate card at {text!r}: {exc}") from exc
            except ValueError as exc:
                raise ValueError(f"Rate card at {text!r} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("Rate card must be a JSON object mapping vendor keys to entries.")
    for key, entry in raw.items():
        _validate_entry(key, entry)
    return dict(raw)


def set_rate_card(source: Any = None) -> dict[str, Any]:
    """Install a rate card for this process and return it. ``{}`` clears it."""
    global _RATE_CARD
    _RATE_CARD = load_rate_card(source)
    return _RATE_CARD


def current_rate_card() -> dict[str, Any]:
    """The rate card in force. Empty when none is configured."""
    return _RATE_CARD


# Load once at import so `FLOE_RATE_CARD=... python agent.py` needs no code
# change. A broken card raises here rather than at the first priced leg, which
# is the difference between a startup failure and a wrong invoice.
_RATE_CARD = load_rate_card()

__all__ = ["RATE_CARD_ENV", "load_rate_card", "set_rate_card", "current_rate_card"]
