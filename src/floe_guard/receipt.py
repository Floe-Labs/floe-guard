"""The per-turn receipt — one shape, local estimate or hosted truth.

``FloeCost`` is the byte-for-byte contract the local meter and hosted Floe both
speak, so graduating from free-local to hosted is a **key swap, not a
re-integration**: the same object appears whether ``usd`` came from the bundled
cost map (``source="estimate"``) or from hosted server-truth
(``source="hosted"``), and ``remaining_usd`` is filled from hosted Floe when a
key is present. Adapters (Pipecat, LiveKit, …) emit this once per turn so a
developer sees *what that call cost* with no extra config — the receipt moment.
"""

from __future__ import annotations

from dataclasses import dataclass

from .pricing import ManualPrice, price_tokens, resolve_price

# The two honest provenances of a cost figure. "estimate" is priced locally from
# the bundled cost map (free, offline, no account); "hosted" is Floe's
# server-truth. Never label an estimate as hosted — accuracy is the whole point.
SOURCE_ESTIMATE = "estimate"
SOURCE_HOSTED = "hosted"


@dataclass(frozen=True)
class FloeCost:
    """One turn's receipt: what it cost, and what's left.

    Attributes:
        usd: cost of this turn, in USD.
        source: ``"estimate"`` (local cost map) or ``"hosted"`` (server-truth).
        model: the model the cost is for, when known.
        remaining_usd: account budget remaining (hosted), or ``None`` when no
            Floe key is present — the local, keyless path still shows cost.
    """

    usd: float
    source: str
    model: str | None = None
    remaining_usd: float | None = None

    def format(self) -> str:
        """A one-line receipt, e.g. ``floe · gpt-4o · $0.0075 est · left $12.34``."""
        tag = "est" if self.source == SOURCE_ESTIMATE else "floe"
        line = f"floe · {self.model or '?'} · ${self.usd:.4f} {tag}"
        if self.remaining_usd is not None:
            line += f" · left ${self.remaining_usd:,.2f}"
        return line


def turn_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    *,
    remaining_usd: float | None = None,
    overrides: dict[str, ManualPrice] | None = None,
) -> FloeCost | None:
    """Per-turn receipt from token usage, priced by the bundled cost map.

    Returns ``None`` if the model cannot be priced — fail closed, never a
    fabricated ``$0``. ``source`` is always ``"estimate"`` here; pass
    ``remaining_usd`` (from :func:`~floe_guard.hosted_remaining_usd`, at whatever
    cadence you like) to show budget without a network call every turn.
    """
    priced = resolve_price(model, overrides)
    if priced is None:
        return None
    usd = price_tokens(priced, prompt_tokens, completion_tokens)
    return FloeCost(usd=usd, source=SOURCE_ESTIMATE, model=model, remaining_usd=remaining_usd)
