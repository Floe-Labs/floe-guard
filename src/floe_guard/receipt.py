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

    def __post_init__(self) -> None:
        if self.source not in (SOURCE_ESTIMATE, SOURCE_HOSTED):
            raise ValueError(
                f"FloeCost.source must be {SOURCE_ESTIMATE!r} or {SOURCE_HOSTED!r}, "
                f"got {self.source!r} — the estimate/hosted distinction is the honesty contract."
            )

    def format(self) -> str:
        """A one-line receipt, e.g. ``floe · gpt-4o · $0.0075 est · left $12.34``.

        Per-call costs are often fractions of a cent, so sub-$0.0001 amounts get
        6 decimals instead of rounding to a misleading ``$0.0000``.
        """
        tag = "est" if self.source == SOURCE_ESTIMATE else "floe"
        amount = f"${self.usd:.6f}" if abs(self.usd) < 0.0001 else f"${self.usd:.4f}"
        line = f"floe · {self.model or '?'} · {amount} {tag}"
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
    prompt_cached_tokens: int = 0,
) -> FloeCost | None:
    """Per-turn receipt from token usage, priced locally by the cost map.

    ``overrides`` (when given) are consulted before the bundled cost map, same as
    :func:`~floe_guard.resolve_price`. Returns ``None`` if the model cannot be
    priced — fail closed, never a fabricated ``$0``. ``source`` is always
    ``"estimate"`` (this prices locally); to show remaining budget, pass
    ``remaining_usd`` yourself — e.g. the result of
    :func:`~floe_guard.hosted_remaining_usd` (a hosted read that needs a Floe
    key), fetched at whatever cadence you like so there's no network call every
    turn. ``turn_cost`` never calls the network.

    ``prompt_cached_tokens`` is the subset of ``prompt_tokens`` served from the
    provider's prompt cache (``prompt_tokens`` is the TOTAL input, inclusive of
    cached). That subset is priced at the model's cache-read rate from the cost
    map, falling back to a conservative multiplier for models without a published
    cache rate — so caching is no longer billed at the full input rate. Clamped
    to ``prompt_tokens``; ``0`` (the default) prices exactly as before.
    """
    priced = resolve_price(model, overrides)
    if priced is None:
        return None
    cached = min(max(0, prompt_cached_tokens), max(0, prompt_tokens))
    non_cached = max(0, prompt_tokens) - cached
    usd = price_tokens(priced, non_cached, completion_tokens, cache_read_input_tokens=cached)
    return FloeCost(usd=usd, source=SOURCE_ESTIMATE, model=model, remaining_usd=remaining_usd)
