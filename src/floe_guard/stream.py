"""Mid-stream budget enforcement for token streams.

``record()`` / ``settle()`` meter a COMPLETED response — by the time they run,
the money is spent. For a streaming response that starts under budget but runs
long, that is one call too late: the guard would only report the overshoot
after the fact. :class:`StreamGuard` closes that gap: feed it the stream's
deltas as they arrive and it re-prices the cumulative call chunk-by-chunk,
raising :class:`~floe_guard.BudgetExceeded` the moment the running call would
cross the ceiling — mid-generation, not post-mortem.

Aborting still settles the tokens consumed so far: the provider bills whatever
was generated before the cancel, so the guard records that partial spend
(fail-closed accounting, and it lands in ``guard.spend_log`` like any other
call), THEN raises. The worst case overshoot shrinks from "the rest of the
stream" to a single chunk.

**Token counts per chunk.** Providers stream text deltas, not token counts, so
:meth:`StreamGuard.feed_text` estimates ~4 characters/token (the usual rule of
thumb) unless you supply ``count_tokens=`` with a real tokenizer. The estimate
only steers the mid-stream cut-off; the final accrual reconciles to the
provider-reported usage when you pass it to :meth:`StreamGuard.finish`.

See ``examples/streaming_guard.py`` for a runnable demo (no API key).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from .errors import UnpriceableModelError, UnpriceableModelWarning
from .guard import BudgetGuard
from .pricing import ManualPrice, price_tokens


def approx_tokens(text: str) -> int:
    """~4 characters/token heuristic; a non-empty delta is at least one token."""
    return max(1, len(text) // 4) if text else 0


class StreamGuard:
    """Meter one streaming response chunk-by-chunk and hard-stop mid-generation.

    Usage (the context manager guarantees the reservation is settled or
    released no matter how the stream ends)::

        handle = guard.reserve(guard.estimate_call(model, prompt_tokens, max_tokens))
        with StreamGuard(guard, model, prompt_tokens=prompt_tokens, reserved=handle) as sg:
            for chunk in stream:
                sg.feed_text(chunk_text(chunk))   # raises BudgetExceeded mid-stream
                consume(chunk)
            sg.finish(completion_tokens=reported_usage)  # reconcile to real usage

    Fail-closed: an unpriceable model raises :class:`UnpriceableModelError` at
    construction (after releasing ``reserved``) when the guard is fail-closed —
    a stream whose spend cannot be measured never starts. When the guard is
    fail-open, feeding is a no-op (nothing to price chunks with) and
    :meth:`finish` routes through ``guard.settle`` so the warn-and-skip policy
    still applies.

    Not thread-safe itself (a stream is consumed sequentially); the underlying
    guard accounting stays lock-protected, so parallel streams each wrap their
    own ``StreamGuard`` against the same guard.
    """

    def __init__(
        self,
        guard: BudgetGuard,
        model: str,
        *,
        prompt_tokens: int = 0,
        reserved: float = 0.0,
        price: ManualPrice | None = None,
        label: str | None = None,
        count_tokens: Callable[[str], int] | None = None,
    ) -> None:
        # Same contract as settle(): a bad handle would corrupt the guard's
        # in-flight tally. Reject it before the stream starts, not at settle time.
        if not math.isfinite(reserved) or reserved < 0:
            raise ValueError(f"reserved must be a finite, non-negative number, got {reserved!r}")
        self._guard = guard
        self._model = model
        self._prompt_tokens = max(0, int(prompt_tokens))
        self._reserved = float(reserved)
        self._price = price
        self._label = label
        self._count = count_tokens or approx_tokens
        self._completion_tokens = 0
        self._closed = False
        # Intra-package seam: StreamGuard is the streaming face of BudgetGuard,
        # so it shares the guard's private resolution/ceiling internals rather
        # than duplicating that logic (or widening the public API).
        self._priced = guard._resolve(model, price)
        if self._priced is None and guard.fail_closed:
            # Same policy as settle(), applied BEFORE any money moves: refuse to
            # stream spend the guard cannot measure.
            warnings.warn(
                f"Cannot price model {model!r}: not in the bundled cost map and no "
                f"manual price given. A stream whose spend cannot be measured "
                f"cannot be mid-stream enforced — pass price=ManualPrice(...) or "
                f"set it in price_overrides.",
                UnpriceableModelWarning,
                stacklevel=2,
            )
            self._closed = True
            guard.release(reserved)
            raise UnpriceableModelError(model)
        # Registered AFTER the fail-closed raise so a refused stream leaves no
        # entry; _settle() unregisters, so entries live exactly as long as the
        # stream. Parallel streams see each other's accrual through this.
        self._key = guard._stream_register(self._reserved)

    def feed_text(self, delta: str) -> None:
        """Meter one text delta (token count via the heuristic/``count_tokens``).

        Raises :class:`~floe_guard.BudgetExceeded` when the cumulative call
        would cross the ceiling — after settling the partial spend already
        incurred, so the running total stays honest.
        """
        self.feed_tokens(self._count(delta))

    def feed_tokens(self, tokens: int) -> None:
        """Meter ``tokens`` more completion tokens. See :meth:`feed_text`."""
        if self._closed:
            raise RuntimeError("stream already settled — create a new StreamGuard per stream")
        self._completion_tokens += max(0, int(tokens))
        if self._priced is None:
            return  # fail-open + unpriceable: nothing to price chunks with
        cumulative = price_tokens(self._priced, self._prompt_tokens, self._completion_tokens)
        if self._guard._stream_would_cross(self._key, cumulative):
            # The tokens in this chunk were already generated (and billed) —
            # settle them before raising so spent_usd reflects reality. The
            # overshoot is at most this one chunk, not the rest of the stream.
            self._settle()
            self._guard._block()

    def finish(
        self,
        *,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> float:
        """Settle the stream. Pass the provider-reported usage (many APIs attach
        it to the final chunk) to reconcile the chunk-count heuristic to truth;
        omitted counts keep the accumulated values. Returns the USD cost.
        """
        if self._closed:
            raise RuntimeError("stream already settled")
        if prompt_tokens is not None:
            self._prompt_tokens = max(0, int(prompt_tokens))
        if completion_tokens is not None:
            self._completion_tokens = max(0, int(completion_tokens))
        return self._settle()

    @property
    def completion_tokens(self) -> int:
        """Completion tokens metered so far (estimate until :meth:`finish`)."""
        return self._completion_tokens

    def _settle(self) -> float:
        self._closed = True
        try:
            return self._guard.settle(
                self._model,
                self._prompt_tokens,
                self._completion_tokens,
                reserved=self._reserved,
                price=self._price,
                label=self._label,
            )
        finally:
            # Settle moved the accrual into spent_usd (or skipped it, fail-open)
            # — either way the registry entry must go, even if settle raised,
            # or a phantom accrual would throttle every other stream forever.
            self._guard._stream_unregister(self._key)

    def __enter__(self) -> StreamGuard:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        # Whatever ended the stream — clean exhaustion without an explicit
        # finish(), a mid-stream error, or the consumer abandoning it — the
        # tokens generated so far were billed: settle them and free the
        # reservation. Already-settled streams (finish() or a feed abort) skip.
        if not self._closed:
            self._settle()


def guard_stream(
    guard: BudgetGuard,
    model: str,
    chunks: Iterable[Any],
    *,
    get_text: Callable[[Any], str] | None = None,
    prompt_tokens: int = 0,
    reserved: float = 0.0,
    price: ManualPrice | None = None,
    label: str | None = None,
    count_tokens: Callable[[str], int] | None = None,
) -> Iterator[Any]:
    """Wrap a stream of chunks with mid-stream budget enforcement.

    Yields each chunk after metering it, so the consumer sees everything that
    was within budget; raises :class:`~floe_guard.BudgetExceeded` mid-stream
    (after settling the partial spend) when the call would cross the ceiling.
    The stream settles on ANY exit, including the consumer breaking out early.

    ``get_text`` extracts the text delta from a chunk. The default handles
    plain-string chunks only and REFUSES (``TypeError``) anything else — a
    silent zero-token fallback would let a whole stream through unmetered,
    which is exactly the failure mode this guard exists to prevent.

    Validation and the fail-closed unpriceable check run eagerly, at call
    time. Once you start iterating, the wrapper owns ``reserved`` and settles
    or releases it on every exit path; a returned-but-never-iterated stream
    leaves the handle with you (release it yourself).
    """
    sg = StreamGuard(
        guard,
        model,
        prompt_tokens=prompt_tokens,
        reserved=reserved,
        price=price,
        label=label,
        count_tokens=count_tokens,
    )

    def _default_get_text(chunk: Any) -> str:
        if isinstance(chunk, str):
            return chunk
        raise TypeError(
            f"guard_stream got a non-string chunk ({type(chunk).__name__}) and no "
            f"get_text= — pass get_text= to extract each chunk's text delta, or the "
            f"stream cannot be metered."
        )

    extract = get_text or _default_get_text

    def _run() -> Iterator[Any]:
        with sg:
            for chunk in chunks:
                sg.feed_text(extract(chunk))
                yield chunk

    return _run()


__all__ = ["StreamGuard", "guard_stream", "approx_tokens"]
