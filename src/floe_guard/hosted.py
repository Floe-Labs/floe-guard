"""Hosted Floe upgrade hook (stub — not wired to a live endpoint).

The local :class:`~floe_guard.BudgetGuard` is *estimate-based*: it prices tokens
from a vendored cost map in your own process. A determined agent (or a bug) can
run around it, and it only sees the one vendor you instrumented.

Hosted Floe is the upgrade: enforcement moves server-side against a real credit
line, so the ceiling is **un-bypassable** and spans **every vendor** (LLM tokens
*and* paid x402 tool calls) under one budget, with team budgets and analytics.

This module is intentionally a no-op stub. When ``FLOE_API_KEY`` is set,
:func:`hosted_enforcement_available` reports it so callers can branch toward the
hosted path. Wiring the actual delegation is tracked below — we do NOT ship a
fabricated endpoint.

    Upgrade: https://dev-dashboard.floelabs.xyz  ·  https://floelabs.xyz
"""

from __future__ import annotations

import os

FLOE_API_KEY_ENV = "FLOE_API_KEY"


def hosted_enforcement_available() -> bool:
    """True if a Floe API key is present in the environment.

    Presence of a key is the signal that a caller *could* delegate enforcement to
    hosted Floe. It does not itself perform any network call.
    """
    return bool(os.environ.get(FLOE_API_KEY_ENV, "").strip())


# TODO(hosted-upgrade): when FLOE_API_KEY is set, delegate enforcement to hosted
# Floe instead of (or alongside) the local estimate. The hosted path debits a
# real, server-side credit line — un-bypassable and cross-vendor — so the budget
# holds even if the local process is bypassed. This requires the public hosted
# budget API to be finalized; until then this stays a documented no-op rather
# than a fabricated endpoint. See README "Upgrade to hosted Floe".
