"""Tests for the offline workload estimator (``floe-guard estimate``).

gpt-4o in the bundled map is $2.5e-6 in / $1e-5 out per token, so the default
1k/1k call prices at exactly $0.0125 — the same stub call the demo uses, which
keeps these tests independent of cost-map drift for other models.
"""

from __future__ import annotations

import pytest

from floe_guard.estimate import run_estimate

_PER_CALL = 0.0125  # gpt-4o: 1000 * $2.5e-6 + 1000 * $1e-5


def test_estimate_prices_a_known_workload(capsys):
    run_estimate("gpt-4o", calls=8, tokens_in=1_000, tokens_out=1_000)
    out = capsys.readouterr().out
    assert f"per call:     ${_PER_CALL:.6f}" in out
    assert f"total:        ${_PER_CALL * 8:.6f}" in out
    # The activation CTA: a copy-pasteable ceiling at the run total.
    assert f"BudgetGuard(limit_usd={_PER_CALL * 8:.6f})" in out


def test_estimate_scales_with_token_counts(capsys):
    run_estimate("gpt-4o", calls=1, tokens_in=2_000, tokens_out=500)
    out = capsys.readouterr().out
    expected = 2_000 * 2.5e-6 + 500 * 1e-5
    assert f"per call:     ${expected:.6f}" in out


def test_estimate_fails_closed_on_unpriceable_model():
    """A model the map cannot price is a clean error, never a $0.00 guess."""
    with pytest.raises(ValueError, match="cannot price"):
        run_estimate("not-a-real-model")


def test_estimate_rejects_bad_arguments():
    with pytest.raises(ValueError, match="--calls"):
        run_estimate("gpt-4o", calls=0)
    with pytest.raises(ValueError, match=">= 0"):
        run_estimate("gpt-4o", tokens_in=-1)
    with pytest.raises(ValueError, match="empty"):
        run_estimate("   ")
