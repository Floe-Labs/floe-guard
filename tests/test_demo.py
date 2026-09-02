"""Tests for the packaged demo (``floe-guard demo`` / ``floe_guard.demo.run_demo``).

The stub call costs ~$0.0125 (gpt-4o, 1k in + 1k out). These guard the request-sized
``check()`` so the demo hard-stops *before* the crossing call at an arbitrary
``--limit-usd``, not just at exact multiples of the per-call cost.
"""

from __future__ import annotations

import re

from floe_guard.demo import run_demo

_STUB_CALL_COST = 0.0125  # gpt-4o: 1000 * $2.5e-6 + 1000 * $1e-5


def test_demo_blocks_the_first_call_when_ceiling_is_below_one_call(capsys):
    """A ceiling below a single stub call must stop at call #1 — the request-sized
    check() blocks it before it runs, so nothing is ever recorded (regression: a
    bare check() would let call #1 through and record spend above the ceiling)."""
    run_demo(limit_usd=_STUB_CALL_COST / 4)  # ~$0.003 — smaller than one call
    out = capsys.readouterr().out
    assert "Loop stopped at call #1" in out
    # The crossing call must NOT have run — no "call #1: +$…" record line.
    assert not re.search(r"call #1: \+\$", out)


def test_cli_demo_ledger_write_failure_returns_nonzero(capsys, monkeypatch):
    """A failed ledger write (read-only CWD / full disk) must exit non-zero with a
    clean message — not an unhandled OSError traceback. The demo already ran; only
    the convenience ledger write failed."""
    from floe_guard.__main__ import main

    real_open = open

    def failing_open(path, *args, **kwargs):
        if str(path).endswith("floe-ledger.jsonl"):
            raise OSError("read-only file system")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", failing_open)
    rc = main(["demo"])
    assert rc == 1
    assert "could not write the spend ledger" in capsys.readouterr().err


def test_demo_never_records_spend_above_the_ceiling(capsys):
    """For a ceiling that is NOT an exact multiple of the per-call cost, the last
    recorded running total must stay at or below the ceiling."""
    limit = 0.033  # between 2 and 3 calls (~$0.025 and ~$0.0375)
    run_demo(limit_usd=limit)
    out = capsys.readouterr().out
    totals = [float(m) for m in re.findall(r"running total \$([0-9.]+)", out)]
    assert totals, "expected at least one recorded call"
    assert max(totals) <= limit + 1e-9, f"recorded spend {max(totals)} exceeded ceiling {limit}"
    assert "Loop stopped at call" in out
