"""The runaway-loop example must demo the stop with no API key and no extras."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "runaway_loop.py"


def test_runaway_loop_example_stops_and_prints_block(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Ensure no account/key is involved.
    monkeypatch.delenv("FLOE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    sys.path.insert(0, str(EXAMPLE.parent))
    try:
        import runaway_loop

        runaway_loop.main()
    finally:
        sys.path.remove(str(EXAMPLE.parent))

    out = capsys.readouterr()
    combined = out.out + out.err
    assert "BUDGET EXCEEDED — call blocked" in combined
    assert "Loop stopped at call" in combined
    assert "OPENAI_API_KEY" not in os.environ
