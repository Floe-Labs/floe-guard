"""The non-model adaptation examples must run offline and visibly shrink a non-model knob.

Issue #50: ``advisory()`` drives any cost axis, not just model choice. Each example
holds the model fixed, so these tests assert two things — the non-model parameter
really did step down, and no model downgrade sneaked in to explain the savings.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _run_example(name: str, capsys: pytest.CaptureFixture[str]) -> str:
    """Import and run an example the way a user would, returning everything it printed."""
    sys.path.insert(0, str(EXAMPLES))
    try:
        importlib.import_module(name).main()
    finally:
        sys.path.remove(str(EXAMPLES))
    out = capsys.readouterr()
    return out.out + out.err


def test_retrieval_depth_example_shrinks_top_k_without_switching_models(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Ensure no account/key is involved.
    monkeypatch.delenv("FLOE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    combined = _run_example("retrieval_depth", capsys)

    # The adapted axis: retrieval depth walks all the way down. Match the step
    # lines, not the header — a header substring would pass even if top_k froze.
    assert "20 chunks  +$" in combined
    assert "12 chunks  +$" in combined
    assert "5 chunks  +$" in combined
    # ...and the model did NOT change, so the savings came from top_k alone.
    assert "model fixed at gpt-4o" in combined
    assert "gpt-4o-mini" not in combined
    assert "held under $0.10" in combined
    assert "OPENAI_API_KEY" not in os.environ


def test_context_size_example_trims_history_and_max_tokens(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FLOE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    combined = _run_example("context_size", capsys)

    # Both context knobs move: the resent transcript and the reply cap. Anchored
    # to the step lines, since the header mentions both caps unconditionally.
    assert "trimming to the last 2 turns" in combined
    assert "turns sent · max_tokens 800" in combined
    assert "turns sent · max_tokens 250" in combined
    assert "model fixed at gpt-4o" in combined
    assert "gpt-4o-mini" not in combined
    assert "held under $0.10" in combined
    assert "OPENAI_API_KEY" not in os.environ


def test_plan_complexity_example_drops_optional_subtasks(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FLOE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    combined = _run_example("plan_complexity", capsys)

    # Reasoning thins out first, then the optional sub-tasks stop running.
    assert "full reasoning" in combined
    assert "reduced reasoning" in combined
    assert "[skipped — optional]" in combined
    # The required sub-tasks are what the taper protects.
    assert "required sub-tasks" in combined
    assert "model fixed at gpt-4o" in combined
    assert "gpt-4o-mini" not in combined
    assert "held under $0.10" in combined
    assert "OPENAI_API_KEY" not in os.environ
