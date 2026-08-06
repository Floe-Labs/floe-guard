"""The non-model adaptation examples must run offline and visibly shrink a non-model knob.

Issue #50: ``advisory()`` drives any cost axis, not just model choice. Each example
holds the model fixed, so these tests assert two things — the non-model parameter
really did step down, and no model downgrade sneaked in to explain the savings.
"""

from __future__ import annotations

import importlib
import os
import re
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _run_example(
    name: str, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> str:
    """Import and run an example the way a user would, returning everything it printed.

    ``monkeypatch.syspath_prepend`` adds the examples dir to ``sys.path`` and undoes
    exactly that at teardown — no manual ``sys.path.remove`` that could drop the
    wrong entry (or raise) if the path was already present.
    """
    monkeypatch.syspath_prepend(str(EXAMPLES))
    importlib.import_module(name).main()
    out = capsys.readouterr()
    return out.out + out.err


def test_retrieval_depth_example_shrinks_top_k_without_switching_models(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify retrieval depth shrinks and the model remains constant."""
    # Ensure no account/key is involved.
    monkeypatch.delenv("FLOE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    combined = _run_example("retrieval_depth", capsys, monkeypatch)

    # The adapted axis: parse the top_k off each *step* line (not the header —
    # a header substring would pass even if top_k froze) and assert the actual
    # observed sequence steps down 20 → 12 → 5 and never climbs back up.
    depths = [int(n) for n in re.findall(r"(\d+) chunks  \+\$", combined)]
    assert depths, "no step lines found in output"
    assert depths == sorted(depths, reverse=True), f"top_k should not increase, got {depths}"
    assert {20, 12, 5} <= set(depths), f"expected all three depths, saw {sorted(set(depths))}"
    # ...and the model did NOT change, so the savings came from top_k alone.
    assert "model fixed at gpt-4o" in combined
    assert "gpt-4o-mini" not in combined
    assert "held under $0.10" in combined
    assert "OPENAI_API_KEY" not in os.environ


def test_context_size_example_trims_history_and_max_tokens(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify context size and reply cap shrink."""
    monkeypatch.delenv("FLOE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    combined = _run_example("context_size", capsys, monkeypatch)

    # Both context knobs move: the resent transcript and the reply cap. Anchored
    # to the step lines, since the header mentions both caps unconditionally.
    assert "trimming to the last 2 turns" in combined
    assert "turns sent · max_tokens 800" in combined
    assert "turns sent · max_tokens 250" in combined
    # The reply cap must step DOWN over time: full (800) precedes the taper (250).
    assert combined.index("max_tokens 800") < combined.index("max_tokens 250")
    assert "model fixed at gpt-4o" in combined
    assert "gpt-4o-mini" not in combined
    assert "held under $0.10" in combined
    assert "OPENAI_API_KEY" not in os.environ


def test_plan_complexity_example_drops_optional_subtasks(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify reasoning steps reduce and optional subtasks are dropped."""
    monkeypatch.delenv("FLOE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    combined = _run_example("plan_complexity", capsys, monkeypatch)

    # Reasoning thins out first, then the optional sub-tasks stop running.
    assert "full reasoning" in combined
    assert "reduced reasoning" in combined
    assert "[skipped — optional]" in combined
    # Reasoning must thin in that order: full precedes reduced.
    assert combined.index("full reasoning") < combined.index("reduced reasoning")
    # The required sub-tasks are what the taper protects (reported as call counts).
    assert "required sub-task calls" in combined
    assert "model fixed at gpt-4o" in combined
    assert "gpt-4o-mini" not in combined
    assert "held under $0.10" in combined
    assert "OPENAI_API_KEY" not in os.environ
