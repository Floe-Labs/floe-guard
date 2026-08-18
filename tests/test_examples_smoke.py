"""Subprocess smoke tests for the no-API-key examples.

Each test runs an example as a real subprocess (no import tricks), removes
FLOE_API_KEY and OPENAI_API_KEY from the environment, asserts a zero exit
code, and checks for a meaningful output marker.

The ordering test for runaway_loop.py additionally verifies that the
"Starting a runaway loop" line appears *before* the budget-exceeded banner
when stdout and stderr are interleaved — the flush=True on every status print
guarantees this even when stdout is redirected/piped.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

TIMEOUT = 60  # seconds


def _run(name: str, *, merge_stderr: bool = False) -> subprocess.CompletedProcess[str]:
    """Run an example as a subprocess without any API keys."""
    env = {k: v for k, v in os.environ.items() if k not in {"FLOE_API_KEY", "OPENAI_API_KEY"}}
    return subprocess.run(
        [sys.executable, str(EXAMPLES / name)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT if merge_stderr else subprocess.PIPE,
        text=True,
        timeout=TIMEOUT,
        env=env,
    )


def test_runaway_loop() -> None:
    result = _run("runaway_loop.py")
    assert result.returncode == 0, result.stderr
    combined = result.stdout + result.stderr
    assert "Loop stopped at call" in combined


def test_runaway_loop_output_ordering() -> None:
    """Starting message must appear before the budget-exceeded banner.

    Merges stdout and stderr into a single stream (the same way a terminal
    or ``2>&1`` pipe would) and checks the relative positions.
    """
    result = _run("runaway_loop.py", merge_stderr=True)
    assert result.returncode == 0, result.stdout
    combined = result.stdout

    assert "Starting a runaway loop" in combined, "expected startup message in output"
    assert "Loop stopped at call" in combined, "expected stop message in output"

    start_pos = combined.index("Starting a runaway loop")
    # The budget-exceeded banner is written to stderr by the library and is
    # deterministic for this demo (the loop always crosses the ceiling), so
    # require it — its absence is a regression, not a pass.
    assert "BUDGET EXCEEDED" in combined, "expected the BUDGET EXCEEDED banner in output"
    banner_pos = combined.index("BUDGET EXCEEDED")
    # After merging, it must come AFTER the startup line.
    assert start_pos < banner_pos, (
        "startup message must appear before the BUDGET EXCEEDED banner; "
        f"got start_pos={start_pos}, banner_pos={banner_pos}"
    )


def test_budget_aware() -> None:
    result = _run("budget_aware.py")
    assert result.returncode == 0, result.stderr
    combined = result.stdout + result.stderr
    assert "Stopped at step" in combined


def test_context_size() -> None:
    result = _run("context_size.py")
    assert result.returncode == 0, result.stderr
    combined = result.stdout + result.stderr
    assert "Stopped at turn" in combined


def test_retrieval_depth() -> None:
    result = _run("retrieval_depth.py")
    assert result.returncode == 0, result.stderr
    combined = result.stdout + result.stderr
    assert "Stopped at step" in combined


def test_plan_complexity() -> None:
    result = _run("plan_complexity.py")
    assert result.returncode == 0, result.stderr
    combined = result.stdout + result.stderr
    assert "Stopped in round" in combined


def test_tool_budget() -> None:
    result = _run("tool_budget.py")
    assert result.returncode == 0, result.stderr
    combined = result.stdout + result.stderr
    assert "stopped after" in combined


def test_step_budget() -> None:
    result = _run("step_budget.py")
    assert result.returncode == 0, result.stderr
    combined = result.stdout + result.stderr
    assert "run complete" in combined


def test_streaming_guard() -> None:
    result = _run("streaming_guard.py")
    assert result.returncode == 0, result.stderr
    combined = result.stdout + result.stderr
    assert "stream cut off" in combined


def test_budget_retry() -> None:
    result = _run("budget_retry.py")
    assert result.returncode == 0, result.stderr
    combined = result.stdout + result.stderr
    assert "retried on cheaper model" in combined


def test_openai_adapter() -> None:
    result = _run("openai_adapter.py")
    assert result.returncode == 0, result.stderr
    combined = result.stdout + result.stderr
    assert "blocked before reaching the client" in combined.lower()


def test_anthropic_adapter() -> None:
    result = _run("anthropic_adapter.py")
    assert result.returncode == 0, result.stderr
    combined = result.stdout + result.stderr
    assert "cache" in combined.lower()


def test_cli_demo() -> None:
    """`floe-guard demo` (python -m floe_guard demo) runs from the installed package."""
    env = {k: v for k, v in os.environ.items() if k not in {"FLOE_API_KEY", "OPENAI_API_KEY"}}
    result = subprocess.run(
        [sys.executable, "-m", "floe_guard", "demo"],
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    combined = result.stdout + result.stderr
    assert "Loop stopped at call" in combined
