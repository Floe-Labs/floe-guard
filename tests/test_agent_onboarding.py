"""Pip-install onboarding for coding agents must verify without a repo checkout.

``AGENTS.md``, ``SKILL.md``, and ``llms.txt`` are the unattended path
(install → wire → verify). ``python examples/runaway_loop.py`` only works
from a clone; the wheel does not ship ``examples/``. The packaged command is
``floe-guard demo`` (see PR #89). README already leads with it.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ONBOARDING = [
    REPO_ROOT / "AGENTS.md",
    REPO_ROOT / "SKILL.md",
    REPO_ROOT / "llms.txt",
]


def test_agent_verify_step_uses_packaged_demo() -> None:
    for path in ONBOARDING:
        text = path.read_text(encoding="utf-8")
        assert "floe-guard demo" in text, (
            f"{path.name}: pip-install verify must use `floe-guard demo` "
            "(examples/runaway_loop.py is checkout-only)"
        )
