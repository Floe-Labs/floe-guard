"""Stop-the-loop demo — runs with NO API key and NO account.

A naive agent loop that calls an LLM forever. The only thing standing between you
and a five-figure overnight bill is ``floe-guard``: it hard-stops the loop before
the call that would cross your $0.10 ceiling.

Run it::

    python examples/runaway_loop.py

Or, straight from the installed package (no repository checkout needed)::

    floe-guard demo

The "LLM" here is a stub that returns fixed token usage — no network, no key. The
cost is priced offline from the bundled cost map, exactly as for a real ``gpt-4o``
call. The demo itself lives in the package (:func:`floe_guard.demo.run_demo`); this
file is a thin wrapper so a repo checkout keeps working.
"""

from __future__ import annotations

from floe_guard.demo import run_demo


def main() -> None:
    """Run the packaged demo. Kept as a stable entry point for callers/tests."""
    run_demo()


if __name__ == "__main__":
    main()
