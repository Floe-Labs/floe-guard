"""floe-guard CLI.

Two commands:

- ``floe-guard demo`` — run the no-key "stop a loop" demo (stub LLM, no account,
  no network) straight from the installed package. ``--limit-usd`` sets the
  ceiling (default $0.10).
- ``floe-guard push`` — the **opt-in** ledger sync. It reads an
  :meth:`~floe_guard.BudgetGuard.export_log` JSONL ledger (from a file or stdin)
  and POSTs it to Floe's Reconcile Mode so your **Coverage Score** can count spend
  the gateway never routed (BYOK / self-hosted / off-path).

Nothing runs on import or in the background — every command is invoked explicitly,
and only ``push`` touches the network (with your key). Zero telemetry otherwise.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .demo import run_demo
from .errors import LedgerSyncError
from .sync import push_ledger


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="floe-guard", description="floe-guard CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    push = sub.add_parser(
        "push",
        help="Opt-in: push an export_log() JSONL ledger to Floe Reconcile Mode (Coverage Score).",
        description=(
            "Push a floe-guard export_log() JSONL ledger to Floe's Reconcile Mode. "
            "Opt-in and explicit — it sends only the priced spend events in the ledger "
            "(no prompts, no content), only with your key. Budget, not balance."
        ),
    )
    push.add_argument(
        "ledger",
        nargs="?",
        default="-",
        help="Ledger JSONL file (export_log() output). Omit or '-' to read stdin.",
    )
    push.add_argument(
        "--key",
        default=None,
        help="Floe agent key (floe_<hex>, read_write). Defaults to $FLOE_API_KEY.",
    )
    push.add_argument(
        "--base-url",
        default=None,
        help="API base URL. Defaults to $FLOE_API_BASE_URL, else the production host.",
    )

    demo = sub.add_parser(
        "demo",
        help="Run the no-key 'stop a loop' demo (no account, no network).",
        description=(
            "Run the runaway-loop demo: a naive agent loop that floe-guard hard-stops "
            "before it crosses a $0.10 ceiling. Stub LLM — no API key, no account, no "
            "network. The same demo as examples/runaway_loop.py, runnable straight from "
            "the installed package."
        ),
    )
    demo.add_argument(
        "--limit-usd",
        type=float,
        default=0.10,
        help="Spend ceiling for the demo, in USD (default: 0.10).",
    )

    args = parser.parse_args(argv)

    if args.command == "demo":
        try:
            run_demo(limit_usd=args.limit_usd)
        except ValueError as exc:
            # e.g. a negative/non-finite --limit-usd; surface a clean CLI error
            # (exit 2) instead of a Python traceback.
            parser.error(str(exc))
        return 0

    if args.command == "push":
        try:
            if args.ledger == "-":
                jsonl = sys.stdin.read()
            else:
                with open(args.ledger, encoding="utf-8") as fh:
                    jsonl = fh.read()
        except OSError as exc:
            print(f"floe-guard push: cannot read {args.ledger!r}: {exc}", file=sys.stderr)
            return 1
        try:
            n = push_ledger(jsonl, args.key, base_url=args.base_url)
        except LedgerSyncError as exc:
            print(f"floe-guard push failed: {exc}", file=sys.stderr)
            return 1
        print(f"Synced {n} spend event(s) to Floe Reconcile Mode.")
        return 0

    return 2  # unreachable: subparser is required


if __name__ == "__main__":
    raise SystemExit(main())
