"""floe-guard CLI.

One command today: ``floe-guard push`` — the **opt-in** ledger sync. It reads an
:meth:`~floe_guard.BudgetGuard.export_log` JSONL ledger (from a file or stdin) and
POSTs it to Floe's Reconcile Mode so your **Coverage Score** can count spend the
gateway never routed (BYOK / self-hosted / off-path). Nothing runs on import or in
the background — ``push`` is a one-shot send you invoke, with your key. Zero
telemetry otherwise.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

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

    args = parser.parse_args(argv)

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
