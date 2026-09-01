"""floe-guard CLI.

Three commands:

- ``floe-guard demo`` — run the no-key "stop a loop" demo (stub LLM, no account,
  no network) straight from the installed package. ``--limit-usd`` sets the
  ceiling (default $0.10).
- ``floe-guard estimate`` — price an agent workload offline from the bundled
  cost map and print the ``BudgetGuard`` ceiling that covers it.
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
from .estimate import run_estimate
from .sync import DASHBOARD_URL, push_ledger


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

    estimate = sub.add_parser(
        "estimate",
        help="Price a workload offline and print the BudgetGuard ceiling that covers it.",
        description=(
            "Price an agent workload (model, calls, tokens per call) from the "
            "bundled cost map — no API key, no account, no network — and print "
            "the BudgetGuard(limit_usd=...) ceiling that covers the run."
        ),
    )
    estimate.add_argument("model", help="Model id as priced by the bundled cost map (e.g. gpt-4o).")
    estimate.add_argument(
        "--calls", type=int, default=1, help="Number of calls in the run (default: 1)."
    )
    estimate.add_argument(
        "--tokens-in", type=int, default=1_000, help="Prompt tokens per call (default: 1000)."
    )
    estimate.add_argument(
        "--tokens-out", type=int, default=1_000, help="Completion tokens per call (default: 1000)."
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

    if args.command == "estimate":
        try:
            run_estimate(
                args.model, calls=args.calls, tokens_in=args.tokens_in, tokens_out=args.tokens_out
            )
        except ValueError as exc:
            # e.g. an unpriceable model or --calls 0; surface a clean CLI error
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
        if n > 0:
            print(f"Synced {n} spend event(s) to Floe Reconcile Mode.")
            print(f"View hosted coverage at {DASHBOARD_URL}")
        elif jsonl.strip():
            # Non-empty ledger, all events already ingested (idempotent re-sync) —
            # hosted coverage exists, so still point at it. A truly empty ledger
            # made no network call at all; there is nothing hosted to point at.
            print(
                f"No new events to sync (all duplicates). "
                f"View hosted coverage at {DASHBOARD_URL}"
            )
        else:
            print("No new events to sync (ledger empty).")
        return 0

    return 2  # unreachable: subparser is required


if __name__ == "__main__":
    raise SystemExit(main())
