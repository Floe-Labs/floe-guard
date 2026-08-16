/**
 * floe-guard CLI.
 *
 * One command today: `floe-guard push` — the **opt-in** ledger sync. It reads a
 * {@link BudgetGuard.exportLog} JSONL ledger (from a file or stdin) and POSTs it to
 * Floe's Reconcile Mode so your **Coverage Score** can count spend the gateway
 * never routed (BYOK / self-hosted / off-path). Nothing runs on import or in the
 * background — `push` is a one-shot send you invoke, with your key. Zero telemetry
 * otherwise.
 *
 * Mirrors `src/floe_guard/__main__.py` in the Python package — same flags, same
 * exit codes, same messages.
 */

import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

import { LedgerSyncError } from "./errors.js";
import { pushLedger } from "./sync.js";

const USAGE = `floe-guard — a local budget guardrail for AI agents

Usage:
  floe-guard push [ledger.jsonl] [--key <floe_...>] [--base-url <url>]

Commands:
  push    Opt-in: push an exportLog() JSONL ledger to Floe Reconcile Mode
          (Coverage Score). Sends only priced spend events — no prompts,
          no content, only with your key. Budget, not balance.

Run 'floe-guard push --help' for push options.`;

const PUSH_USAGE = `Usage: floe-guard push [ledger.jsonl] [options]

Push a floe-guard exportLog() JSONL ledger to Floe's Reconcile Mode. Opt-in and
explicit — it sends only the priced spend events in the ledger (no prompts, no
content), only with your key.

Arguments:
  ledger          Ledger JSONL file (exportLog() output). Omit or '-' to read stdin.

Options:
  --key <key>     Floe agent key (floe_<hex>, read_write). Defaults to $FLOE_API_KEY.
  --base-url <url>  API base URL. Defaults to $FLOE_API_BASE_URL, else the prod host.
  -h, --help      Show this help.`;

interface ParsedPush {
  ledger: string;
  key?: string;
  baseUrl?: string;
  help: boolean;
}

/**
 * Hand-rolled parse for `push` args — positional `ledger` (default `-`), plus
 * `--key`, `--base-url`, `-h/--help`. Dependency-free by design.
 */
function parsePushArgs(args: string[]): ParsedPush {
  const parsed: ParsedPush = { ledger: "-", help: false };
  let sawLedger = false;

  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];
    if (arg === "-h" || arg === "--help") {
      parsed.help = true;
    } else if (arg === "--key") {
      const value = args[i + 1];
      if (value === undefined) throw new Error("--key needs a value");
      parsed.key = value;
      i += 1;
    } else if (arg?.startsWith("--key=")) {
      parsed.key = arg.slice("--key=".length);
    } else if (arg === "--base-url") {
      const value = args[i + 1];
      if (value === undefined) throw new Error("--base-url needs a value");
      parsed.baseUrl = value;
      i += 1;
    } else if (arg?.startsWith("--base-url=")) {
      parsed.baseUrl = arg.slice("--base-url=".length);
    } else if (arg !== undefined && !sawLedger) {
      // First positional (including "-") is the ledger path.
      parsed.ledger = arg;
      sawLedger = true;
    } else {
      throw new Error(`unexpected argument: ${arg}`);
    }
  }
  return parsed;
}

async function readStdin(): Promise<string> {
  const chunks: Uint8Array[] = [];
  for await (const chunk of globalThis.process.stdin) {
    chunks.push(chunk);
  }
  let total = 0;
  for (const c of chunks) total += c.length;
  const joined = new Uint8Array(total);
  let offset = 0;
  for (const c of chunks) {
    joined.set(c, offset);
    offset += c.length;
  }
  return new TextDecoder().decode(joined);
}

/**
 * CLI entry. Returns a process exit code (0 success, 1 read/sync failure, 2 usage
 * error) — the caller maps it to `process.exit`. Import-safe: it does not exit or
 * touch the network on its own.
 */
export async function main(argv: string[]): Promise<number> {
  const [command, ...rest] = argv;

  if (command === undefined) {
    globalThis.process.stderr.write(`${USAGE}\n`);
    return 2;
  }
  if (command === "-h" || command === "--help") {
    globalThis.process.stdout.write(`${USAGE}\n`);
    return 0;
  }
  if (command !== "push") {
    globalThis.process.stderr.write(`floe-guard: unknown command ${JSON.stringify(command)}\n${USAGE}\n`);
    return 2;
  }

  let parsed: ParsedPush;
  try {
    parsed = parsePushArgs(rest);
  } catch (exc) {
    const detail = exc instanceof Error ? exc.message : String(exc);
    globalThis.process.stderr.write(`floe-guard push: ${detail}\n${PUSH_USAGE}\n`);
    return 2;
  }
  if (parsed.help) {
    globalThis.process.stdout.write(`${PUSH_USAGE}\n`);
    return 0;
  }

  let jsonl: string;
  try {
    jsonl = parsed.ledger === "-" ? await readStdin() : readFileSync(parsed.ledger, "utf-8");
  } catch (exc) {
    const detail = exc instanceof Error ? exc.message : String(exc);
    globalThis.process.stderr.write(`floe-guard push: cannot read ${JSON.stringify(parsed.ledger)}: ${detail}\n`);
    return 1;
  }

  try {
    const n = await pushLedger(jsonl, parsed.key, { baseUrl: parsed.baseUrl });
    globalThis.process.stdout.write(`Synced ${n} spend event(s) to Floe Reconcile Mode.\n`);
    return 0;
  } catch (exc) {
    if (exc instanceof LedgerSyncError) {
      globalThis.process.stderr.write(`floe-guard push failed: ${exc.message}\n`);
      return 1;
    }
    throw exc;
  }
}

// Auto-run only when invoked as the CLI entry (dist/cli.js), NOT when imported by
// tests — mirrors Python's `if __name__ == "__main__"`.
const entry = globalThis.process.argv[1];
if (entry !== undefined && import.meta.url === pathToFileURL(entry).href) {
  void main(globalThis.process.argv.slice(2)).then((code) => globalThis.process.exit(code));
}
