/**
 * `floe-guard push` CLI — mirrors the CLI cases in the Python
 * `tests/test_ledger_sync.py`:
 *
 *   - a valid ledger FILE + --key → `pushLedger` runs, POSTs under the key,
 *     prints the synced count, exits 0;
 *   - a non-empty ledger with NO key → LedgerSyncError → exit 1, **zero** network
 *     (fetch never called) — the zero-telemetry / opt-in contract.
 *
 * `fetch` is stubbed (never a live call); the real `pushLedger` runs end-to-end.
 */

import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { main } from "../src/cli.js";

const ONE_EVENT =
  '{"timestamp":1.0,"kind":"tool","model_or_tool":"api","prompt_tokens":null,' +
  '"completion_tokens":null,"cost_usd":0.05}\n';

let dir: string;
let savedKey: string | undefined;
let savedBase: string | undefined;

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "floe-guard-cli-"));
  savedKey = globalThis.process.env.FLOE_API_KEY;
  savedBase = globalThis.process.env.FLOE_API_BASE_URL;
  delete globalThis.process.env.FLOE_API_KEY;
  delete globalThis.process.env.FLOE_API_BASE_URL;
});

afterEach(() => {
  vi.restoreAllMocks();
  rmSync(dir, { recursive: true, force: true });
  if (savedKey === undefined) delete globalThis.process.env.FLOE_API_KEY;
  else globalThis.process.env.FLOE_API_KEY = savedKey;
  if (savedBase === undefined) delete globalThis.process.env.FLOE_API_BASE_URL;
  else globalThis.process.env.FLOE_API_BASE_URL = savedBase;
});

function writeLedger(contents: string): string {
  const path = join(dir, "ledger.jsonl");
  writeFileSync(path, contents);
  return path;
}

describe("floe-guard push", () => {
  it("reads a ledger file and POSTs it under the key, printing the synced count", async () => {
    const ledger = writeLedger(ONE_EVENT);
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response(JSON.stringify({ synced: 1 }), { status: 200 }));
    const stdout = vi.spyOn(globalThis.process.stdout, "write").mockReturnValue(true);

    const rc = await main(["push", ledger, "--key", "floe_abc"]);

    expect(rc).toBe(0);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(String(url)).toContain("/v1/agents/ledger/sync");
    const headers = (init as RequestInit).headers as Record<string, string>;
    expect(headers.Authorization).toBe("Bearer floe_abc");
    expect(stdout).toHaveBeenCalledWith("Synced 1 spend event(s) to Floe Reconcile Mode.\n");
  });

  it("with a non-empty ledger and NO key: exits 1 and makes zero network calls", async () => {
    const ledger = writeLedger(ONE_EVENT);
    const fetchMock = vi.spyOn(globalThis, "fetch");
    const stderr = vi.spyOn(globalThis.process.stderr, "write").mockReturnValue(true);

    const rc = await main(["push", ledger]); // no --key, $FLOE_API_KEY unset

    expect(rc).toBe(1);
    expect(fetchMock).not.toHaveBeenCalled();
    expect(stderr).toHaveBeenCalledWith(
      expect.stringContaining("floe-guard push failed: No Floe API key"),
    );
  });

  it("--help prints usage and exits 0 without touching the network", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch");
    const stdout = vi.spyOn(globalThis.process.stdout, "write").mockReturnValue(true);

    const rc = await main(["push", "--help"]);

    expect(rc).toBe(0);
    expect(fetchMock).not.toHaveBeenCalled();
    expect(stdout).toHaveBeenCalledWith(expect.stringContaining("floe-guard push"));
  });
});
