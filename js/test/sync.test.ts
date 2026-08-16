/**
 * `pushLedger` — the opt-in ledger sync client. Mirrors the `push_ledger` cases in
 * the Python `tests/test_ledger_sync.py`: an empty ledger is a no-op with zero
 * network; a missing key / non-https base / smuggled field all fail-closed BEFORE
 * any send; a non-2xx is surfaced; the happy path returns the server's `synced`.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { LedgerSyncError } from "../src/errors.js";
import { pushLedger } from "../src/sync.js";

const ONE_EVENT = '{"timestamp":1.0,"kind":"tool","model_or_tool":"api","cost_usd":0.01}\n';

let savedKey: string | undefined;
let savedBase: string | undefined;

beforeEach(() => {
  savedKey = globalThis.process.env.FLOE_API_KEY;
  savedBase = globalThis.process.env.FLOE_API_BASE_URL;
  delete globalThis.process.env.FLOE_API_KEY;
  delete globalThis.process.env.FLOE_API_BASE_URL;
});

afterEach(() => {
  vi.restoreAllMocks();
  if (savedKey === undefined) delete globalThis.process.env.FLOE_API_KEY;
  else globalThis.process.env.FLOE_API_KEY = savedKey;
  if (savedBase === undefined) delete globalThis.process.env.FLOE_API_BASE_URL;
  else globalThis.process.env.FLOE_API_BASE_URL = savedBase;
});

describe("pushLedger", () => {
  it("an empty ledger is a no-op and never touches the network", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch");
    expect(await pushLedger("   ")).toBe(0);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("a missing key fails closed before any network", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch");
    await expect(pushLedger(ONE_EVENT)).rejects.toBeInstanceOf(LedgerSyncError);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("refuses a non-https base URL (key + ledger must not leave over http)", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch");
    await expect(
      pushLedger(ONE_EVENT, "floe_abc", { baseUrl: "http://evil.test" }),
    ).rejects.toThrow(/must be an https/);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("rejects a smuggled non-schema field BEFORE any network (privacy contract)", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch");
    const bad =
      '{"timestamp":1.0,"kind":"tool","model_or_tool":"api","cost_usd":0.01,"prompt":"secret"}\n';
    await expect(pushLedger(bad, "floe_abc")).rejects.toThrow(/outside the exportLog/);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("surfaces a 401 as a LedgerSyncError", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ error: "bad key" }), { status: 401 }),
    );
    await expect(pushLedger(ONE_EVENT, "floe_abc")).rejects.toThrow(/401 unauthorized/);
  });

  it("returns the server's accepted count on success", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ synced: 3 }), { status: 200 }),
    );
    expect(await pushLedger(ONE_EVENT, "floe_abc")).toBe(3);
  });

  it("treats an absent 'synced' as zero", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), { status: 200 }),
    );
    expect(await pushLedger(ONE_EVENT, "floe_abc")).toBe(0);
  });
});
