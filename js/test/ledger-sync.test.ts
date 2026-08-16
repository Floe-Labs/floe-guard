/**
 * Opt-in ledger sync — zero-telemetry stays the default.
 *
 * The load-bearing test here is that a guard which never opted in makes **zero**
 * network calls, ever (`fetch` is asserted un-called). Sync sends only the
 * `exportLog()` JSONL, only under the explicit flag, only with a key. Mirrors
 * `tests/test_ledger_sync.py` in the Python package.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { BudgetGuard, LedgerSyncError, pushLedger } from "../src/index.js";

// The package compiles with `types: []` (no @types/node), so `process` is not
// globally typed. Declare the minimal shape the tests touch (real `process` is
// present at runtime under vitest's node environment).
declare const process: { env: Record<string, string | undefined> };

const LEDGER_KEYS = new Set([
  "timestamp",
  "kind",
  "model_or_tool",
  "prompt_tokens",
  "completion_tokens",
  "cost_usd",
  "label",
  "reserved",
]);

function ok(payload: unknown): Response {
  return new Response(JSON.stringify(payload), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

function guardWithSpend(): BudgetGuard {
  const guard = new BudgetGuard(10.0);
  guard.recordTool("api", 0.05); // exactly one ledger event
  return guard;
}

// Sync reads FLOE_API_KEY / FLOE_API_BASE_URL from the env — clear both so a real
// machine key can't leak into the "no key" / default-base assertions.
let savedKey: string | undefined;
let savedBase: string | undefined;
beforeEach(() => {
  savedKey = process.env.FLOE_API_KEY;
  savedBase = process.env.FLOE_API_BASE_URL;
  delete process.env.FLOE_API_KEY;
  delete process.env.FLOE_API_BASE_URL;
});
afterEach(() => {
  if (savedKey === undefined) delete process.env.FLOE_API_KEY;
  else process.env.FLOE_API_KEY = savedKey;
  if (savedBase === undefined) delete process.env.FLOE_API_BASE_URL;
  else process.env.FLOE_API_BASE_URL = savedBase;
  vi.restoreAllMocks();
});

// ── AC1: zero egress without opt-in ───────────────────────────────────────────

describe("zero egress without opt-in", () => {
  it("a full guard lifecycle with no enableSync never touches the network, and sync() rejects", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    const guard = new BudgetGuard(10.0);
    guard.recordTool("api", 0.05);
    guard.record("gpt-4o", 1200, 350);
    guard.advisory();
    guard.exportLog();
    await expect(guard.sync()).rejects.toThrow(/not enabled/);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("disableSync revokes the opt-in: sync() rejects with no network", async () => {
    const guard = guardWithSpend();
    guard.enableSync("floe_abc");
    guard.disableSync();
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    await expect(guard.sync()).rejects.toThrow(/not enabled/);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("an empty ledger syncs nothing and makes no network call", async () => {
    const guard = new BudgetGuard(10.0); // no spend recorded
    guard.enableSync("floe_abc");
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    await expect(guard.sync()).resolves.toBe(0);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("sync() before any enableSync rejects with no network", async () => {
    const guard = guardWithSpend();
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    await expect(guard.sync()).rejects.toThrow(/enableSync/);
    expect(fetchSpy).not.toHaveBeenCalled();
  });
});

// ── AC2: opt-in send is exactly exportLog(), under the key ────────────────────

describe("opt-in send", () => {
  it("sync() POSTs exactly exportLog() under the Bearer key to the sync path", async () => {
    const guard = guardWithSpend();
    guard.enableSync("floe_abc");
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(ok({ synced: 1 }));

    await expect(guard.sync()).resolves.toBe(1);

    expect(fetchSpy).toHaveBeenCalledTimes(1);
    const [url, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
    expect(String(url)).toMatch(/\/v1\/agents\/ledger\/sync$/);
    expect(init.method).toBe("POST");
    const headers = init.headers as Record<string, string>;
    expect(headers.Authorization).toBe("Bearer floe_abc");
    expect(headers["Content-Type"]).toBe("application/x-ndjson");
    // The body is EXACTLY exportLog() — no prompts, no content, no extra fields.
    expect(init.body).toBe(guard.exportLog());
    for (const line of guard.exportLog().split("\n").filter(Boolean)) {
      for (const k of Object.keys(JSON.parse(line))) {
        expect(LEDGER_KEYS.has(k)).toBe(true);
      }
    }
  });

  it("parses the server's {synced} count", async () => {
    const guard = guardWithSpend();
    guard.enableSync("floe_abc");
    vi.spyOn(globalThis, "fetch").mockResolvedValue(ok({ synced: 3 }));
    await expect(guard.sync()).resolves.toBe(3);
  });
});

// ── pushLedger fail-closed ────────────────────────────────────────────────────

describe("pushLedger fail-closed", () => {
  it("missing key throws LedgerSyncError with no network", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    await expect(pushLedger('{"cost_usd":0.01}\n')).rejects.toThrow(LedgerSyncError);
    await expect(pushLedger('{"cost_usd":0.01}\n')).rejects.toThrow(/No Floe API key/);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("an empty/whitespace ledger is a no-op with no network", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    await expect(pushLedger("   ")).resolves.toBe(0);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it.each(["http://credit-api.floelabs.xyz", "file:///etc/passwd", "ftp://x", "not-a-url", "https://"])(
    "refuses a non-https / malformed base URL (%s) — never sends the key or ledger",
    async (badBase) => {
      const fetchSpy = vi.spyOn(globalThis, "fetch");
      await expect(
        pushLedger('{"cost_usd":0.01}\n', "floe_abc", { baseUrl: badBase }),
      ).rejects.toThrow(LedgerSyncError);
      expect(fetchSpy).not.toHaveBeenCalled();
    },
  );

  it.each([
    [401, /401/],
    [403, /403/],
  ])("a non-2xx response (%i) throws LedgerSyncError", async (code, needle) => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response('{"error":"x"}', { status: code as number }),
    );
    await expect(pushLedger('{"cost_usd":0.01}\n', "floe_abc")).rejects.toThrow(needle as RegExp);
  });

  it("a network failure throws LedgerSyncError", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("fetch failed"));
    await expect(pushLedger('{"cost_usd":0.01}\n', "floe_abc")).rejects.toThrow(LedgerSyncError);
  });
});
