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

// A single valid exportLog() event — the fail-closed tests use this so ledger
// validation (which runs before the key/URL/HTTP checks) passes and the specific
// guard under test is what fires.
const ONE_EVENT = '{"timestamp":1.0,"kind":"tool","model_or_tool":"api","cost_usd":0.01}\n';

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
    // Redirects refused — the key + ledger can't be re-sent to another host on a 3xx.
    expect(init.redirect).toBe("error");
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

  it("sends the key it was enabled with, not $FLOE_API_KEY (atomic opt-in snapshot)", async () => {
    // A different key in the env must not win over the one passed to enableSync,
    // and can't leak in if disableSync races the fetch.
    process.env.FLOE_API_KEY = "floe_env_key";
    const guard = guardWithSpend();
    guard.enableSync("floe_explicit");
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(ok({ synced: 1 }));
    await guard.sync();
    const [, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
    expect((init.headers as Record<string, string>).Authorization).toBe("Bearer floe_explicit");
  });

  it("an absent synced count resolves 0", async () => {
    const guard = guardWithSpend();
    guard.enableSync("floe_abc");
    vi.spyOn(globalThis, "fetch").mockResolvedValue(ok({}));
    await expect(guard.sync()).resolves.toBe(0);
  });

  it.each([
    ["fractional", { synced: 1.5 }],
    ["negative", { synced: -1 }],
    ["boolean", { synced: true }],
    ["string", { synced: "3" }],
  ])("rejects a %s synced count (no coercion)", async (_label, payload) => {
    const guard = guardWithSpend();
    guard.enableSync("floe_abc");
    vi.spyOn(globalThis, "fetch").mockResolvedValue(ok(payload));
    await expect(guard.sync()).rejects.toThrow(LedgerSyncError);
  });
});

// ── AC3: the request body is validated against exportLog() before any send ─────

describe("pushLedger validates the ledger before sending", () => {
  it("rejects a line with a smuggled `prompt` field and NEVER calls fetch", async () => {
    const smuggled =
      '{"timestamp":1,"kind":"tool","model_or_tool":"api","cost_usd":0.05,"prompt":"leak me"}\n';
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    await expect(pushLedger(smuggled, "floe_abc")).rejects.toThrow(LedgerSyncError);
    await expect(pushLedger(smuggled, "floe_abc")).rejects.toThrow(/outside the exportLog\(\) schema/);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it.each([
    ["unknown identifier field", '{"timestamp":1,"kind":"tool","model_or_tool":"api","cost_usd":0.05,"user_id":"u1"}'],
    ["missing timestamp", '{"kind":"tool","model_or_tool":"api","cost_usd":0.05}'],
    ["missing kind", '{"timestamp":1,"model_or_tool":"api","cost_usd":0.05}'],
    ["missing model_or_tool", '{"timestamp":1,"kind":"tool","cost_usd":0.05}'],
    ["missing cost_usd", '{"timestamp":1,"kind":"tool","model_or_tool":"api"}'],
    ["bad kind", '{"timestamp":1,"kind":"other","model_or_tool":"api","cost_usd":0.05}'],
    ["non-string model_or_tool", '{"timestamp":1,"kind":"tool","model_or_tool":123,"cost_usd":0.05}'],
    ["negative cost_usd", '{"timestamp":1,"kind":"tool","model_or_tool":"api","cost_usd":-1}'],
    ["non-number cost_usd", '{"timestamp":1,"kind":"tool","model_or_tool":"api","cost_usd":"x"}'],
    ["boolean cost_usd", '{"timestamp":1,"kind":"tool","model_or_tool":"api","cost_usd":true}'],
    ["non-number timestamp", '{"timestamp":"x","kind":"tool","model_or_tool":"api","cost_usd":0.05}'],
    [
      "fractional prompt_tokens",
      '{"timestamp":1,"kind":"llm","model_or_tool":"m","prompt_tokens":1.5,"completion_tokens":null,"cost_usd":0.05}',
    ],
    ["non-string label", '{"timestamp":1,"kind":"tool","model_or_tool":"api","cost_usd":0.05,"label":5}'],
    ["non-number reserved", '{"timestamp":1,"kind":"tool","model_or_tool":"api","cost_usd":0.05,"reserved":"x"}'],
    ["malformed JSON", "not json"],
    ["not a JSON object", "[1,2,3]"],
  ])("rejects a bad-schema line (%s) before any fetch", async (_label, line) => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    await expect(pushLedger(`${line}\n`, "floe_abc")).rejects.toThrow(LedgerSyncError);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("accepts a valid tool + llm ledger (with optional label/reserved)", async () => {
    const valid =
      '{"timestamp":1,"kind":"tool","model_or_tool":"api","prompt_tokens":null,"completion_tokens":null,"cost_usd":0.05,"label":"x"}\n' +
      '{"timestamp":2,"kind":"llm","model_or_tool":"gpt-4o","prompt_tokens":10,"completion_tokens":5,"cost_usd":0.01,"reserved":0.02}\n';
    vi.spyOn(globalThis, "fetch").mockResolvedValue(ok({ synced: 2 }));
    await expect(pushLedger(valid, "floe_abc")).resolves.toBe(2);
  });
});

// ── pushLedger fail-closed ────────────────────────────────────────────────────

describe("pushLedger fail-closed", () => {
  it("missing key throws LedgerSyncError with no network", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    await expect(pushLedger(ONE_EVENT)).rejects.toThrow(LedgerSyncError);
    await expect(pushLedger(ONE_EVENT)).rejects.toThrow(/No Floe API key/);
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
        pushLedger(ONE_EVENT, "floe_abc", { baseUrl: badBase }),
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
    await expect(pushLedger(ONE_EVENT, "floe_abc")).rejects.toThrow(needle as RegExp);
  });

  it("a network failure throws LedgerSyncError", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("fetch failed"));
    await expect(pushLedger(ONE_EVENT, "floe_abc")).rejects.toThrow(LedgerSyncError);
  });
});
