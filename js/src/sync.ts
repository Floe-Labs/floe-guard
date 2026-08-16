/**
 * Opt-in ledger sync — push the local spend ledger to Floe's Reconcile Mode.
 *
 * **Zero-telemetry is the default, and stays the default.** Nothing in this module
 * runs unless the caller *explicitly* opts in with a Floe API key — via
 * {@link BudgetGuard.enableSync} + {@link BudgetGuard.sync}. There is no implicit
 * enablement and no background send: the ledger leaves your process only when you
 * call one of those. Your ledger, your key, your choice.
 *
 * **Why sync at all.** Floe's gateway can't see spend it never routed — BYOK,
 * self-hosted, or off-path LLM/tool calls. Pushing your local ledger into Reconcile
 * Mode is how that spend lands on the ledger and your **Coverage Score** becomes
 * computable. Budget, not balance: this reports *what you already spent* for
 * coverage/attribution; it does not move money or change any wallet balance.
 *
 * **What leaves the process — exactly the {@link BudgetGuard.exportLog} JSONL** and
 * nothing else: one line per spend event, each a priced-cost record — `timestamp`,
 * `kind` (`"llm"`/`"tool"`), `model_or_tool`, `prompt_tokens`, `completion_tokens`,
 * `cost_usd`, and the optional `label` / `reserved` you set. **No prompts, no
 * message content, no identifiers** beyond a `label` you choose.
 *
 * Mirrors `src/floe_guard/sync.py` in the Python package — same behaviour
 * (https-only, key-required, fail-closed), transported over `fetch` instead of
 * `urllib`.
 *
 *     Opt in: https://dev-dashboard.floelabs.xyz  ·  https://floelabs.xyz
 */

import { LedgerSyncError } from "./errors.js";

const FLOE_API_KEY_ENV = "FLOE_API_KEY";
const FLOE_API_BASE_URL_ENV = "FLOE_API_BASE_URL";
const DEFAULT_BASE_URL = "https://credit-api.floelabs.xyz";
const LEDGER_SYNC_PATH = "/v1/agents/ledger/sync";
const DEFAULT_TIMEOUT_MS = 30_000;

// The ONLY keys allowed to leave the process — the exportLog() schema. Enforced
// (not just documented) in validateLedger(): a line carrying anything else is
// rejected, so a hand-supplied file can't smuggle prompts / content / identifiers
// past the privacy contract.
const ALLOWED_KEYS = new Set([
  "timestamp",
  "kind",
  "model_or_tool",
  "prompt_tokens",
  "completion_tokens",
  "cost_usd",
  "label",
  "reserved",
]);
const REQUIRED_KEYS = ["timestamp", "kind", "model_or_tool", "cost_usd"];

/**
 * Read an environment variable without a hard `process` reference — the package
 * compiles with `types: []` (no `@types/node`), so `process` is untyped. Reads
 * live at call time so tests that mutate the env are honoured.
 */
function envVar(name: string): string {
  const proc = (globalThis as { process?: { env?: Record<string, string | undefined> } }).process;
  return proc?.env?.[name] ?? "";
}

/**
 * Parse and validate every ledger line against the `exportLog()` schema.
 *
 * Throws {@link LedgerSyncError} on the first malformed / unknown-field / invalid
 * record, **before** anything is sent — so only priced spend events ever leave the
 * process, even when the ledger came from a hand-edited file. This is the privacy
 * contract *enforced*, not merely asserted: an **extra/unknown key** (a smuggled
 * `prompt`, `user_id`, …) is rejected outright. Mirrors `_validate_ledger` in the
 * Python client.
 *
 * (JS `typeof x === "number"` already excludes booleans and `Number.isInteger`
 * rejects them too, so — unlike Python, where `bool` is an `int` subclass — no
 * explicit boolean guards are needed.)
 */
function validateLedger(jsonl: string): void {
  let i = 0;
  for (const raw of jsonl.split("\n")) {
    i += 1;
    const line = raw.trim();
    if (!line) continue;
    let event: unknown;
    try {
      event = JSON.parse(line);
    } catch (err) {
      throw new LedgerSyncError(`Ledger line ${i} is not valid JSON: ${describeCause(err)}`);
    }
    if (typeof event !== "object" || event === null || Array.isArray(event)) {
      throw new LedgerSyncError(`Ledger line ${i} is not a JSON object.`);
    }
    const ev = event as Record<string, unknown>;
    const extra = Object.keys(ev).filter((k) => !ALLOWED_KEYS.has(k));
    if (extra.length) {
      throw new LedgerSyncError(
        `Ledger line ${i} has fields outside the exportLog() schema: ` +
          `${JSON.stringify(extra.sort())}. Only priced spend events may be synced — ` +
          "no prompts, content, or identifiers.",
      );
    }
    const missing = REQUIRED_KEYS.filter((k) => !(k in ev));
    if (missing.length) {
      throw new LedgerSyncError(
        `Ledger line ${i} is missing required field(s): ${JSON.stringify(missing)}.`,
      );
    }
    if (ev.kind !== "llm" && ev.kind !== "tool") {
      throw new LedgerSyncError(
        `Ledger line ${i}: kind must be 'llm' or 'tool', got ${JSON.stringify(ev.kind)}.`,
      );
    }
    if (typeof ev.model_or_tool !== "string") {
      throw new LedgerSyncError(`Ledger line ${i}: model_or_tool must be a string.`);
    }
    const cost = ev.cost_usd;
    if (typeof cost !== "number" || !Number.isFinite(cost) || cost < 0) {
      throw new LedgerSyncError(
        `Ledger line ${i}: cost_usd must be a finite, non-negative number, got ${JSON.stringify(cost)}.`,
      );
    }
    const ts = ev.timestamp;
    if (typeof ts !== "number" || !Number.isFinite(ts)) {
      throw new LedgerSyncError(
        `Ledger line ${i}: timestamp must be a finite number, got ${JSON.stringify(ts)}.`,
      );
    }
    for (const field of ["prompt_tokens", "completion_tokens"] as const) {
      const value = ev[field];
      // Absent (undefined) or explicit null are fine — tool events carry null; a
      // present value must be an integer.
      if (value !== undefined && value !== null && !Number.isInteger(value)) {
        throw new LedgerSyncError(`Ledger line ${i}: ${field} must be an integer or null.`);
      }
    }
    if ("label" in ev && typeof ev.label !== "string") {
      throw new LedgerSyncError(`Ledger line ${i}: label must be a string.`);
    }
    if ("reserved" in ev) {
      const reserved = ev.reserved;
      if (typeof reserved !== "number" || !Number.isFinite(reserved)) {
        throw new LedgerSyncError(`Ledger line ${i}: reserved must be a finite number.`);
      }
    }
  }
}

/**
 * POST an {@link BudgetGuard.exportLog} JSONL ledger to Reconcile Mode and resolve
 * with the number of events the server accepted.
 *
 * This is the ONLY function that sends the ledger, and it sends only when called.
 * An empty ledger is a no-op (resolves `0` with **no** `fetch` call).
 *
 * @param jsonl the ledger as newline-delimited JSON — `exportLog()` output.
 * @param apiKey Floe agent key (`floe_<hex>`, `read_write`). Defaults to the
 *   `FLOE_API_KEY` env var.
 * @param opts.baseUrl API base. Defaults to `FLOE_API_BASE_URL`, else the prod host.
 * @param opts.timeoutMs request timeout in milliseconds (default 30_000).
 * @throws {LedgerSyncError} missing key, a non-https/malformed base URL, a non-2xx
 *   response, network/timeout, or a malformed response body. The key and ledger are
 *   never sent over a non-https or malformed URL.
 */
export async function pushLedger(
  jsonl: string,
  apiKey?: string,
  opts: { baseUrl?: string; timeoutMs?: number } = {},
): Promise<number> {
  // An empty ledger never touches the network — nothing to send.
  if (!jsonl.trim()) return 0;

  // Enforce the privacy contract BEFORE any network: only exportLog() events —
  // no prompts, content, or identifiers — may leave, even from a hand-supplied file.
  validateLedger(jsonl);

  const key = ((apiKey ?? "") || envVar(FLOE_API_KEY_ENV)).trim();
  if (!key) {
    throw new LedgerSyncError(
      `No Floe API key: pass apiKey or set ${FLOE_API_KEY_ENV}. ` +
        "Sync is opt-in and needs your key.",
    );
  }

  const envBase = envVar(FLOE_API_BASE_URL_ENV).trim();
  const base = ((opts.baseUrl ?? "").trim() || envBase || DEFAULT_BASE_URL).replace(/\/+$/, "");
  let parsed: URL;
  try {
    parsed = new URL(base);
  } catch {
    // The request carries the Floe agent key as a bearer token AND your spend
    // ledger — never send either over a malformed URL.
    throw new LedgerSyncError(
      `Refusing to send the ledger to '${base}': ` +
        "the base URL must be an https:// URL with a host.",
    );
  }
  if (parsed.protocol !== "https:" || !parsed.host) {
    // Same guard for a well-formed but non-https (or hostless) URL.
    throw new LedgerSyncError(
      `Refusing to send the ledger to '${base}': ` +
        "the base URL must be an https:// URL with a host.",
    );
  }
  const url = `${base}${LEDGER_SYNC_PATH}`;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), opts.timeoutMs ?? DEFAULT_TIMEOUT_MS);
  let response: Response;
  try {
    response = await fetch(url, {
      method: "POST",
      // Refuse redirects: a 3xx would otherwise let the key + ledger be re-sent to
      // another host. (The Fetch spec strips `Authorization` on a cross-origin
      // redirect, but `redirect: "error"` is the belt-and-suspenders — nothing is
      // re-sent anywhere, same-origin or not.)
      redirect: "error",
      headers: {
        Authorization: `Bearer ${key}`,
        "Content-Type": "application/x-ndjson",
        Accept: "application/json",
      },
      body: jsonl,
      signal: controller.signal,
    });
  } catch (err) {
    throw new LedgerSyncError(`Could not reach Floe at ${url}: ${describeCause(err)}`);
  } finally {
    clearTimeout(timer);
  }

  if (!response.ok) {
    throw new LedgerSyncError(await describeHttpError(response));
  }

  let payload: unknown;
  try {
    payload = JSON.parse(await response.text());
  } catch (err) {
    throw new LedgerSyncError(`Malformed JSON from Floe at ${url}: ${describeCause(err)}`);
  }
  if (typeof payload !== "object" || payload === null || Array.isArray(payload)) {
    throw new LedgerSyncError(`Unexpected response shape from Floe at ${url}.`);
  }
  // "synced" is the count the server accepted (new rows); "duplicates" (already
  // ingested, idempotent) are not counted here. Absent → 0; present must be a real
  // non-negative integer — reject bool/float/negative/garbage rather than coerce.
  const synced = (payload as Record<string, unknown>).synced;
  if (synced === undefined || synced === null) return 0;
  if (typeof synced !== "number" || !Number.isInteger(synced) || synced < 0) {
    throw new LedgerSyncError(`Invalid 'synced' count from Floe at ${url}: ${JSON.stringify(synced)}.`);
  }
  return synced;
}

function describeCause(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

async function describeHttpError(response: Response): Promise<string> {
  let detail = "";
  try {
    const data: unknown = JSON.parse(await response.text());
    if (
      typeof data === "object" &&
      data !== null &&
      typeof (data as Record<string, unknown>).error === "string"
    ) {
      detail = ` (${(data as Record<string, unknown>).error as string})`;
    }
  } catch {
    // Non-JSON / empty error body — the status code alone is the message.
  }
  const code = response.status;
  if (code === 401) return `Floe rejected the API key (401 unauthorized)${detail}.`;
  if (code === 403) {
    return (
      `Floe refused the sync (403 forbidden)${detail} — the agent may be ` +
      "closed/suspended, or the key is read-only (a read_write key is required)."
    );
  }
  return `Floe returned HTTP ${code}${detail}.`;
}
