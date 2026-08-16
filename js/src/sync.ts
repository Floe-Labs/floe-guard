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
  // ingested, idempotent) are not counted here. Absent / non-numeric → 0.
  const synced = (payload as Record<string, unknown>).synced;
  const n = typeof synced === "number" ? synced : Number(synced);
  return Number.isFinite(n) ? Math.trunc(n) : 0;
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
