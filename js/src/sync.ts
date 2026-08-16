/**
 * Opt-in ledger sync — push the local spend ledger to Floe's Reconcile Mode.
 *
 * **Zero-telemetry is the default, and stays the default.** Nothing in this module
 * runs unless the caller *explicitly* opts in with a Floe API key — via
 * {@link pushLedger} or the `floe-guard push` CLI. There is no implicit enablement
 * and no background send: the ledger leaves your process only when you call it.
 * Your ledger, your key, your choice.
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
 * Mirrors `src/floe_guard/sync.py` in the Python package — same schema, same
 * validation, same fail-closed contract.
 *
 *     Opt in: https://dev-dashboard.floelabs.xyz  ·  https://floelabs.xyz
 */

import { LedgerSyncError } from "./errors.js";

const FLOE_API_KEY_ENV = "FLOE_API_KEY";
const FLOE_API_BASE_URL_ENV = "FLOE_API_BASE_URL";
const DEFAULT_BASE_URL = "https://credit-api.floelabs.xyz";
const LEDGER_SYNC_PATH = "/v1/agents/ledger/sync";

/** Options for {@link pushLedger}. */
export interface PushLedgerOptions {
  /** API base. Defaults to `$FLOE_API_BASE_URL`, else the prod host. */
  baseUrl?: string;
  /** Socket timeout in milliseconds. Defaults to 30_000. */
  timeoutMs?: number;
}

// The ONLY keys allowed to leave the process — the exportLog() schema. Enforced
// (not just documented) below: a line carrying anything else is rejected, so a
// hand-supplied file can't smuggle prompts / content / identifiers past the
// privacy contract.
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

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

/**
 * Parse and validate every ledger line against the `exportLog()` schema.
 *
 * Throws {@link LedgerSyncError} on the first malformed / unknown-field / invalid
 * record, **before** anything is sent — so only priced spend events ever leave the
 * process, even when the ledger came from a hand-edited file or the CLI. This is
 * the privacy contract *enforced*, not merely asserted.
 */
function validateLedger(jsonl: string): void {
  const lines = jsonl.split("\n");
  for (let idx = 0; idx < lines.length; idx += 1) {
    const line = (lines[idx] ?? "").trim();
    if (!line) continue;
    const i = idx + 1;

    let event: unknown;
    try {
      event = JSON.parse(line);
    } catch (exc) {
      const detail = exc instanceof Error ? exc.message : String(exc);
      throw new LedgerSyncError(`Ledger line ${i} is not valid JSON: ${detail}`);
    }
    if (typeof event !== "object" || event === null || Array.isArray(event)) {
      throw new LedgerSyncError(`Ledger line ${i} is not a JSON object.`);
    }
    const record = event as Record<string, unknown>;

    const extra = Object.keys(record).filter((k) => !ALLOWED_KEYS.has(k));
    if (extra.length > 0) {
      throw new LedgerSyncError(
        `Ledger line ${i} has fields outside the exportLog() schema: ` +
          `${JSON.stringify(extra.sort())}. Only priced spend events may be synced — no prompts, ` +
          `content, or identifiers.`,
      );
    }
    const missing = REQUIRED_KEYS.filter((k) => !(k in record));
    if (missing.length > 0) {
      throw new LedgerSyncError(
        `Ledger line ${i} is missing required field(s): ${JSON.stringify(missing.sort())}.`,
      );
    }
    if (record.kind !== "llm" && record.kind !== "tool") {
      throw new LedgerSyncError(
        `Ledger line ${i}: kind must be 'llm' or 'tool', got ${JSON.stringify(record.kind)}.`,
      );
    }
    if (typeof record.model_or_tool !== "string") {
      throw new LedgerSyncError(`Ledger line ${i}: model_or_tool must be a string.`);
    }
    const cost = record.cost_usd;
    if (!isFiniteNumber(cost) || cost < 0) {
      throw new LedgerSyncError(
        `Ledger line ${i}: cost_usd must be a finite, non-negative number, got ${JSON.stringify(cost)}.`,
      );
    }
    if (!isFiniteNumber(record.timestamp)) {
      throw new LedgerSyncError(
        `Ledger line ${i}: timestamp must be a finite number, got ${JSON.stringify(record.timestamp)}.`,
      );
    }
    for (const field of ["prompt_tokens", "completion_tokens"] as const) {
      const value = record[field];
      if (value !== undefined && value !== null && !Number.isInteger(value)) {
        throw new LedgerSyncError(`Ledger line ${i}: ${field} must be an integer or null.`);
      }
    }
    if ("label" in record && typeof record.label !== "string") {
      throw new LedgerSyncError(`Ledger line ${i}: label must be a string.`);
    }
    if ("reserved" in record && !isFiniteNumber(record.reserved)) {
      throw new LedgerSyncError(`Ledger line ${i}: reserved must be a finite number.`);
    }
  }
}

function describeHttpError(status: number, body: string): string {
  let detail = "";
  try {
    const data: unknown = JSON.parse(body);
    if (
      typeof data === "object" &&
      data !== null &&
      typeof (data as Record<string, unknown>).error === "string"
    ) {
      detail = ` (${(data as Record<string, unknown>).error as string})`;
    }
  } catch {
    // non-JSON error body — no extra detail.
  }
  if (status === 401) {
    return `Floe rejected the API key (401 unauthorized)${detail}.`;
  }
  if (status === 403) {
    return (
      `Floe refused the sync (403 forbidden)${detail} — the agent may be ` +
      `closed/suspended, or the key is read-only (a read_write key is required).`
    );
  }
  return `Floe returned HTTP ${status}${detail}.`;
}

/**
 * POST an {@link BudgetGuard.exportLog} JSONL ledger to Reconcile Mode and return
 * the number of events the server accepted.
 *
 * This is the ONLY function that sends the ledger, and it sends only when called.
 * An empty ledger is a no-op (returns `0` without any network call).
 *
 * @param jsonl - the ledger as newline-delimited JSON — `exportLog()` output.
 * @param apiKey - Floe agent key (`floe_<hex>`, `read_write`). Defaults to the
 *   `FLOE_API_KEY` env var.
 * @param options - `baseUrl` (defaults to `FLOE_API_BASE_URL`, else the prod host)
 *   and `timeoutMs` (default 30_000).
 * @throws {@link LedgerSyncError} on a missing key, a non-2xx response, a
 *   network/timeout failure, or a malformed response body.
 */
export async function pushLedger(
  jsonl: string,
  apiKey?: string,
  options: PushLedgerOptions = {},
): Promise<number> {
  // An empty ledger never touches the network — nothing to send.
  if (!jsonl.trim()) return 0;

  // Enforce the privacy contract BEFORE any network: only exportLog() events —
  // no prompts, content, or identifiers — may leave, even from a hand-supplied file.
  validateLedger(jsonl);

  const env = globalThis.process.env;
  const key = (apiKey ?? env[FLOE_API_KEY_ENV] ?? "").trim();
  if (!key) {
    throw new LedgerSyncError(
      `No Floe API key: pass apiKey or set ${FLOE_API_KEY_ENV}. ` +
        `Sync is opt-in and needs your key.`,
    );
  }

  const envBase = (env[FLOE_API_BASE_URL_ENV] ?? "").trim();
  const base = ((options.baseUrl ?? "").trim() || envBase || DEFAULT_BASE_URL).replace(/\/+$/, "");
  let parsed: URL;
  try {
    parsed = new URL(base);
  } catch {
    parsed = new URL("http://");
  }
  if (parsed.protocol !== "https:" || !parsed.host) {
    // The request carries the Floe agent key as a bearer token AND your spend
    // ledger — never send either over a non-https or malformed URL.
    throw new LedgerSyncError(
      `Refusing to send the ledger to ${JSON.stringify(base)}: ` +
        `the base URL must be an https:// URL with a host.`,
    );
  }
  const url = `${base}${LEDGER_SYNC_PATH}`;

  let response: Response;
  try {
    response = await fetch(url, {
      method: "POST",
      // Refuse redirects: a 3xx could re-send the bearer key + ledger to a host
      // we never approved. Treat any redirect as an error.
      redirect: "error",
      signal: AbortSignal.timeout(options.timeoutMs ?? 30_000),
      headers: {
        Authorization: `Bearer ${key}`,
        "Content-Type": "application/x-ndjson",
        Accept: "application/json",
      },
      body: jsonl,
    });
  } catch (exc) {
    const detail = exc instanceof Error ? exc.message : String(exc);
    throw new LedgerSyncError(`Could not reach Floe at ${url}: ${detail}`);
  }

  if (!response.ok) {
    const errBody = await response.text().catch(() => "");
    throw new LedgerSyncError(describeHttpError(response.status, errBody));
  }

  let payload: unknown;
  try {
    payload = await response.json();
  } catch (exc) {
    const detail = exc instanceof Error ? exc.message : String(exc);
    throw new LedgerSyncError(`Malformed JSON from Floe at ${url}: ${detail}`);
  }

  if (typeof payload !== "object" || payload === null || Array.isArray(payload)) {
    throw new LedgerSyncError(`Unexpected response shape from Floe at ${url}.`);
  }
  // "synced" is the count the server accepted (new rows); "duplicates" (already
  // ingested, idempotent) are not counted here. Absent → 0; present must be a real
  // non-negative int (reject bool/float/negative/garbage rather than coerce).
  const synced = (payload as Record<string, unknown>).synced;
  if (synced === undefined || synced === null) return 0;
  if (typeof synced !== "number" || !Number.isInteger(synced) || synced < 0) {
    throw new LedgerSyncError(`Invalid 'synced' count from Floe at ${url}: ${JSON.stringify(synced)}.`);
  }
  return synced;
}
