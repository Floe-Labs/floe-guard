/**
 * Minimal ambient types for the handful of Node built-ins the CLI + sync client
 * touch. The package ships with `types: []` (no `@types/node`) so it stays a lean
 * browser-and-server library; this file declares only the exact surface used —
 * `node:fs` readFileSync, `node:url` pathToFileURL, and the `process` global
 * (env / argv / stdin / std{out,err} / exit) — rather than pulling in the whole
 * Node typings. Keep it to what is actually imported.
 */

declare module "node:fs" {
  export function readFileSync(path: string, encoding: "utf-8"): string;
}

declare module "node:url" {
  export function pathToFileURL(path: string): { readonly href: string };
}

interface FloeNodeWritable {
  write(chunk: string): boolean;
}

interface FloeNodeProcess {
  argv: string[];
  env: Record<string, string | undefined>;
  exit(code?: number): never;
  stdin: AsyncIterable<Uint8Array>;
  stdout: FloeNodeWritable;
  stderr: FloeNodeWritable;
}

// A global `var` declaration is added to `typeof globalThis`, so both `process`
// and `globalThis.process` type-check. The sync client + CLI read it via
// `globalThis.process` to make the runtime dependency explicit at the call site.
declare var process: FloeNodeProcess;
