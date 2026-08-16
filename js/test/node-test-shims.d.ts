/**
 * Test-only ambient types for Node built-ins used by the CLI/sync tests (temp
 * files, env). Kept out of `src/node-runtime.d.ts` so the shipped library shim
 * stays limited to what production code imports. Ambient `declare module` blocks
 * merge, so this augments the `node:fs` surface declared there.
 */

declare module "node:fs" {
  export function writeFileSync(path: string, data: string): void;
  export function mkdtempSync(prefix: string): string;
  export function rmSync(path: string, options?: { recursive?: boolean; force?: boolean }): void;
}

declare module "node:os" {
  export function tmpdir(): string;
}

declare module "node:path" {
  export function join(...parts: string[]): string;
}
