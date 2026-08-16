import { defineConfig } from "tsup";

export default defineConfig([
  {
    entry: [
      "src/index.ts",
      "src/adapters/livekit.ts",
      "src/adapters/vapi.ts",
      "src/adapters/retell.ts",
    ],
    format: ["esm", "cjs"],
    dts: true,
    clean: true,
    sourcemap: true,
    target: "es2022",
  },
  {
    // The `floe-guard` bin — a standalone Node CLI. Built separately so the
    // shebang banner lands ONLY on dist/cli.js (not the importable library
    // entries). ESM-only; no d.ts (it is executed, not imported).
    entry: ["src/cli.ts"],
    format: ["esm"],
    dts: false,
    clean: false,
    sourcemap: true,
    target: "es2022",
    banner: { js: "#!/usr/bin/env node" },
    // Make the emitted bin executable for direct `./dist/cli.js` invocation.
    onSuccess: "chmod +x dist/cli.js",
  },
]);
