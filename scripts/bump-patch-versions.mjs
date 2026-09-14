// Bump the patch version of BOTH packages in place — pyproject.toml (PyPI) and
// js/package.json (npm) — and print "py X, js Y" for the commit message.
//
// Used by the weekly cost-map refresh: version-guard fails any PR that changes
// package source (both cost maps live under it) without a bump, and a refresh
// merged unbumped never publishes. Edits the one version line in each file by
// string replacement so formatting and key order are untouched.
import { readFileSync, writeFileSync } from 'node:fs';

function bumpPatch(v) {
  const parts = v.split('.');
  if (parts.length !== 3 || parts.some((p) => !/^\d+$/.test(p))) {
    throw new Error(`refusing to bump non-X.Y.Z version "${v}"`);
  }
  return `${parts[0]}.${parts[1]}.${Number(parts[2]) + 1}`;
}

const pyPath = 'pyproject.toml';
const py = readFileSync(pyPath, 'utf8');
const pyMatch = py.match(/^version = "([^"]+)"$/m);
if (!pyMatch) throw new Error(`no top-level version line in ${pyPath}`);
const pyNew = bumpPatch(pyMatch[1]);
writeFileSync(pyPath, py.replace(pyMatch[0], `version = "${pyNew}"`));

const jsPath = 'js/package.json';
const js = readFileSync(jsPath, 'utf8');
const jsOld = JSON.parse(js).version;
const jsLine = `"version": "${jsOld}"`;
if (!js.includes(jsLine)) throw new Error(`cannot find ${jsLine} in ${jsPath}`);
const jsNew = bumpPatch(jsOld);
writeFileSync(jsPath, js.replace(jsLine, `"version": "${jsNew}"`));

console.log(`py ${pyNew}, js ${jsNew}`);
