// From js/: npm run build && node ../examples/vapi_streaming_guard.mjs
// No API keys, provider requests, or Vapi account required.
import assert from "node:assert/strict";
import { BudgetGuard } from "../js/dist/index.js";
import { BudgetExceeded, VapiBudgetGuard } from "../js/dist/adapters/vapi.js";

const guard = new BudgetGuard(0.0025, {
  priceOverrides: { demo: { inputCostPerToken: 0, outputCostPerToken: 0.001 } },
  onBlock: () => {},
});
const budget = new VapiBudgetGuard(guard, { model: "demo" });
let closed = false;
let generated = 0;
let forwarded = 0;
async function* source() {
  try {
    for (let i = 0; i < 10; i++) {
      generated++;
      yield { choices: [{ delta: { content: "word" } }] };
    }
  } finally { closed = true; }
}
try {
  for await (const chunk of budget.guardStream(source)) {
    forwarded++;
    console.log("Forwarded:", chunk.choices[0].delta.content);
  }
  assert.fail("Expected a mid-stream budget interruption");
} catch (error) {
  if (!(error instanceof BudgetExceeded)) throw error;
}
assert.equal(generated, 3);
assert.equal(forwarded, 2);
assert.equal(closed, true);
assert.equal(guard.spendLog.length, 1);
assert.ok(Math.abs(guard.spentUsd - 0.003) < 1e-12);
console.log(`Stopped after ${generated} generated chunks; recorded $${guard.spentUsd.toFixed(4)}.`);
console.log("Source closed; the crossing chunk was billed but not forwarded.");
