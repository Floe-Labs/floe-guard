/** Run `cd js && npm ci && npm run build`, then `node ../examples/streaming_guard.mjs`.
 * No API key, provider SDK or network access. Uses the actual built package.
 */
import assert from "node:assert/strict";
import { BudgetExceeded, BudgetGuard, guardStream } from "../js/dist/index.js";

const guard = new BudgetGuard(0.001, {
  priceOverrides: {
    "demo-model": { inputCostPerToken: 0, outputCostPerToken: 0.00001 },
  },
  onBlock: () => {},
});
// As in the Python example, reject an oversized first request before opening
// a source. This uses the existing estimateCall API, not a new streaming API.
const estimate = guard.estimateCall("demo-model", 1000, 100000);
assert.equal(estimate, 1);
assert.throws(() => guard.reserve(estimate), BudgetExceeded);
assert.equal(guard.spentUsd, 0);
assert.equal(guard.remainingUsd, 0.001);
console.log("PASS: oversized first request blocked before generation; $0 spent.");

let sourceClosed = false;
async function* response() {
  try {
    for (;;) yield "a".repeat(40); // ten estimated tokens, $0.0001 per chunk
  } finally {
    sourceClosed = true;
  }
}

let delivered = 0;
try {
  for await (const _chunk of guardStream(guard, "demo-model", response())) delivered++;
  assert.fail("The runaway stream must stop");
} catch (error) {
  if (!(error instanceof BudgetExceeded)) throw error;
  console.log(`Stopped after ${delivered} delivered chunks.`);
  console.log(`Partial spend recorded: $${guard.spentUsd.toFixed(4)} (limit $0.0010).`);
}
assert.equal(delivered, 10);
assert.equal(sourceClosed, true);
assert.equal(guard.spendLog.length, 1);
assert.equal(guard.spendLog[0].completionTokens, 110);
console.log("PASS: crossing chunk recorded, source closed, one ledger entry.");
