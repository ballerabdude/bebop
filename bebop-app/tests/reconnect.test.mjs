import assert from "node:assert/strict";
import test from "node:test";

import { reconnectDelayMs, RECONNECT_BASE_MS, RECONNECT_MAX_MS } from "../src/runtime/backoff.ts";

test("reconnect backoff starts at 250 ms, doubles, caps at 5 s", () => {
  assert.equal(RECONNECT_BASE_MS, 250);
  assert.equal(RECONNECT_MAX_MS, 5000);
  assert.deepEqual(
    [0, 1, 2, 3, 4, 5, 6, 10].map(reconnectDelayMs),
    [250, 500, 1000, 2000, 4000, 5000, 5000, 5000],
  );
});
