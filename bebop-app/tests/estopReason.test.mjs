import assert from "node:assert/strict";
import test from "node:test";

import { humanizeEStopReason, humanizeFlag } from "../src/robot/estopReason.ts";

test("motor fault with a single flag", () => {
  assert.equal(
    humanizeEStopReason("right: motor fault 0x00 (DC_BUS_OVER_VOLTAGE (0x00000100))"),
    "Right motor fault: DC bus over-voltage",
  );
});

test("motor fault with several flags", () => {
  assert.equal(
    humanizeEStopReason("left: motor fault 0x0C (OVER_TEMPERATURE | DC_BUS_UNDER_VOLTAGE)"),
    "Left motor fault: over-temperature, DC bus under-voltage",
  );
});

test("operator stops", () => {
  assert.equal(humanizeEStopReason("operator: operator"), "Stopped by an operator");
  assert.equal(humanizeEStopReason("operator: voice"), "Stopped by an operator (voice)");
});

test("unknown shapes pass through", () => {
  assert.equal(humanizeEStopReason("process exit"), "Process exit");
  assert.equal(
    humanizeEStopReason("left: feedback watchdog: 512 ms since last RX (timeout 200 ms)"),
    "Left: feedback watchdog: 512 ms since last RX (timeout 200 ms)",
  );
  assert.equal(humanizeEStopReason(""), "");
});

test("flag words", () => {
  assert.equal(humanizeFlag("CAN_BUS_OFF"), "CAN bus off");
});
