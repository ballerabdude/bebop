import assert from "node:assert/strict";
import test from "node:test";
import { predictEvictions } from "../src/screens/settings/modelEvictions.ts";

const b = (id, state, gb, kind = "edgellm") => ({ id, state, footprint_gb: gb, kind });

test("no eviction when the new backend fits", () => {
  const list = [b("voice", "ready", 20), b("vla", "stopped", 30)];
  assert.deepEqual(predictEvictions(list, "vla", 100), []);
});

test("evicts residents in order until it fits; tracked never evicted", () => {
  const list = [b("cam", "tracked", 10, "tracked"), b("voice", "ready", 40), b("brain", "ready", 30), b("vla", "stopped", 50)];
  assert.deepEqual(predictEvictions(list, "vla", 100).map((x) => x.id), ["voice"]);
  assert.deepEqual(predictEvictions(list, "vla", 70).map((x) => x.id), ["voice", "brain"]);
});

test("free-memory check also forces eviction", () => {
  const list = [b("voice", "ready", 20), b("vla", "stopped", 30)];
  assert.deepEqual(predictEvictions(list, "vla", 100, 10).map((x) => x.id), ["voice"]);
});

test("already running target evicts nothing", () => {
  const list = [b("voice", "ready", 90), b("vla", "starting", 30)];
  assert.deepEqual(predictEvictions(list, "vla", 100), []);
});
