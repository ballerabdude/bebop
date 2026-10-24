// SetupTransport lifecycle against a fake WebSocket. The StrictMode case:
// mount -> request (socket CONNECTING) -> cleanup disconnect() -> remount ->
// request. The second request must get its own socket and succeed instead
// of inheriting the aborted attempt ("WebSocket error connecting to ...").

import assert from "node:assert/strict";
import test from "node:test";

import { create, fromBinary, toBinary } from "@bufbuild/protobuf";

import {
  AgentResponseSchema,
  ClientRequestSchema,
  ResponseStatus,
  WifiStatusSchema,
} from "../src/proto/bebop_pb.ts";

class FakeWS {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;
  static all = [];
  constructor(url) {
    this.url = url;
    this.readyState = FakeWS.CONNECTING;
    this.sent = [];
    FakeWS.all.push(this);
  }
  send(bytes) {
    this.sent.push(bytes);
  }
  close() {
    const wasConnecting = this.readyState === FakeWS.CONNECTING;
    this.readyState = FakeWS.CLOSED;
    // Browsers fire error + close asynchronously for a socket closed
    // before it was established.
    setTimeout(() => {
      if (wasConnecting) this.onerror?.(new Event("error"));
      this.onclose?.({});
    }, 0);
  }
  // test helpers
  open() {
    this.readyState = FakeWS.OPEN;
    this.onopen?.({});
  }
  replyWifi(ssid) {
    const req = fromBinary(ClientRequestSchema, this.sent.at(-1));
    const msg = create(AgentResponseSchema, {
      requestId: req.requestId,
      status: ResponseStatus.OK,
      payload: {
        case: "wifiStatus",
        value: create(WifiStatusSchema, { connected: true, ssid, ipAddress: "10.0.0.2" }),
      },
    });
    const bytes = toBinary(AgentResponseSchema, msg);
    this.onmessage?.({ data: bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) });
  }
}
globalThis.WebSocket = FakeWS;
globalThis.window ??= globalThis;
globalThis.location ??= { protocol: "http:" };

const { SetupTransport } = await import("../src/setup/setupTransport.ts");
const tick = () => new Promise((r) => setTimeout(r, 5));

test("disconnect during CONNECTING does not poison the next request", async () => {
  FakeWS.all = [];
  const t = new SetupTransport("robot.local");
  const first = t.getWifiStatus();
  const firstResult = first.then(
    () => "resolved",
    (e) => e.message,
  );
  assert.equal(FakeWS.all.length, 1);

  t.disconnect(); // StrictMode cleanup
  const second = t.getWifiStatus(); // StrictMode re-run
  assert.equal(FakeWS.all.length, 2, "remount opens a fresh socket");
  assert.match(FakeWS.all[1].url, /^ws:\/\/robot\.local:9091\/ws$/);

  await tick(); // the aborted socket reports its error/close
  const ws = FakeWS.all[1];
  ws.open();
  await tick();
  assert.equal(ws.sent.length, 1);
  ws.replyWifi("Home");
  const status = await second;
  assert.equal(status.ssid, "Home");
  assert.equal(status.connected, true);
  assert.match(await firstResult, /closed by the client|disconnected/);
  assert.equal(t.isConnected(), true);
});

test("a request after a plain disconnect reconnects", async () => {
  FakeWS.all = [];
  const t = new SetupTransport("robot.local");
  const p = t.getWifiStatus();
  FakeWS.all[0].open();
  await tick();
  FakeWS.all[0].replyWifi("A");
  assert.equal((await p).ssid, "A");

  t.disconnect();
  assert.equal(t.isConnected(), false);
  const q = t.getWifiStatus();
  assert.equal(FakeWS.all.length, 2);
  FakeWS.all[1].open();
  await tick();
  FakeWS.all[1].replyWifi("B");
  assert.equal((await q).ssid, "B");
});

test("a real connect error still surfaces", async () => {
  FakeWS.all = [];
  const t = new SetupTransport("robot.local");
  const p = t.getWifiStatus();
  FakeWS.all[0].readyState = FakeWS.CLOSED;
  FakeWS.all[0].onerror?.(new Event("error"));
  await assert.rejects(p, /WebSocket error connecting to ws:\/\/robot\.local:9091\/ws/);
});
