import assert from "node:assert/strict";
import test from "node:test";
import { createOdomVelocity, driverStatus, signed } from "../src/screens/drive/hudModel.ts";
import { sampleFromReport, summarizeVideoStats } from "../src/components/videoStats.ts";

const close = (a, b, eps = 1e-6) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);

test("odom velocity: straight line at 0.5 m/s converges", () => {
  const est = createOdomVelocity({ tauMs: 1 });
  let out;
  for (let i = 0; i <= 30; i++) out = est.push({ tMs: i * 33, x: 0.5 * i * 0.033, y: 0, theta: 0 });
  close(out.vx, 0.5, 1e-3);
  close(out.wz, 0, 1e-9);
});

test("odom velocity: heading projection and yaw wrap", () => {
  const est = createOdomVelocity({ tauMs: 1 });
  // Facing +y, moving +y -> forward; theta crossing +pi -> -pi turning left.
  est.push({ tMs: 0, x: 0, y: 0, theta: Math.PI - 0.01 });
  const out = est.push({ tMs: 100, x: 0, y: -0.05, theta: -Math.PI + 0.01 });
  close(out.wz, 0.2, 1e-3);
  // Heading ~pi: moving -y is sideways, so forward ~0 (abs small).
  assert.ok(Math.abs(out.vx) < 0.01);
});

test("odom velocity: gaps, duplicates and resets restart at zero", () => {
  const est = createOdomVelocity({ tauMs: 1 });
  est.push({ tMs: 0, x: 0, y: 0, theta: 0 });
  let out = est.push({ tMs: 100, x: 0.1, y: 0, theta: 0 });
  close(out.vx, 1, 1e-3);
  // Duplicate frame (dt < min) keeps the estimate.
  out = est.push({ tMs: 105, x: 0.1, y: 0, theta: 0 });
  close(out.vx, 1, 1e-3);
  // Odometry reset (jump back to 0 by more than maxJump? no: 0.1 m) after a long gap.
  out = est.push({ tMs: 5000, x: 0, y: 0, theta: 0 });
  assert.deepEqual(out, { vx: 0, wz: 0 });
  // A big pose jump (odometry reset mid-drive) also zeroes.
  out = est.push({ tMs: 5100, x: 3, y: 0, theta: 0 });
  assert.deepEqual(out, { vx: 0, wz: 0 });
});

test("odom velocity: a heading-only reset near the origin restarts at zero", () => {
  const est = createOdomVelocity({ tauMs: 1 });
  // Turning in place at 1 rad/s near the origin, theta ~1.5 rad...
  est.push({ tMs: 0, x: 0.1, y: 0.05, theta: 1.5 });
  let out = est.push({ tMs: 33, x: 0.1, y: 0.05, theta: 1.533 });
  close(out.wz, 1, 1e-3);
  // ...then "Reset odometry": pose snaps to (0, 0, 0) within 0.5 m.
  out = est.push({ tMs: 66, x: 0, y: 0, theta: 0 });
  assert.deepEqual(out, { vx: 0, wz: 0 });
  // The next frame differentiates from the reset pose, not the old one.
  out = est.push({ tMs: 99, x: 0, y: 0, theta: 0.033 });
  close(out.wz, 1, 1e-3);
});

test("driverStatus: none, you, human, autonomous (field and name fallback)", () => {
  assert.equal(driverStatus(null).kind, "none");
  const base = { hasActiveOperator: true, youAreActiveOperator: false, activeOperatorName: "" };
  assert.equal(driverStatus({ ...base, youAreActiveOperator: true }).kind, "you");
  const human = driverStatus({ ...base, activeOperatorName: "ipad" });
  assert.equal(human.kind, "human");
  assert.match(human.hint, /"ipad" is driving/);
  const flagged = driverStatus({ ...base, activeOperatorName: "", activeOperatorAutonomous: true });
  assert.equal(flagged.kind, "autonomous");
  assert.equal(flagged.hint, "Voice is driving — move the joystick to take over");
  assert.equal(driverStatus({ ...base, activeOperatorName: "bebop-voice" }).kind, "autonomous");
  // The firmware field wins over the name heuristic.
  assert.equal(
    driverStatus({ ...base, activeOperatorName: "voice", activeOperatorAutonomous: false }).kind,
    "human",
  );
});

test("signed formatting avoids -0.00", () => {
  assert.equal(signed(0.25), "+0.25");
  assert.equal(signed(-1.1), "-1.10");
  assert.equal(signed(-0.0001), "+0.00");
});

function report(objs) {
  return { forEach: (cb) => objs.forEach(cb) };
}

test("video stats: sample + summary from two reports", () => {
  const mk = (t, frames, recv, lost, bytes, jb, jbn, dec) =>
    report([
      { type: "inbound-rtp", kind: "video", timestamp: t, framesDecoded: frames, packetsReceived: recv, packetsLost: lost, bytesReceived: bytes, jitter: 0.004, jitterBufferDelay: jb, jitterBufferEmittedCount: jbn, totalDecodeTime: dec, frameWidth: 848, frameHeight: 480, framesDropped: 1 },
      { type: "transport", selectedCandidatePairId: "p2" },
      { type: "candidate-pair", id: "p1", nominated: true, state: "succeeded", currentRoundTripTime: 0.5 },
      { type: "candidate-pair", id: "p2", currentRoundTripTime: 0.01 },
    ]);
  const a = sampleFromReport(mk(1000, 0, 0, 0, 0, 0, 0, 0), 0);
  const b = sampleFromReport(mk(2000, 30, 990, 10, 125000, 1.5, 30, 0.15), 0);
  assert.equal(b.rtt, 0.01); // the transport's selected pair wins
  const s = summarizeVideoStats(a, b);
  close(s.fps, 30);
  close(s.kbps, 1000);
  close(s.loss, 0.01);
  close(s.jitterMs, 4);
  close(s.jitterBufferMs, 50);
  close(s.decodeMs, 5);
  close(s.latencyMs, 5 + 50 + 5);
  assert.equal(s.width, 848);
  const first = summarizeVideoStats(null, a);
  assert.equal(first.fps, null);
  assert.equal(first.latencyMs, null);
});

test("video stats: no inbound video -> null sample", () => {
  assert.equal(sampleFromReport(report([{ type: "inbound-rtp", kind: "audio" }]), 0), null);
});
