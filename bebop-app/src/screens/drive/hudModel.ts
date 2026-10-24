// Pure helpers behind the Drive HUD (run under `node --test`).
//
//   * `createOdomVelocity`: measured body twist from the odometry pose
//     stream (telemetry carries wheel velocities in rad/s but no wheel
//     radius / track width, so the pose derivative is the measured twist).
//     Wheel-odometry yaw over-reads in-place turns on carpet (AGENTS.md),
//     so the HUD labels it "odom".
//   * `driverStatus`: who holds the drive seat, phrased for the HUD.

export interface OdomSample {
  /// Robot clock (ms) of the telemetry frame.
  tMs: number;
  x: number;
  y: number;
  theta: number;
}

export interface MeasuredTwist {
  vx: number;
  wz: number;
}

function wrapPi(a: number): number {
  let r = a;
  while (r > Math.PI) r -= 2 * Math.PI;
  while (r < -Math.PI) r += 2 * Math.PI;
  return r;
}

/// EMA-smoothed pose derivative. Frames closer than `minDtMs` apart are
/// skipped (duplicated snapshot); a gap over `maxDtMs`, a backwards clock,
/// or an odometry reset (a position jump over `maxJumpM` or a heading jump
/// over `maxTurnRad` in one frame) restarts the estimate at zero.
export function createOdomVelocity({
  tauMs = 250,
  minDtMs = 15,
  maxDtMs = 1000,
  maxJumpM = 0.5,
  maxTurnRad = 0.5,
}: {
  tauMs?: number;
  minDtMs?: number;
  maxDtMs?: number;
  maxJumpM?: number;
  maxTurnRad?: number;
} = {}) {
  let last: OdomSample | null = null;
  let est: MeasuredTwist = { vx: 0, wz: 0 };
  return {
    push(s: OdomSample): MeasuredTwist {
      if (!last) {
        last = s;
        return est;
      }
      const dtMs = s.tMs - last.tMs;
      if (dtMs < minDtMs && dtMs >= 0) return est;
      const dx = s.x - last.x;
      const dy = s.y - last.y;
      const dTheta = wrapPi(s.theta - last.theta);
      if (
        dtMs < 0 ||
        dtMs > maxDtMs ||
        Math.hypot(dx, dy) > maxJumpM ||
        Math.abs(dTheta) > maxTurnRad
      ) {
        last = s;
        est = { vx: 0, wz: 0 };
        return est;
      }
      const dt = dtMs / 1000;
      const mid = last.theta + dTheta / 2;
      // Forward speed: displacement projected on the mean heading.
      const vx = (dx * Math.cos(mid) + dy * Math.sin(mid)) / dt;
      const wz = dTheta / dt;
      const a = 1 - Math.exp(-dtMs / tauMs);
      est = { vx: est.vx + a * (vx - est.vx), wz: est.wz + a * (wz - est.wz) };
      last = s;
      return est;
    },
    reset() {
      last = null;
      est = { vx: 0, wz: 0 };
    },
  };
}

export interface SeatInfo {
  hasActiveOperator: boolean;
  youAreActiveOperator: boolean;
  activeOperatorName: string;
  /// DriveState.active_operator_autonomous (PREEMPT workstream); undefined
  /// on firmware without the field.
  activeOperatorAutonomous?: boolean;
}

export type DriverKind = "none" | "you" | "human" | "autonomous";

export interface DriverStatus {
  kind: DriverKind;
  label: string;
  /// Longer hint for a banner (null when nothing to say).
  hint: string | null;
}

/// Seat holders that are the robot's own agent. Used only when the firmware
/// does not report `active_operator_autonomous`.
const AUTONOMOUS_NAME = /\b(voice|agent|autonomy|autonomous|autopilot|nav)\b/i;

export function driverStatus(seat: SeatInfo | null | undefined): DriverStatus {
  if (!seat || !seat.hasActiveOperator) {
    return { kind: "none", label: "Nobody driving", hint: null };
  }
  if (seat.youAreActiveOperator) return { kind: "you", label: "You are driving", hint: null };
  const name = seat.activeOperatorName.trim();
  const autonomous =
    seat.activeOperatorAutonomous ?? (name !== "" && AUTONOMOUS_NAME.test(name));
  if (autonomous) {
    return {
      kind: "autonomous",
      label: `${name || "Voice"} is driving`,
      hint: `${name || "Voice"} is driving — move the joystick to take over`,
    };
  }
  const who = name ? `"${name}"` : "Another operator";
  return {
    kind: "human",
    label: `${name || "Someone"} is driving`,
    hint: `${who} is driving. You can watch; the seat frees up once their input goes quiet.`,
  };
}

/// Signed fixed-width number for the HUD ("+0.25", "-1.10").
export function signed(v: number, digits = 2): string {
  const r = Number(v.toFixed(digits));
  const z = r === 0 ? 0 : r;
  return z >= 0 ? `+${z.toFixed(digits)}` : z.toFixed(digits);
}
