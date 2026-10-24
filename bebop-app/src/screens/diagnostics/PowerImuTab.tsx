// Diagnostics › Power & IMU: power-board battery/rails and the BNO08x
// orientation cube.

import { useMemo } from "react";

import { PageState } from "../../components/ui";
import { useTelemetry } from "../../robot/session";
import type { ImuView, PowerView } from "../../runtime";

export function PowerImuTab() {
  const { power, imu, loaded } = useTelemetry(
    (s) => ({ power: s?.power ?? null, imu: s?.imu ?? null, loaded: !!s }),
    10,
  );
  if (!loaded) return <PageState kind="loading" />;
  if (!power?.present && !imu?.present) {
    return (
      <PageState kind="empty" title="No power board or IMU configured">
        The firmware config has no <code>power:</code> or <code>imu:</code> block.
      </PageState>
    );
  }
  return (
    <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
      {power?.present ? <PowerCard power={power} /> : null}
      {imu?.present ? <OrientationCard imu={imu} /> : null}
    </div>
  );
}


// ---------------------------------------------------------------------------
// Power-board card
// ---------------------------------------------------------------------------

/// Compact battery / rail-status card. Renders:
///   - SOC bar driven by `stateOfChargePct` (linear from pack voltage)
///   - Battery (VBUS) and motor (VMBUS) voltage readouts
///   - Total motor-branch current (sum of AL/AR/LL/LR)
///   - Per-rail on/off pills (24V / 12V / motor)
///   - Stale / fault banners when the board is misbehaving
///
/// All percent / voltage formatting handles the "no reading yet"
/// case explicitly: when `statusReceived === false` we render placeholder
/// dashes instead of a misleading "0.00 V". The same applies to
/// `stateOfChargePct < 0`, the explicit "unknown" sentinel.
function PowerCard({ power }: { power: PowerView }) {
  const hasData = power.statusReceived;
  const socKnown = hasData && power.stateOfChargePct >= 0;
  const socPct = socKnown ? power.stateOfChargePct : 0;
  const socColor =
    !socKnown
      ? "bg-text-dim/40"
      : socPct >= 60
        ? "bg-success"
        : socPct >= 25
          ? "bg-warning"
          : "bg-danger";

  const voltage = hasData ? power.batteryVoltageV : 0;
  // Per-cell voltage is the most useful single number for a Li-ion
  // operator: 4.2 V = full, 3.7 V = nominal, 3.0 V = empty.
  const cellV = hasData && power.batteryCells > 0
    ? voltage / power.batteryCells
    : 0;

  const faultActive = hasData && power.faultDescription !== "" && power.faultDescription !== "normal";

  return (
    <div className="rounded-card border border-border bg-bg-elev px-3.5 py-3 flex flex-col gap-2.5">
      {/* Header row: title + meta */}
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <div className="flex items-center gap-2">
          <span className="text-caption uppercase tracking-wider text-text-dim font-semibold">
            Power
          </span>
          {power.firmwareVersion ? (
            <span className="text-micro text-text-dim font-mono">
              {power.firmwareVersion}
            </span>
          ) : null}
        </div>
        <div className="flex items-center gap-1.5 text-caption text-text-dim">
          <span>{power.canInterface}</span>
          <span aria-hidden>·</span>
          <span>id 0x{power.powerId.toString(16).toUpperCase().padStart(2, "0")}</span>
          {hasData ? (
            <>
              <span aria-hidden>·</span>
              <span title={`Last status frame ${power.lastStatusAgeMs} ms ago`}>
                {power.lastStatusAgeMs} ms
              </span>
            </>
          ) : null}
        </div>
      </div>

      {/* Status banners */}
      {!hasData ? (
        <div className="text-xs text-text-dim italic">
          Waiting for first status frame from power board…
        </div>
      ) : null}
      {hasData && power.statusStale ? (
        <div className="text-xs text-warning-fg bg-warning/10 border border-warning/30 rounded px-2 py-1">
          Power-board status is stale ({(power.lastStatusAgeMs / 1000).toFixed(1)} s
          since last frame). Verify the {power.canInterface} bus and board power.
        </div>
      ) : null}
      {faultActive ? (
        <div className="text-xs text-danger-fg bg-danger/10 border border-danger/30 rounded px-2 py-1">
          <span className="font-semibold">Power-board fault:</span>{" "}
          <span className="font-mono">{power.faultDescription}</span>
        </div>
      ) : null}

      {/* SOC bar with cell-voltage and pack-voltage readout above */}
      <div className="flex flex-col gap-1.5">
        <div className="flex items-baseline justify-between gap-2 text-label">
          <div className="flex items-baseline gap-2">
            <span className="text-text-dim">Battery</span>
            {socKnown ? (
              <span className="font-semibold tabular-nums text-text">
                {socPct.toFixed(0)}%
              </span>
            ) : (
              <span className="text-text-dim">—</span>
            )}
            {hasData && power.batteryCells > 0 ? (
              <span className="text-caption text-text-dim tabular-nums">
                {power.batteryCells}s · {cellV.toFixed(2)} V/cell
              </span>
            ) : null}
          </div>
          <div className="text-text-dim tabular-nums">
            {hasData ? (
              <>
                <span className="text-text font-semibold">
                  {voltage.toFixed(2)} V
                </span>{" "}
                <span className="text-caption">
                  ({power.packEmptyVoltageV.toFixed(1)}…
                  {power.packFullVoltageV.toFixed(1)})
                </span>
              </>
            ) : (
              <span>—</span>
            )}
          </div>
        </div>
        <div className="h-2 w-full rounded-full bg-bg-elev-2 overflow-hidden relative">
          <div
            className={`h-full transition-[width] ${socColor}`}
            style={{ width: `${Math.max(0, Math.min(100, socPct))}%` }}
            aria-hidden
          />
        </div>
      </div>

      {/* Secondary stats: motor V, total motor current, board temp + rail pills */}
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-x-4 gap-y-1.5 text-label">
        <PowerStat
          label="Motor rail"
          value={hasData ? `${power.motorVoltageV.toFixed(2)} V` : "—"}
          hint="VMBUS — voltage downstream of the soft-start FET"
        />
        <PowerStat
          label="Branch I"
          value={
            hasData ? `${power.totalMotorCurrentA.toFixed(1)} A` : "—"
          }
          hint={
            hasData
              ? `AL ${power.currentAlA.toFixed(1)} · AR ${power.currentArA.toFixed(1)} · ` +
                `LL ${power.currentLlA.toFixed(1)} · LR ${power.currentLrA.toFixed(1)} A`
              : "Sum of AL+AR+LL+LR (per-branch)"
          }
        />
        <PowerStat
          label="Board T"
          value={hasData ? `${power.boardTemperatureC.toFixed(0)} °C` : "—"}
          hint="Power-board internal sensor; board over-temp at 100 °C"
        />
      </div>

      {/* Rail-on pills */}
      {hasData ? (
        <div className="flex items-center gap-1.5 flex-wrap">
          <RailPill label="24V" on={power.rail24vOn} />
          <RailPill label="12V" on={power.rail12vOn} />
          <RailPill label="MOTOR" on={power.motorRailOn} />
          <RailPill label="SOFTSTART" on={power.softStartOn} subtle />
        </div>
      ) : null}
    </div>
  );
}

function PowerStat({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div className="flex flex-col leading-tight" title={hint}>
      <span className="text-micro uppercase tracking-wider text-text-dim">
        {label}
      </span>
      <span className="text-text tabular-nums font-semibold">{value}</span>
    </div>
  );
}

function RailPill({
  label,
  on,
  subtle = false,
}: {
  label: string;
  on: boolean;
  subtle?: boolean;
}) {
  const onClasses = subtle
    ? "bg-text-dim/15 text-text-dim border-border"
    : "bg-success/15 text-success border-success/40";
  const offClasses = "bg-bg-elev-2 text-text-dim/70 border-border";
  return (
    <span
      className={`text-micro uppercase tracking-wider px-1.5 py-0.5 rounded font-semibold border ${
        on ? onClasses : offClasses
      }`}
      title={`${label} rail ${on ? "on" : "off"}`}
    >
      <span className="opacity-70 mr-0.5" aria-hidden>
        {on ? "●" : "○"}
      </span>
      {label}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Orientation card (IMU rotation-vector visualization)
// ---------------------------------------------------------------------------

/// Live orientation indicator driven by the BNO080/BNO085 rotation
/// vector. Renders a labelled box rotating in 3D so the operator can
/// quickly tell whether the robot is upright, leaning, or has flipped,
/// plus numeric roll/pitch/yaw readouts.
///
/// # Frame contract with the firmware
///
/// `imu.quaternion` arrives as **body-frame XYZW (Hamilton)**: the
/// firmware (`bebop-linux/src/imu.rs`) has already composed the
/// `imu.mount:` rotation from `bebop_v2.yaml` and renormalized to unit
/// length before publishing. So we treat the four floats as a unit
/// body-frame attitude and feed them to both the CSS-matrix render and
/// the Euler decomposition without any further rotation.
///
/// The bno080 crate seeds its cached quaternion at all-zeros before
/// the first SHTP report, and that pre-init state still propagates as
/// `received = false`. We gate the whole card on `received && !stale`
/// to avoid rendering an "identity" cube while no real data is flowing
/// — and as a belt-and-braces measure we also fall back to identity
/// if a non-finite or near-zero quaternion ever slips through.
///
/// The Euler readout is for human consumption only and is allowed to
/// flicker near pitch = ±90° (gimbal lock); the cube is singularity-
/// free since the CSS matrix path doesn't go through Euler.
function OrientationCard({ imu }: { imu: ImuView }) {
  const [rawX, rawY, rawZ, rawW] = imu.quaternion;
  const norm = Math.sqrt(
    rawX * rawX + rawY * rawY + rawZ * rawZ + rawW * rawW,
  );
  // Per the firmware contract `norm ≈ 1` whenever `received = true`. We
  // still divide here so a stale / pre-init / non-finite reading
  // renders as identity instead of feeding a degenerate quaternion to
  // either the CSS matrix or the Euler decomposition.
  const haveQuat = imu.received && norm > 1e-6 && Number.isFinite(norm);
  const [qx, qy, qz, qw] = haveQuat
    ? [rawX / norm, rawY / norm, rawZ / norm, rawW / norm]
    : [0, 0, 0, 1];

  const matrix3d = useMemo(
    () => quaternionToCssMatrix3d(qx, qy, qz, qw),
    [qx, qy, qz, qw],
  );
  const [rollDeg, pitchDeg, yawDeg] = useMemo(
    () => quaternionToEulerDeg(qx, qy, qz, qw),
    [qx, qy, qz, qw],
  );

  const usable = haveQuat && !imu.stale;
  const headingDeg = (imu.headingAccuracyRad * 180) / Math.PI;

  return (
    <div className="rounded-card border border-border bg-bg-elev px-3.5 py-3 flex flex-col gap-2.5">
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <div className="flex items-center gap-2">
          <span className="text-caption uppercase tracking-wider text-text-dim font-semibold">
            Orientation
          </span>
          <span className="text-micro text-text-dim font-mono">BNO08x</span>
        </div>
        <div className="flex items-center gap-1.5 text-caption text-text-dim">
          {imu.received ? (
            <span title={`Last quaternion ${imu.lastUpdateAgeMs} ms ago`}>
              {imu.lastUpdateAgeMs} ms
            </span>
          ) : (
            <span className="italic">no fix yet</span>
          )}
          {imu.stale ? (
            <>
              <span aria-hidden>·</span>
              <span className="text-warning-fg font-semibold">
                stale
              </span>
            </>
          ) : null}
        </div>
      </div>

      {/* Cube + readouts: on phones they stack, on md+ they sit side-by-side
          so the 3D widget and the numeric readout share the card row. */}
      <div className="flex flex-col sm:flex-row sm:items-center gap-3">
        <div
          className="self-center shrink-0 [perspective:520px] [perspective-origin:50%_45%]"
          aria-hidden
        >
          <div
            className={`relative w-[110px] h-[110px] [transform-style:preserve-3d] transition-transform duration-100 ease-out ${
              usable ? "opacity-100" : "opacity-40"
            }`}
            style={{ transform: matrix3d }}
          >
            <OrientationFace
              label="FWD"
              color="bg-accent/30 border-accent/70 text-accent"
              transform="translateZ(40px)"
            />
            <OrientationFace
              label="BACK"
              color="bg-bg-elev-2 border-border text-text-dim"
              transform="rotateY(180deg) translateZ(40px)"
            />
            <OrientationFace
              label="L"
              color="bg-bg-elev-2 border-border text-text-dim"
              transform="rotateY(90deg) translateZ(40px)"
            />
            <OrientationFace
              label="R"
              color="bg-bg-elev-2 border-border text-text-dim"
              transform="rotateY(-90deg) translateZ(40px)"
            />
            <OrientationFace
              label="TOP"
              color="bg-success/25 border-success/60 text-success"
              transform="rotateX(90deg) translateZ(40px)"
            />
            <OrientationFace
              label="BOT"
              color="bg-bg-elev-2 border-border text-text-dim"
              transform="rotateX(-90deg) translateZ(40px)"
            />
          </div>
        </div>

        <div className="flex-1 min-w-0 grid grid-cols-3 gap-x-3 gap-y-1.5 text-label">
          <OrientationReadout
            label="Roll"
            value={usable ? `${rollDeg.toFixed(1)}°` : "—"}
            hint="Rotation about body X axis (lean left/right)"
          />
          <OrientationReadout
            label="Pitch"
            value={usable ? `${pitchDeg.toFixed(1)}°` : "—"}
            hint="Rotation about body Y axis (lean forward/back)"
          />
          <OrientationReadout
            label="Yaw"
            value={usable ? `${yawDeg.toFixed(1)}°` : "—"}
            hint="Rotation about body Z axis (heading)"
          />
          <div className="col-span-3 grid grid-cols-4 gap-x-2 gap-y-0 text-caption tabular-nums text-text-dim">
            <span title="quaternion x">
              qx <span className="text-text">{qx.toFixed(2)}</span>
            </span>
            <span title="quaternion y">
              qy <span className="text-text">{qy.toFixed(2)}</span>
            </span>
            <span title="quaternion z">
              qz <span className="text-text">{qz.toFixed(2)}</span>
            </span>
            <span title="quaternion w">
              qw <span className="text-text">{qw.toFixed(2)}</span>
            </span>
          </div>
          {imu.headingAccuracyRad > 0 ? (
            <div className="col-span-3 text-caption text-text-dim">
              Heading accuracy ±{headingDeg.toFixed(1)}° (1σ)
            </div>
          ) : null}
        </div>
      </div>

      {!imu.received ? (
        <div className="text-xs text-text-dim italic">
          Waiting for first rotation-vector frame from BNO08x…
        </div>
      ) : null}
    </div>
  );
}

function OrientationFace({
  label,
  color,
  transform,
}: {
  label: string;
  color: string;
  transform: string;
}) {
  return (
    <div
      className={`absolute inset-0 m-auto w-[80px] h-[80px] border rounded-md flex items-center justify-center text-micro font-semibold tracking-wider [backface-visibility:hidden] ${color}`}
      style={{
        transform,
        top: "50%",
        left: "50%",
        marginLeft: "-40px",
        marginTop: "-40px",
      }}
    >
      {label}
    </div>
  );
}

function OrientationReadout({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div className="flex flex-col leading-tight" title={hint}>
      <span className="text-micro uppercase tracking-wider text-text-dim">
        {label}
      </span>
      <span className="text-text tabular-nums font-semibold">{value}</span>
    </div>
  );
}

/// Build a CSS `matrix3d(...)` transform from a unit XYZW quaternion
/// expressed in **body frame** (REP-103 / FLU: +x forward, +y left,
/// +z up), per the wire-protocol contract documented on `OrientationCard`.
///
/// The cube's local faces are laid out in CSS coordinates:
///
/// - `+Z` = FWD face, toward the viewer.
/// - `+X` = robot-left face, to the operator's right in this rear-view
///   convention.
/// - `-Y` = TOP face, upward on screen.
///
/// Body axes are FLU (`+X forward`, `+Y left`, `+Z up`), so the render path
/// must conjugate the body rotation matrix into CSS basis, not merely flip
/// the screen Y axis:
///
/// ```text
/// css_x =  body_y
/// css_y = -body_z
/// css_z =  body_x
/// ```
///
/// This makes yaw = -90° put the FWD face on the left side of the widget,
/// and makes roll rotate around the visible FWD/BACK axis instead of around
/// the screen's horizontal axis.
function quaternionToCssMatrix3d(
  x: number,
  y: number,
  z: number,
  w: number,
): string {
  const xx = x * x,
    yy = y * y,
    zz = z * z;
  const xy = x * y,
    xz = x * z,
    yz = y * z;
  const wx = w * x,
    wy = w * y,
    wz = w * z;

  // Row-major rotation matrix (m[row][col]).
  const m00 = 1 - 2 * (yy + zz);
  const m01 = 2 * (xy - wz);
  const m02 = 2 * (xz + wy);
  const m10 = 2 * (xy + wz);
  const m11 = 1 - 2 * (xx + zz);
  const m12 = 2 * (yz - wx);
  const m20 = 2 * (xz - wy);
  const m21 = 2 * (yz + wx);
  const m22 = 1 - 2 * (xx + yy);

  // Change basis from body FLU to CSS screen coordinates. If `R` is the
  // body-frame rotation matrix above, this is `C * R * C^-1` with:
  //
  //   C = [ 0  1  0 ]   (css_x =  body_y)
  //       [ 0  0 -1 ]   (css_y = -body_z)
  //       [ 1  0  0 ]   (css_z =  body_x)
  //
  // CSS matrix3d is column-major, so the row-major matrix is packed by
  // columns below.
  const c00 = m11;
  const c01 = -m12;
  const c02 = m10;
  const c10 = -m21;
  const c11 = m22;
  const c12 = -m20;
  const c20 = m01;
  const c21 = -m02;
  const c22 = m00;

  return (
    `matrix3d(` +
    `${c00},${c10},${c20},0,` +
    `${c01},${c11},${c21},0,` +
    `${c02},${c12},${c22},0,` +
    `0,0,0,1)`
  );
}

/// Decompose an XYZW quaternion to ZYX-Tait-Bryan Euler angles (roll,
/// pitch, yaw), in degrees. Singularity-aware: clamps `asin` argument to
/// [-1, 1] so we don't return NaN near pitch = ±90°.
function quaternionToEulerDeg(
  x: number,
  y: number,
  z: number,
  w: number,
): [number, number, number] {
  const sinrCosp = 2 * (w * x + y * z);
  const cosrCosp = 1 - 2 * (x * x + y * y);
  const roll = Math.atan2(sinrCosp, cosrCosp);

  const sinp = 2 * (w * y - z * x);
  const pitch =
    Math.abs(sinp) >= 1 ? (Math.PI / 2) * Math.sign(sinp) : Math.asin(sinp);

  const sinyCosp = 2 * (w * z + x * y);
  const cosyCosp = 1 - 2 * (y * y + z * z);
  const yaw = Math.atan2(sinyCosp, cosyCosp);

  const r2d = 180 / Math.PI;
  return [roll * r2d, pitch * r2d, yaw * r2d];
}
