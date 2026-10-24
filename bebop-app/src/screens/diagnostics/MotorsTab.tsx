// Diagnostics › Motors: firmware mode, wheel arming and calibration, the
// bench joystick, odometry, and CAN bus health. Wheeled chassis only; the
// legged bench (joint dial-in, policy runner, zero calibration) was retired
// from the app. The firmware keeps legged support.

import { useCallback } from "react";

import { ControlProfilePicker } from "../../components/ControlProfilePicker";
import { DriveJoystick } from "../../components/DriveJoystick";
import { GamepadDrive } from "../../components/GamepadDrive";
import {
  Banner,
  Button,
  Card,
  CardHeader,
  PageState,
  SegmentedControl,
  StatusPill,
  Switch,
} from "../../components/ui";
import type { DriveChannel } from "../../input";
import { useDrive } from "../../robot/useDrive";
import { useEStop, useRuntimeAction, useTelemetry } from "../../robot/session";
import type { DriveView, RuntimeMode, RuntimeSnapshot, WheelView } from "../../runtime";

const select = (s: RuntimeSnapshot | null) => s;

type BenchMode = "IDLE" | "DIAL_IN";

const fmt = (v: number): string => (v >= 0 ? `+${v.toFixed(2)}` : v.toFixed(2));

export function MotorsTab() {
  const snapshot = useTelemetry(select, 30);
  const estop = useEStop();
  const { busy, error, setError, run } = useRuntimeAction();

  const motors = snapshot?.motors ?? [];
  const buses = snapshot?.buses ?? [];
  const wheels = snapshot?.wheels ?? [];
  const drive = snapshot?.drive;
  const mode = snapshot?.mode ?? "UNSPECIFIED";
  const estopLatched = estop.latched;
  const wheeled = !!drive?.present && motors.length === 0;
  const driveMode = mode === "DIAL_IN" || mode === "RUN_POLICY";
  const armed = wheels.filter((w) => w.armed).length;
  const otherDriving = !!drive?.hasActiveOperator && !drive?.youAreActiveOperator;
  const canDrive = wheeled && !estopLatched && driveMode && armed > 0 && !otherDriving;
  const input = useDrive({ canDrive, onError: setError });

  const setAllWheels = useCallback(
    (enabled: boolean) =>
      void run(`all-wheels:${enabled}`, (t) => t.setAllWheelsEnabled(enabled)),
    [run],
  );

  if (!snapshot) return <PageState kind="loading">Waiting for telemetry…</PageState>;

  const unhealthy = buses.filter((b) => !b.healthy);

  return (
    <div className="flex flex-col gap-3">
      <Card className="flex flex-col gap-3 lg:flex-row lg:items-center">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
          <span className="text-sm text-text-dim">Mode</span>
          <SegmentedControl<BenchMode>
            label="Firmware mode"
            value={mode === "IDLE" ? "IDLE" : "DIAL_IN"}
            options={[
              { value: "IDLE", label: "Idle" },
              { value: "DIAL_IN", label: "Drive" },
            ]}
            disabled={!!busy || estopLatched}
            onChange={(m) => void run(`mode:${m}`, (t) => t.setMode(m as RuntimeMode))}
          />
          {mode === "RUN_POLICY" ? <StatusPill tone="info">policy</StatusPill> : null}
        </div>
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 lg:ml-auto">
          {buses.map((b) => (
            <span key={b.canInterface} className="flex items-center gap-1 text-label">
              <StatusPill tone={b.healthy ? "ok" : b.state === "" ? "dim" : "err"} dot>
                {b.canInterface}
              </StatusPill>
              <span className="text-text-dim">{b.state || "?"}</span>
            </span>
          ))}
        </div>
      </Card>

      {unhealthy.length > 0 ? (
        <Banner tone="error" title="CAN bus not healthy">
          {unhealthy.map((b) => `${b.canInterface}: ${b.state || "not present"}`).join(", ")}
          {" — check the USB-CAN hub. The runtime re-opens the bus by itself once it is back; if it does not recover, restart the runtime from Diagnostics › Services."}
        </Banner>
      ) : null}
      {wheeled && drive?.operatorStale ? (
        <Banner tone="warning" title="Operator link stale — motion halted">
          No fresh drive commands reached the robot within its operator
          timeout, so it zeroed the twist. Release the controls, then drive again.
        </Banner>
      ) : null}
      {wheeled && otherDriving ? (
        <Banner
          tone="info"
          title={drive?.activeOperatorName ? `"${drive.activeOperatorName}" is driving` : "Another operator is driving"}
        >
          You can watch from here. Driving unlocks once their input goes quiet
          for a couple of seconds. E-STOP always works.
        </Banner>
      ) : null}
      {error ? <Banner tone="error">{error}</Banner> : null}

      {wheeled && drive ? (
        <>
          <DriveCard
            drive={drive}
            wheels={wheels}
            mode={mode}
            estopLatched={estopLatched}
            busy={busy}
            canDrive={canDrive}
            touch={input.touch}
            onToggleWheel={(w, en) => void run(`wheel:${w}`, (t) => t.setWheelEnabled(w, en))}
            onSetAllWheels={setAllWheels}
            onCalibrateWheel={(w) => void run(`calib:${w}`, (t) => t.calibrateWheel(w))}
            onResetOdom={() => void run("odom-reset", (t) => t.resetOdometry())}
          />
          <GamepadDrive
            wheels={wheels}
            mode={mode}
            estopLatched={estopLatched}
            onEStop={() => void estop.trigger("gamepad")}
            onResetEStop={() => void estop.reset()}
            onSetAllWheels={setAllWheels}
          />
        </>
      ) : (
        <Card>
          <CardHeader
            eyebrow="Motors"
            title={motors.length > 0 ? `${motors.length} legged joints` : "No wheels reported"}
            description={
              motors.length > 0
                ? "This robot runs the legged configuration. The legged bench (joint dial-in, policy, zero calibration) is no longer in the app; use the firmware tools."
                : "The firmware reports no wheeled drive."
            }
          />
        </Card>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Differential drive
// ---------------------------------------------------------------------------
// Drive soft limits (the values the joystick / WASD / gamepad bridge
// scale to) come from the active control profile in
// `src/input/profile.ts`, switchable from the picker in the DriveCard
// header and on the gamepad bridge cards. The joystick itself lives in
// `src/components/DriveJoystick.tsx` (shared with the teleop screen).

/// Differential-drive hub: arm controls, live wheel telemetry, odometry,
/// and the drive joystick. Shown only when the firmware reports a `drive:`
/// config (a wheeled chassis with no legged joints).
function DriveCard({
  drive,
  wheels,
  mode,
  estopLatched,
  busy,
  onToggleWheel,
  onSetAllWheels,
  onCalibrateWheel,
  canDrive,
  touch,
  onResetOdom,
}: {
  drive: DriveView;
  wheels: WheelView[];
  mode: RuntimeMode;
  estopLatched: boolean;
  busy: string | null;
  onToggleWheel: (wheel: string, enabled: boolean) => void;
  onSetAllWheels: (enabled: boolean) => void;
  onCalibrateWheel: (wheel: string) => void;
  canDrive: boolean;
  touch: DriveChannel;
  onResetOdom: () => void;
}) {
  const armedWheelCount = wheels.filter((w) => w.armed).length;
  const allArmed = wheels.length > 0 && armedWheelCount === wheels.length;

  return (
    <div className="rounded-card border border-border bg-bg-elev px-3.5 py-3 space-y-4">
      {/* Header row: title + arming controls. */}
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="text-caption uppercase tracking-wider text-text-dim">
            Differential drive
          </div>
          <div className="text-body-sm text-text font-semibold mt-0.5">
            {wheels.length} wheel{wheels.length === 1 ? "" : "s"} ·{" "}
            {armedWheelCount} armed
          </div>
        </div>
        <div className="flex max-w-full flex-wrap items-center gap-2">
          {/* Sensitivity for the on-screen joystick + WASD drive (the
              gamepad bridge cards carry their own picker). */}
          <ControlProfilePicker />
          <Button
            variant="secondary"
            onClick={() => onSetAllWheels(true)}
            disabled={
              !!busy ||
              allArmed ||
              (mode !== "DIAL_IN" && mode !== "RUN_POLICY") ||
              estopLatched ||
              wheels.length === 0
            }
            size="sm"
          >
            Enable wheels
          </Button>
          <Button
            variant="secondary"
            onClick={() => onSetAllWheels(false)}
            disabled={!!busy || armedWheelCount === 0}
            size="sm"
          >
            Disable wheels
          </Button>
        </div>
      </div>

      {!canDrive ? (
        <div className="rounded-card border border-accent/30 bg-accent/5 px-3 py-2 text-label text-text-dim leading-relaxed">
          {estopLatched
            ? "E-STOP is latched — reset it before driving."
            : mode !== "DIAL_IN" && mode !== "RUN_POLICY"
              ? "Switch the mode to Drive, then enable the wheels."
              : armedWheelCount === 0
                ? "Enable the wheels to drive."
                : "Driving is locked: another operator holds the drive seat, or the link is down."}
        </div>
      ) : null}

      {/* Joystick + telemetry side by side on desktop, stacked on phones. */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 items-center">
        <DriveJoystick onTwist={touch.onTwist} onStop={touch.onStop} disabled={!canDrive} />

        <div className="space-y-3">
          {/* Wheels */}
          <div className="rounded-card border border-border bg-bg-elev-2/40 px-3 py-2.5 space-y-2">
            {wheels.length === 0 ? (
              <div className="text-label text-text-dim">No wheels reported.</div>
            ) : (
              wheels.map((w) => (
                <WheelRow
                  key={w.name}
                  wheel={w}
                  busy={busy === `wheel:${w.name}`}
                  calibrating={busy === `calib:${w.name}`}
                  estopLatched={estopLatched}
                  canArm={mode === "DIAL_IN" || mode === "RUN_POLICY"}
                  onToggle={(enabled) => onToggleWheel(w.name, enabled)}
                  onCalibrate={() => onCalibrateWheel(w.name)}
                />
              ))
            )}
          </div>

          {/* Odometry */}
          <div className="rounded-card border border-border bg-bg-elev-2/40 px-3 py-2.5">
            <div className="flex items-center justify-between mb-2">
              <div className="text-caption uppercase tracking-wider text-text-dim">
                Odometry (wheel encoders)
              </div>
              <Button
                variant="ghost"
                onClick={onResetOdom}
                disabled={busy === "odom-reset"}
                size="sm"
              >
                Reset pose
              </Button>
            </div>
            <div className="font-mono text-body-sm text-text leading-relaxed">
              <div>x&nbsp;&nbsp;{fmt(drive.odomX)}&nbsp;m</div>
              <div>y&nbsp;&nbsp;{fmt(drive.odomY)}&nbsp;m</div>
              <div>θ&nbsp;&nbsp;{fmt((drive.odomTheta * 180) / Math.PI)}°</div>
            </div>
            <div className="text-caption text-text-dim mt-1.5">
              cmd vx {fmt(drive.cmdLinearX)} m/s · ω {fmt(drive.cmdAngularZ)} rad/s
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function WheelRow({
  wheel,
  busy,
  calibrating,
  estopLatched,
  canArm,
  onToggle,
  onCalibrate,
}: {
  wheel: WheelView;
  busy: boolean;
  calibrating: boolean;
  estopLatched: boolean;
  canArm: boolean;
  onToggle: (enabled: boolean) => void;
  onCalibrate: () => void;
}) {
  const stale = !wheel.positionReceived || wheel.feedbackStale;
  const faulted = wheel.errorCode !== 0;
  // ODrive AxisState (raw, from heartbeat). Calibration states are 3..=12
  // minus 8 = CLOSED_LOOP_CONTROL (FULL_CALIBRATION_SEQUENCE(3),
  // MOTOR_CALIBRATION(4), SENSOR_CALIBRATION(5), ENCODER_INDEX_SEARCH(6),
  // ENCODER_OFFSET_CALIBRATION(7), LOCKIN_SPIN(9), ENCODER_DIR_FIND(10),
  // HOMING(11), ENCODER_HALL_POLARITY_CALIBRATION(12)). 1 = IDLE.
  const calibrationRunning =
    wheel.axisState >= 3 && wheel.axisState <= 12 && wheel.axisState !== 8;
  // The silent-enable-fail mode that hid a lost encoder calibration: the
  // supervisor ack'd the arm but the axis refused to go closed-loop, so the
  // wheel sat dead with no axis error. Surface it loudly.
  const armedButNotClosedLoop = wheel.armed && wheel.axisState !== 8;
  return (
    <div className="flex items-center gap-2.5">
      <Switch
        checked={wheel.armed}
        label={`Arm ${wheel.name}`}
        onChange={onToggle}
        disabled={busy || estopLatched || (!wheel.armed && !canArm)}
      />
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="text-body-sm text-text font-medium truncate">
            {wheel.name}
          </span>
          <span className="text-caption text-text-dim">id {wheel.nodeId}</span>
          {faulted ? (
            <span className="text-micro px-1.5 py-0.5 rounded-full border border-danger/40 bg-danger/10 text-danger">
              error
            </span>
          ) : calibrationRunning ? (
            <span className="text-micro px-1.5 py-0.5 rounded-full border border-accent/40 bg-accent/10 text-accent">
              calibrating
            </span>
          ) : armedButNotClosedLoop ? (
            <span className="text-micro px-1.5 py-0.5 rounded-full border border-danger/40 bg-danger/10 text-danger">
              axis not closed-loop
            </span>
          ) : stale ? (
            <span className="text-micro px-1.5 py-0.5 rounded-full border border-warning/40 bg-warning/10 text-warning-fg">
              stale
            </span>
          ) : null}
        </div>
        <div className="text-caption text-text-dim font-mono">
          vel {fmt(wheel.velocity)} rad/s · target {fmt(wheel.targetVelocity)}
        </div>
      </div>
      <div className="flex items-center gap-2 shrink-0">
        <div className="text-right text-label text-text-dim font-mono">
          {wheel.velocity >= 0 ? "+" : ""}
          {wheel.velocity.toFixed(2)}
          <span className="text-text-dim/70"> rad/s</span>
        </div>
        {/* CAN full calibration (motor + encoder offset). Only offered on a
            disarmed wheel — the axis spins for ~20-30 s. Not saved to S1 NVM:
            re-run after a power cycle (or persist via odrivetool over USB). */}
        {!wheel.armed ? (
          <Button
            variant="ghost"
            onClick={onCalibrate}
            disabled={calibrating || estopLatched || calibrationRunning}
            size="sm"
            title="Run ODrive full calibration (axis spins ~20-30 s)"
          >
            {calibrationRunning ? "Cal…" : "Calibrate"}
          </Button>
        ) : null}
      </div>
    </div>
  );
}
