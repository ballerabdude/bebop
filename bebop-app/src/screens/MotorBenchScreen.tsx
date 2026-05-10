import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { KeyboardEvent, PointerEvent, ReactNode } from "react";

import { GamepadDriver } from "../components/GamepadDriver";
import { Banner, Button, Spinner } from "../components/ui";
import { getOrCreateRuntimeTransport } from "../runtime";
import type {
  MotorView,
  PowerView,
  RuntimeMode,
  RuntimeSnapshot,
} from "../runtime";
import type { RuntimeTransport } from "../runtime";

interface MotorBenchProps {
  /** IP address of the robot, as reported by `WifiStatus.ip_address` over BLE. */
  robotIp: string;
  /** Optional override for the runtime port. Defaults to 9090. */
  runtimePort?: number;
  onBack: () => void;
  /** When provided, render a "Bluetooth controller" link so the operator
   *  can jump straight from dial-in to gamepad pairing without bouncing
   *  through the dashboard. Only set on the BLE-supported path: the
   *  controller subsystem lives behind `BebopTransport` (which the
   *  IP-only path doesn't have). */
  onOpenControllers?: () => void;
}

const MODE_LABEL: Record<RuntimeMode, string> = {
  UNSPECIFIED: "Unknown",
  IDLE: "Idle",
  DIAL_IN: "Dial-in",
  RUN_POLICY: "Policy",
};

/** Live motor bench: enable/disable per joint, see live state, E-STOP. */
export function MotorBenchScreen({
  robotIp,
  runtimePort = 9090,
  onBack,
  onOpenControllers,
}: MotorBenchProps) {
  const transportRef = useRef<RuntimeTransport | null>(null);
  const [connecting, setConnecting] = useState(true);
  const [snapshot, setSnapshot] = useState<RuntimeSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  // -------------------------------------------------------------- lifecycle
  //
  // The transport is owned by the module-level cache (one per
  // ip:port endpoint), not by this component. That's important for two
  // reasons:
  //
  //   1. React StrictMode runs effects twice in dev. With a per-mount
  //      `new RuntimeTransport()` we'd open two parallel WebSockets and
  //      leak one (the first mount's cleanup races the second mount's
  //      connect). The cache ensures both mounts share one socket.
  //
  //   2. Re-entering motors from the dashboard ("Back" then "Open motor
  //      bench" again) shouldn't tear down and re-establish the WS; the
  //      cached transport stays warm.
  //
  // On unmount we *unsubscribe* listeners but do NOT disconnect — the
  // socket is reusable for the next mount or the next consumer.
  useEffect(() => {
    let cancelled = false;
    const transport = getOrCreateRuntimeTransport(robotIp, runtimePort);
    transportRef.current = transport;
    const offCallbacks: Array<() => void> = [];

    void (async () => {
      setConnecting(true);
      setError(null);
      try {
        // connect() is idempotent on the cached transport; if it's
        // already OPEN this is a no-op.
        await transport.connect(robotIp, runtimePort);
        if (cancelled) return;
        const initial = await transport.getSnapshot();
        setSnapshot(initial);
        await transport.subscribeTelemetry(30);
      } catch (e) {
        if (!cancelled) {
          setError(
            e instanceof Error
              ? e.message
              : `failed to connect: ${String(e)}`,
          );
        }
        return;
      } finally {
        if (!cancelled) setConnecting(false);
      }

      offCallbacks.push(
        transport.onTelemetry((s) => {
          if (!cancelled) setSnapshot(s);
        }),
      );
      offCallbacks.push(
        transport.onEStopLatched(() => {
          if (!cancelled) {
            // Force a fresh snapshot so E-STOP banner shows up immediately
            // even if telemetry is paused.
            void transport.getSnapshot().then((s) => {
              if (!cancelled) setSnapshot(s);
            });
          }
        }),
      );
    })();

    return () => {
      cancelled = true;
      for (const off of offCallbacks) off();
      // Best-effort: stop telemetry pump on unmount so the firmware isn't
      // serving frames into the void. If there's no consumer left after
      // this mount, the socket still stays in the cache, idle.
      void transport.unsubscribeTelemetry().catch(() => {
        /* swallow: socket may already be closed */
      });
      transportRef.current = null;
    };
  }, [robotIp, runtimePort]);

  // -------------------------------------------------------------- actions
  const refreshAfter = useCallback(
    async (label: string, fn: () => Promise<unknown>) => {
      const t = transportRef.current;
      if (!t) return;
      setBusy(label);
      setError(null);
      try {
        await fn();
        const s = await t.getSnapshot();
        setSnapshot(s);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(null);
      }
    },
    [],
  );

  const setMode = useCallback(
    (mode: RuntimeMode) =>
      refreshAfter(`mode:${mode}`, () =>
        transportRef.current!.setMode(mode),
      ),
    [refreshAfter],
  );

  const toggleMotor = useCallback(
    (joint: string, enabled: boolean) =>
      refreshAfter(`motor:${joint}`, () =>
        transportRef.current!.setMotorEnabled(joint, enabled),
      ),
    [refreshAfter],
  );

  const setAll = useCallback(
    (enabled: boolean) =>
      refreshAfter(`all:${enabled}`, () =>
        transportRef.current!.setAllMotorsEnabled(enabled),
      ),
    [refreshAfter],
  );

  // "Run policy" composite action: switch to RUN_POLICY mode first, then
  // arm every joint. Order matters — the firmware's `arm()` only accepts
  // DialIn or RunPolicy, and we want to land in the regime where the
  // policy is actually driving before we enable bus-level torque. The
  // policy runner does not enable motors itself, so without this step
  // RunPolicy silently sends PD commands at disabled motors and looks
  // broken on the operator's first attempt.
  //
  // Also gracefully handles the in-DialIn case: if the operator already
  // dialed-in and motors are armed, switching to RunPolicy preserves the
  // arm state (firmware-side) so the second `setAllMotorsEnabled(true)`
  // is a no-op rather than a bus-level re-enable round-trip.
  const runPolicy = useCallback(
    () =>
      refreshAfter("run-policy", async () => {
        const t = transportRef.current!;
        await t.setMode("RUN_POLICY");
        await t.setAllMotorsEnabled(true);
      }),
    [refreshAfter],
  );

  // Throttled per-joint target sender. The dial-in slider can fire a
  // dozen `setMotorTarget` events per second while dragging; rather than
  // queueing one in-flight request per pixel we keep a single pending
  // value per joint and send the latest one as soon as the previous ack
  // returns. This bounds WS pressure and matches the firmware's 100 Hz
  // slew-limited tick well.
  const inFlightRef = useRef<Set<string>>(new Set());
  const pendingRef = useRef<Map<string, number>>(new Map());

  const sendTarget = useCallback(async (joint: string, value: number) => {
    const t = transportRef.current;
    if (!t) return;
    if (inFlightRef.current.has(joint)) {
      pendingRef.current.set(joint, value);
      return;
    }
    inFlightRef.current.add(joint);
    try {
      await t.setMotorTarget(joint, value);
    } catch (e) {
      // Surface as a transient error banner; don't keep retrying — the
      // firmware will reject for a clear reason (mode change, disarm,
      // E-STOP, out-of-envelope) and the operator should see why.
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      inFlightRef.current.delete(joint);
      const next = pendingRef.current.get(joint);
      if (next !== undefined) {
        pendingRef.current.delete(joint);
        // Schedule the queued send on a microtask so we don't recurse
        // synchronously in the rare case of an immediate ack.
        void Promise.resolve().then(() => sendTarget(joint, next));
      }
    }
  }, []);

  // Re-zero the joint's mechanical origin at its current physical
  // position. This is a *destructive*, persistent flash write on the
  // Robstride motor: it overwrites the motor's stored origin and the
  // only way back is to re-zero somewhere else. Always confirm.
  const reZero = useCallback(
    (joint: string, currentPos: number) =>
      refreshAfter(`zero:${joint}`, async () => {
        const ok = window.confirm(
          `Set mechanical zero for ${joint}?\n\n` +
            `The motor's current physical position (${currentPos.toFixed(3)} rad) ` +
            `will become the new 0 rad reference, persisted to motor flash. ` +
            `Use this only after reassembly when the reported position no ` +
            `longer matches mechanical neutral.\n\n` +
            `This cannot be undone except by re-zeroing again somewhere else.`,
        );
        if (!ok) return;
        await transportRef.current!.setMechanicalZero(joint);
      }),
    [refreshAfter],
  );

  const eStop = useCallback(
    () => refreshAfter("estop", () => transportRef.current!.emergencyStop("operator")),
    [refreshAfter],
  );

  const resetEStop = useCallback(
    () => refreshAfter("reset", () => transportRef.current!.resetEStop()),
    [refreshAfter],
  );

  // -------------------------------------------------------------- render
  if (connecting) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-3 text-text-dim">
        <Spinner large />
        <div className="text-sm">
          Connecting to robot at <code>{robotIp}:{runtimePort}</code>…
        </div>
      </div>
    );
  }

  if (error && !snapshot) {
    return (
      <div className="flex flex-col gap-3">
        <Banner tone="error">
          Couldn&rsquo;t reach the robot runtime at{" "}
          <code>{robotIp}:{runtimePort}</code>.<br />
          {error}
        </Banner>
        <Button variant="secondary" onClick={onBack}>
          Back
        </Button>
      </div>
    );
  }

  const motors = snapshot?.motors ?? [];
  const buses = snapshot?.buses ?? [];
  const power = snapshot?.power;
  const mode = snapshot?.mode ?? "UNSPECIFIED";
  const estopLatched = snapshot?.estopLatched ?? false;
  const estopReason = snapshot?.estopReason ?? "";
  const armedCount = motors.filter((m) => m.armed).length;

  return (
    <div className="flex flex-col gap-3">
      {/* Sticky toolbar: status, mode switcher, bulk controls + E-STOP. On
          desktop it lays out as one wide row so the operator always has
          E-STOP one click away even while scrolling motor rows. On
          mobile it stacks: status pill + bus chips, full-width mode
          segmented control, and a final row with bulk controls and a
          prominent E-STOP — keeping every control reachable with a
          thumb. */}
      <div className="sticky top-0 z-10 -mx-4 px-4 sm:-mx-6 sm:px-6 pt-1 pb-3 bg-bg/85 backdrop-blur-md">
        <div className="rounded-[var(--radius-card)] border border-border bg-bg-elev px-3 py-2.5 flex flex-col gap-3 lg:flex-row lg:items-center lg:gap-5">
          {/* Status: mode pill + bus chips */}
          <div className="flex items-center gap-3 flex-wrap min-w-0">
            <div className="flex items-center gap-2 text-sm">
              <span className="text-text-dim">Mode</span>
              <span
                className={`px-2 py-0.5 rounded-full text-xs font-semibold ${
                  mode === "IDLE"
                    ? "bg-text-dim/20 text-text"
                    : mode === "DIAL_IN"
                      ? "bg-accent/20 text-accent"
                      : "bg-success/20 text-success"
                }`}
              >
                {MODE_LABEL[mode]}
              </span>
            </div>
            <div className="hidden lg:block w-px h-5 bg-border" aria-hidden />
            <div className="flex items-center gap-x-3 gap-y-1 flex-wrap">
              {buses.map((b) => (
                <div
                  key={b.canInterface}
                  className="text-xs flex items-center gap-1"
                >
                  <span
                    className={`inline-block w-2 h-2 rounded-full ${
                      b.healthy
                        ? "bg-success"
                        : b.state === ""
                          ? "bg-text-dim"
                          : "bg-danger"
                    }`}
                    aria-hidden
                  />
                  <span className="text-text">{b.canInterface}</span>
                  <span className="text-text-dim">{b.state || "?"}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="hidden lg:block flex-1" aria-hidden />

          {/* Mode switcher: full-width segmented control on mobile so each
              option is comfortably tappable; auto-width on lg+ where it
              shares the toolbar row with bulk controls. */}
          <div className="flex items-center gap-2">
            <div className="inline-flex w-full lg:w-auto rounded-[var(--radius-card)] bg-bg-elev-2 p-0.5">
              <SegButton
                active={mode === "IDLE"}
                onClick={() => setMode("IDLE")}
                disabled={!!busy || estopLatched}
              >
                Idle
              </SegButton>
              <SegButton
                active={mode === "DIAL_IN"}
                onClick={() => setMode("DIAL_IN")}
                disabled={!!busy || estopLatched}
              >
                Dial-in
              </SegButton>
              <SegButton
                active={mode === "RUN_POLICY"}
                onClick={() => setMode("RUN_POLICY")}
                disabled={!!busy || estopLatched}
              >
                Policy
              </SegButton>
            </div>
          </div>

          {/* Bulk controls + E-STOP. On mobile this row stacks below the
              switcher; on desktop it sits to its right. The E-STOP
              button gets extra emphasis on phones — operators reach for
              it under stress, so it's larger and visually anchored. */}
          <div className="flex items-center justify-between gap-2 lg:justify-end">
            <div className="flex gap-2 flex-1 lg:flex-none">
              <Button
                variant="secondary"
                onClick={() => setAll(true)}
                disabled={
                  !!busy ||
                  (mode !== "DIAL_IN" && mode !== "RUN_POLICY") ||
                  estopLatched
                }
                className="flex-1 lg:flex-none py-2.5! text-sm!"
              >
                Enable all
              </Button>
              <Button
                variant="secondary"
                onClick={() => setAll(false)}
                disabled={!!busy}
                className="flex-1 lg:flex-none py-2.5! text-sm!"
              >
                Disable all
              </Button>
            </div>
            <Button
              onClick={eStop}
              disabled={busy === "estop" || estopLatched}
              className="bg-danger! text-white! hover:bg-[#e94a50]! py-2.5! text-sm! shrink-0 min-w-[88px]"
            >
              E-STOP
            </Button>
          </div>
        </div>
      </div>

      {/* E-STOP banner */}
      {estopLatched ? (
        <Banner tone="error">
          <div className="flex items-start justify-between gap-3">
            <div>
              <div className="font-semibold mb-0.5">E-STOP latched</div>
              <div className="text-xs leading-relaxed">
                {estopReason || "(no reason recorded)"}
              </div>
            </div>
            <Button variant="secondary" onClick={resetEStop} loading={busy === "reset"}>
              Reset
            </Button>
          </div>
        </Banner>
      ) : null}

      {error && snapshot ? <Banner tone="error">{error}</Banner> : null}

      {/* Power-board card. Hidden when the firmware has no `power:`
          block configured (older robots / bring-up rigs). Always visible
          regardless of mode so the operator can keep an eye on battery
          level while dialing in joints. */}
      {power && power.present ? <PowerCard power={power} /> : null}

      {/* Dial-in cheat sheet. Shows only in DialIn mode, summarising the
          discovery loop the YAML is structured around. The intent is to
          remind the operator that the slider is bounded by the *current*
          envelope and that widening the envelope is a YAML edit. */}
      {mode === "DIAL_IN" && !estopLatched ? (
        <div className="rounded-[var(--radius-card)] border border-accent/30 bg-accent/5 px-3.5 py-2.5 text-[12px] text-text-dim leading-relaxed">
          <span className="text-text font-semibold">Position dial-in:</span>{" "}
          arm a joint, then drag the position bar to drive it. The slider
          is bounded by each joint's current{" "}
          <code className="text-text">hard_limits.pos_min/pos_max</code>. To
          test a wider envelope, edit{" "}
          <code className="text-text">firmware/bebop-linux/config/bebop_v2.yaml</code>{" "}
          (raise by ≤25%), restart <code className="text-text">bebop-linux</code>,
          and re-arm. Watch the green dot (live) chase the blue thumb (target);
          a persistent gap means the joint can't reach there.
        </div>
      ) : null}

      {/* Policy-mode status banner. Renders an explicit affordance for
          the "switch to RUN_POLICY + arm every joint" composite action
          because the segmented mode pill alone is too easy to miss
          (and on its own does nothing visible — the policy runner sends
          PD commands at whatever motors are armed, so RunPolicy with
          everything disarmed silently looks broken).
          Three states:
            - Idle / DialIn: outlined "Run policy" CTA + short explainer.
            - RunPolicy with not-all-armed: warning state with a
              one-tap "Enable all" so the operator can recover without
              re-reading the toolbar.
            - RunPolicy with everything armed: green confirmation. */}
      {!estopLatched && mode !== "UNSPECIFIED" ? (
        <PolicyBanner
          mode={mode}
          armedCount={armedCount}
          totalMotors={motors.length}
          busy={busy}
          onRunPolicy={runPolicy}
          onEnableAll={() => setAll(true)}
          onStop={() => setMode("IDLE")}
        />
      ) : null}

      {/* Bluetooth-gamepad bridge. Renders nothing when no pad is
          connected, so it doesn't take up space in the layout for
          users that aren't using a controller. Wired to the same
          callbacks the click-and-drag dial uses so the throttling /
          coalescing logic stays in one place. */}
      <GamepadDriver
        motors={motors}
        mode={mode}
        estopLatched={estopLatched}
        onSetTarget={sendTarget}
        onEStop={eStop}
        onResetEStop={resetEStop}
        onToggleArm={toggleMotor}
      />

      {/* Motor table. On md+ this is a true 6-column grid (joint, limits,
          and four numeric readouts). On phones we collapse to a stacked
          card per joint so nothing overflows the viewport horizontally
          and tap targets stay generous. The header row is hidden on
          mobile because each row labels its own readouts inline. */}
      <div className="rounded-[var(--radius-card)] border border-border bg-bg-elev overflow-hidden">
        <div className={MOTOR_HEADER + " px-3 py-2 text-[11px] uppercase tracking-wider text-text-dim border-b border-border"}>
          <div>Joint</div>
          <div>Limits</div>
          <div className="text-right">Pos (rad)</div>
          <div className="text-right">Vel (rad/s)</div>
          <div className="text-right">Tau (Nm)</div>
          <div className="text-right">T (°C)</div>
        </div>
        {motors.length === 0 ? (
          <div className="px-3 py-4 text-sm text-text-dim">
            No motors reported.
          </div>
        ) : (
          motors.map((m) => (
            <MotorRow
              key={m.jointName}
              motor={m}
              busy={busy === `motor:${m.jointName}`}
              zeroBusy={busy === `zero:${m.jointName}`}
              dialIn={mode === "DIAL_IN"}
              runPolicy={mode === "RUN_POLICY"}
              estopLatched={estopLatched}
              onToggle={(enabled) => toggleMotor(m.jointName, enabled)}
              onSetTarget={(value) => sendTarget(m.jointName, value)}
              onReZero={() => reZero(m.jointName, m.position)}
            />
          ))
        )}
      </div>

      <div className="flex flex-wrap items-center justify-center gap-x-4 gap-y-1 mt-2">
        {onOpenControllers ? (
          <Button variant="ghost" onClick={onOpenControllers}>
            Bluetooth controller
          </Button>
        ) : null}
        <Button variant="ghost" onClick={onBack}>
          Back to dashboard
        </Button>
      </div>
    </div>
  );
}

// Shared grid template for header + rows so the columns stay aligned on
// md+ screens. On mobile the row uses flex-col so each motor reads as a
// stacked card (joint header, limit bars, inline numeric readouts) and
// nothing overflows the narrow viewport. The joint column gets a
// comfortable minmax so names like "hip_pitch_left" always fit; numeric
// columns lock to a fixed width so they line up regardless of magnitude.
const MOTOR_HEADER =
  "hidden md:grid md:grid-cols-[minmax(180px,2fr)_minmax(240px,3fr)_88px_88px_88px_72px] gap-3";
const MOTOR_GRID =
  "flex flex-col gap-2.5 md:grid md:grid-cols-[minmax(180px,2fr)_minmax(240px,3fr)_88px_88px_88px_72px] md:gap-3 md:items-center";

/// Composite "run policy" affordance + live status. See call site for
/// state-by-state behaviour. Lifted out of the main render so the JSX of
/// MotorBenchScreen stays close to the dial-in flow it grew up around.
function PolicyBanner({
  mode,
  armedCount,
  totalMotors,
  busy,
  onRunPolicy,
  onEnableAll,
  onStop,
}: {
  mode: RuntimeMode;
  armedCount: number;
  totalMotors: number;
  busy: string | null;
  onRunPolicy: () => void;
  onEnableAll: () => void;
  onStop: () => void;
}) {
  const runBusy = busy === "run-policy";
  const enableAllBusy = busy === "all:true";
  const stopBusy = busy === "mode:IDLE";

  if (mode !== "RUN_POLICY") {
    return (
      <div className="rounded-[var(--radius-card)] border border-success/30 bg-success/5 px-3.5 py-3 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div className="text-[13px] leading-relaxed">
          <div className="text-text font-semibold mb-0.5">Run trained policy</div>
          <div className="text-text-dim">
            Hands the joints to <code className="text-text">policy.onnx</code>{" "}
            (loaded by <code className="text-text">bebop-linux</code> at boot)
            and enables every motor so the policy can drive. Stops by switching
            back to Idle.
          </div>
        </div>
        <Button
          onClick={onRunPolicy}
          loading={runBusy}
          disabled={!!busy}
          className="bg-success! text-white! hover:brightness-110! py-2.5! text-sm! shrink-0"
        >
          Run policy
        </Button>
      </div>
    );
  }

  // mode === RUN_POLICY
  const everythingArmed = totalMotors > 0 && armedCount === totalMotors;
  if (!everythingArmed) {
    return (
      <div className="rounded-[var(--radius-card)] border border-yellow-500/40 bg-yellow-500/5 px-3.5 py-3 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div className="text-[13px] leading-relaxed">
          <div className="text-yellow-700 dark:text-yellow-300 font-semibold mb-0.5">
            Policy is in control, but only {armedCount}/{totalMotors} motors are
            armed
          </div>
          <div className="text-text-dim">
            The policy is sending PD commands at 100&nbsp;Hz; disabled motors
            ignore them. Enable every joint or fall back to Idle.
          </div>
        </div>
        <div className="flex gap-2 shrink-0">
          <Button
            variant="secondary"
            onClick={onStop}
            loading={stopBusy}
            disabled={!!busy}
            className="py-2.5! text-sm!"
          >
            Stop
          </Button>
          <Button
            onClick={onEnableAll}
            loading={enableAllBusy}
            disabled={!!busy}
            className="bg-success! text-white! hover:brightness-110! py-2.5! text-sm!"
          >
            Enable all
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-[var(--radius-card)] border border-success/40 bg-success/10 px-3.5 py-3 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
      <div className="text-[13px] leading-relaxed">
        <div className="text-success font-semibold mb-0.5 flex items-center gap-2">
          <span
            className="inline-block w-2 h-2 rounded-full bg-success animate-pulse"
            aria-hidden
          />
          Policy is driving — {armedCount}/{totalMotors} motors armed
        </div>
        <div className="text-text-dim">
          Targets are coming from <code className="text-text">policy.onnx</code>{" "}
          at 100&nbsp;Hz. Hard limits and slew caps still apply. Tap Stop or
          E-STOP to take back control.
        </div>
      </div>
      <Button
        variant="secondary"
        onClick={onStop}
        loading={stopBusy}
        disabled={!!busy}
        className="py-2.5! text-sm! shrink-0"
      >
        Stop
      </Button>
    </div>
  );
}

function SegButton({
  active,
  onClick,
  disabled,
  children,
}: {
  active: boolean;
  onClick: () => void;
  disabled?: boolean;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`flex-1 lg:flex-none px-3 py-2 lg:py-1.5 text-sm font-semibold rounded-[calc(var(--radius-card)-4px)] transition-colors disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer ${
        active
          ? "bg-accent text-white shadow-sm"
          : "text-text-dim hover:text-text"
      }`}
    >
      {children}
    </button>
  );
}

// ---------------------------------------------------------------------------

function MotorRow({
  motor,
  busy,
  zeroBusy,
  dialIn,
  runPolicy,
  estopLatched,
  onToggle,
  onSetTarget,
  onReZero,
}: {
  motor: MotorView;
  busy: boolean;
  zeroBusy: boolean;
  dialIn: boolean;
  runPolicy: boolean;
  estopLatched: boolean;
  onToggle: (enabled: boolean) => void;
  onSetTarget: (value: number) => void;
  onReZero: () => void;
}) {
  const posPct = useMemo(
    () => percentOfRange(motor.position, motor.posMin, motor.posMax),
    [motor.position, motor.posMin, motor.posMax],
  );
  const velPct = pctOfMax(motor.velocity, motor.velMax);
  const tauPct = pctOfMax(motor.torque, motor.tauMax);
  const tempPct = motor.tempMax > 0 ? (motor.temperature / motor.tempMax) * 100 : 0;

  const stale = motor.feedbackStale && motor.armed;
  const fault = motor.faultBits !== 0;

  // Toggling on requires DialIn or RunPolicy mode and no E-STOP. The
  // firmware allows arming in either; in RunPolicy this is how the
  // operator gives the policy something to drive.
  const canArm = (dialIn || runPolicy) && !estopLatched;
  // Driving the position slider requires arming + DialIn mode + no E-STOP.
  // The firmware enforces the same rule, but disabling the control
  // upfront is a clearer affordance. In RunPolicy the policy is the
  // driver, so the slider stays hidden.
  const canDrive = dialIn && !estopLatched && motor.armed;
  // Re-zero is allowed only when the joint is disarmed and there's no
  // E-STOP. Mode doesn't matter — the firmware accepts SET_ZERO from
  // either Idle or DialIn (motor must be powered but doesn't need to be
  // in our hold loop). This affordance is most useful right after the
  // operator hits the "outside [pos_min, pos_max]" arming refusal: they
  // back-drive the joint to mechanical neutral, then click re-zero, then
  // re-arm.
  const canReZero = !motor.armed && !estopLatched;
  // Joint's reported pos is currently outside its hard envelope — this is
  // the situation re-zero is designed to fix (typical cause: the encoder
  // origin landed far from mechanical neutral after reassembly). Surface
  // the affordance prominently when it's true, faintly otherwise.
  const posOutOfRange =
    motor.position < motor.posMin - 1e-3 || motor.position > motor.posMax + 1e-3;

  return (
    <div className={MOTOR_GRID + " px-3 py-3 text-sm border-b border-border last:border-b-0 hover:bg-bg-elev-2/40 transition-colors"}>
      <div className="flex items-center gap-3 min-w-0">
        <button
          onClick={() => onToggle(!motor.armed)}
          disabled={busy || (!motor.armed && !canArm)}
          aria-label={`${motor.armed ? "Disarm" : "Arm"} ${motor.jointName}`}
          className={`shrink-0 inline-flex items-center justify-center w-11 h-6 md:w-9 md:h-5 rounded-full transition-colors ${
            motor.armed ? "bg-success" : "bg-text-dim/30"
          } ${busy ? "opacity-60" : ""} disabled:opacity-40`}
        >
          <span
            className={`inline-block w-5 h-5 md:w-4 md:h-4 bg-white rounded-full transition-transform ${
              motor.armed ? "translate-x-2.5 md:translate-x-2" : "-translate-x-2.5 md:-translate-x-2"
            }`}
            aria-hidden
          />
        </button>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 flex-wrap">
            <div className="truncate font-medium text-text">{motor.jointName}</div>
            {/* Re-zero affordance. Hidden while armed (the firmware
                refuses anyway). Two visual modes:
                  - Default (disarmed, in-range): a quiet bordered button
                    so it's obviously interactive without dominating the
                    row.
                  - Highlighted (disarmed, *out of range*): the operator
                    just got refused-to-arm with "pos … outside [min, max]";
                    promote the button so it's the obvious next action. */}
            {canReZero ? (
              <button
                type="button"
                onClick={onReZero}
                disabled={zeroBusy}
                title={
                  "Set this joint's mechanical zero to its current physical position. " +
                  "Use after reassembly when reported position no longer matches " +
                  "mechanical neutral. Persists in motor flash; cannot be undone."
                }
                aria-label={`Set mechanical zero for ${motor.jointName}`}
                className={`shrink-0 inline-flex items-center gap-1 px-2 py-1 rounded text-[11px] font-semibold leading-none transition-colors disabled:opacity-50 disabled:cursor-wait cursor-pointer ${
                  posOutOfRange
                    ? "bg-yellow-500/15 text-yellow-700 dark:text-yellow-300 border border-yellow-500/40 hover:bg-yellow-500/25"
                    : "bg-bg-elev-2 text-text-dim border border-border hover:text-text hover:border-accent hover:bg-accent/10"
                }`}
              >
                <span aria-hidden>⊕</span>
                <span>{zeroBusy ? "Re-zeroing…" : "Set zero"}</span>
              </button>
            ) : null}
          </div>
          <div className="text-[11px] text-text-dim flex flex-wrap items-center gap-x-2 gap-y-0.5 mt-0.5">
            <span>{motor.canInterface}</span>
            <span>id {motor.motorId}</span>
            <span>{motor.model}</span>
            {stale ? (
              <span className="text-danger font-semibold">stale</span>
            ) : null}
            {fault ? (
              <span className="text-danger font-semibold">
                fault 0x{motor.faultBits.toString(16).toUpperCase()}
              </span>
            ) : null}
            {posOutOfRange && canReZero ? (
              <span
                className="text-yellow-700 dark:text-yellow-300 font-semibold"
                title={`Reported position ${motor.position.toFixed(3)} rad is outside the hard envelope [${motor.posMin.toFixed(2)}, ${motor.posMax.toFixed(2)}]; the joint can't be armed until either it's back-driven into range or the mechanical zero is reset.`}
              >
                out of range
              </span>
            ) : null}
          </div>
        </div>
      </div>
      <div className="flex flex-col justify-center gap-1.5 min-w-0">
        {canDrive ? (
          <PositionDial
            position={motor.position}
            target={motor.target}
            posMin={motor.posMin}
            posMax={motor.posMax}
            onCommit={onSetTarget}
          />
        ) : (
          <LimitBar label="pos" pct={posPct} signed />
        )}
        <LimitBar label="vel" pct={velPct} />
        <LimitBar label="tau" pct={tauPct} />
        <LimitBar label="T" pct={tempPct} />
      </div>
      {/* Mobile-only readouts: a single horizontal strip beneath the
          bars so each joint fits in a viewport-width card without
          horizontal scrolling. Hidden on md+ where the four
          readout cells below take over. */}
      <div className="md:hidden grid grid-cols-4 gap-2 text-[12px] tabular-nums pt-1 border-t border-border/50 mt-0.5">
        <MobileReadout label="pos">
          <div>{fmt(motor.position)}</div>
          {canDrive ? (
            <div className="text-[10px] text-text-dim leading-tight">
              → {fmt(motor.target)}
            </div>
          ) : null}
        </MobileReadout>
        <MobileReadout label="vel">{fmt(motor.velocity)}</MobileReadout>
        <MobileReadout label="tau">{fmt(motor.torque)}</MobileReadout>
        <MobileReadout label="T °C">{motor.temperature.toFixed(1)}</MobileReadout>
      </div>
      {/* Desktop-only readout cells. These participate in the parent's
          6-column grid so the columns line up across rows. */}
      <div className="hidden md:block text-right tabular-nums self-center leading-tight">
        <div>{fmt(motor.position)}</div>
        {canDrive ? (
          <div
            className="text-[10px] text-text-dim"
            title="Most recent commanded target"
          >
            → {fmt(motor.target)}
          </div>
        ) : null}
      </div>
      <div className="hidden md:block text-right tabular-nums self-center">
        {fmt(motor.velocity)}
      </div>
      <div className="hidden md:block text-right tabular-nums self-center">
        {fmt(motor.torque)}
      </div>
      <div className="hidden md:block text-right tabular-nums self-center">
        {motor.temperature.toFixed(1)}
      </div>
    </div>
  );
}

function MobileReadout({
  label,
  children,
}: {
  label: string;
  children: ReactNode;
}) {
  return (
    <div className="flex flex-col leading-tight min-w-0">
      <span className="text-[10px] uppercase tracking-wider text-text-dim">
        {label}
      </span>
      <span className="text-text font-semibold truncate">{children}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------

/// Interactive position dial used during dial-in. Renders the joint's
/// hard-limit envelope as a horizontal track with three overlays:
///
/// - `target` thumb: what the operator most recently commanded. Draggable
///   and keyboard-accessible. While the user is dragging we show their
///   draft locally; once they release, we let the telemetry-driven
///   `target` prop catch up (it carries the supervisor's clamped+slewed
///   value, which is the source of truth).
/// - `position` marker: live measured position from telemetry. Read-only.
///   Distance between this and the thumb is the tracking error.
/// - Center tick: 0 rad reference (visible only when 0 is inside the
///   envelope, which is true for every joint in the V2 config).
///
/// The firmware clamps to `[posMin, posMax]` and slew-limits per 100 Hz
/// tick, so the dial can fire commands as fast as pointer events arrive
/// without risking a snap. The parent throttles WS sends to one in-flight
/// request at a time per joint.
const STEP_RAD = 0.005; // matches firmware default `slew.max_pos_step_per_tick`
const NEAR_LIMIT_FRAC = 0.9; // colour the thumb when commanding into the last 10%

function PositionDial({
  position,
  target,
  posMin,
  posMax,
  onCommit,
}: {
  position: number;
  target: number;
  posMin: number;
  posMax: number;
  onCommit: (value: number) => void;
}) {
  const trackRef = useRef<HTMLDivElement>(null);
  const [dragging, setDragging] = useState(false);
  // While dragging we ignore the incoming `target` prop (which lags by a
  // telemetry frame) and show the operator's drag value instead. Released
  // → null → fall back to telemetry-driven target.
  const [draft, setDraft] = useState<number | null>(null);

  const range = Math.max(0, posMax - posMin);
  const value = draft ?? clampNumber(target, posMin, posMax);
  const tgtPct = range > 0 ? ((value - posMin) / range) * 100 : 50;
  const livePct = range > 0
    ? ((clampNumber(position, posMin, posMax) - posMin) / range) * 100
    : 50;
  const showCenter = posMin <= 0 && posMax >= 0 && range > 0;
  const centerPct = showCenter ? ((-posMin) / range) * 100 : 0;

  const trackingErr = Math.abs(value - position);
  const trackingPct = range > 0 ? (trackingErr / range) * 100 : 0;
  const nearLimit =
    range > 0 && Math.abs(value) / Math.max(Math.abs(posMin), Math.abs(posMax)) >= NEAR_LIMIT_FRAC;

  function pxToValue(clientX: number): number {
    const r = trackRef.current?.getBoundingClientRect();
    if (!r || r.width === 0) return value;
    const t = (clientX - r.left) / r.width;
    const raw = posMin + Math.max(0, Math.min(1, t)) * range;
    return quantize(raw, STEP_RAD);
  }

  function commit(v: number) {
    setDraft(v);
    onCommit(v);
  }

  function handlePointerDown(e: PointerEvent<HTMLDivElement>) {
    e.preventDefault();
    (e.target as Element).setPointerCapture(e.pointerId);
    setDragging(true);
    commit(pxToValue(e.clientX));
  }

  function handlePointerMove(e: PointerEvent<HTMLDivElement>) {
    if (!dragging) return;
    commit(pxToValue(e.clientX));
  }

  function handlePointerUp() {
    if (!dragging) return;
    setDragging(false);
    setDraft(null);
  }

  function handleKeyDown(e: KeyboardEvent<HTMLDivElement>) {
    // Coarse step = ~slew/tick so a held arrow key stays inside the slew
    // envelope. Shift = 10x for big jumps.
    let delta = 0;
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") delta = -STEP_RAD;
    else if (e.key === "ArrowRight" || e.key === "ArrowUp") delta = STEP_RAD;
    else if (e.key === "Home") {
      e.preventDefault();
      commit(posMin);
      return;
    } else if (e.key === "End") {
      e.preventDefault();
      commit(posMax);
      return;
    } else if (e.key === "PageDown") {
      delta = -STEP_RAD * 10;
    } else if (e.key === "PageUp") {
      delta = STEP_RAD * 10;
    } else {
      return;
    }
    if (e.shiftKey) delta *= 10;
    e.preventDefault();
    commit(quantize(clampNumber(value + delta, posMin, posMax), STEP_RAD));
  }

  function syncToLive() {
    commit(quantize(clampNumber(position, posMin, posMax), STEP_RAD));
    setDraft(null); // telemetry will re-confirm
  }

  return (
    <div className="flex items-center gap-2">
      <span className="text-[10px] text-text-dim w-7">pos</span>
      <div
        ref={trackRef}
        role="slider"
        tabIndex={0}
        aria-label="Target position"
        aria-valuemin={posMin}
        aria-valuemax={posMax}
        aria-valuenow={value}
        aria-valuetext={`${fmt(value)} rad, live ${fmt(position)} rad`}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
        onKeyDown={handleKeyDown}
        className="relative flex-1 h-5 cursor-pointer touch-none select-none focus:outline-none focus-visible:ring-2 focus-visible:ring-accent rounded-full"
      >
        {/* Track */}
        <div className="absolute inset-x-0 top-1/2 -translate-y-1/2 h-1.5 bg-bg-elev-2 rounded-full" />
        {/* Range fill from center to target */}
        {showCenter ? (
          <div
            className={`absolute top-1/2 -translate-y-1/2 h-1.5 rounded-full ${
              nearLimit ? "bg-danger/70" : "bg-accent/55"
            }`}
            style={{
              left: `${Math.min(centerPct, tgtPct)}%`,
              width: `${Math.abs(tgtPct - centerPct)}%`,
            }}
          />
        ) : (
          <div
            className={`absolute top-1/2 -translate-y-1/2 h-1.5 rounded-full ${
              nearLimit ? "bg-danger/70" : "bg-accent/55"
            }`}
            style={{ left: 0, width: `${tgtPct}%` }}
          />
        )}
        {/* Center tick */}
        {showCenter ? (
          <div
            className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-px h-3 bg-text-dim/40"
            style={{ left: `${centerPct}%` }}
            aria-hidden
          />
        ) : null}
        {/* Live measured position marker */}
        <div
          className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-2 h-2 rounded-full bg-success ring-2 ring-bg-elev"
          style={{ left: `${livePct}%` }}
          aria-hidden
          title={`live ${fmt(position)} rad`}
        />
        {/* Target thumb */}
        <div
          className={`absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-3.5 h-3.5 rounded-full ring-2 ring-bg-elev shadow-md transition-transform ${
            dragging ? "scale-110" : ""
          } ${nearLimit ? "bg-danger" : "bg-accent"}`}
          style={{ left: `${tgtPct}%` }}
          aria-hidden
        />
      </div>
      <button
        type="button"
        onClick={syncToLive}
        title="Set target to live position"
        aria-label="Sync target to live position"
        className="shrink-0 text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded text-text-dim hover:text-text hover:bg-bg-elev-2 cursor-pointer"
      >
        sync
      </button>
      {/* Tracking-error hint: only shows when commanded ≠ live by >5% of
          range. Common cause: you dragged to a target the joint can't
          actually reach (mechanical stop, undersized hold gain). */}
      {trackingPct > 5 ? (
        <span
          className="shrink-0 text-[10px] tabular-nums text-yellow-400 dark:text-yellow-300"
          title="|target − live|"
        >
          Δ{fmt(trackingErr)}
        </span>
      ) : null}
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
          ? "bg-yellow-500 dark:bg-yellow-400"
          : "bg-danger";

  const voltage = hasData ? power.batteryVoltageV : 0;
  // Per-cell voltage is the most useful single number for a Li-ion
  // operator: 4.2 V = full, 3.7 V = nominal, 3.0 V = empty.
  const cellV = hasData && power.batteryCells > 0
    ? voltage / power.batteryCells
    : 0;

  const faultActive = hasData && power.faultDescription !== "" && power.faultDescription !== "normal";

  return (
    <div className="rounded-[var(--radius-card)] border border-border bg-bg-elev px-3.5 py-3 flex flex-col gap-2.5">
      {/* Header row: title + meta */}
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <div className="flex items-center gap-2">
          <span className="text-[11px] uppercase tracking-wider text-text-dim font-semibold">
            Power
          </span>
          {power.firmwareVersion ? (
            <span className="text-[10px] text-text-dim font-mono">
              {power.firmwareVersion}
            </span>
          ) : null}
        </div>
        <div className="flex items-center gap-1.5 text-[11px] text-text-dim">
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
        <div className="text-xs text-yellow-700 dark:text-yellow-300 bg-yellow-500/10 border border-yellow-500/30 rounded px-2 py-1">
          Power-board status is stale ({(power.lastStatusAgeMs / 1000).toFixed(1)} s
          since last frame). Verify the {power.canInterface} bus and board power.
        </div>
      ) : null}
      {faultActive ? (
        <div className="text-xs text-[#ffb5b8] bg-danger/10 border border-danger/30 rounded px-2 py-1">
          <span className="font-semibold">Power-board fault:</span>{" "}
          <span className="font-mono">{power.faultDescription}</span>
        </div>
      ) : null}

      {/* SOC bar with cell-voltage and pack-voltage readout above */}
      <div className="flex flex-col gap-1.5">
        <div className="flex items-baseline justify-between gap-2 text-[12px]">
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
              <span className="text-[11px] text-text-dim tabular-nums">
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
                <span className="text-[11px]">
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
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-x-4 gap-y-1.5 text-[12px]">
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
      <span className="text-[10px] uppercase tracking-wider text-text-dim">
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
      className={`text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded font-semibold border ${
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

function clampNumber(v: number, lo: number, hi: number): number {
  if (Number.isNaN(v)) return lo;
  return v < lo ? lo : v > hi ? hi : v;
}

function quantize(v: number, step: number): number {
  if (step <= 0) return v;
  return Math.round(v / step) * step;
}

function LimitBar({
  label,
  pct,
  signed = false,
}: {
  label: string;
  pct: number;
  signed?: boolean;
}) {
  const clamped = Math.max(0, Math.min(100, Math.abs(pct)));
  const color =
    clamped >= 90
      ? "bg-danger"
      : clamped >= 60
        ? "bg-yellow-500 dark:bg-yellow-400"
        : "bg-success";
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-[10px] text-text-dim w-7">{label}</span>
      {signed ? (
        // Two-sided bar: 50% center, fill toward whichever direction.
        <div className="flex-1 h-1.5 bg-bg-elev-2 rounded-full relative overflow-hidden">
          <div
            className={`absolute top-0 bottom-0 ${color}`}
            style={{
              left: pct < 0 ? `${50 - clamped / 2}%` : "50%",
              width: `${clamped / 2}%`,
            }}
          />
        </div>
      ) : (
        <div className="flex-1 h-1.5 bg-bg-elev-2 rounded-full overflow-hidden">
          <div
            className={`h-full ${color}`}
            style={{ width: `${clamped}%` }}
          />
        </div>
      )}
    </div>
  );
}

function percentOfRange(value: number, min: number, max: number): number {
  // Returns a signed percentage in [-100, +100] where 0 = midpoint of the
  // [min, max] band, ±100 = at the corresponding edge.
  const center = (min + max) / 2;
  const half = Math.abs(max - min) / 2;
  if (half <= 0) return 0;
  return ((value - center) / half) * 100;
}

function pctOfMax(value: number, max: number): number {
  if (max <= 0) return 0;
  return (Math.abs(value) / max) * 100;
}

function fmt(v: number): string {
  return v >= 0 ? `+${v.toFixed(2)}` : v.toFixed(2);
}
