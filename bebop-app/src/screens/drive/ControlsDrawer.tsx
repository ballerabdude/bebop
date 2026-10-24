// Controls for the Drive page that don't belong over the video: wheel arm
// switches, firmware mode, the vision service, odometry reset, the speed
// profile, video playback and the gamepad card.
//
// Many viewers, one driver: while another human holds the drive seat this
// client is a spectator, so everything that would change their drive (arm,
// drive mode, odometry reset, the vision service, which carries their video
// and recording) is disabled. Stop-direction actions (disarm, idle) stay.
//
// Slide-over from the right on phones / tablets, a docked side panel on
// desktop (>=1280). It stays mounted while closed (hidden), because the
// gamepad card handles the pad's buttons (B = E-STOP, A = reset, L3 = arm).

import { useEffect } from "react";
import type { ReactNode } from "react";

import { GamepadDrive } from "../../components/GamepadDrive";
import {
  Banner,
  Button,
  cn,
  SegmentedControl,
  StatusPill,
  Switch,
} from "../../components/ui";
import { CONTROL_PROFILES, setActiveControlProfile, useControlProfile } from "../../input";
import type { ControlProfileId } from "../../input";
import type { RuntimeMode, RuntimeSnapshot, RuntimeTransport, WheelView } from "../../runtime";
import { signed } from "./hudModel";
import { CloseIcon } from "./icons";

type Run = (label: string, fn: (t: RuntimeTransport) => Promise<unknown>) => Promise<void>;

function Section({ title, children, aside }: { title: string; children: ReactNode; aside?: ReactNode }) {
  return (
    <section className="space-y-2.5 border-b border-border px-4 py-4 last:border-b-0">
      <div className="flex items-center justify-between gap-2">
        <h3 className="text-caption font-semibold uppercase tracking-wider text-text-dim">{title}</h3>
        {aside}
      </div>
      {children}
    </section>
  );
}

export function ControlsDrawer({
  open,
  docked,
  onClose,
  snapshot,
  busy,
  error,
  run,
  estopLatched,
  onEStop,
  onResetEStop,
  quickStartable,
  onStartDriving,
  videoEnabled,
  onVideoEnabled,
  onReconnectVideo,
  spectatorOf,
}: {
  open: boolean;
  /// Desktop: a side panel in the page flow instead of an overlay.
  docked: boolean;
  onClose: () => void;
  snapshot: RuntimeSnapshot | null;
  busy: string | null;
  error: string | null;
  run: Run;
  estopLatched: boolean;
  onEStop: () => void;
  onResetEStop: () => void;
  quickStartable: boolean;
  onStartDriving: () => void;
  videoEnabled: boolean;
  onVideoEnabled: (on: boolean) => void;
  /// Tear down and renegotiate every video tile.
  onReconnectVideo: () => void;
  /// Label of the other human driver when this client only watches.
  spectatorOf: string | null;
}) {
  const spectator = spectatorOf !== null;
  const profile = useControlProfile();
  const wheels = snapshot?.wheels ?? [];
  const drive = snapshot?.drive;
  const mode: RuntimeMode = snapshot?.mode ?? "UNSPECIFIED";
  const vision = snapshot?.vision;
  const armed = wheels.filter((w) => w.armed).length;
  const wheeled = !!drive?.present && (snapshot?.motors.length ?? 0) === 0;
  const driveMode = mode === "DIAL_IN";

  // Esc closes the overlay drawer.
  useEffect(() => {
    if (!open || docked) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, docked, onClose]);

  const setAll = (enabled: boolean) =>
    void run(`all-wheels:${enabled}`, (t) => t.setAllWheelsEnabled(enabled));

  const body = (
    <>
      <div className="flex items-center justify-between gap-2 border-b border-border px-4 py-2">
        <h2 className="text-base font-semibold">Controls</h2>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close controls"
          className="-mr-2 flex h-touch w-touch items-center justify-center rounded-card text-text-dim hover:bg-bg-elev-2 hover:text-text"
        >
          <CloseIcon />
        </button>
      </div>
      <div
        className={cn(
          "min-h-0 flex-1 overflow-y-auto overscroll-contain",
          "pb-safe",
        )}
      >
        {error ? (
          <div className="px-4 pt-3">
            <Banner tone="error">{error}</Banner>
          </div>
        ) : null}
        {spectator ? (
          <div className="px-4 pt-3">
            <Banner tone="warning">
              {spectatorOf} — you are watching. Arming, drive mode, odometry and vision are
              theirs until the seat frees up; you can still disarm or go idle.
            </Banner>
          </div>
        ) : null}

        {wheeled ? (
          <Section
            title="Wheels"
            aside={
              <StatusPill tone={armed === 0 ? "warn" : "ok"}>
                {armed}/{wheels.length} armed
              </StatusPill>
            }
          >
            {quickStartable && !spectator ? (
              <Button
                variant="success"
                className="w-full"
                onClick={onStartDriving}
                loading={busy === "start-driving"}
                disabled={!!busy}
              >
                Start driving
              </Button>
            ) : null}
            <div className="space-y-0.5 rounded-card border border-border bg-bg-elev-2/40 px-1.5 py-1">
              {wheels.map((w) => (
                <WheelRow
                  key={w.name}
                  wheel={w}
                  busy={busy === `wheel:${w.name}`}
                  disabled={!w.armed && (estopLatched || !driveMode || spectator)}
                  onToggle={(enabled) =>
                    void run(`wheel:${w.name}`, (t) => t.setWheelEnabled(w.name, enabled))
                  }
                />
              ))}
            </div>
            <div className="grid grid-cols-2 gap-2">
              <Button
                variant="secondary"
                size="sm"
                onClick={() => setAll(true)}
                disabled={
                  !!busy || estopLatched || spectator || !driveMode || armed === wheels.length
                }
                loading={busy === "all-wheels:true"}
              >
                Arm all
              </Button>
              <Button
                variant="secondary"
                size="sm"
                onClick={() => setAll(false)}
                disabled={!!busy || armed === 0}
                loading={busy === "all-wheels:false"}
              >
                Disarm all
              </Button>
            </div>
            {!driveMode ? (
              <p className="text-caption leading-snug text-text-dim">
                Wheels arm only in drive mode.
              </p>
            ) : null}
          </Section>
        ) : null}

        <Section title="Firmware mode">
          <SegmentedControl<RuntimeMode>
            label="Firmware mode"
            className="w-full"
            value={mode === "DIAL_IN" ? "DIAL_IN" : "IDLE"}
            disabled={!!busy || !snapshot}
            onChange={(m) => void run(`mode:${m}`, (t) => t.setMode(m))}
            options={[
              { value: "IDLE", label: "Idle" },
              { value: "DIAL_IN", label: "Drive", disabled: spectator },
            ]}
          />
          <p className="text-caption leading-snug text-text-dim">
            Idle disarms everything. Drive lets the wheels arm.
          </p>
        </Section>

        <Section title="Speed profile">
          <SegmentedControl<ControlProfileId>
            label="Speed profile"
            className="w-full"
            value={profile.id}
            onChange={setActiveControlProfile}
            options={CONTROL_PROFILES.map((p) => ({ value: p.id, label: p.label }))}
          />
          <p className="text-caption leading-snug text-text-dim">
            {profile.description}{" "}
            <span className="font-mono text-text-dim/80">
              {profile.maxLinear} m/s · {profile.maxAngular} rad/s
            </span>
          </p>
        </Section>

        {wheeled ? (
          <Section title="Odometry">
            <div className="flex items-center justify-between gap-3">
              <div className="font-mono text-body-sm tabular-nums text-text">
                x {signed(drive?.odomX ?? 0)} · y {signed(drive?.odomY ?? 0)} m
                <div className="text-caption text-text-dim">
                  θ {signed(((drive?.odomTheta ?? 0) * 180) / Math.PI, 1)}°
                </div>
              </div>
              <Button
                variant="secondary"
                size="sm"
                onClick={() => void run("odom-reset", (t) => t.resetOdometry())}
                loading={busy === "odom-reset"}
                disabled={!!busy || spectator}
              >
                Reset
              </Button>
            </div>
          </Section>
        ) : null}

        {vision?.present ? (
          <Section
            title="Vision service"
            aside={
              <StatusPill tone={vision.running ? "ok" : vision.state === "failed" ? "err" : "dim"}>
                {vision.running ? "running" : vision.state || "stopped"}
              </StatusPill>
            }
          >
            <p className="text-label leading-relaxed text-text-dim">
              {vision.running
                ? "Cameras, recording and place memory are live."
                : "Stopped. Video, recording and place memory need it."}
            </p>
            {vision.detail && !vision.running ? (
              <div className="break-words font-mono text-caption text-text-dim">{vision.detail}</div>
            ) : null}
            <Button
              variant={vision.running ? "secondary" : "success"}
              size="sm"
              className="w-full"
              onClick={() =>
                void run(`vision:${!vision.running}`, (t) => t.setVisionEnabled(!vision.running))
              }
              loading={busy === `vision:${!vision.running}`}
              disabled={!!busy || spectator}
            >
              {vision.running ? "Stop vision" : "Start vision"}
            </Button>
          </Section>
        ) : null}

        <Section title="Video">
          <label className="flex items-center justify-between gap-3">
            <span className="text-body-sm text-text">
              Playback
              <span className="block text-caption text-text-dim">
                Off saves bandwidth; the robot keeps recording.
              </span>
            </span>
            <Switch checked={videoEnabled} onChange={onVideoEnabled} label="Video playback" />
          </label>
          <Button
            variant="secondary"
            size="sm"
            className="w-full"
            onClick={onReconnectVideo}
            disabled={!videoEnabled}
          >
            Reconnect video
          </Button>
        </Section>

        {wheeled ? (
          <div className="px-4 py-3 empty:hidden">
            <GamepadDrive
              wheels={wheels}
              mode={mode}
              estopLatched={estopLatched}
              onEStop={onEStop}
              onResetEStop={onResetEStop}
              onSetAllWheels={(enabled) => {
                if (enabled && spectator) return;
                setAll(enabled);
              }}
            />
          </div>
        ) : null}

        <Section title="Keys">
          <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-label">
            <dt className="font-mono text-text">W A S D / arrows</dt>
            <dd className="text-text-dim">drive (release to stop)</dd>
            <dt className="font-mono text-text">Space</dt>
            <dd className="text-text-dim">E-STOP</dd>
            <dt className="font-mono text-text">1 2 3 4</dt>
            <dd className="text-text-dim">obstacle · recovery · exclude · note</dd>
            <dt className="font-mono text-text">Gamepad</dt>
            <dd className="text-text-dim">hold RT, left stick drives</dd>
          </dl>
        </Section>
      </div>
    </>
  );

  if (docked) {
    return (
      <aside
        aria-label="Drive controls"
        className={cn(
          "flex min-h-0 w-80 shrink-0 flex-col border-l border-border bg-bg-elev",
          !open && "hidden",
        )}
      >
        {body}
      </aside>
    );
  }
  return (
    // Inside the page (absolute), so the shell's status bar and its E-STOP
    // stay uncovered; z-[38] keeps it under the shell's E-STOP.
    <div className={cn("absolute inset-0 z-[38] overflow-hidden", !open && "pointer-events-none")} aria-hidden={!open}>
      <div
        className={cn(
          "absolute inset-0 bg-black/50 transition-opacity duration-200",
          open ? "opacity-100" : "opacity-0",
        )}
        onClick={onClose}
      />
      <aside
        aria-label="Drive controls"
        inert={!open}
        className={cn(
          "absolute inset-y-0 right-0 flex w-[min(22rem,88vw)] flex-col border-l border-border bg-bg-elev shadow-2xl transition-transform duration-200",
          open ? "translate-x-0" : "invisible translate-x-full",
        )}
      >
        {body}
      </aside>
    </div>
  );
}

function WheelRow({
  wheel,
  busy,
  disabled,
  onToggle,
}: {
  wheel: WheelView;
  busy: boolean;
  disabled: boolean;
  onToggle: (enabled: boolean) => void;
}) {
  return (
    <div className="flex items-center gap-1.5">
      <Switch
        checked={wheel.armed}
        label={`Arm ${wheel.name}`}
        onChange={onToggle}
        disabled={busy || disabled}
      />
      <span className="truncate text-label font-medium text-text">{wheel.name}</span>
      {wheel.errorCode !== 0 ? <StatusPill tone="err">error</StatusPill> : null}
      <span className="ml-auto pr-2 font-mono text-caption tabular-nums text-text-dim">
        {signed(wheel.velocity)} rad/s
      </span>
    </div>
  );
}
