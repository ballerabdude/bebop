// Segmented control for switching the active control-sensitivity
// profile (`src/input/profile.ts`). Mounted wherever the user drives:
// the controllers screen (full form, with descriptions), the motor
// bench's drive card (on-screen joystick + WASD), and the two
// gamepad bridge cards — so the setting is always at hand without a
// separate settings screen.

import { CONTROL_PROFILES, setActiveControlProfile, useControlProfile } from "../input";

/// Switch is instant (next gamepad tick / joystick event uses the new
/// values) and persists across app restarts.
export function ControlProfilePicker({ full = false }: { full?: boolean }) {
  const active = useControlProfile();

  return (
    <div
      className={
        full
          ? "flex flex-col gap-2"
          : "flex items-center gap-1 p-0.5 rounded-lg border border-border bg-bg-elev-2"
      }
      role="group"
      aria-label="Control profile"
    >
      {full ? (
        <div className="text-xs text-text-dim uppercase tracking-wider">
          Control profile
        </div>
      ) : null}
      <div className="flex items-center gap-1 flex-wrap">
        {CONTROL_PROFILES.map((p) => (
          <button
            key={p.id}
            type="button"
            aria-pressed={p.id === active.id}
            title={
              full
                ? undefined
                : `Control profile — ${p.description} Deadzone ${(p.stickDeadzone * 100).toFixed(0)}%, dial-in ${p.dialInRate} rad/s, drive ${p.maxLinear} m/s · ${p.maxAngular} rad/s.`
            }
            onClick={() => setActiveControlProfile(p.id)}
            className={
              full
                ? `flex-1 min-w-[90px] px-3 py-2 rounded-lg border text-left cursor-pointer ${
                    p.id === active.id
                      ? "bg-accent/15 border-accent/40 text-accent"
                      : "bg-bg-elev border-border text-text hover:bg-bg-elev-2"
                  }`
                : `px-2 py-0.5 rounded-md text-[11px] font-semibold uppercase tracking-wider cursor-pointer ${
                    p.id === active.id
                      ? "bg-accent/15 text-accent"
                      : "bg-transparent text-text-dim hover:text-text"
                  }`
            }
          >
            <span className={full ? "block text-[13px] font-semibold" : undefined}>
              {p.label}
            </span>
            {full ? (
              <>
                <span className="block text-[11px] font-normal text-text-dim leading-snug mt-0.5">
                  {p.description}
                </span>
                <span className="block text-[10px] font-mono font-normal text-text-dim/80 mt-1">
                  dz {(p.stickDeadzone * 100).toFixed(0)}%
                  {p.expo > 0 ? ` · expo ${p.expo.toFixed(1)}` : ""} · {p.maxLinear} m/s ·{" "}
                  {p.maxAngular} rad/s · {p.dialInRate} rad/s
                </span>
              </>
            ) : null}
          </button>
        ))}
      </div>
    </div>
  );
}
