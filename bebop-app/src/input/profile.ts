// Control-sensitivity profiles for every app-side input path.
//
// The app has three ways to move a robot — gamepad sticks (dial-in and
// drive), the on-screen drive joystick, and the WASD keyboard drive —
// and until now each hard-coded its own deadzones and rate limits.
// This module centralises those knobs into named presets ("Gentle",
// "Standard", "Sport") that the user can switch at runtime. The choice
// persists in localStorage alongside the drive-layout preference and
// applies immediately: the gamepad poller reads the active profile on
// every tick, and the drive consumers (joystick / WASD / gamepad
// bridge) re-read it whenever they compute a twist.
//
// The "Standard" preset reproduces the historical hard-coded values
// exactly (deadzones from `useGamepad`, limits from `GamepadDrive`,
// dial-in rate from `GamepadDriver`), so existing operators see zero
// behaviour change until they opt into another preset.

import { useSyncExternalStore } from "react";

export type ControlProfileId = "gentle" | "standard" | "sport";

export interface ControlProfile {
  id: ControlProfileId;
  /// Short label rendered in pickers.
  label: string;
  /// One-line summary of the feel, shown in the full picker.
  description: string;
  /// Radial deadzone applied to each stick before anything else.
  /// Larger = wider "does absolutely nothing" ring around centre.
  stickDeadzone: number;
  /// Axial deadzone applied to analog triggers.
  triggerDeadzone: number;
  /// Stick response-curve exponent, 0..1. 0 = linear (raw stick),
  /// higher = softer around centre while full deflection still maps
  /// to ±1. Same shaping the ROS2 pilot node applies (`expo` param).
  expo: number;
  /// Soft limit at full deflection for body twists (wheeled drive):
  /// linear speed in m/s. The firmware still clamps to each wheel's
  /// `vel_max` regardless.
  maxLinear: number;
  /// Soft limit at full deflection for body twists: turn rate in
  /// rad/s (+wz = left turn).
  maxAngular: number;
  /// Per-joint dial-in rate (rad/s) at full stick + full trigger on
  /// the legged robot. Coupled with the firmware's
  /// `slew.max_pos_step_per_tick` ceiling.
  dialInRate: number;
}

export const CONTROL_PROFILES: readonly ControlProfile[] = [
  {
    id: "gentle",
    label: "Gentle",
    description:
      "Wide deadzone, soft centre and half-speed limits — first drives and tight spaces.",
    stickDeadzone: 0.18,
    triggerDeadzone: 0.08,
    expo: 0.5,
    maxLinear: 0.5,
    maxAngular: 1.0,
    dialInRate: 1.0,
  },
  {
    id: "standard",
    label: "Standard",
    description:
      "Linear sticks and the original soft limits — matches the classic bebop-app feel.",
    stickDeadzone: 0.12,
    triggerDeadzone: 0.05,
    expo: 0,
    maxLinear: 1.0,
    maxAngular: 2.0,
    dialInRate: 2.0,
  },
  {
    id: "sport",
    label: "Sport",
    description:
      "Narrow deadzone, direct response and 1.5× limits — firmware still clamps per wheel.",
    stickDeadzone: 0.08,
    triggerDeadzone: 0.05,
    expo: 0,
    maxLinear: 1.5,
    maxAngular: 3.0,
    dialInRate: 3.0,
  },
];

export const DEFAULT_PROFILE_ID: ControlProfileId = "standard";

/// Persisted alongside `bebop.driveLayout` etc. — same localStorage
/// convention as the drive-layout picker in `GamepadDrive`.
const PROFILE_STORAGE_KEY = "bebop.controlProfile";

const listeners = new Set<(profile: ControlProfile) => void>();

function profileById(id: ControlProfileId): ControlProfile {
  return CONTROL_PROFILES.find((p) => p.id === id) ?? standardProfile();
}

function standardProfile(): ControlProfile {
  // "standard" is always present; the array fallback is only for
  // exhaustiveness.
  return (
    CONTROL_PROFILES.find((p) => p.id === DEFAULT_PROFILE_ID) ?? CONTROL_PROFILES[0]
  );
}

function loadStoredId(): ControlProfileId {
  try {
    const raw = window.localStorage.getItem(PROFILE_STORAGE_KEY);
    if (raw && CONTROL_PROFILES.some((p) => p.id === raw)) {
      return raw as ControlProfileId;
    }
  } catch {
    /* localStorage may be disabled — default applies */
  }
  return DEFAULT_PROFILE_ID;
}

let active: ControlProfile = profileById(loadStoredId());

/// The profile every input path should scale by. Cheap (object read),
/// so the gamepad poller can call it per tick without caching.
export function getActiveControlProfile(): ControlProfile {
  return active;
}

/// Switch the active profile and persist the choice. Invalid ids are
/// ignored (keeps the stored value authoritative).
export function setActiveControlProfile(id: ControlProfileId): void {
  if (!CONTROL_PROFILES.some((p) => p.id === id)) return;
  const next = profileById(id);
  if (next === active) return;
  active = next;
  try {
    window.localStorage.setItem(PROFILE_STORAGE_KEY, id);
  } catch {
    /* preference just won't persist */
  }
  for (const cb of listeners) cb(next);
}

/// React hook: the active profile, re-rendering on change. The
/// snapshot is a stable object reference per preset, so
/// `useSyncExternalStore` never loops.
export function useControlProfile(): ControlProfile {
  return useSyncExternalStore(subscribeControlProfile, getActiveControlProfile);
}

function subscribeControlProfile(cb: () => void): () => void {
  const wrapped = () => cb();
  listeners.add(wrapped);
  return () => {
    listeners.delete(wrapped);
  };
}

/// Expo stick shaping applied per-axis after the deadzone:
/// `(1-expo)·v + expo·v³`. Keeps the sign, maps ±1 to ±1, and
/// progressively softens the centre as expo grows. Mirrors the
/// `_shape()` curve in `ros2/src/bebop_pilot/scripts/pilot_node.py`.
export function applyExpo(v: number, expo: number): number {
  if (expo <= 0 || v === 0) return v;
  return (1 - expo) * v + expo * v * v * v;
}
