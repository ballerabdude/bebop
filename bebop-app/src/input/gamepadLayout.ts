// Gamepad stick layout for driving, shared by the drive-input arbiter
// (`useDriveInput.ts`, which turns sticks into twists) and the gamepad
// status card (`GamepadDrive.tsx`, which shows and switches it).
// `split` (default): left stick up/down drives, right stick left/right
// turns. `arcade`: the left stick does both, like the on-screen joystick.

import { useSyncExternalStore } from "react";

import type { GamepadDriveLayout } from "./driveArbiter";

const LAYOUT_STORAGE_KEY = "bebop.driveLayout";

const listeners = new Set<() => void>();

function load(): GamepadDriveLayout {
  try {
    return window.localStorage.getItem(LAYOUT_STORAGE_KEY) === "arcade" ? "arcade" : "split";
  } catch {
    return "split";
  }
}

let active: GamepadDriveLayout = load();

export function getGamepadLayout(): GamepadDriveLayout {
  return active;
}

export function setGamepadLayout(next: GamepadDriveLayout): void {
  if (next === active) return;
  active = next;
  try {
    window.localStorage.setItem(LAYOUT_STORAGE_KEY, next);
  } catch {
    /* localStorage may be disabled: the preference just won't persist */
  }
  for (const cb of listeners) cb();
}

function subscribe(cb: () => void): () => void {
  listeners.add(cb);
  return () => {
    listeners.delete(cb);
  };
}

export function useGamepadLayout(): GamepadDriveLayout {
  return useSyncExternalStore(subscribe, getGamepadLayout);
}
