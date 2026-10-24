// Drive-twist stream shared by every drive input (on-screen joystick,
// WASD / arrows, gamepad) on the teleop screen and the motor bench.
//
// Why it exists: the firmware's operator watchdog
// (`server.operator_timeout_ms`, 500 ms) zeroes a non-zero twist that isn't
// refreshed in time. The old per-screen sender awaited every
// SetVelocityCommand's Ack before sending the next twist, and the 10 Hz
// keepalive only refilled its one pending slot, so a single Ack slower than
// 500 ms meant no twist went out at all. Acks queue behind 30 Hz telemetry
// on the downlink, and Wi-Fi carrying four WebRTC camera streams made that
// happen mid-drive (navd_session_20261023_212953: three 0.57-0.68 s halts).
//
// What it does:
//  * never waits for an acknowledgement: `send` is fire-and-forget;
//  * while a drive cycle is open, the newest twist goes out at least every
//    `keepaliveMs` (100 ms), and a changed twist goes out promptly, at most
//    one frame per `minIntervalMs` (~33 ms);
//  * while the socket is backed up (`bufferedAmount() > maxBufferedBytes`)
//    non-zero frames are skipped rather than queued behind stale ones; the
//    newest twist goes out on the next tick once the socket drains;
//  * stops (0, 0) go out immediately: no throttle, no buffer check;
//  * firmware rejections of non-zero twists (another operator holds the
//    drive seat) reach `onError`, at most once per `errorIntervalMs`.
//
// The module is pure: no React, DOM or transport imports. The socket write,
// buffered-byte probe, clock and timers are injected, so the unit tests
// (tests/twistStream.test.mjs) drive it with a fake clock.

/// How often the held twist is re-sent while a drive cycle is open. Must
/// stay well below the firmware's `operator_timeout_ms` (500 ms default,
/// `firmware/bebop-linux/src/config.rs`); 100 ms leaves room for four
/// missed frames before the robot halts.
export const TWIST_KEEPALIVE_MS = 100;

/// Minimum spacing between frames when the twist changes (~30 Hz). The
/// gamepad poll runs at 60 Hz and a pointer drag can run faster; the newest
/// twist always wins.
export const TWIST_MIN_INTERVAL_MS = 33;

/// `WebSocket.bufferedAmount` above which non-zero frames are skipped. A
/// twist frame encodes to 13-17 bytes and a healthy link drains to ~0
/// between ticks, so this is about 15 frames that never left the device.
export const TWIST_MAX_BUFFERED_BYTES = 256;

/// At most one rejection is reported to the UI per this interval. A
/// rejected drive repeats its rejection on every frame (30 Hz).
export const TWIST_ERROR_INTERVAL_MS = 1000;

/// Cap on outstanding error callbacks for fire-and-forget twist frames (see
/// `createReplyRegistry`). About 2 s of frames at the 30 Hz ceiling.
export const TWIST_REPLY_CAPACITY = 64;

export interface Twist {
  vx: number;
  wz: number;
}

export interface TwistStreamDeps {
  /// Write one SetVelocityCommand frame without waiting for a reply.
  /// Returns false when nothing was written (socket not open). For
  /// non-zero twists `onReject` is passed: call it with the firmware's
  /// message if that frame is rejected. Stops get no callback, so they
  /// stay silent.
  send(vx: number, wz: number, onReject?: (message: string) => void): boolean;
  /// Bytes queued on the socket that haven't reached the network yet.
  bufferedAmount(): number;
  /// Monotonic clock, ms.
  now(): number;
  setTimer(fn: () => void, ms: number): unknown;
  clearTimer(handle: unknown): void;
  /// Rate-limited sink for rejections of non-zero twists.
  onError?(message: string): void;
}

export interface TwistStreamOptions {
  keepaliveMs?: number;
  minIntervalMs?: number;
  maxBufferedBytes?: number;
  errorIntervalMs?: number;
}

export interface TwistStream {
  /// Request a body twist (vx m/s, wz rad/s, +wz = left turn). A non-zero
  /// twist opens or continues a drive cycle. (0, 0) is a stop.
  set(vx: number, wz: number): void;
  /// Close the drive cycle and write (0, 0) right away. Still writes after
  /// `dispose()`: a late stop from an unmounting input must reach the robot.
  stop(): void;
  /// True while a drive cycle is open.
  isDriving(): boolean;
  /// Cancel the keepalive and ignore later non-zero twists. Writes nothing.
  dispose(): void;
}

export function createTwistStream(
  deps: TwistStreamDeps,
  options: TwistStreamOptions = {},
): TwistStream {
  const keepaliveMs = options.keepaliveMs ?? TWIST_KEEPALIVE_MS;
  const minIntervalMs = options.minIntervalMs ?? TWIST_MIN_INTERVAL_MS;
  const maxBufferedBytes = options.maxBufferedBytes ?? TWIST_MAX_BUFFERED_BYTES;
  const errorIntervalMs = options.errorIntervalMs ?? TWIST_ERROR_INTERVAL_MS;

  /// Newest requested non-zero twist. `null` = no drive cycle open.
  let target: Twist | null = null;
  /// Last twist written in this cycle (`null` until the first one).
  let written: Twist | null = null;
  /// When the last frame of any kind, stops included, was written.
  let writtenAt = -Infinity;
  /// The single wake-up timer: the next due frame or a congestion re-check.
  let timer: { handle: unknown; at: number } | null = null;
  let lastErrorAt = -Infinity;
  let disposed = false;

  function reportError(message: string): void {
    if (disposed) return;
    const now = deps.now();
    if (now - lastErrorAt < errorIntervalMs) return;
    lastErrorAt = now;
    deps.onError?.(message);
  }

  // A throwing transport must not take the keepalive down with it.
  function write(vx: number, wz: number, onReject?: (message: string) => void): boolean {
    try {
      return deps.send(vx, wz, onReject);
    } catch {
      return false;
    }
  }

  function cancelTimer(): void {
    if (timer === null) return;
    deps.clearTimer(timer.handle);
    timer = null;
  }

  // Keep an armed timer that fires no later than `at`. `pump` works out
  // what is due whenever it fires, so an earlier timer is always safe.
  function wakeAt(at: number, now: number): void {
    if (timer !== null && timer.at <= at) return;
    cancelTimer();
    timer = { at, handle: deps.setTimer(onTimer, Math.max(0, at - now)) };
  }

  function onTimer(): void {
    timer = null;
    pump();
  }

  // Write the newest twist if a frame is due, or sleep until one is.
  function pump(): void {
    if (disposed || target === null) return;
    const now = deps.now();
    const changed =
      written === null || written.vx !== target.vx || written.wz !== target.wz;
    const due = writtenAt + (changed ? minIntervalMs : keepaliveMs);
    if (now < due) {
      wakeAt(due, now);
      return;
    }
    if (deps.bufferedAmount() > maxBufferedBytes) {
      // Socket backed up: a frame now would only queue behind stale ones.
      wakeAt(now + minIntervalMs, now);
      return;
    }
    if (write(target.vx, target.wz, reportError)) {
      written = target;
      writtenAt = now;
    }
    // After a write this is the next keepalive. After a failed write
    // (socket closed) it is the retry, so a twist still held when the
    // link comes back resumes.
    wakeAt(now + keepaliveMs, now);
  }

  function stop(): void {
    target = null;
    written = null;
    cancelTimer();
    if (write(0, 0)) writtenAt = deps.now();
  }

  function set(vx: number, wz: number): void {
    // A NaN or infinite component would reach the firmware as a non-zero
    // twist. Treat it as 0.
    const x = Number.isFinite(vx) ? vx : 0;
    const z = Number.isFinite(wz) ? wz : 0;
    if (x === 0 && z === 0) {
      stop();
      return;
    }
    if (disposed) return;
    target = { vx: x, wz: z };
    pump();
  }

  return {
    set,
    stop,
    isDriving: () => target !== null,
    dispose() {
      disposed = true;
      target = null;
      written = null;
      cancelTimer();
    },
  };
}

/// Bounded id -> value map for replies to fire-and-forget frames. Nothing
/// awaits these frames, so if a reply never arrives its entry would stay
/// forever. Past `capacity` the oldest entry is evicted, which keeps memory
/// bounded however the link behaves.
export interface ReplyRegistry<T> {
  add(id: number, value: T): void;
  /// Remove and return the entry for `id`, if it is still held.
  take(id: number): T | undefined;
  clear(): void;
  readonly size: number;
}

export function createReplyRegistry<T>(capacity: number): ReplyRegistry<T> {
  // Map iteration follows insertion order, so the first key is the oldest.
  const entries = new Map<number, T>();
  return {
    add(id, value) {
      entries.delete(id);
      entries.set(id, value);
      while (entries.size > capacity) {
        const oldest = entries.keys().next();
        if (oldest.done) break;
        entries.delete(oldest.value);
      }
    },
    take(id) {
      if (!entries.has(id)) return undefined;
      const value = entries.get(id);
      entries.delete(id);
      return value;
    },
    clear() {
      entries.clear();
    },
    get size() {
      return entries.size;
    },
  };
}
