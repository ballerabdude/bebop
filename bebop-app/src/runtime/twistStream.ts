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
//    drive seat) reach `onError`, at most once per `errorIntervalMs`;
//  * a stop is only written when this stream owes one: it wrote a non-zero
//    twist since its last stop, or `holdsSeat()` says this client holds the
//    drive seat. A spectator's (0, 0) would otherwise halt whoever is
//    driving. A seat-held stop is written once: repeated neutral reports
//    (a joystick resting in its dead zone, keyups typed into a text field)
//    don't re-send (0, 0) until a non-zero twist has gone out again.
//    `forceStop()` always writes (E-STOP-adjacent use);
//  * after the link drops (a write finds the socket closed, or `linkLost()`)
//    or after `requireNeutral()` (window blur, tab hidden, the firmware's
//    operator watchdog fired), a held input does NOT resume: non-zero
//    twists are ignored until the input returns to neutral, i.e. a
//    `set(0, 0)` or `stop()` arrives. Only a fresh operator action moves
//    the robot again.
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
  /// True while this client holds the firmware's drive seat. A stop is then
  /// written even if this stream hasn't driven since its last stop (once:
  /// not again until a non-zero twist has been written).
  /// Optional: without it only this stream's own driving owes a stop.
  holdsSeat?(): boolean;
  /// Called whenever the re-engage latch (`needsNeutral()`) flips.
  onNeutralRequiredChange?(required: boolean): void;
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
  /// Close the drive cycle and write (0, 0) right away if this stream owes
  /// a stop (it drove since its last stop, or holds the seat). Also clears
  /// the re-engage latch: a stop is the input returning to neutral. Still
  /// writes after `dispose()`: a late stop from an unmounting input must
  /// reach the robot.
  stop(): void;
  /// Always writes (0, 0), owed or not. Unlike `stop()` it is a halt, not
  /// a release: if a drive cycle was open (or the latch was set) the held
  /// input must return to neutral before driving again.
  forceStop(): void;
  /// Stop (if owed) and, when an input is currently held, ignore non-zero
  /// twists until the input returns to neutral. For window blur, tab
  /// hidden, page hide and the firmware's operator-stale signal.
  requireNeutral(): void;
  /// The link dropped: nothing more can be written on this connection and
  /// the firmware has released the seat, so no stop is owed. A held input
  /// must return to neutral before it drives again.
  linkLost(): void;
  /// True while a held input must return to neutral before driving.
  needsNeutral(): boolean;
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
  /// A non-zero twist reached the socket since the last stop.
  let owesStop = false;
  /// Re-engage latch: non-zero twists are ignored until neutral.
  let latched = false;
  /// The last frame written was a stop and no non-zero twist went out
  /// since: a seat-held stop needn't be written again.
  let stopWritten = false;

  function setLatched(next: boolean): void {
    if (latched === next) return;
    latched = next;
    try {
      deps.onNeutralRequiredChange?.(next);
    } catch {
      /* a listener error must not break the stream */
    }
  }

  function seatHeld(): boolean {
    try {
      return deps.holdsSeat?.() ?? false;
    } catch {
      return false;
    }
  }

  function reportError(message: string): void {
    if (disposed) return;
    const now = deps.now();
    if (now - lastErrorAt < errorIntervalMs) return;
    lastErrorAt = now;
    deps.onError?.(message);
  }

  // A throwing transport must not take the keepalive down with it.
  // "closed" = the socket isn't open (send returned false); "threw" = a
  // transient failure worth retrying on the next tick.
  function write(
    vx: number,
    wz: number,
    onReject?: (message: string) => void,
  ): "ok" | "closed" | "threw" {
    try {
      return deps.send(vx, wz, onReject) ? "ok" : "closed";
    } catch {
      return "threw";
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
    const result = write(target.vx, target.wz, reportError);
    if (result === "closed") {
      // The link is down. Don't keep retrying the held twist: when the
      // socket comes back it would drive with no new operator action.
      linkLost();
      return;
    }
    if (result === "ok") {
      written = target;
      writtenAt = now;
      owesStop = true;
      stopWritten = false;
    }
    // After a write this is the next keepalive. After a send that threw
    // it is the retry.
    wakeAt(now + keepaliveMs, now);
  }

  function writeStop(): void {
    if (write(0, 0) === "ok") {
      writtenAt = deps.now();
      stopWritten = true;
    }
  }

  // End the drive cycle; write (0, 0) only if this stream owes it.
  function endCycle(): void {
    const owed = owesStop || (!stopWritten && seatHeld());
    target = null;
    written = null;
    owesStop = false;
    cancelTimer();
    if (owed) writeStop();
  }

  function stop(): void {
    endCycle();
    setLatched(false);
  }

  function requireNeutral(): void {
    const held = target !== null || latched;
    endCycle();
    setLatched(held && !disposed);
  }

  function linkLost(): void {
    const held = target !== null || latched;
    target = null;
    written = null;
    owesStop = false;
    // A new connection may be a new seat: its first seat-held stop counts.
    stopWritten = false;
    cancelTimer();
    if (held && !disposed) setLatched(true);
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
    if (disposed || latched) return;
    target = { vx: x, wz: z };
    pump();
  }

  return {
    set,
    stop,
    forceStop() {
      const held = target !== null || latched;
      target = null;
      written = null;
      owesStop = false;
      cancelTimer();
      writeStop();
      // A held input stays held after an emergency stop: it must be
      // released before the robot moves again.
      setLatched(held && !disposed);
    },
    requireNeutral,
    linkLost,
    needsNeutral: () => latched,
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
