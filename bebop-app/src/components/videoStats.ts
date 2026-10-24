// WebRTC receive-side stats for one VideoFeed (RTCPeerConnection.getStats()).
//
// `sampleFromReport` pulls the cumulative counters out of a stats report
// (the inbound video RTP stream plus the selected candidate pair);
// `summarizeVideoStats` turns two consecutive samples into rates the HUD
// can show: fps, bitrate, packet loss over the interval, jitter and a
// glass-to-glass-ish latency estimate (half the RTT + the jitter-buffer
// delay + decode time; capture/encode on the robot is not visible here).
//
// Pure TS so it runs under `node --test`.

export interface VideoStatsSample {
  /// ms timestamp of the sample (report timestamp or performance.now()).
  t: number;
  framesDecoded: number;
  framesDropped: number;
  framesPerSecond: number | null;
  packetsReceived: number;
  packetsLost: number;
  bytesReceived: number;
  /// Seconds (RTP interarrival jitter estimate).
  jitter: number | null;
  /// Cumulative seconds / count (jitter buffer).
  jitterBufferDelay: number;
  jitterBufferEmittedCount: number;
  /// Cumulative seconds.
  totalDecodeTime: number;
  /// Seconds, selected candidate pair.
  rtt: number | null;
  frameWidth: number;
  frameHeight: number;
}

export interface VideoStats {
  fps: number | null;
  kbps: number | null;
  /// Fraction 0..1 of packets lost over the last interval.
  loss: number | null;
  jitterMs: number | null;
  rttMs: number | null;
  jitterBufferMs: number | null;
  decodeMs: number | null;
  /// rtt/2 + jitter buffer + decode, ms; null until enough is known.
  latencyMs: number | null;
  width: number;
  height: number;
  framesDropped: number;
}

type StatLike = Record<string, unknown> & { type?: string; id?: string };

const n = (v: unknown): number => (typeof v === "number" && Number.isFinite(v) ? v : 0);
const nn = (v: unknown): number | null =>
  typeof v === "number" && Number.isFinite(v) ? v : null;

/// Accepts an RTCStatsReport (Map-like) or any iterable of stat objects.
export function sampleFromReport(
  report: { forEach: (cb: (s: StatLike) => void) => void },
  now: number,
): VideoStatsSample | null {
  let inbound: StatLike | null = null;
  let pair: StatLike | null = null;
  let transportPairId: string | null = null;
  const pairs = new Map<string, StatLike>();
  report.forEach((s) => {
    if (s.type === "inbound-rtp" && (s.kind === "video" || s.mediaType === "video")) {
      inbound = s;
    } else if (s.type === "candidate-pair") {
      if (typeof s.id === "string") pairs.set(s.id, s);
      if (s.selected === true || (s.nominated === true && s.state === "succeeded")) pair = pair ?? s;
    } else if (s.type === "transport" && typeof s.selectedCandidatePairId === "string") {
      transportPairId = s.selectedCandidatePairId;
    }
  });
  if (transportPairId && pairs.has(transportPairId)) pair = pairs.get(transportPairId)!;
  if (!inbound) return null;
  const i = inbound as StatLike;
  const p = pair as StatLike | null;
  return {
    t: nn(i.timestamp) ?? now,
    framesDecoded: n(i.framesDecoded),
    framesDropped: n(i.framesDropped),
    framesPerSecond: nn(i.framesPerSecond),
    packetsReceived: n(i.packetsReceived),
    packetsLost: Math.max(0, n(i.packetsLost)),
    bytesReceived: n(i.bytesReceived),
    jitter: nn(i.jitter),
    jitterBufferDelay: n(i.jitterBufferDelay),
    jitterBufferEmittedCount: n(i.jitterBufferEmittedCount),
    totalDecodeTime: n(i.totalDecodeTime),
    rtt: p ? nn(p.currentRoundTripTime) : null,
    frameWidth: n(i.frameWidth),
    frameHeight: n(i.frameHeight),
  };
}

export function summarizeVideoStats(
  prev: VideoStatsSample | null,
  cur: VideoStatsSample,
): VideoStats {
  const dt = prev ? (cur.t - prev.t) / 1000 : 0;
  const ok = prev !== null && dt > 0;
  const dFrames = ok ? cur.framesDecoded - prev!.framesDecoded : 0;
  const dRecv = ok ? cur.packetsReceived - prev!.packetsReceived : 0;
  const dLost = ok ? cur.packetsLost - prev!.packetsLost : 0;
  const dBytes = ok ? cur.bytesReceived - prev!.bytesReceived : 0;
  const dJb = ok ? cur.jitterBufferDelay - prev!.jitterBufferDelay : 0;
  const dJbN = ok ? cur.jitterBufferEmittedCount - prev!.jitterBufferEmittedCount : 0;
  const dDec = ok ? cur.totalDecodeTime - prev!.totalDecodeTime : 0;

  const fps =
    cur.framesPerSecond !== null
      ? cur.framesPerSecond
      : ok && dFrames >= 0
        ? dFrames / dt
        : null;
  const kbps = ok && dBytes >= 0 ? (dBytes * 8) / 1000 / dt : null;
  const loss = ok && dRecv + dLost > 0 && dLost >= 0 ? dLost / (dRecv + Math.max(0, dLost)) : ok ? 0 : null;
  const jitterBufferMs = ok && dJbN > 0 ? (dJb / dJbN) * 1000 : null;
  const decodeMs = ok && dFrames > 0 && dDec >= 0 ? (dDec / dFrames) * 1000 : null;
  const rttMs = cur.rtt !== null ? cur.rtt * 1000 : null;
  const latencyMs =
    jitterBufferMs !== null
      ? (rttMs ?? 0) / 2 + jitterBufferMs + (decodeMs ?? 0)
      : null;
  return {
    fps,
    kbps,
    loss,
    jitterMs: cur.jitter !== null ? cur.jitter * 1000 : null,
    rttMs,
    jitterBufferMs,
    decodeMs,
    latencyMs,
    width: cur.frameWidth,
    height: cur.frameHeight,
    framesDropped: cur.framesDropped,
  };
}
