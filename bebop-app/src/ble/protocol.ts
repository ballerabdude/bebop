// Shared constants that mirror the Rust agent's BLE protocol.
// Keep in sync with:
//   bebop-agent/src/ble/uuids.rs
//   bebop-agent/src/ble/framing.rs

export const SERVICE_UUID = "b3b0b000-0b3b-4f9b-9b3b-b3b0b3b0b3b0";
export const CHAR_REQUEST_UUID = "b3b0b001-0b3b-4f9b-9b3b-b3b0b3b0b3b0";
export const CHAR_RESPONSE_UUID = "b3b0b002-0b3b-4f9b-9b3b-b3b0b3b0b3b0";
export const CHAR_STATUS_UUID = "b3b0b003-0b3b-4f9b-9b3b-b3b0b3b0b3b0";

// Framing: 1-byte version, 1-byte flags, 2-byte big-endian length, then payload.
export const PROTO_VERSION = 1;
export const FLAG_FRAGMENT = 1 << 0;
export const FLAG_FINAL = 1 << 1;

export function encodeFrames(
  payload: Uint8Array,
  maxPayloadPerFrame: number,
): Uint8Array[] {
  if (payload.length === 0) {
    return [makeFrame(new Uint8Array(0), FLAG_FINAL)];
  }
  const frames: Uint8Array[] = [];
  let offset = 0;
  while (offset < payload.length) {
    const take = Math.min(maxPayloadPerFrame, payload.length - offset);
    const chunk = payload.subarray(offset, offset + take);
    offset += take;
    const flags = offset >= payload.length ? FLAG_FINAL : FLAG_FRAGMENT;
    frames.push(makeFrame(chunk, flags));
  }
  return frames;
}

function makeFrame(payload: Uint8Array, flags: number): Uint8Array {
  const frame = new Uint8Array(4 + payload.length);
  frame[0] = PROTO_VERSION;
  frame[1] = flags;
  frame[2] = (payload.length >> 8) & 0xff;
  frame[3] = payload.length & 0xff;
  frame.set(payload, 4);
  return frame;
}

export class Reassembler {
  private chunks: Uint8Array[] = [];
  private length = 0;

  push(frame: Uint8Array): Uint8Array | null {
    if (frame.length < 4) throw new Error("frame too short");
    const ver = frame[0];
    if (ver !== PROTO_VERSION) {
      throw new Error(`unsupported frame version ${ver}`);
    }
    const flags = frame[1];
    const len = (frame[2] << 8) | frame[3];
    if (frame.length < 4 + len) throw new Error("frame payload truncated");
    const payload = frame.subarray(4, 4 + len);
    this.chunks.push(new Uint8Array(payload));
    this.length += payload.length;

    if ((flags & FLAG_FINAL) !== 0) {
      const out = new Uint8Array(this.length);
      let offset = 0;
      for (const c of this.chunks) {
        out.set(c, offset);
        offset += c.length;
      }
      this.chunks = [];
      this.length = 0;
      return out;
    }
    return null;
  }
}
