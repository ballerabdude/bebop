// Turn the firmware's E-STOP reason (`BreachReason::human()` in
// firmware/bebop-linux/src/safety/limits.rs) into a short operator line:
//
//   "right: motor fault 0x00 (DC_BUS_OVER_VOLTAGE (0x00000100))"
//     -> "Right motor fault: DC bus over-voltage"
//   "operator: operator" -> "Stopped by an operator"
//
// Unknown shapes pass through with the first letter capitalized, so nothing
// the firmware says is ever hidden.

const ACRONYMS = new Set(["DC", "CAN", "AC", "MOS", "ADC", "IMU", "SPI", "UART", "PWM"]);

function capitalize(s: string): string {
  return s ? s[0].toUpperCase() + s.slice(1) : s;
}

/// "DC_BUS_OVER_VOLTAGE" -> "DC bus over-voltage".
export function humanizeFlag(flag: string): string {
  const parts = flag.trim().split(/_+/).filter(Boolean);
  const out: string[] = [];
  for (let i = 0; i < parts.length; i += 1) {
    const p = parts[i];
    const word = ACRONYMS.has(p) ? p : p.toLowerCase();
    if ((word === "over" || word === "under") && i + 1 < parts.length) {
      const next = parts[i + 1];
      out.push(`${word}-${ACRONYMS.has(next) ? next : next.toLowerCase()}`);
      i += 1;
      continue;
    }
    out.push(word);
  }
  return out.join(" ");
}

function humanizeFlags(desc: string): string {
  const cleaned = desc.replace(/\s*\(0x[0-9a-f]+\)/gi, "").trim();
  const flags = cleaned
    .split(/\s*[|,+]\s*/)
    .map((f) => f.trim())
    .filter(Boolean);
  if (flags.length === 0) return cleaned;
  return flags
    .map((f) => (/^[A-Z0-9_]+$/.test(f) ? humanizeFlag(f) : f))
    .join(", ");
}

export function humanizeEStopReason(raw: string): string {
  const r = raw.trim();
  if (!r) return "";
  const op = /^operator:\s*(.*)$/i.exec(r);
  if (op) {
    const why = op[1].trim();
    return !why || why.toLowerCase() === "operator"
      ? "Stopped by an operator"
      : `Stopped by an operator (${why})`;
  }
  const fault = /^(.+?):\s*motor fault 0x[0-9a-f]+\s*\((.*)\)$/i.exec(r);
  if (fault) {
    const flags = humanizeFlags(fault[2]);
    return `${capitalize(fault[1])} motor fault${flags ? `: ${flags}` : ""}`;
  }
  return capitalize(r);
}
