"""Session ledger: one JSON row per mirrored drive session, joined with the
firmware's health log by time window.

    python tools/session_ledger.py                      # datasets/sessions/
    python tools/session_ledger.py DIR [--out FILE]
    python tools/session_ledger.py a.mcap b.mcap --system-dir DIR

For each ``navd_session_*.mcap`` the row carries the recording facts the data
plan asks for (docs/path-model.md, section 9): duration, per-topic counts,
the frame-set rate and its gaps, the camera gyro rate and its device-clock
gaps (the recorder-stall signature), moving / reverse fractions and distinct
turn events from the teleop twist, the odometry path length, and -- when the
recorder wrote them -- the ``/runtime`` count and the ``/intent``, ``/event``
and ``/place`` notes. The health join picks every ``system_*.mcap`` whose
time span overlaps the session (never the ``/session`` marker string: the
firmware mints its own stamp about a second after the recorder) and
summarises its ``/power`` and ``/host`` rows inside the window.

Message types that have no generated bindings here (``bebop.system.*``,
``bebop.navd.Intent`` and friends) are decoded from the schema embedded in
the MCAP, so the tool reads files from any recorder or firmware version.
Images are never decoded; the small channels are streamed, which still
reads the whole file once (chunks mix all topics), so expect a couple of
seconds per gigabyte from the page cache.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bebop_vision.mcap_tools import (ProtoDecoder, field_value,  # noqa: E402
                                     iter_messages, read_info,
                                     repeated_message_fields)

DEFAULT_DIR = Path(__file__).resolve().parents[1] / "datasets" / "sessions"
SESSION_PREFIX = "navd_session_"
SYSTEM_PREFIX = "system_"

FRAME_GAP_S = 0.15
GYRO_GAP_MS = 20.0
MOVING_VX = 0.02
MOVING_WZ = 0.05
REVERSE_VX = -0.02
TURN_WZ = 0.4
TURN_MIN_S = 0.5
TURN_MERGE_S = 1.0


# --- pure statistics (unit-tested) ------------------------------------------

def gap_stats(stamps, threshold):
    """(count, max) of consecutive differences above ``threshold``; stamps and
    threshold share a unit."""
    stamps = sorted(stamps)
    count, worst = 0, 0.0
    for a, b in zip(stamps, stamps[1:]):
        d = b - a
        if d > threshold:
            count += 1
            worst = max(worst, d)
    return count, worst


def rate_hz(stamps_s):
    """Effective rate over the whole span (gaps pull it down)."""
    stamps_s = sorted(stamps_s)
    if len(stamps_s) < 2:
        return 0.0
    span = stamps_s[-1] - stamps_s[0]
    return (len(stamps_s) - 1) / span if span > 0 else 0.0


def nominal_rate_hz(stamps_s):
    """Rate from the median interval: what the source runs at between gaps."""
    stamps_s = sorted(stamps_s)
    if len(stamps_s) < 2:
        return 0.0
    dts = sorted(b - a for a, b in zip(stamps_s, stamps_s[1:]) if b > a)
    if not dts:
        return 0.0
    return 1.0 / dts[len(dts) // 2]


def cmd_stats(stamps_s, vx, wz):
    """Moving / reverse fractions and distinct turn events from a twist
    series. A turn event is |wz| > TURN_WZ held for >= TURN_MIN_S; events of
    one sign closer than TURN_MERGE_S are one event."""
    n = len(vx)
    if n == 0:
        return {"moving_frac": 0.0, "reverse_frac": 0.0,
                "turn_events": 0, "turns_left": 0, "turns_right": 0}
    moving = sum(1 for a, b in zip(vx, wz)
                 if abs(a) > MOVING_VX or abs(b) > MOVING_WZ)
    reverse = sum(1 for a in vx if a < REVERSE_VX)
    events = []
    i = 0
    while i < n:
        if abs(wz[i]) > TURN_WZ:
            j = i
            while j < n and abs(wz[j]) > TURN_WZ and (wz[j] > 0) == (wz[i] > 0):
                j += 1
            if stamps_s[j - 1] - stamps_s[i] >= TURN_MIN_S:
                sign = 1 if wz[i] > 0 else -1
                if events and sign == events[-1][1] and stamps_s[i] - events[-1][2] < TURN_MERGE_S:
                    events[-1] = (events[-1][0], sign, stamps_s[j - 1])
                else:
                    events.append((stamps_s[i], sign, stamps_s[j - 1]))
            i = j
        else:
            i += 1
    return {"moving_frac": round(moving / n, 4),
            "reverse_frac": round(reverse / n, 4),
            "turn_events": len(events),
            "turns_left": sum(1 for e in events if e[1] > 0),
            "turns_right": sum(1 for e in events if e[1] < 0)}


def path_length(xs, ys):
    return float(sum(math.hypot(b[0] - a[0], b[1] - a[1])
                     for a, b in zip(zip(xs, ys), zip(xs[1:], ys[1:]))))


def overlaps(a_start, a_end, b_start, b_end):
    return (a_start is not None and a_end is not None and b_start is not None
            and b_end is not None and a_start <= b_end and b_start <= a_end)


# --- per-file analysis ------------------------------------------------------

NOTE_TOPICS = ("/intent", "/event", "/place")


def analyse_session(path, decoder=None):
    decoder = decoder or ProtoDecoder()
    t0 = time.monotonic()
    info = read_info(path)
    row = {
        "name": info.path.name,
        "path": str(info.path),
        "size_bytes": info.size_bytes,
        "footerless": info.footerless,
        "start_ns": info.start_ns,
        "end_ns": info.end_ns,
        "duration_s": round(info.duration_s, 3),
        "messages": info.message_count,
        "topics": dict(sorted(info.message_counts.items())),
    }
    wanted = [t for t in ("/color_near", "/imu_gyro_near", "/cmd_vel", "/odom",
                          *NOTE_TOPICS)
              if t in info.message_counts]
    color_t, gyro_dev_us, cmd, odom = [], [], [], []
    intents, events, places = [], [], []
    for schema, channel, msg in iter_messages(path, wanted, footerless=info.footerless):
        topic = channel.topic
        if topic == "/color_near":
            color_t.append(msg.log_time / 1e9)
            continue
        if schema is None or schema.encoding != "protobuf":
            continue
        m = decoder.decode(schema, msg.data)
        if topic == "/imu_gyro_near":
            gyro_dev_us.append(int(field_value(m, "device_stamp_us", 0)))
        elif topic == "/cmd_vel":
            cmd.append((msg.log_time / 1e9, float(field_value(m, "vx", 0.0)),
                        float(field_value(m, "wz", 0.0))))
        elif topic == "/odom":
            odom.append((float(field_value(m, "x", 0.0)), float(field_value(m, "y", 0.0))))
        elif topic == "/intent":
            intents.append({"text": field_value(m, "text", ""),
                            "from_place": field_value(m, "from_place", ""),
                            "to_place": field_value(m, "to_place", ""),
                            "stamp_ns": int(field_value(m, "stamp_ns", msg.log_time))})
        elif topic == "/event":
            events.append({"kind": field_value(m, "kind", ""),
                           "note": field_value(m, "note", ""),
                           "stamp_ns": int(field_value(m, "stamp_ns", msg.log_time))})
        elif topic == "/place":
            name = field_value(m, "name", "") or field_value(m, "place_id", "")
            stamp = int(field_value(m, "stamp_ns", msg.log_time))
            if places and places[-1]["name"] == name:
                places[-1]["last_ns"] = stamp
                places[-1]["sightings"] += 1
            else:
                places.append({"name": name, "place_id": field_value(m, "place_id", ""),
                               "score": round(float(field_value(m, "score", 0.0)), 4),
                               "first_ns": stamp, "last_ns": stamp, "sightings": 1})
    gaps, gap_max = gap_stats(color_t, FRAME_GAP_S)
    g_gaps, g_max_us = gap_stats(gyro_dev_us, GYRO_GAP_MS * 1000.0)
    row.update({
        "ticks": len(color_t),
        "tick_hz": round(rate_hz(color_t), 2),
        "frame_gaps": gaps, "frame_gap_max_s": round(gap_max, 3),
        "gyro_hz": round(nominal_rate_hz([u / 1e6 for u in gyro_dev_us]), 1),
        "gyro_gaps": g_gaps, "gyro_gap_max_ms": round(g_max_us / 1000.0, 1),
        "cmd_samples": len(cmd),
        "odom_path_m": round(path_length([o[0] for o in odom], [o[1] for o in odom]), 3),
        "runtime_messages": int(info.message_counts.get("/runtime", 0)),
        # Every /intent (clears included) in order; `intent` is the last
        # non-empty one, so clearing it before disarming keeps the mission.
        "intents": intents,
        "intent": next((i for i in reversed(intents)
                        if i["text"] or i["from_place"] or i["to_place"]), None),
        "events": events, "places": places,
    })
    row.update(cmd_stats([c[0] for c in cmd], [c[1] for c in cmd], [c[2] for c in cmd]))
    row["analysis_ms"] = round((time.monotonic() - t0) * 1000.0, 1)
    return row


def summarise_health(system_files, start_ns, end_ns, decoder=None):
    """Power/host statistics from every system file overlapping the window."""
    decoder = decoder or ProtoDecoder()
    joined, power, host = [], [], []
    for info in system_files:
        if not overlaps(start_ns, end_ns, info.start_ns, info.end_ns):
            continue
        joined.append(info.path.name)
        for schema, channel, msg in iter_messages(
                info.path, ["/power", "/host"], start_ns=start_ns,
                end_ns=end_ns + 1, footerless=info.footerless):
            if schema is None or schema.encoding != "protobuf":
                continue
            m = decoder.decode(schema, msg.data)
            (power if channel.topic == "/power" else host).append(m)
    out = {"files": joined, "rows_power": len(power), "rows_host": len(host)}
    # Placeholder rows (no power board, no status yet, stale cache) carry
    # zeros, not measurements.  Missing flags (older schemas) count as valid.
    power = [m for m in power if field_value(m, "present", True)
             and field_value(m, "status_received", True)
             and not field_value(m, "status_stale", False)]
    if power:
        volts = [float(field_value(m, "battery_voltage_v", 0.0)) for m in power]
        soc = [s for s in (float(field_value(m, "state_of_charge_pct", -1.0))
                           for m in power) if s >= 0.0]          # -1 = unknown
        out.update({
            "battery_v_min": round(min(volts), 2), "battery_v_max": round(max(volts), 2),
            "soc_first": round(soc[0], 1) if soc else None,
            "soc_last": round(soc[-1], 1) if soc else None,
            "board_temp_c_max": round(max(float(field_value(m, "board_temperature_c", 0.0))
                                          for m in power), 1),
            # Bits 20..23 are rail on/off flags, not faults (the firmware's
            # powerboard::describe_faults masks 0x000F_FFFF the same way).
            "fault_any": any(int(field_value(m, "fault_bits", 0)) & 0x000F_FFFF
                             for m in power),
            "faults": sorted({str(field_value(m, "fault_description", ""))
                              for m in power} - {"", "normal"}),
        })
    if host:
        cpu = [float(field_value(m, "cpu_pct", 0.0)) for m in host]
        gpu = [float(field_value(m, "gpu_pct", 0.0)) for m in host
               if field_value(m, "gpu_present", True)]
        hottest = None
        for m in host:
            for name in repeated_message_fields(m):
                for z in getattr(m, name):
                    temp = field_value(z, "temp_c", None)
                    if temp is None:
                        continue
                    if hottest is None or float(temp) > hottest["temp_c"]:
                        hottest = {"name": field_value(z, "name", ""), "temp_c": round(float(temp), 1)}
        out.update({
            "cpu_pct_mean": round(sum(cpu) / len(cpu), 1), "cpu_pct_max": round(max(cpu), 1),
            "gpu_pct_mean": round(sum(gpu) / len(gpu), 1) if gpu else None,
            "gpu_pct_max": round(max(gpu), 1) if gpu else None,
            "hottest_zone": hottest,
            "disk_used_pct_last": round(float(field_value(host[-1], "disk_used_pct", 0.0)), 1),
        })
    return out


def find_files(targets, system_dir=None):
    """(session paths, system infos) from a mix of directories and files."""
    sessions, system_dirs = [], set()
    for t in targets:
        p = Path(t)
        if p.is_dir():
            sessions += sorted(p.glob(f"{SESSION_PREFIX}*.mcap"))
            system_dirs.add(p)
        elif p.is_file():
            sessions.append(p)
            system_dirs.add(p.parent)
        else:
            raise SystemExit(f"no such file or directory: {p}")
    if system_dir is not None:
        system_dirs = {Path(system_dir)}
    system_paths = sorted({q for d in system_dirs for q in d.glob(f"{SYSTEM_PREFIX}*.mcap")})
    return sessions, system_paths


def build_ledger(targets, system_dir=None, log=print):
    sessions, system_paths = find_files(targets, system_dir)
    decoder = ProtoDecoder()
    system_infos = [read_info(p) for p in system_paths]
    rows = []
    for path in sessions:
        t0 = time.monotonic()
        row = analyse_session(path, decoder)
        row["health"] = summarise_health(system_infos, row["start_ns"], row["end_ns"], decoder)
        row["ledger_ms"] = round((time.monotonic() - t0) * 1000.0, 1)
        rows.append(row)
        log(format_row(row))
    return rows


def format_header():
    return ("session                            dur_s ticks  Hz  gaps(max s)  gyro_hz ggaps(max ms)"
            "  moving reverse turns(L/R) path_m  intent               places  batt_V       ms")


def format_row(r):
    h = r.get("health") or {}
    batt = (f"{h['battery_v_min']:.1f}-{h['battery_v_max']:.1f}"
            if "battery_v_min" in h else "-")
    intent = (r["intent"] or {}).get("text") or "-"
    places = ",".join(p["name"] for p in r["places"]) or "-"
    return (f"{r['name'][:34]:34} {r['duration_s']:6.1f} {r['ticks']:5d} {r['tick_hz']:4.1f}"
            f"  {r['frame_gaps']:3d} ({r['frame_gap_max_s']:4.2f})   {r['gyro_hz']:6.1f}"
            f"  {r['gyro_gaps']:3d} ({r['gyro_gap_max_ms']:5.0f})   {r['moving_frac']:5.2f}"
            f"   {r['reverse_frac']:5.2f}   {r['turn_events']:3d} ({r['turns_left']}/{r['turns_right']})"
            f" {r['odom_path_m']:6.1f}  {intent[:20]:20} {places[:7]:7} {batt:11} {r.get('ledger_ms', r['analysis_ms']):7.0f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("targets", nargs="*", default=[str(DEFAULT_DIR)],
                    help="directories of mirrored MCAPs and/or session files "
                         "(default: datasets/sessions)")
    ap.add_argument("--system-dir", default=None,
                    help="where the system_*.mcap health logs live "
                         "(default: next to each session)")
    ap.add_argument("--out", default=None,
                    help="ledger path (default: <first dir>/sessions.jsonl)")
    args = ap.parse_args(argv)
    print(format_header())
    rows = build_ledger(args.targets, args.system_dir)
    first = Path(args.targets[0])
    out = Path(args.out) if args.out else (first if first.is_dir() else first.parent) / "sessions.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    total = sum(r["duration_s"] for r in rows)
    print(f"{len(rows)} session(s), {total:.0f} s of recording -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
