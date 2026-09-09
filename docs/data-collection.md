# Data collection protocol (navd sessions)

How to turn robot drive time into training data, and how to know a
session is good. The pipeline: record on the robot → pull → extract →
label (SAM × depth) → review → count it.

## Why protocol matters

The first 1,624 recorded ticks had **zero active goals** — every tick
said "no goal". Any model that must follow waypoints (or imitate
goal-seeking) learns nothing from goal-less data; imitation can only
reproduce what the data contains. Goal-conditioned legs are the
cheapest, highest-value kind of tick.

## Per-session checklist

Before:
- [ ] Jetson disk has headroom (`df -h /`; segments auto-prune at 20 GB)
- [ ] App connected, estop clear, wheels enable cleanly
- [ ] Know what this session is *for*: one line in the ledger

During (aim for 5–10 minutes of recorded drive time):
- [ ] **Goal-conditioned legs** — set a Navigate waypoint, teleop toward
      it (or drive manually and let it correct), `Clear`, repeat. Target
      ≥10 legs per session, mixed straight/turning, 0.5–3 m.
- [ ] Obstacle variety: drive near real clutter, around it, past it —
      the model only sees what you show it.
- [ ] Lighting variety when possible (day, evening, lights off).
- [ ] Speed range: slow approaches, normal cruising, one decisive stop.
- [ ] A few near-misses (you correct late) — those ticks are gold.
- [ ] Note anything unusual (moved furniture, new obstacle) in the ledger.

After:
- [ ] Stop the recorder cleanly (Ctrl-C, or
      `sudo pkill -TERM -f 'record-nav[d]'` from a separate shell)
- [ ] `just pull-data` (mirrors to `datasets/sessions/`, extracts,
      audits) — check the audit line: 100% color, sensible depth medians
- [ ] Dashboard review pass (`just review`): sample ~20 ticks/session,
      paint corrections only where the fused teacher is actually wrong
- [ ] Ledger entry: session name, date, conditions, tick count, verdict

## Ledger

`docs/data-ledger.md` — one row per session: name, date, place, goals
(# legs), conditions, ticks, review status, notes. When evaluating a
navigation approach, the ledger tells you which questions the data can
answer — and which it can't.

## Commands

```sh
just collect     # ssh to the robot, start the recorder (--auto)
just pull-data   # mirror sessions -> extract -> audit
just review      # dashboard on :8099 (session replay + corrections)
```

Sessions land in `bebop-vision/datasets/sessions/` (extracted under
`datasets/navd-v0/<session>/`). The Jetson keeps only the newest ~20 GB
(`--disk-budget-gb`); pull before it prunes.

## What good coverage looks like

| dimension | minimum | good |
|---|---|---|
| sessions | 5 | 15+ |
| drive time | 15 min | 60+ min |
| goal-conditioned legs | 30 | 150+ |
| obstacle layouts | 3 | 10+ |
| lighting conditions | 2 | 4+ |
| rooms/areas | 2 | every drivable area |

These thresholds are guidance, not gospel — the ledger's job is to make
the gaps visible, not to gate on numbers alone.
