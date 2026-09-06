#!/usr/bin/env python3
"""Generate Foxglove Studio layouts (.json) for Bebop ROS2 MCAP review.

Import the emitted file via Foxglove's Layouts -> Import from file.

Usage:
    python3 make_foxglove_layout.py --layout robot
    python3 make_foxglove_layout.py --layout noise --out my.json
    python3 make_foxglove_layout.py --layout all --force
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import layout_noise, layout_policy, layout_robot, layout_navd

LAYOUTS = {
    layout_robot.NAME: (layout_robot.OUTPUT, layout_robot.build),
    layout_noise.NAME: (layout_noise.OUTPUT, layout_noise.build),
    layout_policy.NAME: (layout_policy.OUTPUT, layout_policy.build),
    layout_navd.NAME: (layout_navd.OUTPUT, layout_navd.build),
}

def write_layout(name, out, *, force):
    if os.path.exists(out) and not force:
        print(f"skip {out} (exists; pass --force to overwrite)")
        return
    _, builder = LAYOUTS[name]
    with open(out, "w") as f:
        json.dump(builder(), f, indent=2)
        f.write("\n")
    print(f"wrote {out}")

def main():
    p = argparse.ArgumentParser(description="Generate Foxglove layout JSON")
    p.add_argument("--layout", choices=[*LAYOUTS.keys(), "all"], default="robot")
    p.add_argument("--out")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    d = os.path.dirname(os.path.abspath(__file__))

    if args.layout == "all":
        if args.out: raise SystemExit("--out only for single layout")
        for name, (fname, _) in LAYOUTS.items():
            write_layout(name, os.path.join(d, fname), force=args.force)
        return

    default_fn, _ = LAYOUTS[args.layout]
    out = args.out or os.path.join(d, default_fn)
    write_layout(args.layout, out, force=args.force)

if __name__ == "__main__":
    main()
