#!/usr/bin/env python3
"""Generate Foxglove Studio layouts (.json) for Bebop MCAP review.

Each layout lives in its own module:
  - layout_robot.py  -> bebop_robot_layout.json
  - layout_noise.py  -> bebop_noise_layout.json

Import the emitted file via Foxglove's layouts sidebar -> "Import from
file". Works in both the desktop app and app.foxglove.dev.

Usage:
    python3 make_foxglove_layout.py --layout robot
    python3 make_foxglove_layout.py --layout noise --out my_noise_layout.json
    python3 make_foxglove_layout.py --layout all --force

    python3 layout_robot.py
    python3 layout_noise.py --force
"""
import argparse
import json
import os
import sys

# Allow imports when run from the repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import layout_noise
import layout_robot

LAYOUTS = {
    layout_robot.NAME: (layout_robot.OUTPUT, layout_robot.build),
    layout_noise.NAME: (layout_noise.OUTPUT, layout_noise.build),
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate one or more Foxglove layout JSON files."
    )
    parser.add_argument(
        "--layout",
        choices=[*LAYOUTS.keys(), "all"],
        default="robot",
        help="Layout to generate. Defaults to robot.",
    )
    parser.add_argument(
        "--out",
        help="Output JSON path. Only valid when generating a single layout.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing layout file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.layout == "all":
        if args.out:
            raise SystemExit("--out can only be used with a single layout")
        for name, (filename, _) in LAYOUTS.items():
            write_layout(name, os.path.join(script_dir, filename), force=args.force)
        return

    default_filename, _ = LAYOUTS[args.layout]
    out = args.out or os.path.join(script_dir, default_filename)
    write_layout(args.layout, out, force=args.force)


if __name__ == "__main__":
    main()
