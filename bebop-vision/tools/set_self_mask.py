"""Draw the chassis self-mask on a live view — prints the YAML snippet.

Runs on the workstation (no camera access): fetches MJPEG snapshots from
the bebop-vision videoserver (:9092), shows the UNMASKED color feed plus
the (already-masked) depth render, lets you freeze a frame and draw the
chassis shape on the color view, converts it into DEPTH-frame pixel
coords (the units rig YAML `self_mask_pixels` expects — orbbec.py zeroes
those pixels of the 848x480 depth frame; the original color-space rects
are the historical bug), and prints the snippet to paste under the
camera in config/orbbec_rig.yaml.

The default draw mode is a POLYGON (click the vertices): the chassis is
a trapezoid in the image, so a tight polygon masks far less floor than
an axis-aligned rect. Pass --rect for the old drag-a-rectangle flow.
Config accepts both forms per entry:
    - [336, 228, 560, 480]              # rect [x0, y0, x1, y1]
    - [[540, 375], [870, 377], ...]     # polygon vertices

Safe to run any time, including mid-recording: it only reads HTTP from
the videoserver, it never opens a camera. Bebop-vision must be running
(main.py --record-navd / --goal-drive).

Usage (workstation, from bebop-vision/):
    .venv/bin/python tools/set_self_mask.py
    .venv/bin/python tools/set_self_mask.py --host http://192.168.0.69:9092

Keys (live view):
    SPACE/c   freeze the frame, then draw
    x         cycle color->depth mapping mode (see below)
    q/ESC     quit
Keys (draw view, polygon mode):
    click     add vertex      Backspace/u  undo vertex
    ENTER/n   close polygon (>= 3 vertices)
    ESC/c     cancel, back to live
Keys (verify view):
    y/ENTER   accept — print the final YAML snippet and exit
    r         redraw
    x         re-convert under the next mapping mode
    l         back to the live view
    q/ESC     quit

The verify view draws your exact drawing on the color window (green) and
the converted shape — what actually lands in the config — on the depth
window (red). The magenta shape is the CURRENT config mask (projected
into the color view / native on depth).

Mapping modes ('x' cycles). The magenta shape on the color window is the
CURRENT config mask projected through the active mode — when the mode is
right it brackets the chassis:
    extrinsic   intrinsics + the camera's color->depth extrinsic at the
                nominal range --range (default 0.7 m) — correct default
    no-trans    rotation only (extrinsic without the 23.6 mm baseline)
    scale       intrinsics-only scaling (the fuse_navd_labels.py
                convention; lands ~20 px off here)
"""

import argparse
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

MODES = ("extrinsic", "no-trans", "scale")

DEPTH_SIZE = (848, 480)      # (w, h) — the frame orbbec.py zeroes into
COLOR_SIZE = (1280, 800)     # near camera color frame


def fetch_snapshot(host, stream, timeout=3.0):
    """GET /snapshot?stream=<name> -> decoded BGR image (or None)."""
    try:
        with urllib.request.urlopen(
                f"{host}/snapshot?stream={stream}", timeout=timeout) as r:
            data = r.read()
    except Exception:
        return None
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def check_server(host, timeout=3.0):
    try:
        with urllib.request.urlopen(f"{host}/healthz", timeout=timeout) as r:
            return r.read().strip() == b"ok"
    except Exception:
        return False


def load_intrinsics(path):
    with open(path) as f:
        return json.load(f)


def load_current_shapes(rig_path, serial):
    """Raw self_mask_pixels entries for `serial` from the rig YAML."""
    try:
        import yaml
    except ImportError:
        return None
    try:
        cfg = yaml.safe_load(open(rig_path))
    except FileNotFoundError:
        return None
    cam = cfg["robots"]["default"]["cameras"].get(serial, {})
    return cam.get("self_mask_pixels") or None


def shape_kind(entry):
    """'rect' (four scalars) or 'poly' (list of [x, y] pairs) — mirrors
    bebop_vision.orbbec.parse_self_mask_entries."""
    scalars = [v for v in entry if isinstance(v, (int, float))]
    return "rect" if len(entry) == 4 and len(scalars) == 4 else "poly"


def _extrinsics(intr, mode):
    if mode not in MODES:
        raise ValueError(f"unknown mapping mode {mode!r}")
    if mode == "scale":
        return None, None
    R = np.asarray(intr["color_to_depth_transform"]["rotation"],
                   np.float64).reshape(3, 3)
    t = None
    if mode == "extrinsic":
        t = np.asarray(intr["color_to_depth_transform"]["translation"],
                       np.float64) / 1000.0   # SDK translation is in mm
    return R, t


def color_to_depth_point(u, v, z_m, intr, mode):
    """Color pixel on the optical ray at range z_m -> depth pixel."""
    R, t = _extrinsics(intr, mode)
    p = np.array([(u - intr["color_cx"]) / intr["color_fx"] * z_m,
                  (v - intr["color_cy"]) / intr["color_fy"] * z_m,
                  z_m])
    if R is not None:
        p = R @ p + (t if t is not None else 0.0)
    if p[2] <= 1e-6:
        return None
    return (p[0] / p[2] * intr["fx"] + intr["cx"],
            p[1] / p[2] * intr["fy"] + intr["cy"])


def depth_to_color_point(u_d, v_d, z_m, intr, mode):
    """Depth pixel on the optical ray at range z_m -> color pixel."""
    R, t = _extrinsics(intr, mode)
    p = np.array([(u_d - intr["cx"]) / intr["fx"] * z_m,
                  (v_d - intr["cy"]) / intr["fy"] * z_m,
                  z_m])
    if R is not None:
        p = R.T @ (p - t) if t is not None else R.T @ p
    if p[2] <= 1e-6:
        return None
    return (p[0] / p[2] * intr["color_fx"] + intr["color_cx"],
            p[1] / p[2] * intr["color_fy"] + intr["color_cy"])


def _clamp_px(p, size):
    w, h = size
    return (min(w - 1, max(0, int(round(p[0])))),
            min(h - 1, max(0, int(round(p[1])))))


def roi_to_depth_rect(x, y, w, h, z_m, intr, mode, depth_size=DEPTH_SIZE):
    """selectROI (x, y, w, h) in color px -> [x0, y0, x1, y1] depth px.

    Maps the four corners along optical rays at z_m, clamps to the depth
    frame, and rounds outward (floor/ceil) so the mask errs on covering.
    """
    dw, dh = depth_size
    corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    pts = [color_to_depth_point(u, v, z_m, intr, mode) for u, v in corners]
    if any(p is None for p in pts):
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [max(0, int(math.floor(min(xs)))), max(0, int(math.floor(min(ys)))),
            min(dw, int(math.ceil(max(xs)))), min(dh, int(math.ceil(max(ys))))]


def poly_to_depth_poly(points, z_m, intr, mode, depth_size=DEPTH_SIZE):
    """Full-res color [[x, y], ...] polygon -> clamped depth polygon."""
    out = []
    for u, v in points:
        p = color_to_depth_point(u, v, z_m, intr, mode)
        if p is None:
            return None
        x, y = _clamp_px(p, depth_size)
        out.append([x, y])
    return out


def depth_rect_to_poly(rect, z_m, intr, mode, color_size=COLOR_SIZE):
    """Depth rect -> int corner polygon in color px (validation overlay)."""
    cw, ch = color_size
    corners = [(rect[0], rect[1]), (rect[2], rect[1]),
               (rect[2], rect[3]), (rect[0], rect[3])]
    pts = [depth_to_color_point(u, v, z_m, intr, mode) for u, v in corners]
    if any(p is None for p in pts):
        return None
    return [_clamp_px(p, color_size) for p in pts]


def depth_poly_to_color_poly(poly, z_m, intr, mode, color_size=COLOR_SIZE):
    """Depth polygon -> color polygon (validation overlay)."""
    out = []
    for u, v in poly:
        p = depth_to_color_point(u, v, z_m, intr, mode)
        if p is None:
            return None
        out.append(_clamp_px(p, color_size))
    return out


def yml_snippet(shape, z_m, mode, serial):
    kind, pts = shape
    if kind == "rect":
        item = f"[{pts[0]}, {pts[1]}, {pts[2]}, {pts[3]}]"
    else:
        item = "[" + ", ".join(f"[{x}, {y}]" for x, y in pts) + "]"
    kind_note = ("rect [x0, y0, x1, y1]" if kind == "rect"
                 else "polygon vertices")
    return (
        f"Paste under the {serial} camera in bebop-vision/config/orbbec_rig.yaml:\n"
        f"\n"
        f"        self_mask_pixels:\n"
        f"          - {item}\n"
        f"\n"
        f"(depth-frame {DEPTH_SIZE[0]}x{DEPTH_SIZE[1]} px, {kind_note};\n"
        f"orbbec.py zeroes those pixels; drawn at nominal range {z_m:.2f} m\n"
        f"via '{mode}')")


def overlay_fill(img, pts, color, alpha=0.30):
    ov = img.copy()
    cv2.fillPoly(ov, [np.array(pts, np.int32)], color)
    return cv2.addWeighted(ov, alpha, img, 1.0 - alpha, 0)


def draw_current_on_color(img, entries, z_m, intr, mode):
    """Project each current config shape into the color view (magenta)."""
    for e in entries:
        if shape_kind(e) == "rect":
            poly = depth_rect_to_poly(e, z_m, intr, mode)
        else:
            poly = depth_poly_to_color_poly(e, z_m, intr, mode)
        if poly:
            cv2.polylines(img, [np.array(poly, np.int32)], True,
                          (255, 0, 255), 2)


def draw_current_on_depth(img, entries):
    """Draw each current config shape natively in depth coords (magenta)."""
    for e in entries:
        if shape_kind(e) == "rect":
            cv2.rectangle(img, (e[0], e[1]), (e[2], e[3]), (255, 0, 255), 2)
        else:
            cv2.polylines(img, [np.array(e, np.int32)], True, (255, 0, 255), 2)


def draw_new_on_color(img, drawn, scale):
    """The user's exact drawing on the color view (green fill + outline).

    Drawn on purpose from `drawn` (full-res color px), NOT from the
    converted depth shape — the conversion maps through the nominal range
    and would wobble the overlay; the color view should show what you
    drew, the depth view shows what the config gets.
    """
    if drawn[0] == "rect":
        x, y, w, h = drawn[1]
        cv2.rectangle(img, (int(x * scale), int(y * scale)),
                      (int((x + w) * scale), int((y + h) * scale)),
                      (0, 255, 0), 2)
        return img
    pts = [[int(x * scale), int(y * scale)] for x, y in drawn[1]]
    img = overlay_fill(img, pts, (0, 200, 0), 0.30)
    cv2.polylines(img, [np.array(pts, np.int32)], True, (0, 255, 0), 2)
    return img


def draw_new_on_depth(img, shape):
    """Drawn shape on the depth view (red fill + outline)."""
    if shape[0] == "rect":
        cv2.rectangle(img, (shape[1][0], shape[1][1]),
                      (shape[1][2], shape[1][3]), (0, 0, 255), 2)
        return img
    img = overlay_fill(img, shape[1], (0, 0, 255), 0.35)
    cv2.polylines(img, [np.array(shape[1], np.int32)], True, (0, 0, 255), 2)
    return img


def capture_polygon(win, disp, scale):
    """Interactive polygon capture on `disp` (display px).

    Left-click adds a vertex; ENTER/'n' closes (>= 3 points); Backspace
    or 'u' undoes; ESC/'c' cancels. Returns full-res [[x, y], ...] or
    None on cancel.
    """
    pts = []
    cur = [None]

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
        elif event == cv2.EVENT_MOUSEMOVE:
            cur[0] = (x, y)

    cv2.setMouseCallback(win, on_mouse)
    try:
        while True:
            frame = disp.copy()
            if pts:
                preview = list(pts) + ([cur[0]] if cur[0] else [])
                frame = overlay_fill(frame, preview, (0, 200, 0), 0.25)
                cv2.polylines(frame, [np.array(preview, np.int32)], False,
                              (0, 255, 0), 2)
                for p in pts:
                    cv2.circle(frame, p, 3, (0, 255, 0), -1)
            cv2.putText(frame, "click vertices | ENTER/n close | "
                               "Backspace undo | ESC cancel",
                        (8, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 4)
            cv2.putText(frame, "click vertices | ENTER/n close | "
                               "Backspace undo | ESC cancel",
                        (8, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1)
            cv2.imshow(win, frame)
            key = cv2.waitKey(30) & 0xFF
            if key in (13, ord("n")):
                if len(pts) >= 3:
                    return [[int(px / scale), int(py / scale)] for px, py in pts]
            elif key in (8, ord("u")):
                if pts:
                    pts.pop()
            elif key in (27, ord("c")):
                return None
    finally:
        cv2.setMouseCallback(win, lambda *a: None)


def display_scaled(img, max_width):
    scale = min(1.0, max_width / img.shape[1])
    if scale < 1.0:
        img = cv2.resize(img, (int(img.shape[1] * scale),
                               int(img.shape[0] * scale)))
    return img, scale


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="http://bebop.local:9092",
                    help="bebop-vision videoserver base URL")
    ap.add_argument("--serial", default="CPBLC53000PE",
                    help="camera serial (rig YAML + intrinsics key)")
    ap.add_argument("--stream", default="color_near",
                    help="videoserver color stream to draw on")
    ap.add_argument("--depth-stream", default="depth_near",
                    help="videoserver depth stream for verification")
    ap.add_argument("--range", type=float, default=0.7,
                    help="nominal chassis range for the projection (m)")
    ap.add_argument("--rig", default=str(ROOT / "config" / "orbbec_rig.yaml"))
    ap.add_argument("--intrinsics-dir", default=str(ROOT / "config"))
    ap.add_argument("--max-width", type=int, default=1280,
                    help="cap the color window width (display scale)")
    ap.add_argument("--mode", choices=MODES, default="extrinsic",
                    help="initial color->depth mapping mode")
    ap.add_argument("--rect", action="store_true",
                    help="drag an axis-aligned rect instead of a polygon")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not check_server(args.host):
        sys.exit(f"videoserver {args.host} is not answering /healthz — "
                 f"is bebop-vision running (main.py --record-navd / --goal-drive)?")
    print(f"[set-self-mask] server {args.host} ok; fetching "
          f"{args.stream} + {args.depth_stream}")

    intr = load_intrinsics(
        Path(args.intrinsics_dir) / f"orbbec_intrinsics_{args.serial}.json")
    current = load_current_shapes(args.rig, args.serial)
    if current:
        print(f"[set-self-mask] current config shapes: {current}")
    else:
        print("[set-self-mask] no current self_mask_pixels in the rig YAML "
              "(nothing to compare against)")
    mode = args.mode

    # First frames + window sanity before committing to the loop.
    color = fetch_snapshot(args.host, args.stream)
    depth = fetch_snapshot(args.host, args.depth_stream)
    if color is None or depth is None:
        sys.exit(f"could not fetch {args.stream}/{args.depth_stream} from "
                 f"{args.host}")
    try:
        cv2.namedWindow("near color")
    except cv2.error:
        sys.exit("no display — run from a desktop session (or X forwarding)")

    frozen = None         # frozen color frame (full res) while drawing
    drawn = None          # what the user drew: ("rect", (x,y,w,h)) full-res
                          # color px, or ("poly", [[x, y], ...]) full-res
    shape = None          # converted ("rect", [x0,y0,x1,y1]) or
                          # ("poly", [[x, y], ...]) in DEPTH px
    state = "live"
    print("[set-self-mask] keys: SPACE/c draw | x mapping mode | q quit")
    while True:
        if state == "live":
            fresh = fetch_snapshot(args.host, args.stream)
            if fresh is not None:
                color = fresh
            fresh = fetch_snapshot(args.host, args.depth_stream)
            if fresh is not None:
                depth = fresh

        cdisp, scale = display_scaled(color, args.max_width)
        if current:
            draw_current_on_color(cdisp, current, args.range, intr, mode)
        if shape:
            cdisp = draw_new_on_color(cdisp, drawn, scale)
        cv2.putText(cdisp, f"mode={mode}  z={args.range:.2f}m", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(cdisp, f"mode={mode}  z={args.range:.2f}m", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("near color", cdisp)

        dvis = cv2.resize(depth, (DEPTH_SIZE[0], DEPTH_SIZE[1]),
                          interpolation=cv2.INTER_NEAREST)
        if current:
            draw_current_on_depth(dvis, current)
        if shape:
            dvis = draw_new_on_depth(dvis, shape)
        cv2.putText(dvis, "depth 848x480 (verify)", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("near depth", dvis)

        key = cv2.waitKey(60 if state == "live" else 120) & 0xFF
        if key in (ord("q"), 27):
            print("[set-self-mask] quit — nothing printed")
            break
        if key == ord("x"):
            mode = MODES[(MODES.index(mode) + 1) % len(MODES)]
            if drawn is not None:
                shape = convert(drawn, args, intr, mode)
                print(yml_snippet(shape, args.range, mode, args.serial))
            continue
        if state == "live" and key in (ord("c"), 32):
            frozen = color.copy()
            color = frozen
            if args.rect:
                x, y, w, h = cv2.selectROI(
                    "drag the chassis rect (ENTER=ok, c=cancel)", cdisp,
                    showCrosshair=True)
                if (x, y, w, h) == (0, 0, 0, 0):
                    continue
                drawn = ("rect", (int(x / scale), int(y / scale),
                                  int(w / scale), int(h / scale)))
                shape = convert(drawn, args, intr, mode)
            else:
                pts = capture_polygon("near color", cdisp, scale)
                if not pts:
                    continue
                drawn = ("poly", pts)
                shape = convert(drawn, args, intr, mode)
            if shape is None:
                print("[set-self-mask] conversion failed (ray behind the "
                      "camera?) — redraw")
                drawn = None
                state = "live"
                continue
            print(yml_snippet(shape, args.range, mode, args.serial))
            state = "verify"
        elif state == "verify":
            if key in (ord("y"), 13):
                print("\n" + yml_snippet(shape, args.range, mode,
                                         args.serial))
                break
            if key == ord("r"):
                shape = None
                drawn = None
                state = "live"
            if key == ord("l"):
                shape = None
                drawn = None
                state = "live"

    cv2.destroyAllWindows()


def convert(drawn, args, intr, mode):
    """User-drawn shape (full-res color px) -> depth-frame shape."""
    if drawn[0] == "rect":
        x, y, w, h = drawn[1]
        r = roi_to_depth_rect(x, y, w, h, args.range, intr, mode)
        return None if r is None else ("rect", r)
    p = poly_to_depth_poly(drawn[1], args.range, intr, mode)
    return None if p is None else ("poly", p)


if __name__ == "__main__":
    main()
