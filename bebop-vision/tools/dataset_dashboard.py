"""Web dashboard for reviewing and hand-correcting the navd training dataset.

Workflow (why this tool exists):
  1. Sessions recorded with `main.py --record-navd` land under
     datasets/navd-v0/<session>/ with per-tick color JPEGs, depth npz and
     60x60 label grids. The `fused` grid (0 blocked / 1 navigable /
     2 caution) is the SAM+depth teacher that training consumes.
  2. The teacher is good but not perfect — SAM misses dark/shiny objects,
     and the `disagree` / `unconfirmed` mining channels mark ticks where
     it probably did. A human reviews those ticks here: paints the correct
     classes on the 60x60 canvas while looking at the camera images and
     the label overlay (which shows which image pixels carry which class,
     via the ray LUT machinery from tools/fuse_navd_labels.py).
  3. Saving writes a `hand` uint8 60x60 array into labels/<stamp>.npz
     (same 0/1/2 semantics as `fused`). The presence of `hand` means
     "human reviewed"; training prefers `hand` over `fused` when present.
     The teacher keys are NEVER overwritten — corrections are additive,
     so re-running the fusion tool or a better SAM can always be diffed
     against what the human decided.
  4. Every save/clear appends an audit line to <session>/hand_edits.jsonl.

Runs as a stdlib-only HTTP server (no new pip deps, no frontend build):
  python tools/dataset_dashboard.py --data datasets/navd-v0 --port 8099
Default port 8099 — 9090/9091/9092 are taken on the robot (firmware,
bebop-agent, bebop-vision videoserver). Binds 127.0.0.1 by default; pass
--host 0.0.0.0 to review from another machine.

Endpoints (all JSON unless noted):
  GET  /                              single-page UI (inline HTML/JS)
  GET  /api/sessions                  session list: tick counts + hand counts
  GET  /api/session/<name>/ticks      per-tick stats for the jump filters
  GET  /api/tick/<name>/<stamp>       full tick payload (images + grids + manifest)
  GET  /api/overlay/<name>/<stamp>    label overlay on the color image
                                      (?role=near|far&src=auto|fused|hand&alpha=0.5)
  POST /api/tick/<name>/<stamp>/hand  body {"grid": 60x60} or {"clear": true}
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

# Grid geometry (see fuse_navd_labels.py / docs/navd.md): 60x60 cells,
# 3 m forward x 3 m wide @ 5 cm. row 0 = far edge, row 59 = at the robot,
# col 0 = robot right. Class semantics shared by `fused` and `hand`.
GRID = 60
N_CELLS = GRID * GRID
CLASSES = (0, 1, 2)          # 0 blocked / 1 navigable / 2 caution

# Class palette in BGR for cv2 rendering (spec: blocked = solid warm,
# navigable = free/green, caution = distinct amber — same convention as
# the videoserver BEV colors). The HTML UI mirrors these as RGB hex.
CLASS_BGR = {0: (47, 47, 211), 1: (67, 160, 46), 2: (0, 165, 255)}
CLASS_RGB_HEX = {0: "#d32f2f", 1: "#2ea043", 2: "#ffa500"}

# Depth view matches the repo convention (videoserver.render_depth):
# 0-4 m turbo colormap, invalid (0) = black, half-res PNG.
DEPTH_MAX_MM = 4000
DEPTH_VIEW = (424, 240)
GRID_KEYS = ("fused", "teacher", "sem_near", "sem_far", "floor_near",
             "floor_far", "disagree", "unconfirmed", "hand")

# Where the ray LUTs live (config/raylut_{role}.npz, built by
# fuse_navd_labels.py). They map each stride-4 color pixel to the BEV cell
# its flat-floor landing lands in (`land_cells`, flat 0..3599, -1 invalid).
# Import the real machinery; fall back to a local loader so the dashboard
# still runs if the tool import fails for any reason.
_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TOOLS_DIR.parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))
try:
    from fuse_navd_labels import load_lut as _tool_load_lut  # noqa: E402
except Exception:                                            # pragma: no cover
    _tool_load_lut = None


def _default_load_lut(role):
    """Load config/raylut_<role>.npz, or None if it does not exist.

    Prefers the shared loader from fuse_navd_labels (which resolves config/
    against its own ROOT); falls back to this repo's config dir so the
    dashboard is self-contained.
    """
    if _tool_load_lut is not None:
        try:
            return _tool_load_lut(role)
        except Exception:
            pass
    p = _REPO_ROOT / "config" / f"raylut_{role}.npz"
    if not p.exists():
        return None
    with np.load(p) as z:
        return {k: v for k, v in z.items()}


def _b64(data):
    """bytes -> base64 ascii str (JPEG/PNG passthrough for <img src=...>)."""
    return base64.b64encode(data).decode("ascii")


def render_depth_png(depth_mm):
    """uint16 (480, 848) mm depth -> half-res turbo-colormapped PNG bytes.

    Same look as videoserver.render_depth (0-4 m, invalid = black) but
    PNG-encoded for the review UI (no JPEG mush on hard depth edges).
    """
    valid = depth_mm > 0
    v = np.clip(depth_mm.astype(np.float32) / DEPTH_MAX_MM, 0.0, 1.0) * 255
    vis = cv2.applyColorMap(v.astype(np.uint8), cv2.COLORMAP_TURBO)
    vis[~valid] = 0
    vis = cv2.resize(vis, DEPTH_VIEW, interpolation=cv2.INTER_AREA)
    ok, png = cv2.imencode(".png", vis,
                           [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    return png.tobytes() if ok else None


def class_palette_bgr():
    """(N_CLASSES+1, 3) BGR lookup; index N_CELLS sentinel = transparent bg."""
    pal = np.zeros((N_CELLS + 1, 3), np.uint8)
    for cls, bgr in CLASS_BGR.items():
        pal[cls] = bgr
    return pal


def grid_shape_for_lut(lut, img_h, img_w):
    """(rows, cols) of the stride-subsampled pixel grid for a color size.

    build_lut() subsamples the color image every PIXEL_STRIDE, so n_px ==
    (H/s) * (W/s) for the stride s it was built with (4 today). We do not
    store s, so detect it from the decoded image size; fall back to a
    square-ish estimate if nothing divides exactly.
    """
    n_px = int(np.asarray(lut["n_px"]).ravel()[0]) \
        if np.asarray(lut["n_px"]).ndim else int(lut["n_px"])
    for s in (2, 4, 8):
        if img_h % s == 0 and img_w % s == 0 and (img_h // s) * (img_w // s) == n_px:
            return img_h // s, img_w // s
    est = max(1, int(round((img_h * img_w / n_px) ** 0.5)))
    return max(1, img_h // est), max(1, img_w // est)


def build_overlay(lut, grid, img_h, img_w):
    """(60x60 class grid, ray LUT) -> (overlay BGRA, valid mask) at image size.

    Every stride-4 color pixel whose landing cell is valid gets its BEV
    cell's class color, nearest-upscaled back to full resolution. Pixels
    with no landing cell (sky / outside the 3x3 m grid) stay transparent.
    """
    lc = np.asarray(lut["land_cells"]).astype(np.int64)
    # palette lookup with a transparent sentinel class for invalid landings
    pal = class_palette_bgr()
    safe = np.where(lc >= 0, lc, N_CELLS)          # N_CELLS -> sentinel row
    small = pal[safe]
    valid = (lc >= 0).astype(np.uint8)
    rows, cols = grid_shape_for_lut(lut, img_h, img_w)
    if rows * cols != len(safe):                   # defensive: bad shape fit
        return None, None
    up = cv2.resize(small.reshape(rows, cols, 3), (img_w, img_h),
                    interpolation=cv2.INTER_NEAREST)
    vmask = cv2.resize(valid.reshape(rows, cols), (img_w, img_h),
                       interpolation=cv2.INTER_NEAREST)
    bgra = cv2.cvtColor(up, cv2.COLOR_BGR2BGRA)
    bgra[..., 3] = vmask * 255
    return bgra, vmask > 0


class DatasetDashboard:
    """All dashboard logic, socket-free: methods take parsed names/bodies.

    The HTTP handler below is a thin shim over this class; tests call the
    methods directly against a synthetic session tree.
    """

    def __init__(self, data_root, load_lut=_default_load_lut):
        self.root = Path(data_root)
        self._load_lut = load_lut
        self._luts = {}            # role -> lut dict | None (cached)
        self._manifests = {}       # session name -> {stamp_ns: manifest row}
        self._legacy = {}          # session name -> bool (no fused labels)

    # ------------------------------------------------------------ sessions

    def _session_dir(self, name):
        """Resolved session dir, refusing path traversal in <name>."""
        d = (self.root / name).resolve()
        if d.parent != self.root.resolve() or not d.is_dir():
            raise FileNotFoundError(name)
        return d

    def _label_paths(self, name):
        return sorted((self._session_dir(name) / "labels").glob("*.npz"))

    def _hand_count(self, name):
        """Number of ticks whose labels npz already carries a `hand` array.

        Cheap: np.load only reads the zip directory for a key check, no
        array decompression.
        """
        n = 0
        for p in self._label_paths(name):
            try:
                with np.load(p) as z:
                    if "hand" in z.files:
                        n += 1
            except Exception:
                pass
        return n

    def _is_legacy(self, name):
        """True when the session predates the label pipeline: its label
        npz files carry only the recorded `teacher` grid and no `fused`
        target (early bring-up captures — nothing to review or paint).
        Cached per session; one npz header read on first touch."""
        if name not in self._legacy:
            d = self._session_dir(name)
            legacy = True
            for p in sorted((d / "labels").glob("*.npz"))[:1]:
                try:
                    with np.load(p) as z:
                        legacy = "fused" not in z.files
                except Exception:
                    legacy = True
            self._legacy[name] = legacy
        return self._legacy[name]

    def list_sessions(self):
        """[{"name", "ticks", "hand", "legacy"}] for every session with
        labels. legacy = pre-label-pipeline session (teacher-only npz):
        viewable, but painting is disabled in the UI."""
        out = []
        for d in sorted(self.root.iterdir()):
            if not (d / "labels").is_dir():
                continue
            out.append({
                "name": d.name,
                "ticks": len(list((d / "labels").glob("*.npz"))),
                "hand": self._hand_count(d.name),
                "legacy": self._is_legacy(d.name),
            })
        return out

    # --------------------------------------------------------------- ticks

    def list_ticks(self, name):
        """Per-tick summary rows driving the UI filters.

        Stats are the cheap ones a reviewer jumps on: class fractions of
        `fused`, cell counts of the mining channels (`disagree`,
        `unconfirmed`) and whether a hand correction exists.
        """
        rows = []
        for p in self._label_paths(name):
            stamp = int(p.stem)
            with np.load(p) as z:
                counts = np.bincount(z["fused"].ravel(), minlength=3)
                row = {
                    # "stamp" as STRING: stamp_ns values are 19 digits —
                    # beyond JS Number.MAX_SAFE_INTEGER (2^53), so JSON
                    # numbers round in the browser (…002 -> …000) and every
                    # file lookup misses. The string is the URL-safe key;
                    # "stamp_ns" stays for display/back-compat.
                    "stamp": f"{stamp:020d}",   # exact; matches filenames
                    "stamp_ns": stamp,
                    "hand": "hand" in z.files,
                    "f0": float(counts[0]) / N_CELLS,
                    "f1": float(counts[1]) / N_CELLS,
                    "f2": float(counts[2]) / N_CELLS,
                    "disagree_cells": int((z["disagree"] > 0).sum())
                    if "disagree" in z.files else 0,
                    "unconfirmed_cells": int((z["unconfirmed"] > 0).sum())
                    if "unconfirmed" in z.files else 0,
                }
            rows.append(row)
        return rows

    def _manifest_row(self, name, stamp):
        """Manifest row for one stamp (lazy per-session cache; None if absent)."""
        if name not in self._manifests:
            cache = {}
            mf = self._session_dir(name) / "manifest.jsonl"
            if mf.exists():
                with open(mf) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except ValueError:
                            continue
                        if "stamp_ns" in row:
                            cache[int(row["stamp_ns"])] = row
            self._manifests[name] = cache
        return self._manifests[name].get(int(stamp))

    def tick_payload(self, name, stamp):
        """Everything the UI needs for one tick, in one JSON response."""
        d = self._session_dir(name)
        s20 = f"{int(stamp):020d}"
        grids = {}
        lab = d / "labels" / f"{s20}.npz"
        if lab.exists():
            with np.load(lab) as z:
                for k in GRID_KEYS:
                    if k in z.files:
                        grids[k] = z[k].astype(np.uint8).tolist()
        payload = {
            "name": name,
            "stamp": str(int(stamp)),   # URL-safe exact key (see list_ticks)
            "stamp_ns": int(stamp),
            "legacy": self._is_legacy(name),
            "color_near": None,
            "color_far": None,
            "depth_near": None,
            "depth_far": None,
            "grids": grids,
            "manifest": self._manifest_row(name, stamp),
        }
        for role in ("near", "far"):
            jpg = d / ("color" if role == "near" else "color_far") / f"{s20}.jpg"
            if jpg.exists():
                payload[f"color_{role}"] = _b64(jpg.read_bytes())
            dep = d / "depth" / f"{s20}.npz"
            if dep.exists():
                with np.load(dep) as z:
                    if role in z.files:
                        payload[f"depth_{role}"] = _b64(render_depth_png(z[role]))
        return payload

    # ------------------------------------------------------------ overlays

    def _lut(self, role):
        """Cached LUT for a role (None if config/raylut_<role>.npz is absent,
        e.g. the far camera on a synthetic or early dataset)."""
        if role not in self._luts:
            try:
                self._luts[role] = self._load_lut(role)
            except Exception:
                self._luts[role] = None
        return self._luts[role]

    def overlay_payload(self, name, stamp, role="near", src="auto", alpha=0.5):
        """Class-colored label overlay on the color image (LUT machinery).

        Returns {"role", "src", "overlay": png-b64 (RGBA, transparent where
        no landing cell), "blend": jpeg-b64 (color with the overlay
        composited at `alpha`)} or {"error": ...} when there is no LUT /
        image. `src` picks the displayed grid: auto = hand if present else
        fused (what training would use); the response echoes the EFFECTIVE
        source ("hand" or "fused"), not the requested parameter.
        """
        d = self._session_dir(name)
        s20 = f"{int(stamp):020d}"
        sub = "color" if role == "near" else "color_far"
        img = cv2.imread(str(d / sub / f"{s20}.jpg"))
        if img is None:
            return {"error": f"no {sub} image for stamp {stamp}"}
        lut = self._lut(role)
        if lut is None or "land_cells" not in lut:
            return {"error": f"no ray LUT for role {role}"}
        lab = d / "labels" / f"{s20}.npz"
        if not lab.exists():
            return {"error": f"no labels for stamp {stamp}"}
        with np.load(lab) as z:
            if src == "hand" and "hand" not in z.files:
                return {"error": f"no hand grid for stamp {stamp}"}
            if src == "hand" or (src == "auto" and "hand" in z.files):
                grid, eff_src = z["hand"], "hand"
            else:
                grid, eff_src = z["fused"], "fused"
        bgra, vmask = build_overlay(lut, grid, img.shape[0], img.shape[1])
        if bgra is None:
            return {"error": "LUT shape does not match image size"}
        ok, png = cv2.imencode(".png", bgra,
                               [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        blend = img.copy()
        if vmask.any():
            blend[vmask] = np.clip(
                (1.0 - alpha) * blend[vmask].astype(np.float32)
                + alpha * bgra[vmask][..., :3].astype(np.float32),
                0, 255).astype(np.uint8)
        ok2, jpeg = cv2.imencode(".jpg", blend,
                                 [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return {"role": role, "src": eff_src,
                "overlay": _b64(png.tobytes()) if ok else None,
                "blend": _b64(jpeg.tobytes()) if ok2 else None}

    # ---------------------------------------------------------- hand edits

    @staticmethod
    def _class_counts(grid):
        """Per-class cell counts of a grid (audit trail payload)."""
        c = np.bincount(np.asarray(grid).ravel().astype(np.int64),
                        minlength=3)
        return {str(i): int(c[i]) for i in range(3)}

    @staticmethod
    def _write_labels(path, grids):
        """Atomic npz rewrite: temp file in the same dir + os.replace.

        Training may be reading these files concurrently; os.replace is
        atomic on POSIX so readers see either the old or the new file, and
        every pre-existing key is preserved (only `hand` is added/removed).
        """
        # numpy appends ".npz" unless the name already ends with it, so the
        # temp name must too (write tmp, then atomic rename over the real file)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(tmp, **grids)
        os.replace(tmp, path)

    @staticmethod
    def _audit(sess_dir, stamp, changed, before, after):
        """Append one hand_edits.jsonl line (best effort — never fatal)."""
        try:
            line = json.dumps({
                "stamp_ns": int(stamp),
                "stamp": str(int(stamp)),
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "changed_cells": int(changed),
                "class_counts_before": before,
                "class_counts_after": after,
            })
            with open(sess_dir / "hand_edits.jsonl", "a") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def save_hand(self, name, stamp, grid):
        """Store a human correction as labels/<stamp>.npz key `hand`.

        Accepts 60x60 nested lists or a flat 3600 list with values in
        0/1/2. All existing npz keys are preserved untouched; only `hand`
        is added/replaced. Returns (ok, message).
        """
        try:
            d = self._session_dir(name)
        except FileNotFoundError:
            return False, "unknown session"
        g = np.asarray(grid, dtype=np.int64)
        if g.shape not in ((GRID, GRID), (N_CELLS,)):
            return False, "grid must be 60x60 (or 3600 flat)"
        g = g.reshape(GRID, GRID)
        if not np.isin(g, CLASSES).all():
            return False, "grid values must be 0/1/2"
        lab = d / "labels" / f"{int(stamp):020d}.npz"
        if not lab.exists():
            return False, f"no labels npz for stamp {stamp}"
        with np.load(lab) as z:
            grids = {k: z[k] for k in z.files}
        base = grids["hand"] if "hand" in grids else grids.get("fused")
        if base is not None:
            before = self._class_counts(base)
            changed = int((g != base.astype(np.int64)).sum())
        else:
            before = {str(c): 0 for c in CLASSES}
            changed = N_CELLS
        grids["hand"] = g.astype(np.uint8)
        self._write_labels(lab, grids)
        self._audit(d, stamp, changed, before, self._class_counts(g))
        return True, "saved"

    def clear_hand(self, name, stamp):
        """Delete the `hand` key (revert to teacher), preserving the rest."""
        try:
            d = self._session_dir(name)
        except FileNotFoundError:
            return False, "unknown session"
        lab = d / "labels" / f"{int(stamp):020d}.npz"
        if not lab.exists():
            return False, f"no labels npz for stamp {stamp}"
        with np.load(lab) as z:
            grids = {k: z[k] for k in z.files}
        if "hand" not in grids:
            return True, "no hand to clear"
        before = self._class_counts(grids["hand"])
        del grids["hand"]
        after = self._class_counts(grids["fused"]) if "fused" in grids \
            else {str(c): 0 for c in CLASSES}
        self._write_labels(lab, grids)
        self._audit(d, stamp, N_CELLS, before, after)
        return True, "cleared"


# ------------------------------------------------------------------ HTTP

_ROUTES = (
    ("sessions", re.compile(r"^/api/sessions$")),
    ("ticks", re.compile(r"^/api/session/([^/]+)/ticks$")),
    ("tick", re.compile(r"^/api/tick/([^/]+)/(\d+)$")),
    ("hand", re.compile(r"^/api/tick/([^/]+)/(\d+)/hand$")),
    ("overlay", re.compile(r"^/api/overlay/([^/]+)/(\d+)$")),
)


class DashboardHandler(BaseHTTPRequestHandler):
    """Thin HTTP shim over DatasetDashboard (instance lives on the server)."""

    dashboard = None          # set by main(); tests can set it too

    def log_message(self, fmt, *args):    # one line, stderr-friendly
        sys.stderr.write("[dashboard] " + fmt % args + "\n")

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _page(self):
        body = PAGE_HTML.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        m = _ROUTES[1][1].match(u.path)             # ticks
        if m:
            return self._json(self.dashboard.list_ticks(m.group(1)))
        m = _ROUTES[0][1].match(u.path)             # sessions
        if m:
            return self._json({"sessions": self.dashboard.list_sessions()})
        m = _ROUTES[2][1].match(u.path)             # tick payload
        if m:
            try:
                return self._json(self.dashboard.tick_payload(m.group(1),
                                                              m.group(2)))
            except FileNotFoundError:
                return self._json({"error": "unknown session"}, 404)
        m = _ROUTES[4][1].match(u.path)             # overlay
        if m:
            q = parse_qs(u.query)
            out = self.dashboard.overlay_payload(
                m.group(1), m.group(2),
                role=q.get("role", ["near"])[0],
                src=q.get("src", ["auto"])[0],
                alpha=float(q.get("alpha", ["0.5"])[0]))
            return self._json(out, 404 if "error" in out else 200)
        if u.path == "/" or u.path.startswith("/index"):
            return self._page()
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        m = _ROUTES[3][1].match(urlparse(self.path).path)
        if not m:
            return self._json({"error": "not found"}, 404)
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return self._json({"error": "bad JSON body"}, 400)
        name, stamp = m.group(1), m.group(2)
        if body.get("clear"):
            ok, msg = self.dashboard.clear_hand(name, stamp)
        elif "grid" in body:
            ok, msg = self.dashboard.save_hand(name, stamp, body["grid"])
        else:
            return self._json({"error": "need 'grid' or 'clear'"}, 400)
        return self._json({"ok": ok, "message": msg}, 200 if ok else 400)


def main():
    ap = argparse.ArgumentParser(
        description="navd dataset review / hand-correction dashboard")
    ap.add_argument("--data",
                    default=str(_REPO_ROOT / "datasets" / "navd-v0"),
                    help="dataset root (one dir per session)")
    ap.add_argument("--port", type=int, default=8099,
                    help="HTTP port (default 8099; 9090-9092 are taken)")
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (default loopback)")
    args = ap.parse_args()
    DashboardHandler.dashboard = DatasetDashboard(args.data)
    srv = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"[dashboard] serving {args.data} on "
          f"http://{args.host}:{args.port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


# Single-page UI: inline HTML/CSS/JS, vanilla, keyboard driven. The paint
# canvas draws the grid camera-aligned (row 0 = far = top; grid col 0 =
# robot right renders at canvas RIGHT, matching the overlay blend on the
# images) and mirrors the column back to grid indices before posting.
PAGE_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>navd dataset dashboard</title>
<style>
body{background:#14181d;color:#d6dde6;font:13px/1.4 monospace;margin:0}
header{display:flex;gap:10px;align-items:center;padding:6px 10px;
       background:#1b2129;flex-wrap:wrap}
select,button{background:#232b35;color:#d6dde6;border:1px solid #39434f;
              border-radius:4px;padding:3px 8px;font:inherit;cursor:pointer}
button:hover{background:#2c3641}
.dirty{color:#ffb454;font-weight:bold}
main{display:flex;gap:12px;padding:10px}
#left{width:520px}
#left img{display:block;background:#000;max-width:520px;margin-bottom:4px}
#left img.missing{background:repeating-linear-gradient(45deg,#151a20,#151a20
                  8px,#0d1116 8px,#0d1116 16px);min-height:110px;width:100%}
#stats{padding:4px 10px;color:#9fb0c0;white-space:pre-wrap}
#right{flex:1}
#canvas{background:#0a0d10;cursor:crosshair;image-rendering:pixelated}
.cls{display:inline-block;width:12px;height:12px;vertical-align:-1px;margin:0 3px 0 10px}
#handstate{font-weight:bold}
</style></head><body>
<header>
 <b>navd review</b>
 <select id="sess"></select>
 <span id="tickpos">-</span>
 <button onclick="nav(-1)" title="left arrow">&lt;</button>
 <button onclick="nav(1)" title="right arrow">&gt;</button>
 <button onclick="jump('disagree')" title="d">j: disagree</button>
 <button onclick="jump('caution')" title="c">j: caution</button>
 <button onclick="jump('unreviewed')" title="n">j: unreviewed</button>
 <button onclick="jump('first')" title="f">j: first</button>
 <span id="handstate"></span>
 <button onclick="toggleSrc()" title="t" id="toggle">show: teacher</button>
 <button onclick="toggleOverlay()" title="o">overlay</button>
 <label>alpha <input id="alpha" type="range" min="0" max="100" value="50"></label>
</header>
<div id="stats"></div>
<main>
 <div id="left">
  <div>near color <span id="ovstate"></span></div>
  <img id="imgnear">
  <div>depth near</div><img id="depnear" style="max-width:340px">
  <div style="display:flex;gap:6px"><div><div>far color</div><img id="imgfar" style="max-width:250px"></div>
  <div><div>depth far</div><img id="depfar" style="max-width:250px"></div></div>
 </div>
 <div id="right">
  <div>
   <span class="cls" style="background:#d32f2f"></span>blocked (1)
   <span class="cls" style="background:#2ea043"></span>navigable (2)
   <span class="cls" style="background:#ffa500"></span>caution (3)
   brush <span id="brush">2</span>
   <button onclick="undo()" title="u">undo</button>
   <button onclick="revert()" title="r">revert</button>
   <button onclick="save()" title="s" id="save">Save</button>
   <button onclick="clearHand()" id="clearHand">Clear hand</button>
  </div>
  <canvas id="canvas" width="480" height="480"></canvas>
  <div id="msg" style="color:#9fb0c0"></div>
 </div>
</main>
<script>
"use strict";
// state: client-side until Save; Save POSTs the full grid (server stores
// `hand`; teacher keys are never touched).
let ticks = [], idx = -1, sess = null, showHand = false, overlayOn = false;
let dirty = false, grid = null, undoStack = [], brush = 2, cls = 0;
let painting = false;   // mouse button currently down on the canvas
let editable = false;   // fused/hand data present -> paint+save allowed
const CLS_HEX = ["#d32f2f", "#2ea043", "#ffa500"];
const BG_HEX = "#0a0d10";
const CV = document.getElementById("canvas"), CX = CV.getContext("2d");
const CELL = 8;   // 60 * 8 = 480 px canvas

const $ = id => document.getElementById(id);
function api(path, opt) { return fetch(path, opt).then(r => r.json()); }
function msg(t) { $("msg").textContent = t; }

// Monotonic request token: every async render path checks it after each
// await and bails when superseded — otherwise a slow fetch from the
// previous session/tick resolves late and paints stale data under the new
// header (seen 2026-09-07: session switch showed the old session's stamp
// with every image 404).
let reqSeq = 0;
const PLACEHOLDER =
  "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7";
function setImg(el, b64, mime) {
  if (b64) {
    el.src = "data:" + mime + ";base64," + b64;
    el.classList.remove("missing");
  } else {
    el.src = PLACEHOLDER;
    el.classList.add("missing");
  }
}

async function boot() {
  const s = await api("/api/sessions");
  const sel = $("sess");
  sel.innerHTML = "";
  for (const x of s.sessions) {
    const o = document.createElement("option");
    o.value = x.name;
    o.textContent = x.name + " (" + x.ticks + " ticks, " + x.hand + " hand"
                    + (x.legacy ? ", LEGACY" : "") + ")";
    sel.appendChild(o);
  }
  if (s.sessions.length) { sel.value = s.sessions[0].name; await loadSession(); }
  sel.onchange = loadSession;
}
async function loadSession() {
  const my = ++reqSeq;
  sess = $("sess").value;
  ticks = await api("/api/session/" + sess + "/ticks");
  if (my !== reqSeq) return;
  if (!Array.isArray(ticks)) ticks = [];
  idx = 0;
  overlayOn = false;
  await show(my);
}
function nav(d) {
  if (!ticks.length) return;
  idx = Math.min(Math.max(idx + d, 0), ticks.length - 1);
  show();
}
async function show(my) {
  if (my === undefined) my = ++reqSeq;
  if (idx < 0 || idx >= ticks.length) return;
  dirty = false; undoStack = []; painting = false;
  const p = await api("/api/tick/" + sess + "/" + (ticks[idx].stamp || ticks[idx].stamp_ns));
  if (my !== reqSeq) return;   // superseded mid-fetch: drop the payload
  window._payload = p;
  const g = p.grids || {};
  const hasHand = !!g.hand;
  // painting only on fused/hand data; legacy (teacher-only) is read-only.
  // `painting` is the mouse-down flag (mouse handlers) — keep separate.
  editable = !p.legacy && !!(g.fused || g.hand);
  const src = hasHand ? g.hand : (g.fused || g.teacher);
  grid = src ? src.map(r => r.slice())
             : Array.from({ length: 60 }, () => Array(60).fill(0));
  showHand = hasHand;   // prefill the paint canvas with hand if it exists
  $("tickpos").textContent = (idx + 1) + "/" + ticks.length;
  $("handstate").textContent = p.legacy ? "[legacy]"
    : (hasHand ? "[hand]" : "[teacher]");
  $("toggle").textContent = "show: " + (showHand ? "hand" : "teacher");
  draw(); render(p); updateSave();
  msg(p.legacy
      ? "legacy session: no fused labels (teacher only) — view only; run "
        + "tools/sam_floor_label.py + tools/fuse_navd_labels.py to review"
      : "");
}
function render(p) {
  const g = p.grids || {};
  const sum = k => g[k] ? g[k].flat().reduce((a, b) => a + (b ? 1 : 0), 0) : 0;
  const fr = k => {
    if (!g[k]) return "-";
    const c = [0, 0, 0]; g[k].flat().forEach(v => c[v]++);
    return "b" + (100 * c[0] / 3600).toFixed(0) + " n" + (100 * c[1] / 3600).toFixed(0) +
           " c" + (100 * c[2] / 3600).toFixed(0);
  };
  const m = p.manifest || {};
  setImg($("imgnear"), p.color_near, "image/jpeg");
  setImg($("imgfar"), p.color_far, "image/jpeg");
  setImg($("depnear"), p.depth_near, "image/png");
  setImg($("depfar"), p.depth_far, "image/png");
  $("stats").textContent =
    "stamp " + (p.stamp || p.stamp_ns) + "   " + sess + (p.legacy ? "   [LEGACY: teacher only]" : "") + "\n" +
    "fused " + fr("fused") + "    hand " + (g.hand ? fr("hand") : "-") + "\n" +
    "disagree " + sum("disagree") + " cells    unconfirmed " + sum("unconfirmed") + " cells\n" +
    "cmd_vel " + JSON.stringify(m.cmd_vel || {}) + "\n" +
    "odom " + JSON.stringify(m.odom || {}) + "\n" +
    "goal " + JSON.stringify(m.goal || {});
}
function draw() {
  for (let r = 0; r < 60; r++) for (let c = 0; c < 60; c++) {
    // camera-aligned: grid col 0 (robot right) draws at canvas RIGHT
    const v = grid[r][c];
    CX.fillStyle = v == 3 ? BG_HEX : CLS_HEX[v];
    CX.fillRect((59 - c) * CELL, r * CELL, CELL, CELL);
  }
}
function updateSave() {
  $("save").textContent = dirty ? "* Save" : "Save";
  $("save").classList.toggle("dirty", dirty);
  $("save").disabled = !editable;
  $("clearHand").disabled = !editable;
}
function pushUndo() {
  undoStack.push(grid.map(r => r.slice()));
  if (undoStack.length > 200) undoStack.shift();
}
function undo() { if (undoStack.length) { grid = undoStack.pop(); dirty = true; draw(); updateSave(); } }
function revert() { dirty = false; show(); }
async function refreshSessionList() { await boot(); }
async function toggleSrc() {
  if (!window._payload) return;
  const g = window._payload.grids || {};
  if (!g.hand) { msg("no hand grid for this tick (save first)"); showHand = false; }
  else showHand = !showHand;
  const src = showHand && g.hand ? g.hand : (g.fused || grid);
  grid = src.map(r => r.slice());
  dirty = false; undoStack = [];
  $("toggle").textContent = "show: " + (showHand ? "hand" : "teacher");
  draw(); updateSave();
}
async function toggleOverlay() {
  if (!ticks.length || idx < 0) return;
  overlayOn = !overlayOn;
  if (!overlayOn) { $("imgnear").src = "data:image/jpeg;base64," + window._payload.color_near; $("ovstate").textContent = ""; return; }
  const q = "?role=near&src=" + (showHand ? "hand" : "auto") +
            "&alpha=" + ($("alpha").value / 100);
  const o = await api("/api/overlay/" + sess + "/" + (ticks[idx].stamp || ticks[idx].stamp_ns) + q);
  if (o.blend) {
    $("imgnear").src = "data:image/jpeg;base64," + o.blend;
    $("ovstate").textContent = "(label overlay: " + o.src + ")";
  } else { $("ovstate").textContent = "(overlay: " + (o.error || "unavailable") + ")"; }
}
$("alpha").onchange = () => { if (overlayOn) toggleOverlay().then(toggleOverlay).then(toggleOverlay); };
function paint(ev) {
  if (!editable) return;
  const rect = CV.getBoundingClientRect();
  const cx = Math.floor((ev.clientX - rect.left) / rect.width * 60);
  const cy = Math.floor((ev.clientY - rect.top) / rect.height * 60);
  if (cx < 0 || cy < 0 || cx > 59 || cy > 59) return;
  pushUndo();
  const h = Math.floor(brush / 2);
  for (let r = cy - h; r <= cy + h; r++) for (let c = cx - h; c <= cx + h; c++) {
    if (r < 0 || c < 0 || r > 59 || c > 59) continue;
    grid[r][59 - c] = cls;   // mirror the column back to grid space
  }
  dirty = true; draw(); updateSave();
}
CV.onmousedown = e => { if (editable) { painting = true; paint(e); } };
CV.onmousemove = e => { if (painting) paint(e); };
window.onmouseup = () => { painting = false; };
async function save() {
  if (idx < 0 || !editable) return;
  const flat = grid.map(r => r.map(v => v == 3 ? 2 : v));  // unpainted keeps caution
  const r = await api("/api/tick/" + sess + "/" + (ticks[idx].stamp || ticks[idx].stamp_ns) + "/hand", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ grid: flat })
  });
  if (r.ok) {
    dirty = false; updateSave(); msg("saved (hand stored; teacher untouched)");
    ticks[idx].hand = true;
    $("handstate").textContent = "[hand]"; window.showHandSaved = true;
    const i = $("sess").selectedIndex;
    if (i >= 0) $("sess").options[i].textContent =
      $("sess").options[i].textContent.replace(/(\d+) hand/, (m, n) => (Number(n) + 1) + " hand");
  } else msg("save failed: " + r.message);
}
async function clearHand() {
  if (idx < 0 || !editable) return;
  const r = await api("/api/tick/" + sess + "/" + (ticks[idx].stamp || ticks[idx].stamp_ns) + "/hand", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ clear: true })
  });
  if (r.ok) { dirty = false; show(); msg("hand cleared (reverted to teacher)"); }
  else msg("clear failed: " + r.message);
}
function jump(kind) {
  if (!ticks.length) return;
  if (kind == "first") { idx = 0; show(); return; }
  for (let k = 1; k <= ticks.length; k++) {
    const i = (idx + k) % ticks.length, t = ticks[i];
    const hit = (kind == "disagree" && t.disagree_cells > 0) ||
                (kind == "caution" && t.f2 > 0.35 && !t.hand) ||
                (kind == "unreviewed" && !t.hand);
    if (hit) { idx = i; show(); return; }
  }
  msg("no tick matches jump: " + kind);
}
window.onkeydown = e => {
  if (e.target.tagName == "SELECT" || e.target.tagName == "INPUT") return;
  const k = e.key;
  if (k == "ArrowRight") nav(1);
  else if (k == "ArrowLeft") nav(-1);
  else if (k == "1") cls = 0; else if (k == "2") cls = 1; else if (k == "3") cls = 2;
  else if (k == "[") { brush = Math.max(1, brush - 1); $("brush").textContent = brush; }
  else if (k == "]") { brush = Math.min(5, brush + 1); $("brush").textContent = brush; }
  else if (k == "u") undo();
  else if (k == "s") save();
  else if (k == "r") revert();
  else if (k == "t") toggleSrc();
  else if (k == "o") toggleOverlay();
  else if (k == "d") jump("disagree");
  else if (k == "c") jump("caution");
  else if (k == "n") jump("unreviewed");
  else if (k == "f") jump("first");
};
boot();
</script></body></html>"""

if __name__ == "__main__":
    main()
