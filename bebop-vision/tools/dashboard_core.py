"""Headless dataset core: sessions, ticks, overlays, hand edits.

Single source of truth for reading/writing the navd-v0 session tree —
used by tools/dashboard_api.py (the FastAPI backend serving the React
dashboard) and by the tests. The original stdlib-server + inline-HTML
tool (tools/dataset_dashboard.py) was retired 2026-09-07; this module
keeps the battle-tested logic with no HTTP surface of its own.
"""

import base64
import json
import os
import sys
import time
from pathlib import Path

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
# the videoserver BEV colors). The dashboard mirrors these as RGB hex.
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
        self._keycounts = {}       # session name -> ((n, mtime), counts)
        self._sam_cov = {}         # (name, role, n) -> sampled coverage

    # ------------------------------------------------------------ sessions

    def _session_dir(self, name):
        """Resolved session dir, refusing path traversal in <name>."""
        d = (self.root / name).resolve()
        if d.parent != self.root.resolve() or not d.is_dir():
            raise FileNotFoundError(name)
        return d

    def _label_paths(self, name):
        return sorted((self._session_dir(name) / "labels").glob("*.npz"))

    def _key_counts(self, name):
        """{'fused': n, 'hand': n} over the session's label npz files.

        Cheap per file: np.load only reads the zip directory for a key
        check, no array decompression. Cached per session and invalidated
        by (file count, newest mtime) so the pipeline view can poll
        without rescanning unchanged sessions.
        """
        paths = self._label_paths(name)
        key = (len(paths),
               max((p.stat().st_mtime for p in paths), default=0.0))
        cached = self._keycounts.get(name)
        if cached and cached[0] == key:
            return cached[1]
        counts = {"fused": 0, "hand": 0}
        for p in paths:
            try:
                with np.load(p) as z:
                    counts["fused"] += "fused" in z.files
                    counts["hand"] += "hand" in z.files
            except Exception:
                pass
        self._keycounts[name] = (key, counts)
        return counts

    def _hand_count(self, name):
        """Number of ticks whose labels npz already carries a `hand` array."""
        return self._key_counts(name)["hand"]

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

    # ------------------------------------------------------- SAM artifacts

    def _self_pixels(self, role, lut):
        """LUT pixels looking at the robot's own chassis (self-view
        polygon from the rig config). The robot is not scene — the gate
        view never paints it. Best effort: empty on any config failure.
        """
        try:
            import fuse_navd_labels as fuse
            cfg = fuse.load_cfg()
            cams = cfg["cameras"]
            serial = next(s for s, c in cams.items() if c["role"] == role)
            return np.asarray(fuse.self_pixel_mask(serial, cams, lut), bool)
        except Exception:
            return np.zeros(len(lut["land_cells"]), bool)

    def _gate_classes(self, role, mask, depth_path, img_h, img_w):
        """Per-LUT-pixel SAM+depth gate classes (mode "gate").

        Mirrors the fuse_navd_labels.py per-pixel decision exactly:
          0 green        SAM floor AND the ray's measured depth lands
                         within FLOOR_TOL of the flat-floor prediction
                         (these pixels became navigable confirmations)
          1 red          non-floor pixel, off-ground surface or no depth
                         at all (blocked evidence; fuse sweeps the whole
                         band when depth is missing)
          2 transparent  depth-consistent ground SAM missed (stays
                         unconfirmed), sky / no-landing pixels, and the
                         robot's own self-view pixels

        Returns {"small": u8 (rows, cols), "counts": [floor, blocked]}
        or {"error": ...} when the LUT/depth/config machinery is
        unavailable (raw `sam` mode still works everywhere).
        """
        try:
            import fuse_navd_labels as fuse
            lut = self._lut(role)
            if lut is None or "land_cells" not in lut:
                return {"error": f"no ray LUT for role {role} (gate mode)"}
            rows, cols = grid_shape_for_lut(lut, img_h, img_w)
            if rows * cols != len(lut["land_cells"]):
                return {"error": "LUT shape does not match image size"}
            with np.load(depth_path) as z:
                if role not in z.files:
                    return {"error": f"no {role} depth npz for gate mode"}
                depth_img = z[role]
            d_m = depth_img[lut["v_d"], lut["u_d"]].astype(np.float32) * 1e-3
            valid = (d_m > 0.05) & (d_m < 6.0)
            on_ground = valid & \
                (np.abs(np.asarray(lut["t_g"], np.float32) - d_m)
                 <= fuse.FLOOR_TOL_M)
            stride = max(1, img_h // rows)
            self_pix = self._self_pixels(role, lut)
            act = mask[::stride, ::stride].ravel() & ~self_pix
            lc = np.asarray(lut["land_cells"]).astype(np.int64)
            has = lc >= 0                       # pixel has a landing cell
            small = np.full(rows * cols, 2, np.uint8)     # clear default
            small[has & act & on_ground] = 0              # floor -> navigable
            small[has & ~act & ~on_ground & ~self_pix] = 1   # blocked evidence
            return {"small": small.reshape(rows, cols),
                    "counts": [int((small == 0).sum()),
                               int((small == 1).sum())]}
        except Exception as exc:   # noqa: BLE001 — gate is best-effort
            return {"error": f"gate mode unavailable: "
                             f"{type(exc).__name__}: {exc}"}

    def sam_overlay_payload(self, name, stamp, role="near", mode="sam",
                            alpha=0.5):
        """Step-3 artifact view: the raw SAM floor mask over the camera
        image (mode "sam") or the SAM+depth per-pixel gate (mode "gate").

        Returns {"role", "mode", "overlay": png-b64 RGBA (transparent
        off-mask), "blend": jpeg-b64 (composited at `alpha`),
        "floor_frac", "gate": {"floor_px", "blocked_px"} | None} or
        {"error": ...}. The client applies opacity in CSS to `overlay`;
        `blend` is the server-side composite for a single-shot look.
        """
        d = self._session_dir(name)
        s20 = f"{int(stamp):020d}"
        sub = "color" if role == "near" else "color_far"
        img = cv2.imread(str(d / sub / f"{s20}.jpg"))
        if img is None:
            return {"error": f"no {sub} image for stamp {stamp}"}
        mp = d / ("sam_floor" if role == "near" else "sam_floor_far") \
            / f"{s20}.npz"
        if not mp.exists():
            return {"error": f"no SAM floor mask for stamp {stamp} ({role})"
                             " — run tools/sam_floor_label.py"}
        with np.load(mp) as z:
            mask = z["mask"]
        if mask.shape[:2] != img.shape[:2]:
            return {"error": f"SAM mask {mask.shape[:2]} != image "
                             f"{img.shape[:2]} for stamp {stamp}"}
        h, w = img.shape[:2]
        if mode == "gate":
            gate = self._gate_classes(role, mask, d / "depth" / f"{s20}.npz",
                                      h, w)
            if "error" in gate:
                return gate
            # index 2 (clear) must exist for the gather; its alpha is 0
            pal = np.array([CLASS_BGR[1], CLASS_BGR[0], (0, 0, 0)], np.uint8)
            up = cv2.resize(pal[gate["small"]], (w, h),
                            interpolation=cv2.INTER_NEAREST)
            vmask = cv2.resize((gate["small"] < 2).astype(np.uint8),
                               (w, h), interpolation=cv2.INTER_NEAREST) > 0
            bgra = cv2.cvtColor(up, cv2.COLOR_BGR2BGRA)
            bgra[..., 3] = vmask * 255
            gate_out = {"floor_px": gate["counts"][0],
                        "blocked_px": gate["counts"][1]}
        else:
            bgra = np.zeros((h, w, 4), np.uint8)
            bgra[mask] = (*CLASS_BGR[1], 255)     # green = raw SAM floor
            gate_out = None
        ok, png = cv2.imencode(".png", bgra,
                               [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        blend = img.copy()
        sel = bgra[..., 3] > 0
        if sel.any():
            blend[sel] = np.clip(
                (1.0 - alpha) * blend[sel].astype(np.float32)
                + alpha * bgra[sel][..., :3].astype(np.float32),
                0, 255).astype(np.uint8)
        ok2, jpeg = cv2.imencode(".jpg", blend,
                                 [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return {"role": role, "mode": mode,
                "overlay": _b64(png.tobytes()) if ok else None,
                "blend": _b64(jpeg.tobytes()) if ok2 else None,
                "floor_frac": round(float(mask.mean()), 4),
                "gate": gate_out}

    # ------------------------------------------------------ pipeline view

    def _sam_dir_stats(self, name, role, sample=16):
        """{done, total, complete, coverage} for one camera's SAM masks.

        Coverage is the mean mask density over an evenly spaced sample
        of frames (loading every 800x1280 mask every poll would be far
        too slow); cached per (session, role, file count) so it is
        recomputed only while the SAM pass is adding files.
        """
        d = self._session_dir(name)
        out = d / ("sam_floor" if role == "near" else "sam_floor_far")
        files = sorted(out.glob("*.npz")) if out.is_dir() else []
        total = len(self._label_paths(name))
        cov = None
        if files:
            ck = (name, role, len(files))
            cov = self._sam_cov.get(ck)
            if cov is None:
                step = max(1, len(files) // sample)
                vals = []
                for p in files[::step][:sample]:
                    try:
                        with np.load(p) as z:
                            vals.append(float(z["mask"].mean()))
                    except Exception:
                        pass
                cov = round(float(np.mean(vals)), 4) if vals else None
                self._sam_cov[ck] = cov
        return {"done": len(files), "total": total,
                "complete": bool(files) and len(files) == total,
                "coverage": cov}

    def pipeline_status(self, mcap_dir=None):
        """Per-session artifact status for each pipeline stage.

        Stages: record (raw MCAP in the captures dir), extract (session
        tree + manifest), sam (mask counts + sampled coverage per
        camera), fuse (labels npz carrying `fused`), review (hand
        edits). Train/export/runtime are global artifacts — the API
        serves those from /api/pipeline/train, not per session.
        """
        mcap_dir = Path(mcap_dir) if mcap_dir else \
            self.root.parent / "sessions"
        out = []
        for d in sorted(self.root.iterdir()):
            if not (d / "labels").is_dir():
                continue
            name = d.name
            ticks = len(self._label_paths(name))
            kc = self._key_counts(name)
            mp = mcap_dir / f"{name}.mcap"
            if mp.exists():
                st = mp.stat()
                record = {"present": True,
                          "size_mb": round(st.st_size / 1e6, 1),
                          "mtime": int(st.st_mtime)}
            else:
                record = {"present": False, "size_mb": None, "mtime": None}
            out.append({
                "name": name,
                "ticks": ticks,
                "record": record,
                "extract": {"ticks": ticks,
                            "manifest": (d / "manifest.jsonl").exists()},
                "sam": {"near": self._sam_dir_stats(name, "near"),
                        "far": self._sam_dir_stats(name, "far")},
                "fuse": {"fused": kc["fused"], "total": ticks,
                         "complete": ticks > 0 and kc["fused"] == ticks},
                "review": {"hand": kc["hand"]},
            })
        return out

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

    # ---------------------------------------------------------- grid texture

    def _scatter_camera(self, name, stamp, role, tex, covered):
        """Accumulate one camera's color pixels into BEV cells via its ray
        LUT: tex[r, c] = mean RGB of stride-4 pixels whose flat-floor ray
        lands there. Cells already covered (near camera processed first)
        are left untouched. Returns True on success."""
        d = self._session_dir(name)
        s20 = f"{int(stamp):020d}"
        sub = "color" if role == "near" else "color_far"
        img = cv2.imread(str(d / sub / f"{s20}.jpg"))
        if img is None:
            return False
        lut = self._lut(role)
        if lut is None or "land_cells" not in lut:
            return False
        lc = np.asarray(lut["land_cells"]).astype(np.int64)
        rows, cols = grid_shape_for_lut(lut, img.shape[0], img.shape[1])
        if rows * cols != len(lc):
            return False
        stride = max(1, img.shape[0] // rows)
        px = cv2.cvtColor(img[::stride, ::stride],
                          cv2.COLOR_BGR2RGB).reshape(-1, 3).astype(np.float32)
        valid = lc >= 0
        flat, px = lc[valid], px[valid]
        w = np.bincount(flat, minlength=N_CELLS).astype(np.float32)
        has = w > 0
        take = has & ~covered.ravel()
        mean3 = np.zeros((N_CELLS, 3), np.float32)
        for ch in range(3):
            sums = np.bincount(flat, weights=px[:, ch], minlength=N_CELLS)
            mean3[:, ch] = np.divide(sums, np.maximum(w, 1.0))
        tex[take] = mean3[take]
        covered.ravel()[has] = True
        return True

    def build_texture(self, name, stamp):
        """Combined near+far camera texture on the 60x60 grid (near wins
        where both cover). {"png": b64 RGB PNG or None, "coverage": float}
        for the paint canvas underlay; {"error": ...} when unavailable."""
        try:
            self._session_dir(name)
        except FileNotFoundError:
            raise
        tex = np.zeros((N_CELLS, 3), np.float32)
        covered = np.zeros((GRID, GRID), bool)
        ok = any(self._scatter_camera(name, stamp, role, tex, covered)
                 for role in ("near", "far"))
        if not ok or not covered.any():
            return {"png": None, "coverage": 0.0,
                    "error": "no ray LUT / no color image for this tick"}
        img = tex.reshape(GRID, GRID, 3).astype(np.uint8)
        ok, png = cv2.imencode(".png", img,
                               [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
        return {"png": _b64(png.tobytes()) if ok else None,
                "coverage": round(float(covered.mean()), 4)}


