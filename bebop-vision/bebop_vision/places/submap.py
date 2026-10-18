"""Per-place local submap parsing and RLE helpers.

A submap is a *local patch* taken from one capture session's gravity-aligned
structure snapshot (``GET /structure`` on the live-SLAM service, ``:9096``).
The stored grid, walls and anchor all live in that session's ``map`` frame:

* ``anchor_x``/``anchor_y``/``anchor_yaw_deg`` is the robot/camera pose at
  capture, in that same frame.  The place's conceptual origin is this anchor.
  The anchor is pinned to the **first** capture; later captures only add
  evidence, so the accumulated raster stays in one coherent frame.
  Anchor-relative coordinates are derived as ``R(-yaw) * (p - anchor)`` and
  never stored back as if they were survey coordinates.
* A re-capture does not replace the stored patch: :func:`merge_submaps`
  unions the occupied/free cell sets and rebins them into a union bounding
  box at the stored resolution.  Cells are placed by their centre in the
  capture frame (``origin + (i + 0.5) * resolution_m``), which is the same
  frame as the pinned anchor.
* Nothing here reads wheel odometry.  ``places.odom_*`` is legacy capture
  metadata and is deliberately not part of this module.
* There is no cross-session alignment: two captures from different sessions
  have unrelated ``map`` frames until relocalization exists, so the union is
  only geometrically meaningful within one session.

The encoding matches the structure module's wire format: row-major base64
little-endian uint16 run lengths, starting with a 0-run.  It is duplicated
here (instead of importing ``slam.structure``) so the places store stays
independent of the live-SLAM package.
"""
from __future__ import annotations

import base64
import math
from dataclasses import dataclass

import numpy as np

# A structure grid is internally capped at 60k cells, but its bounding box can
# be larger.  Reject absurd payloads before decoding them.
MAX_GRID_CELLS = 4_000_000
MAX_WALLS = 20_000
MAX_GUIDE_POINTS = 10_000


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be a finite number")
    return out


def _optional_number(value, name: str):
    if value is None:
        return None
    return _number(value, name)


def _integer(value, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


def encode_rle(mask: np.ndarray) -> str:
    """Row-major run lengths (uint16), starting with a 0-run.

    Byte-for-byte compatible with ``StructureMap._rle_base64``; used by tests
    and by any future submap merge.
    """
    bits = np.asarray(mask, dtype=np.uint8).reshape(-1)
    if bits.size == 0 or not bits.any():
        return ""
    change = np.flatnonzero(bits[1:] != bits[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [bits.size]))
    runs = (ends - starts).astype("<u2")
    if bits[0] == 1:
        runs = np.concatenate((np.array([0], dtype="<u2"), runs))
    return base64.b64encode(runs.tobytes()).decode("ascii")


def _decode_runs(encoded: str, name: str) -> np.ndarray:
    if not isinstance(encoded, str):
        raise ValueError(f"{name} must be a base64 string")
    if not encoded:
        return np.zeros(0, dtype=np.int64)
    try:
        raw = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError) as exc:
        raise ValueError(f"{name} is not valid base64") from exc
    if len(raw) % 2:
        raise ValueError(f"{name} has an odd byte length")
    return np.frombuffer(raw, dtype="<u2").astype(np.int64)


def decode_rle(encoded: str, width: int, height: int) -> np.ndarray:
    """Decode and validate a row-major uint16 RLE mask.

    Returns a ``(height, width)`` boolean array.  Raises ``ValueError`` when
    the payload does not cover exactly ``width * height`` cells or contains a
    zero run anywhere except the leading 0-run the encoder emits for a mask
    that starts with a set bit.
    """
    width = int(width)
    height = int(height)
    total = width * height
    runs = _decode_runs(encoded, "rle")
    if runs.size == 0:
        # The structure encoder emits "" for an all-zero mask, regardless of
        # the grid size (there is no set bit to describe).
        return np.zeros((height, width), dtype=bool)
    zero_at = np.flatnonzero(runs == 0)
    if zero_at.size and not (zero_at.size == 1 and int(zero_at[0]) == 0):
        raise ValueError("rle contains a zero-length run")
    covered = int(runs.sum())
    if covered != total:
        raise ValueError(f"rle covers {covered} cells, expected {total}")
    if runs[0] == 0:
        runs = runs[1:]
        values = np.empty(runs.size, dtype=np.uint8)
        values[0::2] = 1
        values[1::2] = 0
    else:
        values = np.empty(runs.size, dtype=np.uint8)
        values[0::2] = 0
        values[1::2] = 1
    return np.repeat(values, runs).astype(bool).reshape(height, width)


def rle_true_count(encoded: str) -> int:
    """Number of set bits in a validated row-major RLE mask."""
    runs = _decode_runs(encoded, "rle")
    if runs.size == 0:
        return 0
    # Runs alternate 0-run, 1-run, ... (a leading 0-run shifts by one but the
    # set bits are still at odd indices).
    return int(runs[1::2].sum())


@dataclass(frozen=True)
class ParsedSubmap:
    """Validated fields extracted from one structure snapshot."""

    anchor_x: float
    anchor_y: float
    anchor_yaw_deg: float
    origin_x: float
    origin_y: float
    resolution_m: float
    width: int
    height: int
    occupied_rle: str
    free_rle: str
    walls: list[list[float]]
    gravity: dict
    camera_height_m: float


@dataclass(frozen=True)
class MergedSubmap:
    """Accumulated grid geometry in the pinned capture frame.

    ``origin_*``, ``walls`` and the masks are all in the frame of the first
    capture (the place's pinned anchor frame).  ``width``/``height`` describe
    the union bounding box at ``resolution_m``.
    """

    origin_x: float
    origin_y: float
    resolution_m: float
    width: int
    height: int
    occupied_rle: str
    free_rle: str
    walls: list[list[float]]


def _wall_matches(a, b, tolerance: float) -> bool:
    """True when two ``[x0, y0, x1, y1]`` segments describe the same line.

    Endpoint order is irrelevant: a wall seen from the opposite side is the
    same piece of evidence, not a second wall.
    """
    tol = float(tolerance)
    if (abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol
            and abs(a[2] - b[2]) <= tol and abs(a[3] - b[3]) <= tol):
        return True
    return (abs(a[0] - b[2]) <= tol and abs(a[1] - b[3]) <= tol
            and abs(a[2] - b[0]) <= tol and abs(a[3] - b[1]) <= tol)


def merge_walls(stored, incoming, resolution_m: float) -> list[list[float]]:
    """Union two wall lists, dropping near-duplicate segments.

    The wall extractor needs the per-cell height evidence that a stored
    raster no longer carries, so walls are accumulated as per-capture
    evidence rather than re-fitted from occupancy alone.  Re-observing a wall
    therefore never duplicates it, and an empty capture never removes one.
    """
    out = [list(w) for w in (stored or [])]
    tolerance = max(1e-6, float(resolution_m) / 2.0)
    for wall in incoming or []:
        candidate = list(wall)
        if not any(_wall_matches(candidate, seen, tolerance) for seen in out):
            out.append(candidate)
    return out


def _rebin_cells(mask: np.ndarray, origin_x: float, origin_y: float,
                 resolution_m: float, *, base_origin_x: float,
                 base_origin_y: float, base_resolution_m: float):
    """Map a grid's set cells onto the stored grid's lattice.

    Each set cell is represented by its centre in the capture frame,
    ``origin + (i + 0.5) * resolution_m``, then assigned to the base-lattice
    cell that contains it.  Returns integer ``(rows, cols)`` index arrays.
    """
    rows, cols = np.nonzero(np.asarray(mask, dtype=bool))
    if cols.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    world_x = float(origin_x) + (cols + 0.5) * float(resolution_m)
    world_y = float(origin_y) + (rows + 0.5) * float(resolution_m)
    # The epsilon keeps exact lattice centres (i + 0.5) from landing one cell
    # low because of binary floating-point round-off.
    base_cols = np.floor(
        (world_x - float(base_origin_x)) / float(base_resolution_m)
        + 1e-9).astype(np.int64)
    base_rows = np.floor(
        (world_y - float(base_origin_y)) / float(base_resolution_m)
        + 1e-9).astype(np.int64)
    return base_rows, base_cols


def merge_submaps(stored: ParsedSubmap, incoming: ParsedSubmap) -> MergedSubmap:
    """Accumulate ``incoming`` evidence into the stored submap.

    The union of both occupied sets is the new occupied layer; free space is
    the union of both free sets minus occupied cells (an obstacle always wins
    over free space).  Set cells are rebinned into a union bounding box at
    the stored resolution, so captures with different ``origin``,
    ``width``/``height`` (or even resolution) align by their capture-frame
    centres instead of overwriting one another.

    The real monotonicity invariant is **known cells** (occupied ∪ free):
    merging never turns a known cell back into unknown.  ``free_count`` alone
    is allowed to fall when a newly observed obstruction reclassifies free
    space as occupied.

    A capture with no cells at all leaves the stored geometry untouched; its
    walls, if any, are still merged.  This is what makes an empty/weak
    snapshot unable to shrink a better map.
    """
    base_ox = float(stored.origin_x)
    base_oy = float(stored.origin_y)
    base_res = float(stored.resolution_m)

    stored_occ = decode_rle(stored.occupied_rle, stored.width, stored.height)
    stored_free = decode_rle(stored.free_rle, stored.width, stored.height)
    incoming_occ = decode_rle(
        incoming.occupied_rle, incoming.width, incoming.height)
    incoming_free = decode_rle(
        incoming.free_rle, incoming.width, incoming.height)

    walls = merge_walls(stored.walls, incoming.walls, base_res)

    stored_set = stored_occ | stored_free
    incoming_set = incoming_occ | incoming_free
    if not stored_set.any() and not incoming_set.any():
        return MergedSubmap(
            origin_x=base_ox,
            origin_y=base_oy,
            resolution_m=base_res,
            width=int(stored.width),
            height=int(stored.height),
            occupied_rle=stored.occupied_rle,
            free_rle=stored.free_rle,
            walls=walls,
        )

    # Stored cells already sit on the base lattice: their indices are 0..w-1.
    stored_rows, stored_cols = np.nonzero(stored_set)
    # Incoming cells are projected onto that lattice.
    inc_rows, inc_cols = _rebin_cells(
        incoming_set, incoming.origin_x, incoming.origin_y,
        incoming.resolution_m, base_origin_x=base_ox,
        base_origin_y=base_oy, base_resolution_m=base_res)

    all_rows = np.concatenate((stored_rows, inc_rows))
    all_cols = np.concatenate((stored_cols, inc_cols))
    min_row = int(all_rows.min())
    max_row = int(all_rows.max())
    min_col = int(all_cols.min())
    max_col = int(all_cols.max())
    height = max_row - min_row + 1
    width = max_col - min_col + 1
    if width * height > MAX_GRID_CELLS:
        raise ValueError(
            f"merged grid has {width * height} cells; maximum is "
            f"{MAX_GRID_CELLS}")

    merged_occ = np.zeros((height, width), dtype=bool)
    merged_free = np.zeros((height, width), dtype=bool)

    def _paste(mask, rows, cols):
        if rows.size:
            mask[rows - min_row, cols - min_col] = True

    stored_occ_rows, stored_occ_cols = np.nonzero(stored_occ)
    stored_free_rows, stored_free_cols = np.nonzero(stored_free)
    _paste(merged_occ, stored_occ_rows, stored_occ_cols)
    _paste(merged_free, stored_free_rows, stored_free_cols)
    inc_occ = _rebin_cells(
        incoming_occ, incoming.origin_x, incoming.origin_y,
        incoming.resolution_m, base_origin_x=base_ox,
        base_origin_y=base_oy, base_resolution_m=base_res)
    inc_free = _rebin_cells(
        incoming_free, incoming.origin_x, incoming.origin_y,
        incoming.resolution_m, base_origin_x=base_ox,
        base_origin_y=base_oy, base_resolution_m=base_res)
    _paste(merged_occ, *inc_occ)
    _paste(merged_free, *inc_free)
    merged_free &= ~merged_occ

    return MergedSubmap(
        origin_x=base_ox + min_col * base_res,
        origin_y=base_oy + min_row * base_res,
        resolution_m=base_res,
        width=width,
        height=height,
        occupied_rle=encode_rle(merged_occ),
        free_rle=encode_rle(merged_free),
        walls=walls,
    )


def parse_guide_path(path) -> list[list[float]] | None:
    """Validate a caller-supplied guide path ([[x, y], ...], metres)."""
    if path is None:
        return None
    if not isinstance(path, (list, tuple)):
        raise ValueError("guide_path must be a list of [x, y] pairs")
    if len(path) > MAX_GUIDE_POINTS:
        raise ValueError(f"guide_path must have <= {MAX_GUIDE_POINTS} points")
    out: list[list[float]] = []
    for i, point in enumerate(path):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"guide_path[{i}] must be an [x, y] pair")
        out.append([_number(point[0], f"guide_path[{i}].x"),
                    _number(point[1], f"guide_path[{i}].y")])
    return out


def parse_structure_snapshot(payload, pose=None) -> ParsedSubmap:
    """Validate a ``GET /structure`` payload for storage as a place submap.

    Accepts either the snapshot itself or a wrapper ``{"structure": {...},
    "pose": {...}}``.  The pose defaults to the snapshot's own ``pose`` block
    (the visual map pose), never to wheel odometry.  A snapshot without valid
    gravity/floor is rejected: a submap must be gravity-aligned.
    """
    if not isinstance(payload, dict):
        raise ValueError("map payload must be a JSON object")
    snapshot = payload.get("structure", payload)
    if not isinstance(snapshot, dict):
        raise ValueError("structure must be a JSON object")

    if pose is None:
        if isinstance(payload.get("pose"), dict):
            pose = payload["pose"]
        elif isinstance(snapshot.get("pose"), dict):
            pose = snapshot["pose"]
    if not isinstance(pose, dict):
        raise ValueError("pose is required (x, y, yaw_deg)")
    if pose.get("valid") is False:
        raise ValueError("pose is not valid; refusing to anchor a submap")
    anchor_x = _number(pose.get("x"), "pose.x")
    anchor_y = _number(pose.get("y"), "pose.y")
    anchor_yaw = _number(pose.get("yaw_deg"), "pose.yaw_deg")

    gravity = snapshot.get("gravity")
    if not isinstance(gravity, dict) or gravity.get("valid") is not True:
        raise ValueError("structure.gravity.valid must be true "
                         "(a submap must be gravity-aligned)")
    up = gravity.get("up")
    if not isinstance(up, (list, tuple)) or len(up) != 3:
        raise ValueError("structure.gravity.up must be a 3-vector")
    up_out = [_number(up[i], f"gravity.up[{i}]") for i in range(3)]
    gravity_out = {
        "valid": True,
        "up": up_out,
        "roll_deg": _optional_number(gravity.get("roll_deg"),
                                     "gravity.roll_deg"),
        "pitch_deg": _optional_number(gravity.get("pitch_deg"),
                                      "gravity.pitch_deg"),
    }
    if gravity.get("inlier_ratio") is not None:
        gravity_out["inlier_ratio"] = _number(
            gravity.get("inlier_ratio"), "gravity.inlier_ratio")

    floor = snapshot.get("floor")
    if not isinstance(floor, dict) or floor.get("valid") is not True:
        raise ValueError("structure.floor.valid must be true "
                         "(a submap needs a floor estimate)")
    camera_height = _number(floor.get("camera_height_m"),
                            "floor.camera_height_m")
    if camera_height <= 0:
        raise ValueError("floor.camera_height_m must be positive")

    grid = snapshot.get("grid")
    if not isinstance(grid, dict):
        raise ValueError("structure.grid is required")
    resolution = _number(grid.get("resolution_m"), "grid.resolution_m")
    if resolution <= 0:
        raise ValueError("grid.resolution_m must be positive")
    width = _integer(grid.get("width"), "grid.width")
    height = _integer(grid.get("height"), "grid.height")
    if width * height > MAX_GRID_CELLS:
        raise ValueError(
            f"grid has {width * height} cells; maximum is {MAX_GRID_CELLS}")
    if grid.get("encoding") != "rle_base64":
        raise ValueError("grid.encoding must be 'rle_base64'")
    occupied_rle = grid.get("occupied_rle", "")
    free_rle = grid.get("free_rle", "")
    # Full round-trip validation: a truncated/overlong mask is rejected here,
    # not stored and decoded wrongly by a consumer later.
    decode_rle(occupied_rle, width, height)
    decode_rle(free_rle, width, height)

    origin = grid.get("origin")
    if not isinstance(origin, (list, tuple)) or len(origin) != 2:
        raise ValueError("grid.origin must be an [x, y] pair")
    origin_x = _number(origin[0], "grid.origin[0]")
    origin_y = _number(origin[1], "grid.origin[1]")

    raw_walls = snapshot.get("walls", [])
    if not isinstance(raw_walls, list):
        raise ValueError("structure.walls must be a list")
    if len(raw_walls) > MAX_WALLS:
        raise ValueError(f"structure.walls must have <= {MAX_WALLS} entries")
    walls: list[list[float]] = []
    for i, wall in enumerate(raw_walls):
        if not isinstance(wall, (list, tuple)) or len(wall) != 4:
            raise ValueError(f"walls[{i}] must be [x0, y0, x1, y1]")
        walls.append([_number(wall[j], f"walls[{i}][{j}]")
                      for j in range(4)])

    return ParsedSubmap(
        anchor_x=anchor_x,
        anchor_y=anchor_y,
        anchor_yaw_deg=anchor_yaw,
        origin_x=origin_x,
        origin_y=origin_y,
        resolution_m=resolution,
        width=width,
        height=height,
        occupied_rle=occupied_rle,
        free_rle=free_rle,
        walls=walls,
        gravity=gravity_out,
        camera_height_m=camera_height,
    )
