"""Synthetic tests for the learned master-BEV labels (no SAM, no hardware).

The scene is ray-cast through the same mount model as master_bev, so the
depth->body projection, color->depth sampling, floor plane, evidence
thresholds and conflict priority are exercised end to end.  SAM is replaced
by a synthetic hit-type map (floor / box).
"""

import json

import numpy as np
import pytest

from bebop_vision.bev_student_data import (read_index, split_by_drive,
                                           validate_session)
from bebop_vision.bev_student_labels import (
    FLOOR, HAZARD, IGNORE, OBSTACLE, WALL, CameraSpec, GridSpec,
    LabelConfig, camera_evidence, ego_mask, fit_plane, fuse_camera_evidence,
    height_above_plane, prepare_observation, prompt_class, resolve_labels,
    validate_label_arrays)
from bebop_vision.master_bev import mount_rotation

W, H = 64, 48
FX = FY = 50.0
CX, CY = 32.0, 24.0
GRID = GridSpec(range_m=4.0, width_m=2.0, cell_m=0.05)
LABEL_CFG = LabelConfig(min_range_m=0.30, floor_band_m=0.20)


def make_spec(role="far", pitch=-30.0, height=1.0):
    return CameraSpec(
        role=role, serial="TEST", width_d=W, height_d=H,
        fx_d=FX, fy_d=FY, cx_d=CX, cy_d=CY,
        width_c=W, height_c=H, fx_c=FX, fy_c=FY, cx_c=CX, cy_c=CY,
        stride_x=1, stride_y=1,
        R_body=mount_rotation(pitch, 0.0),
        t_body=np.array([0.0, 0.0, height], np.float32),
        R_dc=np.eye(3, dtype=np.float32),
        t_dc=np.zeros(3, np.float32),
    )


def render_scene(*, boxes=(), floors=((0.0, -20.0, 20.0),), spec=None):
    """Ray-cast a depth image + hit-type map (1 floor, 2 box)."""
    spec = spec or make_spec()
    R = spec.R_body.astype(np.float64)
    t = spec.t_body.astype(np.float64)
    u = (np.arange(W) - CX).astype(np.float64)
    v = (np.arange(H) - CY).astype(np.float64)
    uu, vv = np.meshgrid(u, v)
    dir_cam = np.stack([uu / FX, vv / FY, np.ones_like(uu)], axis=-1)
    D = dir_cam @ R.T
    depth = np.full((H, W), np.inf)
    hit_type = np.zeros((H, W), np.uint8)

    for z0, x0, x1 in floors:
        with np.errstate(divide="ignore", invalid="ignore"):
            tz = (z0 - t[2]) / D[..., 2]
        px = t[0] + tz * D[..., 0]
        py = t[1] + tz * D[..., 1]
        hit = (np.isfinite(tz) & (tz > 0) & (tz < 20.0)
               & (px >= x0) & (px < x1) & (np.abs(py) < 20.0))
        better = hit & (tz < depth)
        depth = np.where(better, tz, depth)
        hit_type[better] = 1

    for x0, x1, y0, y1, z0, z1 in boxes:
        tmin = np.full((H, W), -np.inf)
        tmax = np.full((H, W), np.inf)
        for lo, hi, o, A in ((x0, x1, t[0], D[..., 0]),
                             (y0, y1, t[1], D[..., 1]),
                             (z0, z1, t[2], D[..., 2])):
            with np.errstate(divide="ignore", invalid="ignore"):
                ta = (lo - o) / A
                tb = (hi - o) / A
            tmin = np.maximum(tmin, np.minimum(ta, tb))
            tmax = np.minimum(tmax, np.maximum(ta, tb))
        hit = ((tmin < tmax) & (tmax > 0) & (tmin > 0) & (tmin < 20.0))
        better = hit & (tmin < depth)
        depth = np.where(better, tmin, depth)
        hit_type[better] = 2

    depth_mm = np.where(np.isfinite(depth), depth * 1000.0, 0.0)
    return depth_mm.astype(np.uint16), hit_type


def label_scene(depth, hit_type, spec=None, cfg=None):
    spec = spec or make_spec()
    cfg = cfg or LABEL_CFG
    floor_conf = (hit_type == 1).astype(np.float32)
    obs_conf = (hit_type == 2).astype(np.float32)
    prep = prepare_observation(
        spec, depth, floor_conf=floor_conf, wall_conf=None,
        obstacle_conf=obs_conf, cfg=cfg, grid=GRID)
    plane = fit_plane(prep, cfg)
    assert plane["ok"], "synthetic floor plane must fit"
    ev = camera_evidence(prep, plane, cfg, GRID)
    return fuse_camera_evidence({"near": ev}, cfg, GRID,
                                ego=ego_mask(cfg, GRID))


def rc(x, y):
    return GRID.cell_of(x, y)


# --- grid convention -------------------------------------------------------

def test_grid_is_120x60_at_5cm():
    g = GridSpec()
    assert (g.rows, g.cols) == (120, 60)


def test_grid_convention_row0_far_col0_right():
    row, col = rc(3.0, 0.0)
    assert row == 20 and col == 20  # 1 m before the far edge, centre
    row, col = rc(1.0, 0.5)
    assert row == 60 and col == 30  # half-width left


# --- prompt mapping --------------------------------------------------------

def test_prompt_categories():
    assert prompt_class("floor") == FLOOR
    assert prompt_class("door frame") == WALL
    assert prompt_class("cardboard box") == OBSTACLE
    assert prompt_class("staircase") == HAZARD
    assert prompt_class("spaceship") is None


# --- projection / evidence -------------------------------------------------

def test_synthetic_floor_and_box_project_to_expected_cells():
    box = (1.45, 1.55, -0.20, 0.20, 0.00, 0.40)
    depth, hit = render_scene(boxes=(box,))
    out = label_scene(depth, hit)

    r, c = rc(3.0, 0.0)
    assert out.coverage[r, c]
    assert out.semantic[r, c] == FLOOR

    # The visible box face is its near side, x=1.45.
    r, c = rc(1.42, 0.0)
    assert out.coverage[r, c]
    assert out.semantic[r, c] == OBSTACLE
    assert np.sum(out.semantic == OBSTACLE) < 80


def test_hazard_from_step_down():
    box = (1.99, 2.01, -2.0, 2.0, -0.30, 0.0)  # vertical riser
    floors = ((0.0, -20.0, 2.0), (-0.30, 2.0, 20.0))
    depth, hit = render_scene(boxes=(box,), floors=floors)
    out = label_scene(depth, hit)
    # The riser occludes the near part of the lower floor; the first
    # visible drop cells are where the grazing ray clears its top (~2.6 m).
    cells = [rc(x, 0.0) for x in (2.7, 2.9, 3.1)]
    assert any(out.semantic[r, c] == HAZARD for r, c in cells)


def test_hazard_requires_floor_confirmation():
    box = (1.99, 2.01, -2.0, 2.0, -0.30, 0.0)
    floors = ((0.0, -20.0, 2.0), (-0.30, 2.0, 20.0))
    depth, _hit = render_scene(boxes=(box,), floors=floors)
    spec = make_spec()
    prep = prepare_observation(spec, depth, cfg=LABEL_CFG, grid=GRID)
    plane = fit_plane(prep, LABEL_CFG)
    ev = camera_evidence(prep, plane, LABEL_CFG, GRID)
    out = fuse_camera_evidence(
        {"far": ev}, LABEL_CFG, GRID, ego=ego_mask(LABEL_CFG, GRID))
    assert not np.any(out.semantic == HAZARD)


def test_config_loads_grid_and_ego():
    from bebop_vision.bev_student_labels import LabelConfig, specs_from_config
    from bebop_vision.orbbec import load_rig_config
    cfg = load_rig_config()["robots"]["default"]
    lc = LabelConfig.from_config(cfg)
    assert lc.ego_x_range_m == (0.0, 0.55)
    assert lc.hazard_require_floor is True
    specs, _cfg, grid, _lc = specs_from_config(roles=("near",))
    assert grid.shape == (120, 60)
    assert specs["near"].width_d == 848
    assert specs["near"].width_c == 1280


def test_ego_is_a_long_chassis_rectangle():
    mask = ego_mask(LabelConfig(), GridSpec())
    rr, cc = np.where(mask)
    assert (rr.max() - rr.min() + 1, cc.max() - cc.min() + 1) == (11, 8)
    assert int(mask.sum()) == 88


def test_conflict_priority_hazard_wall_obstacle_floor():
    cov = np.ones((4, 4), bool)
    ego = np.zeros((4, 4), bool)
    floor = np.ones((4, 4), np.float32)
    wall = np.zeros((4, 4), np.float32)
    obs = np.zeros((4, 4), np.float32)
    haz = np.zeros((4, 4), np.float32)
    wall[1, 1] = 1.0
    obs[1, 1] = 1.0
    haz[1, 1] = 1.0
    wall[2, 2] = 1.0
    obs[2, 2] = 1.0
    sem, _ = resolve_labels(floor, wall, obs, haz, cov, ego,
                            LabelConfig(), GridSpec(4 * 0.05, 4 * 0.05))
    assert sem[1, 1] == HAZARD
    assert sem[2, 2] == WALL
    assert sem[0, 0] == FLOOR


def test_ego_and_unobserved_are_ignore():
    cov = np.ones((3, 3), bool)
    ego = np.zeros((3, 3), bool)
    ego[0, 0] = True
    cov[2, 2] = False
    floor = np.ones((3, 3), np.float32)
    sem, _ = resolve_labels(floor, np.zeros((3, 3), np.float32),
                            np.zeros((3, 3), np.float32),
                            np.zeros((3, 3), np.float32),
                            cov, ego, LabelConfig(),
                            GridSpec(3 * 0.05, 3 * 0.05))
    assert sem[0, 0] == IGNORE
    assert sem[2, 2] == IGNORE
    assert sem[1, 1] == FLOOR


def test_height_above_plane():
    n = np.array([0.0, 0.0, 1.0])
    p = np.array([[0.0, 0.0, 0.1], [1.0, 2.0, -0.2]])
    h = height_above_plane(p, n, 0.0)
    np.testing.assert_allclose(h, [0.1, -0.2])


# --- contract --------------------------------------------------------------

def test_validate_label_contract():
    sem = np.zeros(GRID.shape, np.uint8)
    ego = np.zeros(GRID.shape, bool)
    cov = np.ones(GRID.shape, bool)
    validate_label_arrays(sem, ego, cov, GRID)

    ego[0, 0] = True
    with pytest.raises(ValueError):
        validate_label_arrays(sem, ego, cov, GRID)

    sem[0, 0] = IGNORE
    validate_label_arrays(sem, ego, cov, GRID)

    cov[0, 1] = False
    with pytest.raises(ValueError):
        validate_label_arrays(sem, ego, cov, GRID)


# --- extraction sampling ---------------------------------------------------

def test_extract_sample_grid_is_10hz():
    from tools.mcap_extract import _sample_stamps
    # ~21.5 Hz source ticks; the old minimum-interval filter gave ~8 Hz.
    stamps = [int(i * 46.5e6) for i in range(100)]
    picked = _sample_stamps(stamps, 10.0)
    span_s = (picked[-1] - picked[0]) / 1e9
    hz = (len(picked) - 1) / span_s
    assert 9.0 <= hz <= 10.5
    assert all(b > a for a, b in zip(picked, picked[1:]))


# --- drive split -----------------------------------------------------------

def _fake_session(root, name, n=2):
    sd = root / "sessions" / name
    (sd / "labels").mkdir(parents=True)
    rows = []
    for i in range(n):
        stamp = 1_000_000 + i
        sem = np.zeros(GRID.shape, np.uint8)
        ego = np.zeros(GRID.shape, bool)
        cov = np.ones(GRID.shape, bool)
        np.savez_compressed(sd / "labels" / f"{stamp:020d}.npz",
                            semantic=sem, ego=ego, coverage=cov)
        rows.append({"stamp_ns": stamp, "stamp_us": stamp // 1000,
                     "session": name, "frame": i,
                     "color_near": f"color_near/{stamp:020d}.jpg",
                     "color_far": f"color_far/{stamp:020d}.jpg",
                     "depth_near": f"depth_near/{stamp:020d}.png",
                     "depth_far": f"depth_far/{stamp:020d}.png",
                     "label": None, "split": None})
    with open(sd / "index.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return sd


def test_split_by_drive_never_leaks_frames(tmp_path):
    names = ["drive_a", "drive_b", "drive_c"]
    for name in names:
        _fake_session(tmp_path, name, n=3)
    # Missing image files are reported, but labels are what the split uses.
    payload = split_by_drive(tmp_path, val_fraction=1 / 3, seed=0)
    assert payload["val"] and payload["train"]
    assert set(payload["val"]).isdisjoint(payload["train"])

    for name in names:
        rows = read_index(tmp_path / "sessions" / name)
        splits = {r["split"] for r in rows}
        assert len(splits) == 1
        assert splits.pop() in ("train", "val")

    report = validate_session(tmp_path / "sessions" / names[0], grid=GRID)
    assert report["labeled"] == 3
    assert report["frames"] == 3
