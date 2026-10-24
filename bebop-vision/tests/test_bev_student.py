"""Synthetic tests for the SAM 3.1 x depth labeler (no SAM, no hardware).

The scene is ray-cast through the same mount model as master_bev, so the
depth->body projection, color->depth sampling, floor plane, evidence
thresholds and conflict priority are exercised end to end.  SAM is replaced
by a synthetic hit-type map (floor / box).
"""

import dataclasses
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


def test_rectangular_grid_keeps_metric_coordinates_and_ego():
    grid = GridSpec(rows_override=360, cols_override=640)
    assert grid.shape == (360, 640)
    assert grid.cell_x_m == pytest.approx(6 / 360)
    assert grid.cell_y_m == pytest.approx(3 / 640)
    x, y = grid.centers()
    for row, col in ((0, 0), (180, 320), (359, 639)):
        assert grid.cell_of(x[row], y[col]) == (row, col)
    ego = ego_mask(LabelConfig(), grid)
    rr, cc = np.where(ego)
    assert np.all((x[rr] >= 0) & (x[rr] <= 0.55))
    assert np.all(np.abs(y[cc]) <= 0.22)
    assert ego.sum() * grid.cell_x_m * grid.cell_y_m == pytest.approx(
        0.55 * 0.44, abs=0.005)
    with pytest.raises(ValueError):
        GridSpec(rows_override=360)


def test_rectangular_labels_project_native_points_and_keep_drop_hazards():
    grid = GridSpec(range_m=4.0, width_m=2.0,
                    rows_override=360, cols_override=640)
    spec = make_spec()
    depth, hits = render_scene(
        # one real stair riser (18 cm): deeper depth-only drops are treated
        # as glossy-floor reflections unless SAM says stairs
        boxes=((1.99, 2.01, -2.0, 2.0, -0.18, 0.0),),
        floors=((0.0, -20.0, 2.0), (-0.18, 2.0, 20.0)), spec=spec)
    prep = prepare_observation(spec, depth, floor_conf=(hits == 1).astype(np.float32),
                               obstacle_conf=(hits == 2).astype(np.float32),
                               cfg=LABEL_CFG, grid=grid)
    plane = fit_plane(prep, LABEL_CFG)
    assert plane["ok"]
    evidence = camera_evidence(prep, plane, LABEL_CFG, grid)
    label = fuse_camera_evidence({"near": evidence}, LABEL_CFG, grid)
    validate_label_arrays(label.semantic, label.ego, label.coverage, grid)
    assert np.any(label.semantic == HAZARD)
    assert np.any(label.semantic == FLOOR)
    # Every covered cell must come from an actual deprojected depth sample.
    covered = set(np.flatnonzero(label.coverage))
    projected = set(prep.flat[prep.in_grid])
    assert covered <= projected
    for idx in np.flatnonzero(prep.in_grid)[::25]:
        assert grid.cell_of(*prep.points[idx, :2]) == (
            prep.rows[idx], prep.cols[idx])


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
    box = (1.99, 2.01, -2.0, 2.0, -0.18, 0.0)  # one vertical stair riser
    floors = ((0.0, -20.0, 2.0), (-0.18, 2.0, 20.0))
    depth, hit = render_scene(boxes=(box,), floors=floors)
    out = label_scene(depth, hit)
    # The riser occludes the near part of the lower floor; the first
    # visible drop cells are where the grazing ray clears its top (~2.4 m).
    xc, _ = GRID.centers()
    band = out.semantic[(xc >= 2.4) & (xc <= 4.0)]
    assert np.any(band == HAZARD)
    assert not np.any(out.semantic[xc < 1.9] == HAZARD)


def test_stairs_prompt_marks_above_plane_as_obstacle():
    # An ascending stair/step surface is above the floor plane.  The
    # stairs/step prompt must make it obstacle evidence, not floor/ignore.
    box = (1.45, 1.55, -0.20, 0.20, 0.00, 0.40)
    depth, hit = render_scene(boxes=(box,))
    spec = make_spec()
    prep = prepare_observation(
        spec, depth, hazard_conf=(hit == 2).astype(np.float32),
        cfg=LABEL_CFG, grid=GRID)
    plane = fit_plane(prep, LABEL_CFG)
    ev = camera_evidence(prep, plane, LABEL_CFG, GRID)
    assert ev["occ_w"].sum() > 0.0
    assert ev["floor_w"].sum() < ev["cov_w"].sum()


def test_hazard_requires_floor_confirmation():
    box = (1.99, 2.01, -2.0, 2.0, -0.18, 0.0)
    floors = ((0.0, -20.0, 2.0), (-0.18, 2.0, 20.0))
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
    # Chassis rectangle measured from the near camera (2026-10-19).
    assert lc.ego_x_range_m == (-0.10, 0.29)
    assert lc.ego_y_range_m == (-0.27, 0.17)
    assert lc.cull_rect_m is None and lc.cull_ego_margin_m == 0.02
    assert lc.hazard_ego_margin_m == 0.15
    assert lc.ego_mask_file == "ego_mask.npz"
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


def test_fit_ground_plane_handles_sparse_point_sets():
    """A near-floor band with fewer than iters*3 points must still fit
    (regression: numpy refused to sample 96 triplets without replacement
    from ~100 points and the stride-1 relabel crashed mid-drive)."""
    from bebop_vision.master_bev import fit_ground_plane

    rng = np.random.default_rng(0)
    n = 100  # between min_inliers (50) and iters * 3 (288)
    pts = np.column_stack([rng.uniform(0.5, 3.0, n), rng.uniform(-1, 1, n),
                           rng.normal(0.0, 0.003, n)])
    fit = fit_ground_plane(pts, rng=rng)
    assert fit is not None
    nvec, d, count = fit
    assert nvec[2] > 0.99 and abs(d) < 0.02 and count >= 50


def _coverage_rows_known(label, grid, x0, x1, half_width=0.3):
    """Covered fraction of the band [x0, x1) forward, |y| < half_width."""
    r0 = int((grid.range_m - x1) / grid.cell_x_m)
    r1 = int((grid.range_m - x0) / grid.cell_x_m)
    y = (np.arange(grid.cols) + 0.5) * grid.cell_y_m - grid.width_m / 2
    return float(label.coverage[r0:r1][:, np.abs(y) < half_width].mean())


def test_footprint_splat_fills_sampling_gaps_but_keeps_walls_thin():
    """On a fine grid the point projection leaves unobserved stripes between
    depth rows; the footprint splat fills them.  A vertical face is a depth
    discontinuity, so it must not smear forward."""
    grid = GridSpec(range_m=4.0, width_m=2.0,
                    rows_override=240, cols_override=400)   # 1.67 x 0.5 cm
    spec = make_spec(pitch=-20.0, height=1.0)
    box = (2.0, 2.4, -0.4, 0.4, 0.0, 0.5)
    depth, hits = render_scene(boxes=(box,), spec=spec)
    floor_conf = (hits == 1).astype(np.float32)
    obs_conf = (hits == 2).astype(np.float32)

    def run(cfg):
        prep = prepare_observation(spec, depth, floor_conf=floor_conf,
                                   obstacle_conf=obs_conf, cfg=cfg, grid=grid)
        plane = fit_plane(prep, cfg)
        assert plane["ok"]
        ev = camera_evidence(prep, plane, cfg, grid)
        return prep, fuse_camera_evidence({"near": ev}, cfg, grid)

    point_cfg = dataclasses.replace(LABEL_CFG)
    splat_cfg = dataclasses.replace(LABEL_CFG, splat_footprint=True)
    prep_p, lab_p = run(point_cfg)
    prep_s, lab_s = run(splat_cfg)
    assert prep_p.splat_src is None and prep_s.splat_src is not None
    validate_label_arrays(lab_s.semantic, lab_s.ego, lab_s.coverage, grid)

    # The floor in front of the box (1.0-2.0 m) is striped under point
    # projection and (nearly) solid under the splat.
    assert _coverage_rows_known(lab_p, grid, 1.0, 1.9) < 0.3
    assert _coverage_rows_known(lab_s, grid, 1.0, 1.9) > 0.9
    # Splat only adds coverage; every point-projected cell is still covered.
    assert np.all(lab_s.coverage[lab_p.coverage])
    # Semantics agree wherever both runs classified the cell (the splat also
    # turns covered-but-unclassified point cells into classified ones, which
    # is the intended gain, not a disagreement).
    both = (lab_p.semantic != IGNORE) & (lab_s.semantic != IGNORE)
    assert np.mean(lab_p.semantic[both] == lab_s.semantic[both]) > 0.95
    # The box's front face is a vertical wall of obstacle points at x=2.0;
    # the splat must not push obstacle evidence in front of it.
    x = grid.range_m - (np.arange(grid.rows) + 0.5) * grid.cell_x_m
    obs_rows = np.flatnonzero((lab_s.semantic == OBSTACLE).any(axis=1))
    assert x[obs_rows].min() > 2.0 - 0.06
    # Every splat cell belongs to an in-grid sample and lies in the grid.
    assert np.all(prep_s.in_grid[prep_s.splat_src])
    assert prep_s.splat_flat.min() >= 0
    assert prep_s.splat_flat.max() < grid.rows * grid.cols


def test_splat_patch_is_bounded_by_max_extent():
    """Neighbours across a depth jump or beyond splat_max_m do not extend a
    patch: a sample never covers more than (max/cell)+1 cells per axis."""
    grid = GridSpec(range_m=4.0, width_m=2.0,
                    rows_override=240, cols_override=400)
    spec = make_spec(pitch=-20.0, height=1.0)
    depth, hits = render_scene(boxes=((2.0, 2.4, -0.4, 0.4, 0.0, 0.5),),
                               spec=spec)
    cfg = dataclasses.replace(LABEL_CFG, splat_footprint=True, splat_max_m=0.05)
    prep = prepare_observation(spec, depth, floor_conf=(hits == 1).astype(np.float32),
                               cfg=cfg, grid=grid)
    counts = np.bincount(prep.splat_src, minlength=len(prep.points))
    max_rows = int(0.05 / grid.cell_x_m) + 2
    max_cols = int(0.05 / grid.cell_y_m) + 2
    assert counts.max() <= max_rows * max_cols


def test_label_config_reads_splat_from_rig_yaml():
    cfg = LabelConfig.from_config(
        {"bev_student": {"label": {"splat_footprint": True, "splat_max_m": 0.2}}})
    assert cfg.splat_footprint is True and cfg.splat_max_m == 0.2
    assert LabelConfig.from_config({}).splat_footprint is False


def test_rect_cull_keeps_floor_beside_chassis_and_drops_inside():
    """The rectangular near-body cull replaces the 0.55 m dead disc: floor
    beside the chassis (inside the old disc) survives, points inside the
    rectangle do not."""
    spec = make_spec(pitch=-60.0, height=1.0)
    depth, hits = render_scene(spec=spec)
    floor_conf = (hits == 1).astype(np.float32)
    disc_cfg = dataclasses.replace(LABEL_CFG, min_range_m=0.55)
    rect = (-0.30, 0.35, -0.28, 0.17)
    rect_cfg = dataclasses.replace(LABEL_CFG, min_range_m=0.55, cull_rect_m=rect)
    p_disc = prepare_observation(spec, depth, floor_conf=floor_conf, cfg=disc_cfg, grid=GRID)
    p_rect = prepare_observation(spec, depth, floor_conf=floor_conf, cfg=rect_cfg, grid=GRID)
    r_disc = np.hypot(p_disc.points[:, 0], p_disc.points[:, 1])
    assert r_disc.min() >= 0.55
    x, y = p_rect.points[:, 0], p_rect.points[:, 1]
    inside = (x >= rect[0]) & (x <= rect[1]) & (y >= rect[2]) & (y <= rect[3])
    assert not inside.any()
    beside = (np.hypot(x, y) < 0.55) & (y > rect[3])
    assert beside.sum() > 0, "floor beside the chassis must survive the rect cull"
    assert len(p_rect.points) > len(p_disc.points)


def test_label_config_cull_rect_from_yaml_and_validation():
    cfg = LabelConfig.from_config(
        {"bev_student": {"label": {"cull_rect_m": [-0.3, 0.35, -0.28, 0.17]}}})
    assert cfg.cull_rect_m == (-0.3, 0.35, -0.28, 0.17)
    with pytest.raises(ValueError):
        LabelConfig.from_config({"bev_student": {"label": {"cull_rect_m": [0.3, -0.3, 0, 1]}}})
    assert LabelConfig.from_config({}).cull_rect_m is None
