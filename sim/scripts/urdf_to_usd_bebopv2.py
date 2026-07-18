# pyright: reportMissingImports=false
"""Headless URDF → USD pipeline for the Bebop V2 robot.

One-command replacement for the GUI flow (Isaac Sim → File → Import →
URDF, then `post_import_bebopv2.py` in the Script Editor). It:

  1. converts `ros2/src/bebopv2_description/urdf/bebopv2.urdf` to a
     layered USD asset at `sim/usd/bebopv2/` using Isaac Lab's
     `UrdfConverter` (same importer the GUI uses — importer 3.0 writes
     `<usd_dir>/<urdf-stem>/<urdf-stem>.usda` + `payloads/`), with
     importer settings matching the table in `ros2/README.md` →
     "URDF importer settings",
  2. applies the post-import fixes from `post_import_bebopv2.py`
     (floating base, spawn lift, IMU prim, foot contact-report API)
     directly to the USD file's root layer, and
  3. prints a sanity report (joint order, per-link masses, measured
     sole height vs. the configured `LIFT_Z`) so a future feet/leg
     redesign that silently changes the spawn geometry is caught here.

The whole `<usd_dir>/<urdf-stem>` directory is DELETED before each run:
it is a generated artifact (recoverable from git), and the importer does
not clean up renamed payload files between runs.

Usage (inside `bebop_isaac_lab`, or via `just lab-urdf-to-usd`):

    /workspace/isaaclab/isaaclab.sh -p \
        /workspace/bebop_bot/sim/scripts/urdf_to_usd_bebopv2.py --headless

Path defaults come from `BEBOP_PROJECT_ROOT` (set to `/workspace/bebop_bot`
in the compose file). Override with `--urdf` / `--usd-dir` if needed.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import sys
import xml.etree.ElementTree as ET

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# CLI + app launch (must happen before importing isaaclab.sim / omni modules)
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.environ.get("BEBOP_PROJECT_ROOT", "/workspace/bebop_bot")
DEFAULT_URDF = f"{PROJECT_ROOT}/ros2/src/bebopv2_description/urdf/bebopv2.urdf"
DEFAULT_USD_DIR = f"{PROJECT_ROOT}/sim/usd"

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--urdf", default=DEFAULT_URDF, help="input URDF (flat, absolute mesh paths)")
parser.add_argument("--usd-dir", default=DEFAULT_USD_DIR, help="output dir for the layered USD asset")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Everything below runs with the simulator app live
# ---------------------------------------------------------------------------

from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402

# post_import_bebopv2.py lives next to this script and guards its Script
# Editor entry point, so importing it here is side-effect free.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import post_import_bebopv2 as post  # noqa: E402

# Settle clearance between the zero-pose sole plane and the spawn height,
# matching the old asset (sole -0.7668 m, lift 0.8 m -> ~33 mm). Keep in
# sync with the derivation comment on `post.LIFT_Z`.
SOLE_CLEARANCE_M = 0.035


def _urdf_joint_names(urdf_path: str) -> list[str]:
    """Joint names in URDF document order (the order xacro emitted)."""
    root = ET.parse(urdf_path).getroot()
    return [j.attrib["name"] for j in root.iter("joint")]


def _usd_joint_names(stage, robot_path: str) -> list[str]:
    """Joint prim names under `<robot>/Physics`, in traversal order."""
    physics_scope = stage.GetPrimAtPath(f"{robot_path}/Physics")
    if not physics_scope.IsValid():
        return []
    return [
        child.GetName()
        for child in physics_scope.GetChildren()
        if "Joint" in child.GetTypeName()
    ]


def _link_masses(stage, robot_path: str) -> list[tuple[str, float]]:
    """(link name, mass kg) for every prim under Geometry with MassAPI."""
    out = []
    geom_root = stage.GetPrimAtPath(f"{robot_path}/Geometry")
    for prim in Usd.PrimRange(geom_root):
        mass_api = UsdPhysics.MassAPI(prim)
        mass_attr = mass_api.GetMassAttr()
        if mass_attr.IsValid() and mass_attr.HasAuthoredValue():
            out.append((prim.GetName(), mass_attr.Get()))
    return out


def _measure_sole_z(stage, robot_path: str) -> float:
    """Min-z of the robot's composed visual bbox, in the root frame.

    With the fresh import sitting at the origin (pre-lift), this is the
    sole plane height relative to `base_link` — the number `LIFT_Z` is
    derived from.
    """
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bound = cache.ComputeWorldBound(stage.GetPrimAtPath(robot_path))
    return bound.ComputeAlignedRange().GetMin()[2]


def convert(urdf_path: str, usd_dir: str) -> str:
    """Run the URDF importer; return the generated `.usda` path."""
    stem = pathlib.PurePath(urdf_path).stem
    robot_dir = os.path.join(usd_dir, stem)
    if os.path.isdir(robot_dir):
        print(f"[convert] removing existing generated asset: {robot_dir}")
        shutil.rmtree(robot_dir)

    cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=usd_dir,
        # Floating base — the post-import fixup would disable a fixed root
        # joint anyway. Matches the committed asset.
        fix_base=False,
        # No fixed joints exist in this URDF; set explicitly to mirror
        # Isaac Lab's convert_urdf.py default behaviour.
        merge_fixed_joints=False,
        force_usd_conversion=True,
        # Preserve the drive values the importer derives from the URDF
        # (None/None). Isaac Lab's actuator cfg (DCMotorCfg in
        # exp_standing.py) overrides drive gains at spawn, so these are
        # cosmetic for training.
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None),
        ),
        # Remaining fields stay at defaults, which match the documented
        # GUI settings in ros2/README.md: collision_from_visuals=False
        # (URDF declares explicit collision geometry), self_collision=False,
        # merge_mesh=False, collision_type="Convex Hull",
        # run_asset_transformer=True (layered payloads output).
    )
    converter = UrdfConverter(cfg)
    print(f"[convert] generated USD: {converter.usd_path}")
    return converter.usd_path


def print_sanity_report(stage, robot_path: str, urdf_path: str) -> None:
    print("-" * 72)
    urdf_joints = _urdf_joint_names(urdf_path)
    usd_joints = _usd_joint_names(stage, robot_path)
    missing = [n for n in urdf_joints if n not in usd_joints]
    extra = [n for n in usd_joints if n not in urdf_joints]
    print(f"[sanity] joints: {len(usd_joints)} in USD, {len(urdf_joints)} in URDF")
    if missing:
        print(f"    ERROR — URDF joints missing from USD: {missing}")
    if extra:
        print(f"    USD-only joints (importer-added, e.g. root joint): {extra}")
    # NOTE: USD prim traversal order is NOT the articulation joint order
    # and URDF document order never was the reference — Isaac Lab resolves
    # joint order at spawn, where `preserve_order=True` on the action and
    # observation terms forces firmware order and the assertion in
    # bebop_v2_actions.py fails fast on any mismatch. This list is for
    # eyeballing completeness only.
    print(f"[sanity] USD Physics-scope traversal order: {usd_joints}")

    masses = _link_masses(stage, robot_path)
    print("[sanity] link masses (kg):")
    for name, mass in masses:
        print(f"    {name:<24} {mass:.4f}")
    print(f"    {'TOTAL':<24} {sum(m for _, m in masses):.4f}")

    sole_z = _measure_sole_z(stage, robot_path)
    recommended = -sole_z + SOLE_CLEARANCE_M
    print("[sanity] spawn height check (measured pre-lift, zero pose):")
    print(f"    sole plane min-z in root frame : {sole_z:+.4f} m")
    print(f"    recommended lift (-z + {SOLE_CLEARANCE_M} m) : {recommended:.4f} m")
    print(f"    configured LIFT_Z              : {post.LIFT_Z:.4f} m")
    if abs(post.LIFT_Z - recommended) > 0.01:
        print(
            "    WARNING: LIFT_Z differs from the measured recommendation by "
            ">10 mm. The feet design (or leg kinematics) changed — update "
            "LIFT_Z in post_import_bebopv2.py and the spawn heights in "
            "exp_standing.py / exp_mirror.py."
        )
    print("-" * 72)


def main() -> None:
    urdf_path = os.path.abspath(args_cli.urdf)
    usd_dir = os.path.abspath(args_cli.usd_dir)
    if not os.path.isfile(urdf_path):
        raise FileNotFoundError(
            f"URDF not found: {urdf_path}\n"
            "Regenerate it first with `just ros2-urdf` (see ros2/README.md)."
        )

    usd_path = convert(urdf_path, usd_dir)

    # Pure-pxr stage access: deterministic in headless mode (no async
    # omni.usd context open), and the IMU is authored via the pxr fallback
    # inside apply_fixes.
    stage = Usd.Stage.Open(usd_path)
    robot_path = post._find_robot_path(stage)
    print(f"[post_import] robot root prim: {robot_path}")

    print_sanity_report(stage, robot_path, urdf_path)
    post.apply_fixes(stage, robot_path, imu_via_kit_command=False)

    stage.GetRootLayer().Save()
    print(f"[done] fixes baked into root layer: {stage.GetRootLayer().realPath}")
    print("[done] next: `just lab-play --num_envs 1` to verify the asset "
          "loads and the joint-order assertion passes.")


if __name__ == "__main__":
    main()
    simulation_app.close()
