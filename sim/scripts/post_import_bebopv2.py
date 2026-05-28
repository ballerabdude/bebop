# pyright: reportMissingImports=false
"""Post-import fixup for the Bebop V2 robot in Isaac Sim.

Run this *inside Isaac Sim's Script Editor* immediately after using the
URDF importer to convert `ros2/src/bebopv2_description/urdf/bebopv2.urdf`
to USD. The importer leaves the asset in three states that are wrong
for a free-standing biped:

  1. A **fixed root joint** is added that anchors `base_link` to world.
     For a walking biped we want `base_link` to be a free-floating
     dynamic body so PhysX simulates it under gravity.
  2. The robot prim sits at world origin (0, 0, 0). Because the URDF's
     `base_link` origin is at the **hip**, half the robot is below the
     ground plane on first Play. We lift the whole robot by 0.65 m so
     it spawns standing on the floor.
  3. There's no IMU sensor on `base_link`, but the on-robot stack and
     the trained policy both consume `/imu/data` (orientation, angular
     velocity, linear acceleration of the base frame). We attach an
     IMU prim so the asset matches the real robot wherever it's
     loaded — Isaac Sim standalone, Isaac Lab, etc.
  4. The two foot rigid bodies (`foot_left_1`, `foot_right_1`) lack
     `PhysxContactReportAPI`, which Isaac Lab's `ContactSensor` requires
     to surface ground-contact events. Isaac Lab's spawn-time helper
     `sim.utils.schemas.activate_contact_sensors` walks the asset and
     adds the API to each rigid body it finds — BUT its walker stops
     descending once it hits the first rigid body
     (assumes "nested rigid bodies are not allowed by SDK", which is
     true for flat humanoid USDs like H1 but not for URDF-imported
     articulations where the kinematic tree IS the body tree). On our
     V2 USD the walker stops at `base_link` and never reaches the
     feet. Pre-baking the contact-report API onto the foot prims in
     USD fixes this once and for all.

All fixes are idempotent — re-running the script is a no-op if the
fixes are already in place.

Usage
-----

In the Script Editor (Window → Script Editor in Isaac Sim):

    1. Drop the imported `bebopv2` USD into your stage if it isn't
       already there.
    2. Paste this whole file in and hit Run.
    3. Press Play — the robot should fall under gravity and stand on
       the ground plane.

You can also load it via the Editor's "Open" button — it lives at
`sim/scripts/post_import_bebopv2.py` in the repo (which is bind-mounted
into the Isaac Lab/Sim container at
`/workspace/bebop_bot/sim/scripts/post_import_bebopv2.py`).
"""

from __future__ import annotations

import omni.kit.commands
import omni.usd
from pxr import Gf, UsdGeom, UsdPhysics

ROBOT_CANDIDATES = ["/World/bebopv2", "/bebopv2"]
LIFT_Z = 0.8  # meters — half the leg length, so the foot lands on z=0

# Foot rigid bodies (relative to `<robot>/Geometry/base_link`) that need
# `PhysxContactReportAPI` baked in for Isaac Lab's `ContactSensor` to
# detect ground contacts. See `_enable_foot_contact_report` for the
# full rationale.
FOOT_BODY_RELATIVE_PATHS = (
    "hip_flexion_left_1/hip_abduction_left_1/knee_flexion_left_1/foot_left_1",
    "hip_flexion_right_1/hip_abduction_right_1/knee_flexion_right_1/foot_right_1",
)

# IMU mount on base_link. Translation is in the base_link frame; tweak
# to match the physical BNO085 mount on the real robot if you have it.
IMU_PRIM_NAME = "Imu_Sensor"
IMU_FREQUENCY_HZ = 200.0
IMU_TRANSLATION = Gf.Vec3d(0.0, 0.0, 0.0)
IMU_ORIENTATION = Gf.Quatd(1.0, 0.0, 0.0, 0.0)  # identity (w, x, y, z)


def _find_robot_path(stage) -> str:
    for path in ROBOT_CANDIDATES:
        if stage.GetPrimAtPath(path).IsValid():
            return path
    raise RuntimeError(
        f"Robot not found at any of {ROBOT_CANDIDATES}. "
        "Drop the imported `bebopv2` USD onto the stage first."
    )


def _disable_root_joint(stage, robot_path: str) -> None:
    root_joint_path = f"{robot_path}/Physics/root_joint"
    root_joint_prim = stage.GetPrimAtPath(root_joint_path)
    if not root_joint_prim.IsValid():
        print(f"[post_import] no root joint at {root_joint_path}; skipping")
        return

    joint = UsdPhysics.Joint(root_joint_prim)
    enabled_attr = joint.GetJointEnabledAttr()
    if not enabled_attr.IsValid():
        enabled_attr = joint.CreateJointEnabledAttr()
    enabled_attr.Set(False)
    print(f"[post_import] disabled fixed root joint: {root_joint_path}")


def _ensure_dynamic_base(stage, robot_path: str) -> None:
    base_link_path = f"{robot_path}/Geometry/base_link"
    rb = UsdPhysics.RigidBodyAPI.Get(stage, base_link_path)
    if not rb:
        print(f"[post_import] no RigidBodyAPI on {base_link_path}; skipping")
        return

    kin_attr = rb.GetKinematicEnabledAttr()
    if kin_attr.IsValid() and kin_attr.Get():
        kin_attr.Set(False)
        print(f"[post_import] set kinematicEnabled=False on {base_link_path}")


def _lift_robot(stage, robot_path: str, dz: float = LIFT_Z) -> None:
    robot_prim = stage.GetPrimAtPath(robot_path)
    xformable = UsdGeom.Xformable(robot_prim)

    translate_op = None
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
            break
    if translate_op is None:
        translate_op = xformable.AddTranslateOp()

    translate_op.Set(Gf.Vec3d(0.0, 0.0, float(dz)))
    print(f"[post_import] translate {robot_path} → (0, 0, {dz})")


def _add_imu(stage, robot_path: str) -> None:
    base_link_path = f"{robot_path}/Geometry/base_link"
    base_link_prim = stage.GetPrimAtPath(base_link_path)
    if not base_link_prim.IsValid():
        print(f"[post_import] no base_link at {base_link_path}; skipping IMU")
        return

    imu_path = f"{base_link_path}/{IMU_PRIM_NAME}"
    if stage.GetPrimAtPath(imu_path).IsValid():
        print(f"[post_import] IMU already present at {imu_path}; skipping")
        return

    # `IsaacSensorCreateImuSensor` is the stable, version-portable way
    # to add an IMU prim under a parent rigid body. It returns
    # (success, prim) — we only care about the success flag.
    #
    # NOTE: the `visualize` kwarg was removed in Isaac Sim 5.x; do not
    # add it back. If you need debug visualization, toggle it on the
    # prim's `IsaacSensorAPI` after creation, or set `debug_vis=True`
    # in Isaac Lab's `ImuCfg`.
    success, _ = omni.kit.commands.execute(
        "IsaacSensorCreateImuSensor",
        path=f"/{IMU_PRIM_NAME}",
        parent=base_link_path,
        sensor_period=1.0 / IMU_FREQUENCY_HZ,
        translation=IMU_TRANSLATION,
        orientation=IMU_ORIENTATION,
    )
    if success:
        print(
            f"[post_import] added IMU at {imu_path} "
            f"({IMU_FREQUENCY_HZ:.0f} Hz)"
        )
    else:
        print(f"[post_import] WARNING: IsaacSensorCreateImuSensor failed for {imu_path}")


def _enable_foot_contact_report(stage, robot_path: str) -> None:
    """Apply `PhysxContactReportAPI` to each foot rigid body.

    Isaac Lab's `sim.utils.schemas.activate_contact_sensors` (the helper
    that runs at spawn time when `UsdFileCfg.activate_contact_sensors=True`)
    walks the asset tree and adds the API to each rigid body it finds —
    but its walker stops descending once it encounters the first
    rigid body (see comment ``"nested rigid bodies are not allowed by
    SDK"`` in `isaaclab/sim/schemas/schemas.py::activate_contact_sensors`).
    For our URDF-imported V2 USD the walker stops at `base_link` and
    never reaches the leg/foot bodies, so the `ContactSensor` errors out
    with ``could not find any bodies with contact reporter API`` even
    when `activate_contact_sensors=True` is set on the spawn cfg.

    Pre-baking the API onto the foot prims fixes this regardless of
    what the spawn-time walker does, and is also visible to PhysX at
    parse time (no runtime schema mutation needed).
    """
    base_link_path = f"{robot_path}/Geometry/base_link"
    for rel_path in FOOT_BODY_RELATIVE_PATHS:
        prim_path = f"{base_link_path}/{rel_path}"
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"[post_import] no foot prim at {prim_path}; skipping")
            continue

        applied = list(prim.GetAppliedSchemas())
        for schema in ("PhysxRigidBodyAPI", "PhysxContactReportAPI"):
            if schema not in applied:
                prim.AddAppliedSchema(schema)

        threshold_attr = prim.GetAttribute("physxContactReport:threshold")
        if not threshold_attr.IsValid():
            threshold_attr = prim.CreateAttribute(
                "physxContactReport:threshold",
                "float",  # type: ignore[arg-type]
            )
        threshold_attr.Set(0.0)

        sleep_attr = prim.GetAttribute("physxRigidBody:sleepThreshold")
        if not sleep_attr.IsValid():
            sleep_attr = prim.CreateAttribute(
                "physxRigidBody:sleepThreshold",
                "float",  # type: ignore[arg-type]
            )
        sleep_attr.Set(0.0)

        print(f"[post_import] enabled contact-report API on {prim_path}")


def main() -> None:
    stage = omni.usd.get_context().get_stage()
    robot_path = _find_robot_path(stage)

    _disable_root_joint(stage, robot_path)
    _ensure_dynamic_base(stage, robot_path)
    _lift_robot(stage, robot_path)
    _add_imu(stage, robot_path)
    _enable_foot_contact_report(stage, robot_path)

    print(
        "[post_import] done. Press Play — the robot should fall under "
        "gravity onto the ground plane."
    )


main()
