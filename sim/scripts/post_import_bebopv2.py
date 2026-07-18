# pyright: reportMissingImports=false
"""Post-import fixup for the Bebop V2 robot in Isaac Sim.

This module is used two ways:

1. **GUI / Script Editor** (documented manual flow): run it *inside Isaac
   Sim's Script Editor* immediately after using the URDF importer to
   convert `ros2/src/bebopv2_description/urdf/bebopv2.urdf` to USD. Paste
   the whole file in and hit Run (the Script Editor executes it as
   `__main__`), or open it via the Editor's "Open" button — it lives at
   `sim/scripts/post_import_bebopv2.py` in the repo (bind-mounted into
   the Isaac containers at
   `/workspace/bebop_bot/sim/scripts/post_import_bebopv2.py`).

2. **Headless pipeline**: `sim/scripts/urdf_to_usd_bebopv2.py` imports
   :func:`apply_fixes` and runs it against the freshly converted USD
   stage — no GUI needed. That is the preferred flow; see
   `ros2/README.md` → "Importing the URDF into Isaac Sim (URDF → USD)".

Either way, the importer leaves the asset in states that are wrong for a
free-standing biped:

  1. A **fixed root joint** is added that anchors `base_link` to world.
     For a walking biped we want `base_link` to be a free-floating
     dynamic body so PhysX simulates it under gravity.
  2. The robot prim sits at world origin (0, 0, 0). Because the URDF's
     `base_link` origin is at the **hip**, half the robot is below the
     ground plane on first Play. We lift the whole robot by
     :data:`LIFT_Z` so it spawns standing on the floor.
  3. There's no IMU sensor on `base_link`, but the on-robot stack and
     the trained policy both consume `/imu/data` (orientation, angular
     velocity, linear acceleration of the base frame). We attach an
     IMU prim so the asset matches the real robot wherever it's
     loaded — Isaac Sim standalone, Isaac Lab, etc. (Isaac Lab's own
     `ImuCfg` reads the `base_link` rigid body directly and does not
     consume this prim; it exists for Isaac Sim standalone parity.)
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

All fixes are idempotent — re-running them is a no-op if the fixes are
already in place.

After running in the Script Editor: press Play — the robot should fall
under gravity and stand on the ground plane.
"""

from __future__ import annotations

import omni.kit.commands
import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdPhysics

ROBOT_CANDIDATES = ["/World/bebopv2", "/bebopv2"]

# Spawn height of the robot root, chosen so the feet land ON the ground
# plane — not inside it, not hovering above it.
#
# Derivation (Jul 2026 feet redesign): with all joints at zero, the foot
# sole plane sits 0.7302 m below `base_link` (min-z vertex of the foot
# STLs x the 0.001 mesh scale; the STL world frame == base_link frame
# because each leg's joint-origin z-offsets exactly cancel the foot
# mesh's visual-origin z). The previous feet measured 0.7668 m, which
# matched the old 0.8 m lift (~33 mm settle clearance). We keep the same
# ~35 mm clearance: 0.7302 + 0.035 = 0.7652 -> 0.765.
#
# `sim/scripts/urdf_to_usd_bebopv2.py` re-measures this from the
# converted asset on every run and warns if it drifts. Keep the value in
# sync with `InitialStateCfg.pos` in `bebop_training/experiments/
# exp_standing.py` and `MIRROR_BASE_POS` in `exp_mirror.py`.
LIFT_Z = 0.765  # meters — sole-at-zero (0.7302) + 0.035 settle clearance

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


def _add_imu_pxr(stage, base_link_path: str, imu_path: str) -> None:
    """Author the IMU prim with plain pxr (no Kit commands).

    Used by the headless pipeline, where the Script-Editor Kit command
    may be unavailable. Replicates exactly the prim that
    `IsaacSensorCreateImuSensor` produces (attr-for-attr, checked against
    the previously committed GUI-imported asset).
    """
    prim = stage.DefinePrim(imu_path, "IsaacImuSensor")
    # custom=False: these are IsaacImuSensor schema attrs — author them as
    # such (matches what IsaacSensorCreateImuSensor writes), not as custom
    # user attributes.
    prim.CreateAttribute("enabled", Sdf.ValueTypeNames.Bool, custom=False).Set(True)
    prim.CreateAttribute("sensorPeriod", Sdf.ValueTypeNames.Float, custom=False).Set(
        1.0 / IMU_FREQUENCY_HZ
    )
    prim.CreateAttribute(
        "angularVelocityFilterWidth", Sdf.ValueTypeNames.Int, custom=False
    ).Set(1)
    prim.CreateAttribute(
        "linearAccelerationFilterWidth", Sdf.ValueTypeNames.Int, custom=False
    ).Set(1)
    prim.CreateAttribute(
        "orientationFilterWidth", Sdf.ValueTypeNames.Int, custom=False
    ).Set(1)

    xformable = UsdGeom.Xformable(prim)
    xformable.AddTranslateOp().Set(IMU_TRANSLATION)
    # AddOrientOp/AddScaleOp author float-precision ops by default (the
    # current IsaacImuSensor schema expects quatf/float3), so convert —
    # the older Kit command wrote quatd/double3, which is equivalent.
    xformable.AddOrientOp().Set(Gf.Quatf(IMU_ORIENTATION))
    xformable.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
    print(f"[post_import] authored IMU prim at {imu_path} via pxr")


def _add_imu(stage, robot_path: str, use_kit_command: bool = True) -> None:
    base_link_path = f"{robot_path}/Geometry/base_link"
    base_link_prim = stage.GetPrimAtPath(base_link_path)
    if not base_link_prim.IsValid():
        print(f"[post_import] no base_link at {base_link_path}; skipping IMU")
        return

    imu_path = f"{base_link_path}/{IMU_PRIM_NAME}"
    if stage.GetPrimAtPath(imu_path).IsValid():
        print(f"[post_import] IMU already present at {imu_path}; skipping")
        return

    if not use_kit_command:
        _add_imu_pxr(stage, base_link_path, imu_path)
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
        print(
            f"[post_import] WARNING: IsaacSensorCreateImuSensor failed for "
            f"{imu_path}; falling back to pxr-authored prim"
        )
        _add_imu_pxr(stage, base_link_path, imu_path)


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


def apply_fixes(stage, robot_path: str, imu_via_kit_command: bool = True) -> None:
    """Apply all post-import fixes to `stage` (idempotent).

    `robot_path` is the robot root prim (e.g. `/bebopv2`). Pass
    `imu_via_kit_command=False` when running headless without the sensor
    Kit extensions loaded — the IMU prim is then authored directly with
    pxr (same result, see `_add_imu_pxr`).
    """
    _disable_root_joint(stage, robot_path)
    _ensure_dynamic_base(stage, robot_path)
    _lift_robot(stage, robot_path)
    _add_imu(stage, robot_path, use_kit_command=imu_via_kit_command)
    _enable_foot_contact_report(stage, robot_path)


def main() -> None:
    stage = omni.usd.get_context().get_stage()
    robot_path = _find_robot_path(stage)
    apply_fixes(stage, robot_path)

    print(
        "[post_import] done. Press Play — the robot should fall under "
        "gravity onto the ground plane."
    )


# The Isaac Sim Script Editor executes pasted/opened code as `__main__`,
# so the GUI flow is unaffected by this guard — while the headless
# pipeline (`urdf_to_usd_bebopv2.py`) can now import this module without
# triggering it.
if __name__ == "__main__":
    main()
