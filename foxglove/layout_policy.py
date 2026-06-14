"""Foxglove layout: 3D URDF panel plus policy I/O debug plots.

A debugging companion to the robot layout. It keeps the animated URDF
panel on the left but fills the rest of the screen with plots of exactly
what the policy *sees* (`/policy/observation`) and what it *emits*
(`/policy/action`), so you can correlate inputs to outputs tick by tick.

Observation vector layout (firmware/bebop-linux/src/observation.rs::build,
OBS_DIM = 49):

    [ 0.. 3)  base_ang_vel        (scaled body gyro)
    [ 3.. 6)  projected_gravity   (gravity in body frame)
    [ 6..14)  joint_pos_rel       (q - q_default, scaled)
    [14..22)  joint_vel_rel       (q_dot, scaled)
    [22..46)  last_action         (24-dim previous raw NN output: pos|kp|kd)
    [46..49)  velocity_commands   (vx, vy, wz)

Action message layout (bebop_msgs/PolicyAction):

    raw_action[24]            (raw NN output: pos[0:8] | kp[8:16] | kd[16:24])
    position_targets_rad[8]   (decoded position targets)
    kp[8]                     (decoded, clamped)
    kd[8]                     (decoded, clamped)
"""

from layout_common import foxglove_doc, plot, obs_path, action_path
from layout_robot import _robot_3d_panel

NAME = "policy"
OUTPUT = "bebop_policy_layout.json"

# Compact joint labels for dense legends.
SHORT_JOINTS = [
    "hip_flex_L", "hip_flex_R", "hip_abd_L", "hip_abd_R",
    "knee_L", "knee_R", "foot_L", "foot_R",
]

# Observation index ranges (see module docstring).
OBS_ANG_VEL = range(0, 3)
OBS_GRAVITY = range(3, 6)
OBS_JPOS = range(6, 14)
OBS_JVEL = range(14, 22)
OBS_CMD = range(46, 49)

XYZ = ["x", "y", "z"]
CMD = ["vx", "vy", "wz"]


def _stack(ids):
    """Nest a list of panel ids into Foxglove's binary split tree, evenly."""
    n = len(ids)
    if n == 1:
        return ids[0]
    return {
        "direction": "column",
        "first": ids[0],
        "second": _stack(ids[1:]),
        "splitPercentage": 100.0 / n,
    }


def build():
    robot_id = "3D!robot"

    # --- Policy inputs (observation vector) ---
    obs_angvel_id = "Plot!obs_angvel"
    obs_gravity_id = "Plot!obs_gravity"
    obs_jpos_id = "Plot!obs_jpos"
    obs_jvel_id = "Plot!obs_jvel"
    obs_cmd_id = "Plot!obs_cmd"

    # --- Policy outputs (action message) ---
    act_target_id = "Plot!act_target"
    act_kp_id = "Plot!act_kp"
    act_kd_id = "Plot!act_kd"

    config_by_id = {
        robot_id: _robot_3d_panel(),

        obs_angvel_id: plot(
            [obs_path(i, XYZ[k]) for k, i in enumerate(OBS_ANG_VEL)],
            title="IN: base angular velocity (obs[0:3], scaled)",
        ),
        obs_gravity_id: plot(
            [obs_path(i, XYZ[k]) for k, i in enumerate(OBS_GRAVITY)],
            title="IN: projected gravity (obs[3:6])",
        ),
        obs_jpos_id: plot(
            [obs_path(i, SHORT_JOINTS[k]) for k, i in enumerate(OBS_JPOS)],
            title="IN: joint pos rel (obs[6:14], scaled)",
        ),
        obs_jvel_id: plot(
            [obs_path(i, SHORT_JOINTS[k]) for k, i in enumerate(OBS_JVEL)],
            title="IN: joint vel (obs[14:22], scaled)",
        ),
        obs_cmd_id: plot(
            [obs_path(i, CMD[k]) for k, i in enumerate(OBS_CMD)],
            title="IN: velocity commands (obs[46:49])",
        ),

        act_target_id: plot(
            [action_path("position_targets_rad", i, SHORT_JOINTS[i]) for i in range(8)],
            title="OUT: position targets (rad)",
        ),
        act_kp_id: plot(
            [action_path("kp", i, SHORT_JOINTS[i]) for i in range(8)],
            title="OUT: kp (decoded, clamped)",
        ),
        act_kd_id: plot(
            [action_path("kd", i, SHORT_JOINTS[i]) for i in range(8)],
            title="OUT: kd (decoded, clamped)",
        ),
    }

    inputs_col = _stack([
        obs_angvel_id, obs_gravity_id, obs_jpos_id, obs_jvel_id, obs_cmd_id,
    ])
    outputs_col = _stack([act_target_id, act_kp_id, act_kd_id])

    # 3D URDF on the left; policy inputs and outputs as two plot columns
    # filling the rest of the width.
    layout = {
        "direction": "row",
        "first": robot_id,
        "second": {
            "direction": "row",
            "first": inputs_col,
            "second": outputs_col,
            "splitPercentage": 50,
        },
        "splitPercentage": 34,
    }

    return foxglove_doc(config_by_id, layout)


if __name__ == "__main__":
    import argparse, json, os
    parser = argparse.ArgumentParser(description=f"Generate {OUTPUT}")
    parser.add_argument("--out", help=f"Output path (default: {OUTPUT})")
    parser.add_argument("--force", action="store_true", help="Overwrite existing file")
    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out = args.out or os.path.join(script_dir, OUTPUT)
    if os.path.exists(out) and not args.force:
        raise SystemExit(f"{out} exists; pass --force to overwrite")
    with open(out, "w") as f:
        json.dump(build(), f, indent=2)
        f.write("\n")
    print(f"wrote {out}")
