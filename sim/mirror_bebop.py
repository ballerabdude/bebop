"""Mirror live Bebop V2 hardware pose inside Isaac Lab.

Connects to the robot runtime WebSocket API (``bebop-linux`` on port 9090),
subscribes to telemetry, and each sim tick teleports the Isaac articulation
to the reported joint positions and base orientation. Base XY position stays
fixed at the spawn pose — the real robot has no odometry.

Example::

    # Interactive Omniverse viewport (default)
    /workspace/isaaclab/isaaclab.sh -p mirror_bebop.py \\
        --robot-host 192.168.0.69 \\
        --telemetry-hz 30

    # Lighter Newton renderer instead of the Kit viewport
    /workspace/isaaclab/isaaclab.sh -p mirror_bebop.py \\
        --robot-host 192.168.0.69 \\
        --viz newton
"""

from __future__ import annotations

import argparse
import socket
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Mirror hardware pose in Isaac Lab.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-BebopV2-Mirror-v0",
    help="Registered mirror task (default Isaac-BebopV2-Mirror-v0).",
)
parser.add_argument(
    "--robot-host",
    type=str,
    default="127.0.0.1",
    help="Robot IP or hostname running bebop-linux (default 127.0.0.1).",
)
parser.add_argument(
    "--robot-port",
    type=int,
    default=9090,
    help="Runtime WebSocket port (default 9090).",
)
parser.add_argument(
    "--telemetry-hz",
    type=int,
    default=30,
    help="Telemetry subscription rate in Hz (default 30, max 100).",
)
parser.add_argument(
    "--print-interval",
    type=int,
    default=120,
    help="Print one joint sanity line every N sim steps (0 disables).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# Pose mirroring is for visual joint validation — default to the Kit viewport
# unless the caller picks another backend (e.g. ``--viz newton``).
if args_cli.visualizer is None:
    args_cli.visualizer = ["kit"]


def _preflight_robot_reachable(host: str, port: int) -> None:
    try:
        socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        print(f"ERROR: cannot resolve {host!r}: {exc}", file=sys.stderr)
        if host.endswith(".local"):
            print(
                "mDNS names like bebop.local often fail inside the Isaac container.",
                file=sys.stderr,
            )
            print(
                "Use the robot LAN IP instead, e.g. --robot-host 192.168.0.69",
                file=sys.stderr,
            )
        raise SystemExit(1) from exc

    try:
        with socket.create_connection((host, port), timeout=3.0):
            pass
    except OSError as exc:
        print(
            f"ERROR: cannot reach {host}:{port} ({exc}). "
            "Is bebop-linux running?",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


_preflight_robot_reachable(args_cli.robot_host, args_cli.robot_port)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import bebop_training  # noqa: F401  — registers tasks
from bebop_training.experiments.exp_mirror import MIRROR_BASE_POS
from bebop_training.experiments.exp_standing import JOINT_NAMES_ALL
from bebop_training.runtime_ws import RuntimeTelemetryClient, motor_position_live, motors_in_joint_order


def _apply_telemetry(
    robot,
    joint_ids,
    snapshot,
    device,
    *,
    last_positions: dict[str, float],
    last_velocities: dict[str, float],
) -> list[str]:
    positions, velocities, missing = motors_in_joint_order(
        snapshot,
        JOINT_NAMES_ALL,
        last_positions=last_positions,
        last_velocities=last_velocities,
    )
    for name, pos, vel in zip(JOINT_NAMES_ALL, positions, velocities):
        motor = snapshot.motors.get(name)
        if motor is not None and motor_position_live(motor):
            last_positions[name] = pos
            last_velocities[name] = vel

    env_ids = torch.tensor([0], device=device, dtype=torch.long)

    joint_pos = robot.data.joint_pos.clone()
    joint_vel = robot.data.joint_vel.clone()
    joint_pos[0, joint_ids] = torch.tensor(positions, device=device, dtype=joint_pos.dtype)
    joint_vel[0, joint_ids] = torch.tensor(velocities, device=device, dtype=joint_vel.dtype)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    imu = snapshot.imu
    if imu.received and not imu.stale:
        qx, qy, qz, qw = imu.quaternion_xyzw
    else:
        qx, qy, qz, qw = (0.0, 0.0, 0.0, 1.0)

    root_pose = torch.tensor(
        [
            [
                MIRROR_BASE_POS[0],
                MIRROR_BASE_POS[1],
                MIRROR_BASE_POS[2],
                qx,
                qy,
                qz,
                qw,
            ]
        ],
        device=device,
        dtype=joint_pos.dtype,
    )
    robot.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)
    root_vel = torch.zeros((1, 6), device=device, dtype=joint_pos.dtype)
    robot.write_root_com_velocity_to_sim(root_vel, env_ids=env_ids)

    return missing


def main() -> int:
    client = RuntimeTelemetryClient(
        args_cli.robot_host,
        args_cli.robot_port,
        rate_hz=args_cli.telemetry_hz,
    )
    client.start()
    print(
        f"Connecting to ws://{args_cli.robot_host}:{args_cli.robot_port}/ws "
        f"at {args_cli.telemetry_hz} Hz..."
    )

    try:
        client.wait_for_frame(timeout_s=10.0)
    except TimeoutError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if client.last_error:
            print(f"Last connection error: {client.last_error}", file=sys.stderr)
        print(
            "Ensure bebop-linux is running and reachable on the runtime port.",
            file=sys.stderr,
        )
        if args_cli.robot_host.endswith(".local"):
            print(
                "If you used bebop.local, retry with the robot LAN IP "
                "(e.g. --robot-host 192.168.0.69).",
                file=sys.stderr,
            )
        client.stop()
        simulation_app.close()
        return 1

    task_spec = gym.spec(args_cli.task)
    cfg_entry_point = task_spec.kwargs.get("env_cfg_entry_point")
    if not callable(cfg_entry_point):
        raise ValueError(f"Env config entry point {cfg_entry_point} is not callable.")
    env_cfg = cfg_entry_point()

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    robot = env.unwrapped.scene["robot"]
    device = env.unwrapped.device
    joint_ids, _ = robot.find_joints(JOINT_NAMES_ALL, preserve_order=True)
    if isinstance(joint_ids, slice):
        joint_ids = list(range(*joint_ids.indices(robot.num_joints)))
    elif hasattr(joint_ids, "tolist"):
        joint_ids = joint_ids.tolist()
    else:
        joint_ids = list(joint_ids)

    action = torch.zeros(env.unwrapped.num_envs, 0, device=device)
    step = 0
    missing_warned: set[str] = set()
    last_positions = {name: 0.0 for name in JOINT_NAMES_ALL}
    last_velocities = {name: 0.0 for name in JOINT_NAMES_ALL}

    print("Mirror running. Move joints on hardware (e.g. DialIn) to validate.")

    while simulation_app.is_running():
        with torch.inference_mode():
            env.step(action)

        snapshot = client.latest()
        if snapshot is not None:
            missing = _apply_telemetry(
                robot,
                joint_ids,
                snapshot,
                device,
                last_positions=last_positions,
                last_velocities=last_velocities,
            )
            for name in missing:
                if name not in missing_warned:
                    print(f"WARNING: no telemetry yet for joint {name}")
                    missing_warned.add(name)

            if args_cli.print_interval > 0 and step % args_cli.print_interval == 0:
                positions, _, _ = motors_in_joint_order(
                    snapshot,
                    JOINT_NAMES_ALL,
                    last_positions=last_positions,
                    last_velocities=last_velocities,
                )
                pairs = ", ".join(
                    f"{name}={pos:+.3f}" for name, pos in zip(JOINT_NAMES_ALL, positions)
                )
                imu = snapshot.imu
                imu_note = (
                    f"imu=({imu.quaternion_xyzw[0]:+.3f},"
                    f"{imu.quaternion_xyzw[1]:+.3f},"
                    f"{imu.quaternion_xyzw[2]:+.3f},"
                    f"{imu.quaternion_xyzw[3]:+.3f})"
                    if imu.received
                    else "imu=missing"
                )
                print(f"[mirror step {step}] {pairs} | {imu_note}")

        step += 1

    client.stop()
    env.close()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
