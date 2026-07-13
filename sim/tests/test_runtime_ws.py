"""Tests for runtime WebSocket telemetry helpers."""

from bebop_training.experiments.exp_standing import JOINT_NAMES_ALL
from bebop_training.runtime_ws import (
    ImuSample,
    MotorSample,
    TelemetrySnapshot,
    motor_position_live,
    motors_in_joint_order,
)


def test_motors_in_joint_order_maps_by_name():
    snapshot = TelemetrySnapshot(
        host_unix_ms=1,
        motors={
            "knee_flexion_right_joint": MotorSample(
                joint_name="knee_flexion_right_joint",
                position_rad=0.8,
                velocity_rad_s=0.1,
                position_received=True,
            ),
            "hip_flexion_left_joint": MotorSample(
                joint_name="hip_flexion_left_joint",
                position_rad=-0.2,
                velocity_rad_s=0.0,
                position_received=True,
            ),
        },
        imu=ImuSample(
            present=True,
            received=True,
            stale=False,
            quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        ),
    )

    positions, velocities, missing = motors_in_joint_order(snapshot, JOINT_NAMES_ALL)

    assert positions[0] == -0.2
    assert positions[1] == 0.0
    assert positions[4] == 0.0
    assert positions[5] == 0.8
    assert velocities[5] == 0.1
    assert "hip_flexion_right_joint" in missing


def test_motors_in_joint_order_holds_last_when_not_received():
    snapshot = TelemetrySnapshot(
        host_unix_ms=1,
        motors={
            "hip_flexion_left_joint": MotorSample(
                joint_name="hip_flexion_left_joint",
                position_rad=0.0,
                velocity_rad_s=0.0,
                position_received=False,
            ),
        },
    )
    positions, _, _ = motors_in_joint_order(
        snapshot,
        JOINT_NAMES_ALL,
        last_positions={"hip_flexion_left_joint": 0.5},
    )
    assert positions[0] == 0.5


def test_motors_in_joint_order_uses_position_when_feedback_fresh_without_received_flag():
    snapshot = TelemetrySnapshot(
        host_unix_ms=1,
        motors={
            "hip_flexion_left_joint": MotorSample(
                joint_name="hip_flexion_left_joint",
                position_rad=-0.173,
                velocity_rad_s=0.01,
                position_received=False,
                feedback_stale=False,
                armed=True,
            ),
        },
    )
    positions, velocities, missing = motors_in_joint_order(snapshot, JOINT_NAMES_ALL)
    assert positions[0] == -0.173
    assert velocities[0] == 0.01
    assert motor_position_live(snapshot.motors["hip_flexion_left_joint"])
