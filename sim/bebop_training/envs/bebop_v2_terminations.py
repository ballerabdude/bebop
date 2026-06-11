"""Custom termination conditions for the Bebop V2 articulation."""

from __future__ import annotations

import torch
import warp as wp

from isaaclab.managers import SceneEntityCfg


def _ensure_tensor(value, env_device: str | None = None) -> torch.Tensor:
    """Coerce an Isaac Lab asset/sensor data field to a torch tensor.

    Isaac Lab 3.0 returns ``ProxyArray`` (a thin ``wp.array`` wrapper) from
    ``asset.data.*`` properties; pre-3.0 returned raw ``wp.array`` or
    ``torch.Tensor``. This helper accepts all three and never copies:

    * ``torch.Tensor``  -> returned as-is.
    * ``ProxyArray``    -> ``.torch`` (zero-copy view, avoids the deprecation
      warning that ``torch.as_tensor`` would emit).
    * ``wp.array``      -> ``wp.to_torch`` (zero-copy view).
    * Anything else     -> ``torch.as_tensor`` fallback (mostly for tests).
    """
    if isinstance(value, torch.Tensor):
        return value
    torch_view = getattr(value, "torch", None)
    if isinstance(torch_view, torch.Tensor):
        return torch_view
    if isinstance(value, wp.array):
        return wp.to_torch(value)
    return torch.as_tensor(
        value,
        dtype=torch.float32,
        device=env_device if env_device is not None else "cpu",
    )


def base_link_on_ground(
    env,
    asset_cfg: SceneEntityCfg,
    ground_height_threshold: float = 0.30,
) -> torch.Tensor:
    """Terminate when ``base_link`` drops near ground level (fallen).

    ``base_link`` sits at the top of the robot (~0.65 m when standing). A
    threshold around 0.30 m indicates the torso has clearly fallen toward
    the ground.
    """
    robot = env.scene[asset_cfg.name]
    body_pos_w = _ensure_tensor(robot.data.body_pos_w, env_device=getattr(env, "device", None))
    base_link_height = body_pos_w[:, asset_cfg.body_ids[0], 2]
    return base_link_height <= ground_height_threshold


def feet_both_airborne(
    env,
    sensor_names: list[str],
    max_air_time: float = 0.5,
) -> torch.Tensor:
    """Terminate when *both* feet have been off the ground too long.

    Uses each foot ``ContactSensor``'s ``current_air_time`` (available because
    the sensors are configured with ``track_air_time=True``): the time, per
    foot, since it last lost contact. The episode ends only when *every* foot
    has been airborne longer than ``max_air_time`` at the same instant — i.e.
    the robot is genuinely airborne / has fallen off its feet.

    This deliberately does NOT fire during a recovery step: lifting one foot
    to step keeps the other planted, so its air time stays at 0 and the
    conjunction is False. It is an earlier, cleaner "fell" signal than the
    slow ``base_link_on_ground`` height check and complements the symmetric
    pitch-envelope termination.

    Each foot has its own single-body sensor (the feet are not siblings in
    this nested kinematic tree), so this takes the list of per-foot sensor
    scene keys and ANDs their airborne flags.

    Args:
        sensor_names: per-foot ``ContactSensor`` scene keys (each tracks one
            foot body).
        max_air_time: seconds all feet may be simultaneously airborne before
            the episode is terminated.
    """
    device = getattr(env, "device", None)
    both_airborne = None
    for sensor_name in sensor_names:
        contact_sensor = env.scene.sensors[sensor_name]
        air_time = _ensure_tensor(
            contact_sensor.data.current_air_time, env_device=device
        )
        airborne = air_time[:, 0] > max_air_time
        both_airborne = airborne if both_airborne is None else (both_airborne & airborne)
    return both_airborne


def imu_pitch_out_of_bounds(
    env,
    imu_name: str = "imu",
    pitch_forward_gx_max: float = 0.342,
    pitch_back_gx_min: float = -0.342,
) -> torch.Tensor:
    """Terminate when torso pitch exceeds the hardware fall envelope.

    Uses IMU ``projected_gravity_b[0]`` (same convention as policy obs /
    ``firmware/bebop-linux``). For small angles, ``g_x ≈ sin(pitch)``:

    * ``g_x > pitch_forward_gx_max`` — pitched forward past the limit
      (default ``sin(20°) ≈ 0.342``).
    * ``g_x < pitch_back_gx_min`` — pitched back past the limit
      (default ``-sin(20°) ≈ -0.342``).

    Ends episodes before the slow ``base_link_on_ground`` check when the
    torso is already in a pose that falls on the real robot.
    """
    imu = env.scene[imu_name]
    proj_grav = _ensure_tensor(
        imu.data.projected_gravity_b, env_device=getattr(env, "device", None)
    )
    g_x = proj_grav[:, 0]
    return (g_x > pitch_forward_gx_max) | (g_x < pitch_back_gx_min)


def imu_roll_out_of_bounds(
    env,
    imu_name: str = "imu",
    roll_gy_limit: float = 0.342,
) -> torch.Tensor:
    """Terminate when torso *roll* (sideways tilt) exceeds the fall envelope.

    Uses IMU ``projected_gravity_b[1]`` — the body-frame +y (left) gravity
    component, same convention as the policy obs / firmware. For small angles
    ``g_y ≈ sin(roll)``, so ``|g_y| > roll_gy_limit`` means the torso has
    tipped sideways past the limit (default ``sin(20°) ≈ 0.342``). Symmetric:
    a left or right tip is treated identically.

    Why this exists: ``imu_pitch_out_of_bounds`` only watches the fore/aft
    (``g_x``) axis, so a purely sideways topple never fired an early
    termination — the episode kept collecting ``alive`` reward while the robot
    slowly tipped over, until the slow ``base_link_on_ground`` height check
    finally caught it. That long "tipping but still alive" tail gives the
    policy no clean gradient to *recover* from a sideways disturbance. This
    term provides the symmetric lateral counterpart to the pitch envelope so a
    sideways lean past ``roll_gy_limit`` ends the episode promptly, the same
    way a fore/aft lean does.
    """
    imu = env.scene[imu_name]
    proj_grav = _ensure_tensor(
        imu.data.projected_gravity_b, env_device=getattr(env, "device", None)
    )
    g_y = proj_grav[:, 1]
    return (g_y > roll_gy_limit) | (g_y < -roll_gy_limit)
