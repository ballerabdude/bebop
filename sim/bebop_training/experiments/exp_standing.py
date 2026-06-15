"""Minimal quiet-standing experiment for Bebop V2.

Train the simplest possible policy that just stands still and keeps the
torso upright at the zero velocity command, then deploy the exported ONNX
to the real robot and debug before adding the next feature. This file is the
single source of truth for the standing task config; add one knob at a time
(pushes / reactive recovery, stepping rewards, observation noise, …) and
validate on hardware after each training run.

This is the deliberately stripped-down CLEAN-SLATE baseline: only the
bare-minimum reward (alive + termination_penalty + torso_upright), no
mid-episode pushes, no reactive-recovery shaping, no privileged foot-contact
sensing, and only small torso-CoM (CoG) randomization. The goal is to get a
clean standing policy first and then add ONE knob at a time, reading each
change's effect before stacking the next. Actuator friction/armature
randomization and the small torso-CoM curriculum are the only domain
randomization kept on.

Kept because the real robot needs them:
  * IMU (BNO085) — projected gravity and gyro for balance.
  * Joint encoders — ``joint_pos`` / ``joint_vel`` observations.
  * MIT-mode variable impedance — 24-dim action (8 pos + 8 kp + 8 kd), same
    decode as ``firmware/bebop-linux/config/bebop_v2.yaml``.
  * 49-dim observation layout matching ``observation.rs`` (base_ang_vel,
    projected_gravity, joint_pos, joint_vel, last_action, cmd_vel — note
    no base_lin_vel; the robot cannot observe it).
  * DCMotor actuators — explicit PD + torque-speed saturation (not implicit
    PhysX drives, which can stand at zero pose without a policy).
  * IMU sample rate (200 Hz) and action delay (20 ms) matching the real
    loop (``bebop_v2.yaml`` ``rotation_vector_period_ms = 5``).

Reward set (pitch-imbalance quiet stand):
  * ``alive``               — +1/tick while the episode has not terminated;
                              the positive survival carrot.
  * ``termination_penalty`` — one-shot ``is_terminated`` penalty on a *fall*
                              (excludes the time_out truncation), the hard
                              "don't topple" lever the heavy torso needs.
  * ``torso_upright``       — ``flat_orientation_l2``: symmetric L2
                              penalty on the xy projected-gravity
                              components, i.e. any torso tilt (forward,
                              back, or sideways). No hardcoded target lean;
                              the policy is simply rewarded for keeping the
                              torso vertical. The core balance shaping term.
  * ``feet_straight``       — FIRST knob added back: L1 hip-abduction
                              deviation penalty. Pins the hip-abduction
                              joints to the zero default so the policy can't
                              ADDUCT the legs together (props legs against
                              each other / merges contact patches into one
                              base) or splay them out — both static-stance
                              cheats that dodge real two-leg balancing.
  * ``action_l2``           — SECOND knob added back: raw action-magnitude
                              penalty (Σ raw² over all 24 channels). Anchors
                              the action distribution so its std can't run
                              away — without it, clipped actions + the entropy
                              bonus drove Loss/entropy to grow unbounded (worst
                              in fixed-gain, where the 16 inert kp/kd channels
                              have no other gradient).
  * ``gain_l2``             — THIRD knob: raw kp/kd magnitude penalty, centered
                              on midpoint gains. This keeps variable impedance
                              away from rails unless balance really needs it.
  * ``gain_rate``           — FOURTH knob: tick-to-tick change penalty on the
                              16 kp/kd channels (smooth, slowly-varying
                              impedance).
  * ``position_rate``       — FIFTH knob: tick-to-tick change penalty on the
                              8 position channels. Playback of the converged
                              entropy-0.001 run showed a ~5-6 Hz pos-target
                              limit cycle riding the 0.020 rad/tick slew
                              limiter; action_l2 (level) and gain_rate
                              ([8:24] only) leave that direction unpenalized.
  * ``stationary_pose``     — SIXTH knob: bounded positive reward for near-zero
                              joint velocity. The position-rate run removed the
                              violent target limit cycle but still learned a
                              moving hip/knee balance; this biases the optimum
                              toward a quiet locked stand without an unbounded
                              penalty during recovery.
  * ``default_joint_pose``  — SEVENTH knob: bounded positive reward for actual
                              joint positions near the configured default
                              posture, so the learned equilibrium is tied to
                              the standing standard used by the action decoder.
  * ``forward_lean``        — EIGHTH knob: non-privileged IMU penalty for
                              parking the heavy torso forward over the toes,
                              the real-world fall mode that sim can otherwise
                              hide with a rigid-foot contact patch.
  * ``base_ang_vel_xy``     — NINTH knob: torso roll/pitch angular-velocity
                              penalty (``ang_vel_xy_l2``). Damps the ~1 Hz
                              body sway that appeared once the ankle/hipflex
                              stiffness was restored (underdamped natural
                              inverted-pendulum mode; the RS02 ankle kd ceiling
                              limits joint-level damping). Uses the IMU gyro the
                              policy already observes, so it is non-privileged.
  All other shaping / regularization terms (joint_pos_limits, joint_effort,
  joint_motion, feet_ankle_motion, track_lin_vel_xy / track_ang_vel_z,
  lin_vel_z_l2, action_rate_l2 / joint_acc_l2) remain REMOVED
  for this quiet-stand task —
  re-add one at a time and validate each.

Terminations:
  * ``base_link_ground_contact`` — torso height near the floor (fallen); the
                              SOLE fall terminator (plus ``time_out``). The IMU
                              pitch/roll tilt-cliff terminations are disabled so
                              the policy cannot farm ``alive`` up to a tilt
                              boundary and bail — it must actually stay up.

Reset randomization:
  * Joints sampled uniformly per-joint via
    ``reset_joints_uniform_within_limits`` with
    ``range_fraction = RESET_JOINT_RANGE_FRACTION`` (0.25), i.e. within
    25% of the distance from the default pose to each joint's soft
    limit, preserving the asymmetric knee / hip-abduction shape. This is
    a deliberately tight band around the nominal pose so most episodes
    survive and the policy gets dense standing signal. Set the fraction
    to 1.0 to recover the full-configuration-box behaviour (many
    unrecoverable inits from ``base_link.z = 0.8 m``, diluted signal);
    widen it gradually once the robot reliably stands. Watch
    ``mean_episode_length`` and the ``base_link_ground_contact``
    termination rate when changing it.
  * Base pitch symmetric ±15°, inside the ±30° fall limit so resets measure
    heavy-torso pitch recovery rather than terminating on the cliff. Roll and
    yaw are pinned to zero — the real robot is never dropped sideways or
    upside-down; it starts upright with the feet an inch off the ground.
  * Initial angular velocity perturbed only in pitch by ±0.6 rad/s, keeping the
    task focused on fore/aft torso imbalance instead of general tumbling.

Domain randomization (kept):
  * ``randomize_actuator_params`` — per-episode scaling of joint
    friction/armature about the sysid-measured nominal, covering
    unit-to-unit / thermal / wear spread and the unmeasured left side.
  * ``randomize_torso_com`` — per-reset torso CoM offset, ramped slowly from a
    tiny range to ±2 cm horizontal. This is the first sim-to-real robustness
    knob after the no-noise still-stand converged; it is intentionally much
    smaller than the old ±6 cm CoG randomization that destabilized training.

Deliberately off (add back one at a time after hardware validation):
  * Mid-episode pushes + reactive-recovery shaping (``push_robot``, the
    push curriculum, the asymmetric ``forward_lean`` guard).
  * Privileged foot-contact sensing (foot ``ContactSensor`` s, the
    ``feet_slide`` anti-slip reward, the ``feet_both_airborne``
    termination) — re-add once basic standing is solid on hardware.
  * Explicit air-time / foot-placement stepping rewards; pose-lock
    (``stationary_pose_exp``) shaping.

Observation noise (ON): small hardware-noise corruption on every measured
channel — BNO gyro (base_ang_vel), motor-reported joint velocity, and (added
Jun 2026 for sim-to-real robustness) projected gravity and joint position.
The position/gravity channels were previously kept clean because the still
capture shows them ~constant, but a policy that keys on perfectly clean,
zero-latency gravity/encoder signals learns a razor-sharp high-gain reaction
that rings against the real (noisy, slightly laggy) IMU + encoders; small noise
forces a lower-gain, transfer-robust stand. last_action and cmd_vel are exact
policy/command outputs, so they get no noise.

Deployment checklist (every run):
  1. Export ONNX from the training run.
  2. Confirm ``pos_scale`` (0.5), ``max_pos_step_per_tick`` (0.020), and
     ``POLICY_*`` clamps match ``bebop_v2.yaml``.
  3. Pose the robot near the trained init (joints ≈ 0, torso upright,
     within ~±8° pitch) before RunPolicy.
  4. Log raw actions + decoded targets on hardware; compare to sim play mode.
"""

import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
import isaaclab.envs.mdp as mdp

from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as TermTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ImuCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from ..envs.bebop_v2_actions import VariableImpedanceJointActionCfg
from ..envs.bebop_v2_events import (
    push_magnitude_curriculum,
    randomize_torso_com_uniform,
    reset_joints_uniform_within_limits,
    torso_com_curriculum,
)
from ..envs.bebop_v2_rewards import (
    action_gain_l2,
    action_gain_rate_l2,
    action_position_rate_l2,
    default_joint_pose_exp,
    forward_lean_penalty,
    stationary_pose_exp,
)
from ..envs.bebop_v2_terminations import (
    base_link_on_ground,
    imu_pitch_out_of_bounds,
    imu_roll_out_of_bounds,
)


# IMU pitch convention (body FLU, same as obs / firmware):
#   proj_grav[0] = -sin(pitch); |proj_grav[0]| grows with tilt either way.
# The episode terminates symmetrically at ±PITCH_FALL_LIMIT_DEG of torso
# pitch from vertical; there is no hardcoded target lean — the policy is
# rewarded (via flat_orientation_l2) for staying upright.
#
# ±30° gives a quiet stand plenty of room for a small drift without dying
# on the cliff, while the slower `base_link_ground_contact` check remains the
# true "fallen" terminator. Tighten toward ~15-20° for a stricter stand once
# the robot balances reliably; widen it again when adding reactive recovery.
PITCH_FALL_LIMIT_DEG = 30.0
PITCH_FALL_LIMIT_GX = math.sin(math.radians(PITCH_FALL_LIMIT_DEG))

# Sideways (roll) fall envelope, the lateral counterpart to the pitch limit.
# imu_pitch_out_of_bounds only watches the fore/aft axis, so without this a
# purely sideways topple is never terminated early — it keeps earning `alive`
# while slowly tipping until the slow base_link height check fires.
# imu_roll_out_of_bounds ends the episode at ±ROLL_FALL_LIMIT_DEG of sideways
# lean (``|proj_grav[1]| > sin(limit)``). Kept equal to the pitch limit so a
# sideways lean is treated the same as a fore/aft one.
ROLL_FALL_LIMIT_DEG = 30.0
ROLL_FALL_LIMIT_GY = math.sin(math.radians(ROLL_FALL_LIMIT_DEG))

# Initial torso pitch imbalance (radians, for reset_root_state_uniform's
# pose_range["pitch"], an Euler angle in rad — NOT a projected gravity
# component). Symmetric ±15° spawn, still inside the ±30° fall envelope, trains
# the policy on the heavy-torso pitch failure mode instead of only a near-level
# static stand.
PITCH_INIT_RAD = math.radians(15.0)

# Initial torso pitch-rate imbalance (rad/s). The heavy torso falls through
# pitch, so inject pitch angular velocity on reset and keep roll/yaw rates
# pinned to avoid diluting this quiet-stand task into general tumbling recovery.
PITCH_RATE_INIT_RAD_S = 0.6

# Fraction of each joint's soft range (measured from the default pose to
# each limit) used when sampling reset poses. 1.0 = full configuration
# box (many unrecoverable inits, diluted signal); a small value like 0.25
# keeps resets near the nominal pose so most episodes survive and the
# policy gets dense standing signal. Widen toward 1.0 once it stands.
RESET_JOINT_RANGE_FRACTION = 0.25

# Mid-episode push envelope for ``BebopV2StandingPushCfg`` (the reactive-
# recovery variant). ``push_by_setting_velocity`` adds these (m/s) root-velocity
# impulses at random intervals so the policy must catch a shove instead of
# parking in a fragile static lean. Lateral (y) pushes are the key signal
# against the one-foot lean (a centered two-foot stance is far more y-push
# robust); fore/aft (x) keeps the pitch recovery honest. Magnitudes mirror the
# old ``push_robot`` envelope (±0.4 m/s) and the play_bebop interactive push
# default. Ramped in over training by ``push_magnitude_curriculum`` (starts at
# 40% of this range). Widen once the robot reliably catches the full envelope.
PUSH_VELOCITY_RANGE = {"x": (-0.4, 0.4), "y": (-0.4, 0.4)}

# Target standing base_link height (m) for the OPTIONAL ``base_height_l2``
# reward (currently commented out in RewardsCfg). This is an UNVERIFIED
# estimate from the base_link_ground_contact termination comment (~0.65 m);
# MEASURE the true settled-stand height (play mode → base_link world z) and
# set this before enabling the term, or it will pull the torso to a wrong
# height.
BASE_HEIGHT_TARGET = 0.65

# Small torso center-of-mass robustness range. The old ±6 cm fore/aft CoG
# randomization made the policy re-infer an unobservable balance point every
# reset and tracked a mid-training regression. Re-introduce this as a measured,
# slow robustness knob: horizontal ±2 cm only (battery/electronics placement,
# harness, payload tolerance), no vertical offset. The curriculum starts at 25%
# of this range (±5 mm) and reaches full range after 100k control steps.
TORSO_COM_RANGE = {
    "x": (-0.02, 0.02),
    "y": (-0.02, 0.02),
    "z": (0.0, 0.0),
}
TORSO_COM_START_FRACTION = 0.25
TORSO_COM_CURRICULUM_STEPS = 100_000

# Observation corruption matched to the current hanging/still hardware capture.
# Keep this much smaller than generic locomotion defaults: the standing task
# first needs a clean equilibrium, then robustness to the real noise floor.
GYRO_NOISE_RAD_S = 0.01
JOINT_VEL_NOISE_RAD_S = 0.12
# Jun 2026 sim-to-real: also corrupt projected gravity and joint position.
# These were kept clean (the still capture shows them ~constant), but a policy
# that keys on a perfectly clean, zero-latency gravity / encoder signal learns a
# razor-sharp high-gain reaction that rings against the real (noisy, slightly
# laggy) IMU + encoders. Small noise forces a more robust, lower-gain stand.
#   * projected_gravity: ±0.02 (unit-vector component ≈ ±1.1° tilt)
#   * joint_pos:         ±0.01 rad (encoder quantization / jitter)
PROJ_GRAV_NOISE = 0.02
JOINT_POS_NOISE_RAD = 0.01

# Per-episode randomized action transport delay, in 100 Hz policy ticks
# (1 tick = 10 ms). The deployed loop's true latency (CAN round-trip + motor PD
# + feedback) is not a single fixed value, and a stand tuned for exactly one
# delay can limit-cycle when the real delay differs. Training across 10-40 ms
# forces a latency-robust (more damped) policy. Replaces the old fixed 20 ms
# (``action_delay_steps=2``).
ACTION_DELAY_RANGE = (1, 4)

# Joint order must match firmware/bebop-linux/src/observation.rs::JOINT_NAMES.
JOINT_NAMES_ALL = [
    "hip_flexion_left_joint",
    "hip_flexion_right_joint",
    "hip_abduction_left_joint",
    "hip_abduction_right_joint",
    "knee_flexion_left_joint",
    "knee_flexion_right_joint",
    "foot_left_joint",
    "foot_right_joint",
]

# Per-joint kp/kd clamps for the 24-dim MIT action. Must mirror the per-joint
# policy_gain_clamps in firmware/bebop-linux/config/bebop_v2.yaml.
#
# Jun 2026 sim-to-real NARROWING: the previous wide bands (hip_abd 40-300,
# knee 30-250, foot 80-250) let the policy run very high stiffness AND let the
# under-determined raw gain channels inject large physical kp/kd swings every
# tick (a raw-channel noise of std ~0.35 mapped to ±45 kp on hip_abd). A
# quiet-in-sim stand then rang on hardware: high loop gain has little stability
# margin against the real latency / torque-ripple / sensor noise the DCMotor
# sim model doesn't capture. Lowering the kp CEILINGS cuts loop gain (bigger
# margin), tightening the RANGES cuts the per-tick gain-noise the policy feeds
# back through last_action + physics, and lifting the kd FLOORS adds damping.
# This is still variable impedance — just not stiff enough to ring. MUST stay
# mirrored with the per-joint policy_gain_clamps in bebop_v2.yaml.
#   order: [hipflexL, hipflexR, hipabdL, hipabdR, kneeL, kneeR, footL, footR]
#
# Jun 14 ANKLE/HIPFLEX RESTORE: the Jun 2026 narrowing helped kill the
# oscillation, but the deployed smoothed policy then could not hold the heavy
# torso up — the hardware capture showed the FOOT kp pegged at its 180 ceiling
# (mean 138, raw_kp positive = asking for more) and HIPFLEX kp pegged at 80,
# and the torso slowly toppling. Those are the load-bearing fore/aft balance
# joints. Now that the rate penalties (position_rate / gain_rate) handle
# oscillation, we don't need the gain CAP to do it too, so the ankle + hipflex
# ceilings are restored (foot kp_max 180->250, kp_min 80->100 so the ankle is
# never soft; hipflex kp_max 80->120). Knee / hip_abd are unchanged — they had
# headroom (not pegged). NOTE: the RS02 ankle still has a hard 17 N·m torque
# limit, so kp lets it use that authority firmly but cannot exceed it.
POLICY_KP_MIN = [20.0, 20.0, 40.0, 40.0, 30.0, 30.0, 100.0, 100.0]
POLICY_KP_MAX = [120.0, 120.0, 150.0, 150.0, 150.0, 150.0, 250.0, 250.0]
POLICY_KD_MIN = [2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
POLICY_KD_MAX = [5.0, 5.0, 8.0, 8.0, 8.0, 8.0, 5.0, 5.0]

# Robstride motor friction / armature and T-N curve corners.
#
# Coulomb friction and reflected rotor inertia (armature) are MEASURED via
# the actuator sysid tool (firmware/bebop-linux/src/bin/sysid.rs +
# sim/bebop_training/tools/sysid_fit.py), right-side joints, bench / free
# shaft. Cross-check: hip and knee share the RS04 and their measured rotor
# inertia matched to 0.4% (0.0310 vs 0.0312) — armature is a motor property,
# so this is expected; friction differs (~12%) because it is set by each
# joint's bench assembly, not the motor.
JOINT_FRICTION_HIP_FLEX = 0.567
JOINT_FRICTION_HIP_ABD = 0.373
JOINT_FRICTION_KNEE_FLEX = 0.633
JOINT_FRICTION_FOOT = 0.159

JOINT_ARMATURE_HIP_FLEX = 0.0310
JOINT_ARMATURE_HIP_ABD = 0.0114
JOINT_ARMATURE_KNEE_FLEX = 0.0312
JOINT_ARMATURE_FOOT = 0.0038

# Stall torque (saturation_effort) and no-load speed (velocity_limit) remain
# datasheet values. The sysid bench runs could NOT measure them faithfully:
# stall runs were not mechanically blocked, and no-load runs saturated the
# Robstride velocity encoder full-scale (RS04 ±15, RS03 ±20, RS02 ±30 rad/s),
# which is below the true no-load speed. Re-measure with a brake fixture +
# --allow-cap-override before changing these.
MOTOR_STALL_TORQUE_RS04 = 120.0
MOTOR_STALL_TORQUE_RS03 = 60.0
MOTOR_STALL_TORQUE_RS02 = 17.0

MOTOR_NOLOAD_VEL_RS04 = 20.9
MOTOR_NOLOAD_VEL_RS03 = 20.4
MOTOR_NOLOAD_VEL_RS02 = 42.9


def _midpoint(lo: list[float], hi: list[float]) -> list[float]:
    return [0.5 * (a + b) for a, b in zip(lo, hi)]


_KP_MID = _midpoint(POLICY_KP_MIN, POLICY_KP_MAX)
_KD_MID = _midpoint(POLICY_KD_MIN, POLICY_KD_MAX)


BEBOP_V2_STANDING_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/workspace/bebop_bot/sim/usd/bebopv2/bebopv2.usda",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={joint_name: 0.0 for joint_name in JOINT_NAMES_ALL},
        joint_vel={joint_name: 0.0 for joint_name in JOINT_NAMES_ALL},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hip_flexion": DCMotorCfg(
            joint_names_expr=[
                "hip_flexion_left_joint",
                "hip_flexion_right_joint",
            ],
            saturation_effort=MOTOR_STALL_TORQUE_RS04,
            effort_limit_sim=84.0,
            velocity_limit_sim=12.0,
            velocity_limit=MOTOR_NOLOAD_VEL_RS04,
            stiffness=_KP_MID[0],
            damping=_KD_MID[0],
            armature=JOINT_ARMATURE_HIP_FLEX,
            friction=JOINT_FRICTION_HIP_FLEX,
        ),
        "hip_abduction": DCMotorCfg(
            joint_names_expr=[
                "hip_abduction_left_joint",
                "hip_abduction_right_joint",
            ],
            saturation_effort=MOTOR_STALL_TORQUE_RS03,
            effort_limit_sim=42.0,
            velocity_limit_sim=12.0,
            velocity_limit=MOTOR_NOLOAD_VEL_RS03,
            stiffness=_KP_MID[2],
            damping=_KD_MID[2],
            armature=JOINT_ARMATURE_HIP_ABD,
            friction=JOINT_FRICTION_HIP_ABD,
        ),
        "knee_flexion": DCMotorCfg(
            joint_names_expr=[
                "knee_flexion_left_joint",
                "knee_flexion_right_joint",
            ],
            saturation_effort=MOTOR_STALL_TORQUE_RS04,
            effort_limit_sim=84.0,
            velocity_limit_sim=12.0,
            velocity_limit=MOTOR_NOLOAD_VEL_RS04,
            stiffness=_KP_MID[4],
            damping=_KD_MID[4],
            armature=JOINT_ARMATURE_KNEE_FLEX,
            friction=JOINT_FRICTION_KNEE_FLEX,
        ),
        "foot": DCMotorCfg(
            joint_names_expr=["foot_left_joint", "foot_right_joint"],
            saturation_effort=MOTOR_STALL_TORQUE_RS02,
            effort_limit_sim=17.0,
            velocity_limit_sim=20.0,
            velocity_limit=MOTOR_NOLOAD_VEL_RS02,
            stiffness=_KP_MID[6],
            damping=_KD_MID[6],
            armature=JOINT_ARMATURE_FOOT,
            friction=JOINT_FRICTION_FOOT,
        ),
    },
)


@configclass
class ActionsCfg:
    joint_pos = VariableImpedanceJointActionCfg(
        asset_name="robot",
        joint_names=JOINT_NAMES_ALL,
        # CRITICAL sim-to-real: force the action's joint order to EXACTLY
        # JOINT_NAMES_ALL (= firmware observation.rs::JOINT_NAMES). The Newton
        # articulation resolves joints in RIGHT-before-LEFT pair order
        # ([hipflex_R, hipflex_L, ...]) while the firmware contract is
        # LEFT-before-RIGHT ([hipflex_L, hipflex_R, ...]). With the default
        # preserve_order=False the policy would train on the articulation order
        # and deploy mirror-swapped (legs L<->R) against firmware. The matching
        # joint_pos/joint_vel observation terms below ALSO pin preserve_order so
        # the whole policy I/O is in firmware order. (The action-term assertion
        # in VariableImpedanceJointAction.__init__ guards this.)
        preserve_order=True,
        pos_scale=0.5,
        use_default_offset=True,
        max_pos_step_per_tick=0.020,
        action_delay_steps=2,
        action_delay_range=ACTION_DELAY_RANGE,
        kp_min=POLICY_KP_MIN,
        kp_max=POLICY_KP_MAX,
        kd_min=POLICY_KD_MIN,
        kd_max=POLICY_KD_MAX,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # Observation noise (applied because enable_corruption=True below).
        # Every measured channel is corrupted; only the exact policy/command
        # echoes (last_action, velocity_commands) stay clean:
        #   * base_ang_vel (BNO gyro)        : ±0.01 rad/s
        #   * projected_gravity (IMU fusion) : ±0.02  (≈ ±1.1° tilt)
        #   * joint_pos (encoder)            : ±0.01 rad
        #   * joint_vel (motor feedback)     : ±0.12 rad/s
        # Jun 2026 sim-to-real: projected_gravity / joint_pos used to be clean
        # (the still capture shows them ~constant), but that let the policy
        # learn a razor-sharp high-gain reaction on a perfect signal which rang
        # on the real, noisy/laggy IMU + encoders. Small noise forces a more
        # robust, lower-gain stand.
        #
        # NOTE: base_lin_vel is intentionally NOT an observation. Bebop V2
        # has no wheel odometry or VIO estimator, so the real robot can only
        # ever feed zeros for it (see firmware policy_runner.rs). Training
        # against the sim's privileged ground-truth base velocity created a
        # 3-dim sim-to-real gap that destabilized deployment. The standing
        # task is at zero command, so base linear velocity carries no policy
        # signal anyway — it lives only in the (privileged) tracking rewards.
        base_ang_vel = ObsTerm(
            func=mdp.imu_ang_vel,
            params={"asset_cfg": SceneEntityCfg("imu")},
            noise=Unoise(n_min=-GYRO_NOISE_RAD_S, n_max=GYRO_NOISE_RAD_S),
        )
        projected_gravity = ObsTerm(
            func=mdp.imu_projected_gravity,
            params={"asset_cfg": SceneEntityCfg("imu")},
            noise=Unoise(n_min=-PROJ_GRAV_NOISE, n_max=PROJ_GRAV_NOISE),
        )
        # joint_pos / joint_vel MUST be emitted in JOINT_NAMES_ALL order (=
        # firmware observation.rs::JOINT_NAMES), NOT the Newton articulation
        # order (which is right-before-left per pair). Pinning joint_names +
        # preserve_order=True here keeps these observations in lock-step with
        # the action term (which also sets preserve_order=True) and with the
        # firmware obs builder. Without this the policy's joint sensing would be
        # L<->R mirror-swapped on hardware. (default_offset for joint_pos_rel is
        # the per-joint default pose, looked up by id, so reordering is safe.)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=JOINT_NAMES_ALL, preserve_order=True
                )
            },
            noise=Unoise(n_min=-JOINT_POS_NOISE_RAD, n_max=JOINT_POS_NOISE_RAD),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=JOINT_NAMES_ALL, preserve_order=True
                )
            },
            noise=Unoise(n_min=-JOINT_VEL_NOISE_RAD_S, n_max=JOINT_VEL_NOISE_RAD_S),
        )
        actions = ObsTerm(func=mdp.last_action)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )

        def __post_init__(self):
            # Apply the per-term noise models above. Without this the noise=
            # configs are silently ignored.
            self.enable_corruption = True

    def __post_init__(self):
        self.policy = self.PolicyCfg()


@configclass
class EventCfg:
    # Per-joint uniform sampling across each joint's soft limits. With
    # soft_joint_pos_limit_factor = 0.9, this covers ~90% of the URDF
    # range for every joint, including the asymmetric knee
    # (-1.41, +0.71 rad) and hip-abduction (-0.31, +0.16 rad on the
    # right, mirror on the left) ranges that a symmetric offset can't
    # represent. The VariableImpedanceJointAction's reset() reads the
    # post-event joint pose into the slew tracker, so whatever we
    # sample here becomes the policy's true initial condition (it
    # won't get yanked back to zero on tick 1).
    reset_joints = EventTerm(
        func=reset_joints_uniform_within_limits,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES_ALL),
            "velocity_range": (-0.5, 0.5),
            "range_fraction": RESET_JOINT_RANGE_FRACTION,
        },
    )
    # Base-orientation randomization. Only torso pitch is perturbed: the real
    # failure starts with the heavy torso falling fore/aft, not with a sideways
    # or yawed drop. The wider pitch and pitch-rate reset forces the policy to
    # catch that imbalance instead of only learning an almost-level static pose.
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                # Symmetric ±15° pitch spawn, inside the ±30° fall limit so
                # resets measure pitch recovery rather than terminating on the
                # cliff.
                "pitch": (-PITCH_INIT_RAD, PITCH_INIT_RAD),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (-PITCH_RATE_INIT_RAD_S, PITCH_RATE_INIT_RAD_S),
                "yaw": (0.0, 0.0),
            },
        },
    )

    # Actuator dynamics randomization. The JOINT_FRICTION_* / JOINT_ARMATURE_*
    # constants are sysid-measured on the right-side joints only, so the left
    # side and unit-to-unit / thermal / wear variation are unmodeled. Re-sample
    # each episode (scale about the measured nominal) so the policy is robust to
    # the spread instead of overfitting one bench measurement:
    #   - friction: x[0.5, 1.6]  (assembly-dependent; widest empirical spread)
    #   - armature: x[0.8, 1.25] (motor-rotor property; tighter, but widened for
    #                             sim-to-real margin on the unmeasured left side)
    # Jun 2026 sim-to-real: widened from x[0.7,1.4] / x[0.9,1.1]. The hardware
    # ring is a transfer/robustness failure, so the policy needs to see a wider
    # actuator-dynamics spread (not just one bench measurement) to learn a stand
    # that survives the real motor response rather than overfitting the sim one.
    randomize_actuator_params = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES_ALL),
            "friction_distribution_params": (0.5, 1.6),
            "armature_distribution_params": (0.8, 1.25),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # Small torso-CoM (CoG) randomization. This is intentionally much gentler
    # than the old ±6 cm fore/aft range that destabilized training: per reset,
    # sample the base_link CoM inside a horizontal ±2 cm box, with the
    # curriculum below starting at ±5 mm and ramping slowly. The custom event
    # sets nominal + offset every reset (non-accumulating), so it is safe to use
    # in reset mode.
    randomize_torso_com = EventTerm(
        func=randomize_torso_com_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": dict(TORSO_COM_RANGE),
        },
    )


@configclass
class RewardsCfg:
    """Standing reward: stay alive and recover heavy-torso pitch imbalance.

    The hardware failure is dominated by the heavy torso pitching forward/back,
    so this task now emphasizes uprightness and forward-lean avoidance over a
    base-stillness target. Smoothness and joint-velocity terms still discourage
    chatter, but the policy is allowed to move the base/joints to catch pitch.
    """

    # +1/tick while the episode has not terminated — the positive survival
    # signal. ``mdp.is_alive`` is weighted by the env step_dt internally, so
    # this is the carrot that makes "keep standing" worth more than ending the
    # episode early.
    alive = RewTerm(func=mdp.is_alive, weight=1.0)

    # Hard "don't fall" signal: a one-shot penalty on the step the episode
    # terminates for a *fall* (here, ``base_link_ground_contact``).
    # ``mdp.is_terminated`` excludes the time_out truncation, so surviving to
    # the 20 s limit is NOT penalized — only toppling is. -200 is the standard
    # Isaac Lab value; lower toward ~-50 if the policy becomes over-cautious
    # (freezes / refuses to move).
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # The single balance shaping term. ``flat_orientation_l2`` penalizes the
    # squared xy components of projected gravity, i.e. any torso tilt (forward,
    # back, or sideways) symmetrically — no hardcoded target lean, just "stay
    # upright." Weight tightened (-2.0 -> -4.0): the hardware target is a
    # straighter, more level torso, so even small persistent pitch/roll offsets
    # should lose to the zero-tilt equilibrium.
    torso_upright = RewTerm(func=mdp.flat_orientation_l2, weight=-4.0)

    # Jun 15 knob (kills the ~1 Hz body sway): penalize torso roll/pitch
    # ANGULAR VELOCITY. After the ankle/hipflex stiffness was restored the
    # robot finally held itself upright on hardware, but it swayed ±~9.5° at
    # ~1 Hz (the natural inverted-pendulum mode) — the balance loop is
    # underdamped, partly because the RS02 ankle kd is hard-capped at 5. Every
    # other reward only penalizes torso tilt *position* (torso_upright /
    # forward_lean) or *joint* velocity (stationary_pose / joint_vel), so the
    # torso could swing THROUGH upright at speed for free. ``ang_vel_xy_l2``
    # taxes Σ(ω_x² + ω_y²) of the base — the same gyro signal the IMU measures
    # and the policy already observes (base_ang_vel), so it is non-privileged
    # and transfers. This directly damps the sway. Raise if it still wobbles on
    # hardware; lower if it becomes too stiff to catch the pitch resets.
    base_ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.10)

    # FIRST knob added back (fixes an observed cheat): hip-abduction deviation
    # penalty. With nothing pinning the hip-abduction joints, the policy learned
    # to ADDUCT both legs — pulling the feet together so the legs prop against
    # each other and the two contact patches merge into one wide static base,
    # "standing" without actually balancing on two independent legs. (The
    # mirror-image splay-OUTWARD cheat is the same exploit in the other
    # direction.) ``joint_deviation_l1`` is symmetric — it taxes |q - default|
    # on both hip-abduction joints — so pinning them to the zero default holds
    # the legs at their nominal stance width, killing both the legs-together and
    # the splay cheats.
    #
    # Safe to use a firm weight (-3.0) here because this stand has NO lateral /
    # roll disturbance: hip abduction is the only side-to-side recovery
    # actuator, but with no sideways push it never needs to move, so pinning it
    # to zero costs nothing real. Raise toward -5.0 if any leg-together / splay
    # persists in playback. WHEN lateral pushes / reactive recovery are added
    # back, soften this (the policy must be free to widen its stance to catch a
    # sideways shove) — prefer penalizing abduction *velocity* over position.
    feet_straight = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-3.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "hip_abduction_left_joint",
                    "hip_abduction_right_joint",
                ],
            ),
        },
    )

    # SECOND knob added back (fixes the entropy runaway): raw action-magnitude
    # penalty. With NO action regularization, nothing anchored the policy's
    # action distribution: raw actions are clipped to [-1, 1] before physics, so
    # once the Gaussian std grew large enough to saturate the rails, growing it
    # further cost the task nothing while the entropy bonus kept paying for more
    # entropy -> std (and Loss/entropy) ran away linearly without bound. It is
    # worst in the FIXED-GAIN variant: the 16 kp/kd channels are inert and
    # untouched by any reward, so their std explodes freely. ``action_l2``
    # penalizes Σ raw² over all 24 channels, so large std -> large expected
    # sampled magnitude -> penalty -> the gradient pushes std back down,
    # anchoring the distribution (and pulling raw toward 0 = default pose +
    # midpoint gains, a sane prior). Strengthened after variable-impedance
    # playback still showed large position outputs and gain drift.
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.02)

    # THIRD knob (variable-impedance center): raw kp/kd magnitude penalty only.
    # ``action_l2`` spreads its gradient over all 24 channels; this isolates the
    # 16 gain channels so the quiet-stand optimum is midpoint gains unless the
    # policy earns enough balance reward to move a gain away from zero. This is
    # the main anti-rail term for variable impedance.
    gain_l2 = RewTerm(func=action_gain_l2, weight=-0.03)

    # FOURTH knob added back (smoothness): tick-to-tick CHANGE penalty on the 16
    # variable-impedance kp/kd channels only. ``action_l2`` above is a *level*
    # penalty (Σ raw² over all 24 channels) — it pulls the action toward the
    # neutral pose + midpoint gains but says nothing about how fast a channel
    # moves, so a gain can still flip hard tick-to-tick (e.g. kp +0.036 ->
    # +0.102 -> +0.081) as long as it averages near 0. That gain chatter is the
    # dominant jitter source in the deployed logs. ``action_gain_rate_l2`` is a
    # *rate* penalty (Σ (gain_t - gain_{t-1})² over channels [8:24]) that taxes
    # only the change, leaving the level free — so it enforces smooth,
    # slowly-varying impedance without fighting action_l2's mid-stiffness prior.
    # Strengthened substantially after playback showed gain channels still
    # changing while projected gravity was nearly constant. Watch the term's
    # value in TensorBoard; lower if the policy cannot catch pitch resets.
    # Jun 14 smoothing: -0.10 -> -0.20 to suppress residual gain chatter that
    # rode along with the position bang-bang in the first robust-transfer run.
    gain_rate = RewTerm(func=action_gain_rate_l2, weight=-0.20)

    # FIFTH knob added back (kills the position limit cycle): tick-to-tick
    # CHANGE penalty on the 8 POSITION channels. Deterministic playback of the
    # entropy-0.001 run showed the pos targets oscillating at ~5-6 Hz with
    # ~0.1 rad commanded swing (e.g. foot_right raw -0.665 -> -0.473 -> -0.434
    # -> -0.635) — far past the 0.020 rad/tick slew limit, so the policy was
    # riding the rate limiter as a bang-bang velocity controller instead of
    # holding a quiet pose. Neither existing regularizer touches this mode:
    # ``action_l2`` is a *level* penalty (the oscillation averages to a
    # constant level, so it's nearly free) and ``gain_rate`` only covers the
    # kp/kd channels [8:24] — after it smoothed the gains, the chatter migrated
    # to the one unpenalized direction left, the pos-channel rate.
    # ``action_position_rate_l2`` taxes Σ (pos_t - pos_{t-1})² over channels
    # [0:8] only. Strengthened because the policy was still changing raw
    # position targets by several hundredths per tick around an already-level
    # torso. Lower if the policy becomes too sluggish to catch pitch.
    # Jun 14 smoothing: -0.05 -> -0.30. The first robust-transfer policy stood
    # upright with tame gains but the deployed capture showed the position
    # targets STILL exceeding the 0.020 rad/tick slew limit on 50-78% of ticks
    # (a ~1.8 Hz leg limit cycle from bang-banging the setpoint). This is the
    # primary anti-bang-bang lever; raise further if the hardware still hunts,
    # back off if mean_episode_length drops (can't catch the pitch resets).
    position_rate = RewTerm(func=action_position_rate_l2, weight=-0.30)

    # SIXTH knob added back (quiet stand): bounded positive reward for low joint
    # velocity. The position-rate run removed the violent setpoint limit cycle,
    # but playback still showed a dynamic hip/knee balance with persistent torso
    # angular velocity instead of a settled stance. ``stationary_pose_exp`` is
    # exp(-Σ joint_vel² / std²), bounded in [0, 1], so it strongly prefers a
    # still, locked pose without adding an unbounded cost during a legitimate
    # recovery transient. Tightened after playback showed a level torso with
    # persistent hip motion; lower the weight or widen std if pitch recovery
    # becomes too stiff.
    stationary_pose = RewTerm(
        func=stationary_pose_exp,
        weight=1.0,
        params={"std": 0.75},
    )

    # SIXTH-b knob (Jun 14 smoothing): unbounded joint-velocity penalty. The
    # bounded ``stationary_pose_exp`` above SATURATES toward 0 once the legs
    # hunt (its exp tail is flat), so it provides almost no gradient at the
    # ~0.5 rad/s, ~1.8 Hz limit cycle seen in the first robust-transfer
    # deployment. ``joint_vel_l2`` (Σ joint_vel²) has its LARGEST gradient
    # exactly there, so it actively damps the residual oscillation that the
    # saturated exp reward can't see. Kept small so it doesn't punish the
    # legitimate pitch-recovery transient from the ±15° reset; raise if the
    # hardware still hunts, lower if recovery becomes sluggish.
    # Jun 15: -0.005 -> -0.015. Joint velocities rose (std 0.7-1.3 rad/s) when
    # the gain ceilings were restored, so the joint-level damping is bumped in
    # step with the new base_ang_vel_xy torso-sway penalty.
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.015)

    # SEVENTH knob (configured standing posture): bounded positive reward for
    # actual joint positions near ``asset.data.default_joint_pos``. That default
    # is the same nominal stand used by ``VariableImpedanceJointAction`` as the
    # center of the raw position action mapping, so this ties the learned
    # equilibrium to the configured standing standard instead of whatever
    # crouched/asymmetric pose happens to survive. Keep bounded so pitch
    # recovery can still leave the posture temporarily.
    default_joint_pose = RewTerm(
        func=default_joint_pose_exp,
        weight=0.75,
        params={"std": 0.35},
    )

    # EIGHTH knob (heavy-torso forward fall): non-privileged IMU penalty for
    # parking the torso forward over the toes. The real robot's heavy torso and
    # weak ankle cannot hold the sim-only toe-balance strategy, so this shapes
    # the policy away from that fall basin while still allowing brief forward
    # excursions during recovery.
    forward_lean = RewTerm(func=forward_lean_penalty, weight=-3.0)

    # NOTE: ``feet_load_symmetry`` (the privileged contact-force anti-one-foot-
    # lean penalty) was REMOVED. It reads per-foot contact forces the robot
    # cannot observe and, with torso-CoM randomization off, it overfit the
    # single sim CoM and shifted the policy onto a sim-specific balance that
    # transferred badly (saturated / thrashing actions on hardware) while the
    # run also failed to converge (Policy/mean_std stuck high). The leaning
    # cheat is instead attacked NON-privileged-ly by mid-episode pushes in
    # ``BebopV2StandingPushCfg`` (a one-foot lean is fragile to a sideways
    # shove), so the policy is pressured toward a centered, push-robust stance
    # without keying on an unobservable contact-force signal.


@configclass
class TerminationsCfg:
    time_out = TermTerm(func=mdp.time_out, time_out=True)
    # IMU tilt terminations DISABLED for now. Ending the episode at a ±30°
    # tilt cliff let the policy collect the easy ``alive`` reward right up to
    # the boundary without ever learning to truly recover its balance — once
    # that crutch dominated, training collapsed. With these off, only a real
    # fall (base_link near the floor) ends the episode, so the policy must
    # actually keep the torso up to keep surviving instead of cheating against
    # the tilt cliff. Re-enable (and re-tighten toward 15-20°) once the robot
    # reliably balances off the ground-contact backstop alone.
    # imu_pitch_out_of_bounds = TermTerm(
    #     func=imu_pitch_out_of_bounds,
    #     params={
    #         "pitch_forward_gx_max": PITCH_FALL_LIMIT_GX,
    #         "pitch_back_gx_min": -PITCH_FALL_LIMIT_GX,
    #     },
    # )
    # # Lateral counterpart to the pitch envelope: end the episode at ±30° of
    # # sideways (roll) lean (``|proj_grav[1]| > sin(30°)``). Without this a
    # # purely sideways topple is never terminated early — it keeps earning
    # # `alive` while slowly tipping until the slow base_link height check fires,
    # # giving the policy no clean signal that it has tipped over sideways.
    # imu_roll_out_of_bounds = TermTerm(
    #     func=imu_roll_out_of_bounds,
    #     params={"roll_gy_limit": ROLL_FALL_LIMIT_GY},
    # )
    # Sole "fallen" check: torso height near the floor. With the IMU tilt
    # checks disabled this is the only fall terminator — a tip-over or a slow
    # vertical collapse both eventually drop base_link below this height and
    # end the episode (paired with the one-shot ``termination_penalty``).
    base_link_ground_contact = TermTerm(
        func=base_link_on_ground,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "ground_height_threshold": 0.30,
        },
    )


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        debug_vis=False,
        rel_standing_envs=1.0,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
    )


@configclass
class CurriculumCfg:
    """Curricula for robustness knobs added after the quiet stand converges."""

    # Ramp torso-CoM randomization from a tiny near-nominal box to the full
    # ±2 cm horizontal range. CoM is unobserved, so widening slowly preserves
    # dense standing signal while making the final policy less tied to one
    # exact torso/battery balance point.
    torso_com = CurrTerm(
        func=torso_com_curriculum,
        params={
            "term_name": "randomize_torso_com",
            "full_com_range": dict(TORSO_COM_RANGE),
            "start_fraction": TORSO_COM_START_FRACTION,
            "num_curriculum_steps": TORSO_COM_CURRICULUM_STEPS,
        },
    )


@configclass
class BebopV2StandingCfg(ManagerBasedRLEnvCfg):
    decimation = 2
    episode_length_s = 20.0

    scene = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)
    observations = ObservationsCfg()
    actions = ActionsCfg()
    commands = CommandsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    events = EventCfg()
    curriculum = CurriculumCfg()

    def __post_init__(self):
        self.viewer.eye = [2.5, 2.5, 2.5]
        self.viewer.lookat = [0.0, 0.0, 0.0]

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        # Leave contact processing on. The simple stand has no foot
        # ContactSensors, but keeping this False is harmless and is required
        # the moment the privileged foot-contact sensing (feet_slide /
        # feet_both_airborne) is added back.
        self.sim.disable_contact_processing = False

        self.scene.robot = BEBOP_V2_STANDING_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.terrain = terrain_gen.TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="average",
                restitution_combine_mode="average",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        )

        self.scene.light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
        )

        self.scene.imu = ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link",
            update_period=0.005,
            debug_vis=False,
            offset=ImuCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(0.0, 0.0, 0.0, 1.0),
            ),
        )

        # NOTE: the per-foot ContactSensors were REMOVED with the
        # ``feet_load_symmetry`` reward (their only consumer). ``spawn`` keeps
        # ``activate_contact_sensors=True`` and ``disable_contact_processing``
        # stays False (both harmless) so re-adding privileged foot-contact
        # sensing (feet_slide / feet_load_symmetry / a stepping reward) later is
        # a one-line change.


@configclass
class BebopV2StandingFixedGainCfg(BebopV2StandingCfg):
    """Quiet-stand variant with the variable-impedance gains FROZEN.

    Identical task, rewards, terminations, and domain randomization as
    ``BebopV2StandingCfg``, with one change: the policy's 16 kp/kd action
    channels are ignored and physics uses fixed per-joint gains (the midpoint
    of each ``POLICY_KP_MIN/MAX`` / ``POLICY_KD_MIN/MAX`` band, i.e. ``_KP_MID``
    / ``_KD_MID``). The policy therefore only has to learn the 8 joint position
    targets.

    Why this is the recommended *first* hardware stand:

    * The variable-impedance gains are the dominant source of the chattering
      seen in the deployed logs — the kp channels flip wildly tick-to-tick
      (e.g. abduction kp 177 -> 162 -> 185, knee kp 71 -> 30 -> 152). Removing
      16 of the 24 action dimensions removes that thrashing and leaves a much
      easier, smoother control problem (position-only PD around fixed gains).
    * It honours the file's "add one knob at a time" philosophy: get a clean
      fixed-gain stand on hardware first, then switch back to
      ``BebopV2StandingCfg`` to re-introduce variable impedance.

    Deployment stays drop-in: the action vector is still 24-dim, so the ONNX
    I/O and the 49-dim observation (``last_action`` is 24-wide) match firmware
    unchanged. The fixed gains equal the firmware ``decode_policy_action``
    output at ``raw_kp = raw_kd = 0``, and the (now inert) gain channels are
    pushed toward 0 by ``action_l2``, so the robot deploys with the same gains
    trained here WITHOUT any firmware change. (If you override ``kp_fixed`` /
    ``kd_fixed`` away from the midpoints, you must also pin the matching
    ``policy_gain_clamps`` in ``firmware/bebop-linux/config/bebop_v2.yaml`` so
    the real robot uses the same gains.)
    """

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.freeze_gains = True
        # Default the frozen gains to the per-joint midpoints, which is the
        # firmware raw=0 decode -> no firmware change required.
        self.actions.joint_pos.kp_fixed = _KP_MID
        self.actions.joint_pos.kd_fixed = _KD_MID
        # NOTE: the asymmetric forward_lean shaping reward was removed for the
        # clean-slate baseline (only alive + termination_penalty + torso_upright
        # remain, inherited from RewardsCfg). Re-add it as one deliberate change
        # if playback shows a sustained non-transferable forward toe-lean.


@configclass
class BebopV2StandingPushCfg(BebopV2StandingCfg):
    """Variable-impedance stand WITH mid-episode pushes (reactive recovery).

    Identical task / rewards / terminations / domain randomization as
    ``BebopV2StandingCfg``, plus three coupled changes that turn the quiet
    stand into a push-robust stand. This is the deliberate Step-2 follow-up to
    the clean ``Standing-v0`` baseline (validate Standing-v0 converges and
    transfers first, THEN train this):

    1. ``push_robot`` — a random-interval ``push_by_setting_velocity`` event
       that adds ±0.4 m/s fore/aft + lateral root-velocity shoves
       (``PUSH_VELOCITY_RANGE``). This is the NON-privileged replacement for the
       removed ``feet_load_symmetry`` penalty: a one-foot / asymmetric lean is
       fragile to a sideways shove, so pushes pressure the policy toward a
       centered, two-foot stance WITHOUT keying on an unobservable contact-force
       signal that overfits the sim CoM. Pushes are also a genuine robustness
       feature the hardware needs, not a sim cheat-patch.
    2. ``push_level`` curriculum — ``push_magnitude_curriculum`` ramps the push
       envelope from 40% to 100% over ~150k control steps, so the policy first
       masters standing and only then faces the full shove.
    3. Softened ``feet_straight`` (-3.0 -> -0.5). Hip abduction is the ONLY
       side-to-side recovery actuator; with lateral pushes the policy MUST be
       free to widen its stance to catch a sideways shove, so the firm pin that
       was safe in the no-disturbance stand would now punish the only recovery
       it has. The weight is lowered (not removed) so the legs still resting
       near the nominal width is mildly preferred, but recovery abduction is
       allowed. Consider switching to an abduction-*velocity* penalty if a
       persistent splay shows up in playback.

    Pair with ``BebopPPOPushCfg`` (higher ``entropy_coef``): pushes enlarge the
    reward landscape, so the actor needs more exploration than the quiet-stand
    baseline to keep finding recoveries instead of collapsing prematurely.
    """

    def __post_init__(self):
        super().__post_init__()

        # Free hip abduction for lateral recovery (see class docstring).
        self.rewards.feet_straight.weight = -0.5

        # Random-interval root-velocity shoves. interval_range_s picks a fresh
        # wait between pushes per env, so over a 20 s episode the policy sees a
        # handful of disturbances at varied phases of its stand. velocity_range
        # is seeded to the full envelope; the curriculum below rescales it each
        # reset (starting at 40%).
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(4.0, 8.0),
            params={"velocity_range": dict(PUSH_VELOCITY_RANGE)},
        )

        # Ramp the push magnitude in over training (40% -> 100%).
        self.curriculum.push_level = CurrTerm(
            func=push_magnitude_curriculum,
            params={
                "term_name": "push_robot",
                "full_velocity_range": dict(PUSH_VELOCITY_RANGE),
                "start_fraction": 0.4,
                "num_curriculum_steps": 150_000,
            },
        )
