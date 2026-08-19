"""Minimal quiet-standing experiment for Bebop V2.

Trains the simplest standing policy (zero velocity command, upright torso),
deploys the exported ONNX to the real robot, and debugs before adding the
next feature. Add one knob at a time and validate on hardware after each run.

Joint order and per-joint gain clamps MUST mirror
``firmware/bebop-linux/config/bebop_v2.yaml`` and
``firmware/bebop-linux/src/observation.rs::JOINT_NAMES``.
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
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from ..envs.bebop_v2_actions import VariableImpedanceJointActionCfg
from ..envs.bebop_v2_actuator_net import RobstrideResponseActuatorCfg
from ..envs.bebop_v2_events import (
    push_magnitude_curriculum,
    randomize_torso_com_uniform,
    reset_joints_uniform_within_limits,
    torso_com_curriculum,
)
from ..envs.bebop_v2_rewards import (
    action_gain_rate_l2,
    action_position_rate_l2,
    bilateral_joint_symmetry_l2,
    com_over_support_reward,
    feet_flat_orientation_l2,
    joint_deviation_l1_balance_gated,
    joint_vel_l2,
    stationary_pose_exp,
    torso_pitch_asymmetric_reward,
    torso_pitch_zero_penalty,
    torso_settle_in_band_l2,
)
from ..envs.bebop_v2_terminations import (
    base_link_on_ground,
    imu_pitch_out_of_bounds,
    imu_roll_out_of_bounds,
)


# Symmetric ±30° tilt envelope. Wider than a strict stand so small drift doesn't
# die on the cliff; the slower base_link height check is the true fall terminator.
PITCH_FALL_LIMIT_DEG = 30.0
PITCH_FALL_LIMIT_GX = math.sin(math.radians(PITCH_FALL_LIMIT_DEG))
ROLL_FALL_LIMIT_DEG = 30.0
ROLL_FALL_LIMIT_GY = math.sin(math.radians(ROLL_FALL_LIMIT_DEG))

# Initial torso pitch (rad) and pitch-rate (rad/s) on reset. Symmetric ±15° /
# ±0.6 rad/s trains pitch-recovery authority (restored Jul 10 2026 after the
# no-perturbation diagnostic run). Recovery skill comes from these reset
# perturbations (and later the Push variant), NOT from extra reward terms.
PITCH_INIT_RAD = math.radians(15.0)
PITCH_RATE_INIT_RAD_S = 0.6

# Balance band (g_x) for the torso: near-upright with a slight FORWARD bias.
#
# Jul 17 2026 geometry audit (URDF link masses + foot sole STL), replacing
# the 8-12 deg BACK-lean band that preceded it:
#   * the ankle axis sits only 23 mm ahead of the heel edge of the sole
#     (the toe edge is +140 mm ahead — a toe-heavy, human-like foot);
#   * at the upright zero pose the whole-robot CoM (~14 kg, torso 6.7 kg)
#     already sits 48 mm BEHIND the ankle axis = 25 mm behind the heel
#     edge: straight-leg upright standing is statically IMPOSSIBLE. The
#     robot must flex the hips ~5 deg (slant the legs back, moving the
#     ankle under the CoM) and/or pitch the torso ~5 deg forward;
#   * the old 8-12 deg back-lean band put the CoM 131-172 mm behind the
#     ankle — 6-7x the heel margin. No posture in that band is statically
#     sustainable. Capture 20260717_213006 (run 2026-07-17_13-07-06
#     ckpt-4999) showed the consequence on hardware: the policy slid to
#     the feasible posture NEAREST the unreachable band — g_x mean -0.027
#     with hips flexed ~0.06 rad, CoM within a few mm of the heel edge —
#     where any backward wobble crosses the heel and the robot tips back
#     irrecoverably ("unstable once it adjusts pitch, has to be held").
#     Fighting the impossible target all episode also doubled its chatter:
#     24.4% slew-exceedance vs 6.3% for the Jul 15 quiet stand.
#
# The band therefore straddles the statically sustainable posture:
# near-upright (the hip anchor below is relaxed so the policy can make
# the ~5 deg geometric compensation), biased slightly FORWARD because the
# toe side carries the support margin (140 mm) while the heel side
# (23 mm) is the fall direction. g_x = -sin(pitch); g_x > 0 = forward.
BALANCE_BAND_GX_MIN = -0.02   # 1.1 deg back (heel side — stay off it)
BALANCE_BAND_GX_MAX = +0.05   # 2.9 deg forward (toe side — the safe side)

# Asymmetric Gaussian shoulders outside the band. The BACK (below) shoulder
# is wide (0.15 -> restoring slope reaches ~17 deg back) so a backward-
# tipped robot still has gradient home; the FORWARD shoulder is tighter
# (0.08) — forward excursions are self-limiting over the 140 mm toe margin.
BALANCE_EDGE_STD_BELOW = 0.15
BALANCE_EDGE_STD_ABOVE = 0.08
# Quadratic tails beyond the band, both directions (gain on
# relu(overshoot)^2, deadband at the band edge). The Gaussian shoulders'
# gradient dies ~2 std out; the tails keep a nonzero restoring gradient at
# large tilt — the learnable signal for "recover once the pitch starts to
# run away", which the pre-Jul-17 reward lacked (flat alive + termination
# cliff only). Mild gain: far-field direction, not a wall.
BALANCE_FWD_PENALTY_GAIN = 1.0
BALANCE_BWD_PENALTY_GAIN = 1.0
# Roll tolerance inside the band (g_y; ~0.15 ≈ 8.6° at 1/e).
BALANCE_ROLL_STD = 0.15

# Center of the balance band in g_x units — the balance target the movement
# penalties gate on. Jul 20 2026 retarget: set to 0.0 (torso locked at
# exactly 0° tilt) for the knee+ankle CoG-control strategy. The anti-chatter
# gates now peak exactly where the policy is supposed to live, and forward/
# backward gate relaxation is symmetric about the true target.
#
# History (preserved for context): pre-Jul-20 this was +0.015, the midpoint
# of the forward-biased band [MIN=-0.02, MAX=+0.05] that the Jul 17 2026
# geometry audit set because straight-leg upright standing is statically
# impossible (CoM 48 mm behind the ankle axis) and the sustainable posture
# needed ~5° of forward torso pitch OR hip flexion. The Jul 20 strategy
# resolves that the other way: the torso is pinned at 0° and the LEGS
# (hips/knees/ankles) articulate to put the feet under the CoM. The
# MIN/MAX constants above are now dormant (only the commented-out band
# terms read them) and are left in place to avoid churn; the gate center
# no longer derives from them.
BALANCE_BAND_GX_CENTER = 0.0
# Width of the balance gate Gaussian (in g_x units; ~0.10 ≈ 5.7° at 1/e).
# Tightened 0.15 -> 0.10 (Jul 15 2026) after capture 20260715_122500 showed
# the 0.15 gate opened too early (at moderate tilt where the policy should
# still be smooth), enabling a flailing limit cycle. At 0.10 the gate only
# relaxes the movement penalties at genuinely dangerous tilt (>6°), keeping
# the anti-chatter pressure on through the normal recovery regime.
BALANCE_GATE_STD = 0.10
# Floor for the balance gate (0..1). Prevents the movement penalties from
# vanishing entirely at tilt, which the policy otherwise exploits by
# manufacturing tilt to unlock chatter (capture 20260715_122500: vel_std
# 0.69-1.34, slew-exceedance 56.9%). At gate_floor=0.2 the -10.0 position_rate
# still contributes -2.0·rate at full tilt — enough to keep recovery smooth.
BALANCE_GATE_FLOOR = 0.20
# Width of the CoM-excursion balance gate (metres) — the gate signal used by
# the knee+ankle strategy's movement penalties (com_gate=True on
# position_rate / joint_vel). The gate fires at full strength when the CoM
# proxy (base_link xy) is over the foot midpoint and decays to BALANCE_GATE_FLOOR
# as the CoM heads off the support polygon. 0.06 m ≈ the old 5.7° tilt-gate
# width (0.65 m torso height × tan(5.7°)), so the anti-chatter pressure
# relaxes over roughly the same balance excursion as before — but keyed on
# the actual balance criterion (CoM over feet) instead of torso pitch, which
# the knee+ankle strategy holds at 0°. Tune UP toward 0.08 if small CoM
# corrections feel over-taxed (gate closes too fast, recovery too expensive);
# tune DOWN if the policy starts manufacturing CoM swings to unlock chatter.
BALANCE_GATE_DIST_STD = 0.06

# Bilateral joint pairs for the symmetry reward. Every L/R pair on this robot
# is sign-MIRRORED in the URDF (right-side flexion joints use a flipped -Y
# axis; hip_abduction uses the same +X axis but mirrored limits). A symmetric
# stance therefore reads q_L = -q_R for ALL four pairs, and the asymmetry
# residual is the SUM (q_L + q_R), not the difference. See the per-pair axis
# table in ``bilateral_joint_symmetry_l2``'s docstring and the verification
# against ``ros2/src/bebopv2_description/urdf/bebopv2.urdf``.
BILATERAL_SYMMETRY_PAIRS = [
    ("hip_flexion_left_joint", "hip_flexion_right_joint"),
    ("hip_abduction_left_joint", "hip_abduction_right_joint"),
    ("knee_flexion_left_joint", "knee_flexion_right_joint"),
    ("foot_left_joint", "foot_right_joint"),
]
# Weight on the bilateral-symmetry penalty (Jul 15 2026). Capture
# 20260715_214224 of run 2026-07-15_12-35-39 ckpt-4000 showed severe
# bilateral asymmetry on hardware despite the (correct, sum-convention)
# symmetry term existing in the code: hip_flexion L+R=-0.82, foot L+R=+0.49
# (both far from the symmetric 0). Root cause: the symmetry term was NOT
# in the active RewardsCfg — the only symmetry enforcement was the weak
# -0.3 all-joint joint_pos_anchor, whose gradient is diluted 8 ways and
# cannot enforce per-pair mirroring. This dedicated term is heavily
# weighted (-2.0) and balance-gated so it fires at full strength when the
# robot is balanced (the stance we want symmetric) and relaxes toward
# gate_floor when tilted (the policy is free to use an asymmetric catch).
# Why -2.0: at the capture's mean asymmetry (Σ(q_L+q_R)² ≈ 0.82²+0.49²
# ≈ 0.91 for the two worst pairs alone) this contributes ≈ -1.8/tick,
# comparable to the alive reward (+1.0) — finally enough to make an
# asymmetric stance expensive relative to the survival gradient. The
# gate_floor=0.2 keeps a -0.36/tick floor at full tilt so the policy
# can't manufacture tilt to suppress the constraint.
BILATERAL_SYMMETRY_WEIGHT = -2.0

# Fraction of each joint's soft range used when sampling reset poses. A small
# value keeps resets near the nominal pose so most episodes survive; widen
# toward 1.0 once the robot stands reliably.
RESET_JOINT_RANGE_FRACTION = 0.25

# Mid-episode push envelope for the reactive-recovery variant (m/s).
# Widened ±0.4 -> ±0.7 (Jul 18 2026) after capture model_3000.mcap (run
# 2026-07-18_17-51-11 ckpt-3000, hardware) showed the ±0.4-trained policy
# sitting settled at g_x mean -0.196 (11 deg back) and NOT recovering: the
# ±0.4 m/s instantaneous shove (<=5.6 N.s on the 14 kg robot) never pushed
# the CoM far enough off the support to make recovery worth learning, so
# the policy converged to "absorb and stay tilted". At ±0.7 the resultant
# impulse reaches ~1.0 m/s (~14 N.s) — enough to force an actual catch.
PUSH_VELOCITY_RANGE = {"x": (-0.7, 0.7), "y": (-0.7, 0.7)}

# Target standing base_link height (m). UNVERIFIED — measure the true settled
# height before enabling the optional base_height_l2 reward.
BASE_HEIGHT_TARGET = 0.65

# Torso CoM robustness range (m). Horizontal ±2 cm only; curriculum ramps from
# 25% of this range to full over 100k control steps.
TORSO_COM_RANGE = {
    "x": (-0.02, 0.02),
    "y": (-0.02, 0.02),
    "z": (0.0, 0.0),
}
TORSO_COM_START_FRACTION = 0.25
TORSO_COM_CURRICULUM_STEPS = 100_000

# Observation noise calibrated to the real hardware noise floor (capture
# 20260714_031815, DIAL_IN mode, motors armed, robot still). The previous
# values were 4-100x too high (worst on projected_gravity and joint_pos),
# which over-regularized the policy — it learned to discount the very tilt
# signal it needs on hardware. Projected_gravity is the policy's primary
# balance cue; over-noising it trained a half-blind-to-tilt policy that then
# leaned 7.7 deg sideways on hardware without correcting.
#
# Real still-state stds (rad or rad/s):
#   gyro wx/wy         0.0023-0.0028  -> sim   0.01   (4x high, mild)
#   joint_pos          0.0001-0.0007  -> sim   0.01   (15-100x high, severe)
#   joint_vel          0.036-0.072    -> sim   0.12   (1.7-3.3x high, mild)
#   proj_grav gx/gy    0.0003-0.0004  -> sim   0.02   (50-65x high, severe)
#
# Calibrated values: 2-3x the still-state std to cover the policy-active
# vibration regime (capture 20260714_030338 RUN_POLICY: gyro std ~0.2 rad/s,
# ~50x the still state, so the active floor is what the policy must weather).
# The bigger numbers reflect what the sensors actually look like when the
# robot is balancing, not the desk-still floor.
GYRO_NOISE_RAD_S = 0.006
JOINT_VEL_NOISE_RAD_S = 0.10
PROJ_GRAV_NOISE = 0.005
JOINT_POS_NOISE_RAD = 0.003

# Per-episode randomized action transport delay in 100 Hz policy ticks.
# Training across 10-40 ms forces a latency-robust (more damped) policy.
ACTION_DELAY_RANGE = (1, 4)

# Sole material (polyurethane). The Jul 2026 feet have grippy polyurethane
# soles; at the default terrain friction (mu=1.0) the sim feet slide under
# the lateral/sagittal shear a real sole would hold, so the policy never
# needs to pick a foot up — it can shuffle/skid to recover. The real soles
# essentially cannot slide on the deployment floor, so that shuffle strategy
# does not transfer: hardware CANNOT skid, and a policy that leans on
# sliding gets stuck. Modeling the grip forces the policy to learn
# lift-and-place footwork ("move the foot through air") instead of dragging.
#
# mu ~1.7-2.1 for polyurethane on a hard floor vs the current 1.0. Sampled
# per env from the range below (make_consistent keeps mu_dynamic <=
# mu_static per PhysX's constraint). Values are boot-time bucketed
# (num_buckets) and re-assigned per episode reset, so training sees a spread
# of grip levels rather than one exact coefficient — robust to the true
# sole/floor pairing, whatever it measures out to.
FOOT_FRICTION_STATIC_RANGE = (1.7, 2.1)
FOOT_FRICTION_DYNAMIC_RANGE = (1.7, 2.1)
FOOT_FRICTION_BUCKETS = 64

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
# order: [hipflexL, hipflexR, hipabdL, hipabdR, kneeL, kneeR, footL, footR]
# Foot (ankle) kp_min raised 100 -> 212.5 (Jul 9 2026): capture of run
# 2026-07-08_03-55-34 ckpt-4500 showed foot kp parked at ~111, right at the
# old floor — the RS02 (~17 N·m, weakest joint) was being commanded the
# softest setting available and the ankle got pushed around when the robot
# tried to stand. Raising the floor to 75% of the band (100 + 0.75*(250-100)
# = 212.5) forces the policy to keep ankle authority high; the policy can
# still tune kp within [212.5, 250] but can't go soft. Mirror in firmware
# bebop_v2.yaml foot_left_joint / foot_right_joint.
POLICY_KP_MIN = [20.0, 20.0, 40.0, 40.0, 30.0, 30.0, 212.5, 212.5]
POLICY_KP_MAX = [120.0, 120.0, 150.0, 150.0, 150.0, 150.0, 250.0, 250.0]
POLICY_KD_MIN = [2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
POLICY_KD_MAX = [5.0, 5.0, 8.0, 8.0, 8.0, 8.0, 5.0, 5.0]

# Robstride motor friction / armature (sysid-measured on right-side joints) and
# datasheet stall torque / no-load speed (re-measure with a brake fixture before
# changing the saturation_effort / velocity_limit values).
JOINT_FRICTION_HIP_FLEX = 0.567
JOINT_FRICTION_HIP_ABD = 0.373
JOINT_FRICTION_KNEE_FLEX = 0.633
JOINT_FRICTION_FOOT = 0.159

JOINT_ARMATURE_HIP_FLEX = 0.0310
JOINT_ARMATURE_HIP_ABD = 0.0114
JOINT_ARMATURE_KNEE_FLEX = 0.0312
JOINT_ARMATURE_FOOT = 0.0038

MOTOR_STALL_TORQUE_RS04 = 120.0
MOTOR_STALL_TORQUE_RS03 = 60.0
MOTOR_STALL_TORQUE_RS02 = 17.0

# RS02 ankle torque authority. The ankle is capable of the full 17 Nm RS02
# stall torque, and the Jul 20 2026 retarget to a knee+ankle CoG-control
# strategy WANTS the ankle as an active balance actuator (ankle strategy:
# shift the CoP within the foot via ankle torque). The sim effort limit is
# therefore the stall value, not the continuous rating.
#
# History (preserved for context): Jul 10 2026 capped this at 6 Nm (the RS02
# continuous rating @ 100 rpm per datasheet). With the limit at 17 Nm the
# sim ankle could hold peak torque INDEFINITELY, so training converged on an
# ankle-supported lean (capture 20260710_033855 showed ~37 Nm commanded at
# the right foot — 2x stall). The real motor saturates and gets backdriven,
# hence the sim-to-real gap, and the 6 Nm cap forced the policy to balance
# with the hips over the foot instead of leaning on the ankle. That prior
# strategy is being reversed: the user explicitly wants the ankle to share
# the balance load with the knees. The constant name keeps the historical
# ``CONT`` label; the value is now the stall torque. Watch ankle torque p95
# in captures — if it parks at the limit the policy is leaning on the ankle
# the way it did under the pre-Jul-10 uncapped config, and the cap should be
# reconsidered.
MOTOR_CONT_TORQUE_RS02 = 17.0

MOTOR_NOLOAD_VEL_RS04 = 20.9
MOTOR_NOLOAD_VEL_RS03 = 20.4
MOTOR_NOLOAD_VEL_RS02 = 42.9


def _midpoint(lo: list[float], hi: list[float]) -> list[float]:
    return [0.5 * (a + b) for a, b in zip(lo, hi)]


_Kp_MID = _midpoint(POLICY_KP_MIN, POLICY_KP_MAX)
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
        # Spawn z: sole sits 0.7302 m below base_link at zero pose (Jul 2026
        # feet redesign; old feet were 0.7668 → z=0.8). +0.035 m settle
        # clearance. Keep in sync with LIFT_Z in
        # sim/scripts/post_import_bebopv2.py.
        pos=(0.0, 0.0, 0.765),
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
            stiffness=_Kp_MID[0],
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
            stiffness=_Kp_MID[2],
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
            stiffness=_Kp_MID[4],
            damping=_KD_MID[4],
            armature=JOINT_ARMATURE_KNEE_FLEX,
            friction=JOINT_FRICTION_KNEE_FLEX,
        ),
        "foot": DCMotorCfg(
            joint_names_expr=["foot_left_joint", "foot_right_joint"],
            saturation_effort=MOTOR_STALL_TORQUE_RS02,
            # Ankle torque authority — see MOTOR_CONT_TORQUE_RS02 (Jul 20 2026:
            # raised to the full 17 Nm stall so the ankle is an active balance
            # actuator under the knee+ankle CoG strategy).
            effort_limit_sim=MOTOR_CONT_TORQUE_RS02,
            velocity_limit_sim=20.0,
            velocity_limit=MOTOR_NOLOAD_VEL_RS02,
            stiffness=_Kp_MID[6],
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
        # Newton resolves joints right-before-left; firmware is left-before-right.
        # preserve_order=True keeps the action I/O in firmware order.
        preserve_order=True,
        pos_scale=0.5,
        use_default_offset=True,
        # Recovery-bandwidth limits, relaxed Jul 22 2026 (round 6, user-
        # approved, mirrored in firmware bebop_v2.yaml): four configs hit
        # the same wall at ~±0.4-0.5 m/s pushes — surviving a hard shove
        # needs actuation bandwidth the round-4/5 values didn't provide.
        # Slew 0.020 -> 0.030 rad/tick (2 -> 3 rad/s setpoint rate; a
        # 0.5 rad catch-step swing takes 0.17 s instead of 0.25 s).
        max_pos_step_per_tick=0.030,
        action_delay_steps=2,
        action_delay_range=ACTION_DELAY_RANGE,
        # Hard anti-snap guarantee on the kp/kd channels (Jul 22 2026):
        # EMA-filtered gains CANNOT chatter, so fast gain modulation is
        # physically useless and the gain_rate reward tax only has to
        # govern exploration NOISE (its round-5 role), never the
        # survival strategy. tau 0.15 -> 0.08 s (round 6): still ~15x
        # attenuation of tick-rate snapping (ripple 6.5%), but the -3 dB
        # bandwidth rises to ~2 Hz so ankle/hip impedance can stiffen in
        # ~0.08 s — fast enough to matter inside a push transient.
        # FIRMWARE MUST mirror this EMA at decode time.
        gain_ema_tau_s=0.08,
        kp_min=POLICY_KP_MIN,
        kp_max=POLICY_KP_MAX,
        kd_min=POLICY_KD_MIN,
        kd_max=POLICY_KD_MAX,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # base_lin_vel is intentionally NOT observed: the real robot has no
        # odometry/VIO and can only feed zeros, so training against the sim's
        # privileged ground-truth base velocity created a sim-to-real gap.
        base_ang_vel = ObsTerm(
            func=mdp.imu_ang_vel,
            params={"asset_cfg": SceneEntityCfg("imu")},
            noise=Unoise(n_min=-GYRO_NOISE_RAD_S, n_max=GYRO_NOISE_RAD_S),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-PROJ_GRAV_NOISE, n_max=PROJ_GRAV_NOISE),
        )
        # preserve_order=True keeps joint sensing in lock-step with the action
        # term and firmware obs builder (left-before-right).
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
            self.enable_corruption = True

    def __post_init__(self):
        self.policy = self.PolicyCfg()


@configclass
class EventCfg:
    reset_joints = EventTerm(
        func=reset_joints_uniform_within_limits,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES_ALL),
            "velocity_range": (-0.5, 0.5),
            "range_fraction": RESET_JOINT_RANGE_FRACTION,
        },
    )
    # Only torso pitch (and pitch rate) are perturbed at reset — the real
    # failure mode is the heavy torso falling fore/aft, not a sideways drop.
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
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

    # Actuator dynamics randomization — sysid measured right-side joints only;
    # re-sample each episode to cover left-side / unit / thermal / wear spread.
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

    # Small torso-CoM randomization (horizontal ±2 cm box). Curriculum ramps from
    # ±5 mm to full range; non-accumulating so it is safe in reset mode.
    randomize_torso_com = EventTerm(
        func=randomize_torso_com_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": dict(TORSO_COM_RANGE),
        },
    )

    # Grippy polyurethane soles (Jul 18 2026). Raises ONLY the two foot
    # bodies' contact friction from the default 1.0 to the polyurethane
    # range (~1.7-2.1, sampled per env); every other body keeps the default
    # material. With mu=1.0 the sim feet slid under shear the real soles
    # hold, letting the policy shuffle/skid to recover — a strategy that
    # cannot transfer to hardware (the real soles can't slide), and one
    # that leaves the feet planted when they should step. High grip removes
    # the skid crutch so recovery has to come from lift-and-place footwork.
    randomize_foot_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=("foot_left_1", "foot_right_1")
            ),
            "static_friction_range": FOOT_FRICTION_STATIC_RANGE,
            "dynamic_friction_range": FOOT_FRICTION_DYNAMIC_RANGE,
            "restitution_range": (0.0, 0.0),
            "num_buckets": FOOT_FRICTION_BUCKETS,
            "make_consistent": True,
        },
    )

    # Link-mass randomization (Jul 24 2026, paper-derived — Dowdy & Chagas
    # Vaz, arXiv:2607.18135, Table I: Link Mass scale 0.8-1.3). Covers
    # unit-to-unit mass spread, harness/cable/battery-state variation, and
    # URDF mass error — the axis this config did not randomize at all (the
    # torso-CoM event moves the CoM but keeps total mass/inertia fixed;
    # this scales them). Complements randomize_torso_com, does not replace
    # it. recompute_inertia=True scales the inertia tensors with the mass
    # ratios so dynamics stay consistent.
    randomize_link_masses = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_params": (0.8, 1.3),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )


@configclass
class RewardsCfg:
    """0°-torso knee+ankle CoG-control standing reward (Jul 20 2026 retarget):
    pin the torso at 0° tilt (no pitch/roll balancing via the heavy torso)
    and balance the center of gravity by articulating the knees and ankles
    (the "ankle/knee strategy"). Top priority is stability — minimize
    tick-to-tick action changes and oscillation; small foot movements / a
    recovery step to keep balance are acceptable. Trained on the push task
    (``Isaac-BebopV2-Standing-Push-v0``) so the policy learns counter-
    balancing under ±0.7 m/s fore/aft AND lateral shoves.

    Jul 20 2026 strategy shift: the prior design (Jul 17 retarget, see
    History below) asked the policy to hold a near-upright slightly-forward
    torso pitch band and balance by pitching the torso. That did not balance
    well — the heavy torso is the wrong actuator for fine CoG control. The
    new design pins the torso at 0° via an always-on, bounded tilt penalty
    (``torso_pitch_penalty``, now pitch+roll) and lets the LEGS do the
    balancing: ``com_over_support`` rewards keeping the CoM over the feet,
    which forces the knees/ankles/hips to articulate to put the feet under
    the CoM. The static-impossibility geometry (upright zero pose puts the
    CoM 48 mm behind the ankle axis, 25 mm behind the heel edge — see the
    BALANCE_BAND_GX_* constants block) is resolved by leg articulation
    (~5° hip flexion and/or knee+ankle crouch), NOT by torso pitch. Any
    joint-position anchor to the all-zeros pose (``hip_flexion_anchor``,
    ``joint_pos_anchor``) is therefore OFF — it would directly fight the
    leg articulation the strategy requires.

    Ankle authority: the RS02 foot ``effort_limit_sim`` is set to the full
    17 Nm stall torque (was capped at 6 Nm continuous pre-Jul-20) so the
    ankle can participate as an active balance actuator — the ankle
    strategy needs that torque to shift the CoP within the foot. Watch
    ankle torque p95 in captures: if it parks at the limit the policy is
    leaning on the ankle the way the pre-Jul-10 uncapped config did, and
    the cap should be reconsidered.

    Anti-chatter trio + the gain filter + the std governor (the #1
    priority): the movement penalties attack tick-to-tick motion and
    oscillation from different sides, each with documented solo failure
    modes when absent. ``position_rate`` and ``joint_vel`` are gated by
    the CoM-excursion balance gate (see the Gate machinery section
    below) — full strength at balance, relaxed during recovery;
    ``base_ang_vel_xy`` is NOT gated. Fast kp/kd gain chatter is handled
    STRUCTURALLY by the ``gain_ema_tau_s`` low-pass filter in the action
    term (round 4), while ``gain_rate`` (re-added round 5) taxes the RAW
    gain-channel rate as an anti-noise governor on the shared log-std:

    * ``position_rate`` (CoM-gated, -10.0) — action-side: tick-to-tick
      change of the 8 position setpoints. The only term that isolates the
      setpoint chatter. Quadratic → slow ankle/knee micro-corrections are
      nearly free (0.005 rad/tick ≈ -0.004/tick at gate 1); slew-limit
      flipping is taxed. The CoM gate opens to its 0.2 floor during a
      shove so recovery motion stays 5× cheaper than chatter at steady
      state.
    * ``joint_vel`` (CoM-gated, -1.0) — plant-side: unbounded L2 on
      joint velocity. Keeps a linear-in-v gradient all the way to v=0,
      which is what kills small limit cycles (the bounded
      ``standing_stillness`` carrot's gradient vanishes as v → 0).
    * ``base_ang_vel_xy`` (NOT gated, -0.30) — torso-side: IMU gyro
      ω_x² + ω_y². Damps the whole-body sway-on-the-ankles limit cycle
      that survives when joint velocities are small (and that this
      strategy will excite, since the torso is pinned and residual
      oscillation shows up as torso pitch/roll RATE).
    * gain chatter — STRUCTURAL, not a reward: ``gain_ema_tau_s=0.08``
      in the action term EMA-filters the decoded kp/kd, so the gains
      physics sees cannot snap (100 Hz square-wave ripple ≈ 6.5%).
    * ``gain_rate`` (NOT gated, -2.0) — anti-noise governor on the raw
      gain channels. Safe since the EMA (fast gain modulation is
      physically useless, so the survival strategy lives in the position
      channels and this tax hits only exploration noise); NECESSARY
      because the shared log-std otherwise explodes with no task pressure
      on the filtered gain channels (run 2026-07-22_20-49-06: std 75k →
      4.7e15). See the round-5 note at the term below.

    Active reward set (10 terms):
    * ``alive`` (+4.0, raised +1.0 → +2.0 → +4.0 Jul 21-22 2026) — survival
      carrot. Raised to fix the NEGATIVE net reward-while-alive budget: round-2
      (alive +1.0) had ~-0.96/step at the peak (eplen 1739 → 565 collapse);
      round-3b (alive +2.0) had ~-0.647/step (eplen 1154 → 690 collapse). At
      +4.0 the net flipped positive (+1.34/step) and run 2026-07-22_10-55-08
      reached eplen 1904 (95%) with reward +23 and termination ~0. Also kills
      the freeze exploit: with a positive net per step, surviving long (only
      active balance can, against pushes) beats freezing (topples fast).
    * ``termination_penalty`` (-200.0) — fall = catastrophic (the only
      fall signal; IMU tilt terminations are disabled for push training).
    * ``torso_pitch_penalty`` (``torso_pitch_zero_penalty``, -2.0) —
      always-on bounded inverted Gaussian on torso TILT (pitch + roll)
      centered exactly at 0°, saturating at ±6° magnitude. Prices tilt
      in every state without ever blocking recovery (max cost is the
      weight, not an unbounded quadratic). Extended Jul 20 2026 from
      pitch-only to pitch+roll because the push task shoves laterally
      too — a roll lean can be settled into just like a pitch lean.
      Saturation tightened 10° → 6° (Jul 20 2026 round 2) after run
      2026-07-21_03-21-08 settled into a persistent ~6° lean: at 10°
      saturation a 6° lean only cost ~-1.1/step, which the (then
      un-gated) stillness carrot happily paid; at 6° saturation the same
      lean costs the full -1.9/step, so "hold a small lean" is clearly
      worse than "correct to 0°". Far-field (>6°) gradient comes from
      ``com_over_support`` + ``alive``, by design.
    * ``com_over_support`` (+0.4, stillness-gated) — bounded [0,1] reward
      for keeping the torso CoM (base_link xy) over the foot midpoint.
      The "control the CoG with the legs" carrot — without it the policy
      could hold g_x=0 with the CoM parked behind the 23 mm heel margin
      (a falling pose that earns full pitch credit). Stillness-gated so
      the policy can't earn it by swinging the CoM through the target;
      earns partial credit at moderate motion so small steps stay
      affordable ("move around a little to keep balance" is OK).
    * ``standing_stillness`` (``stationary_pose_exp``, +1.0, upright-gated) —
      bounded [0,1] carrot for zero joint velocity, multiplied by an
      uprightness Gaussian (``exp(-(g_x²+g_y²)/0.12²)``) so it only pays full
      when still AND near-upright. Saturates to 0 during a push transient so
      it never punishes recovery motion. Upright gate added Jul 20 2026
      (round 2): the un-gated carrot rewarded holding *whatever* pose the
      policy settled into — including a tilted one — and run
      2026-07-21_03-21-08 exploited that to freeze in a still ~6° lean. A
      CoM-on-support balance gate was tried Jul 21 2026 (round 3) and
      REVERTED: it was ineffective (a frozen balanced statue has the CoM
      over the feet, so the gate is 1.0 — it doesn't kill the freeze) and
      harmful (it zeroed the carrot during early learning when the robot is
      always falling, making the reward too sparse — run 2026-07-21_22-27-00
      regressed to eplen 71 with std exploding to 4.19). The freeze exploit
      is instead addressed by raising ``alive`` (see below).
    * ``position_rate`` / ``joint_vel`` / ``base_ang_vel_xy`` / ``gain_rate``
      — the anti-chatter trio + std governor above (fast gain chatter is
      also handled structurally by the gain_ema_tau_s filter).
    * ``feet_flat`` (``feet_flat_orientation_l2``, -5.0, stillness-gated) —
      Σ_feet sin²(sole tilt from horizontal). Forces both soles parallel
      to the ground (the ankle-strategy contact patch), NOT a foot-joint
      anchor: in the leg-articulated stance the shank slants, so a flat
      sole needs q_foot ≈ ±5°; the term targets the sole *orientation*
      and lets the ankle pick the angle.

    Explicitly OFF (deliberate, would fight the strategy):
    * ``torso_posture`` / ``torso_settle`` — the forward-biased band
      contradicts a 0° target; superseded by ``torso_pitch_penalty``.
    * ``hip_flexion_anchor`` / ``joint_pos_anchor`` — anchor to the
      all-zeros pose, which is the statically impossible straight-leg
      pose; at 0° torso the legs *must* articulate, so any anchor to zero
      fights the strategy head-on.
    * ``bilateral_symmetry`` — OFF for simplicity. Compatible with a
      symmetric crouch (it penalizes q_L+q_R, not |q|), but kept off
      until a capture shows the one-leg-loading failure mode it exists
      for (capture 20260715_214224: hip L+R=-0.82, foot L+R=+0.49). 
      TRIPWIRE: re-enable at -2.0 if ``analyze_capture.py`` shows
      |L+R| > ~0.2 on hip_flexion or foot pairs after lateral shoves.

    Gate machinery (Jul 20 2026 round 2 — CoM-excursion gate): the gated
    movement penalties (``position_rate``, ``joint_vel``) use
    ``gate = max(gate_floor, exp(-(dist / gate_dist_std)²))`` where ``dist``
    is the CoM-off-support distance ``‖base_link_xy − foot_midpoint_xy‖``
    (``_com_off_support_dist``), ``gate_dist_std = BALANCE_GATE_DIST_STD =
    0.06 m``, and ``gate_floor = 0.20``. This REPLACES the pre-round-2 torso
    -tilt gate (``exp(-((g_x - center)² + g_y²) / gate_std²)``), which was
    keyed on the very signal this strategy holds at 0°: the knee+ankle
    strategy recovers from pushes while keeping the torso LEVEL, so the
    tilt gate stayed closed (gate ≈ 1) during exactly the recovery motion it
    should have been relaxing for — the policy was charged full anti-chatter
    price to recover, and run 2026-07-21_03-21-08 plateaued at ~25% eplen
    with gain_rate/position_rate the dominant penalties. Keying on CoM
    excursion — the actual balance criterion — opens the gate during a
    level-torso recovery, making the desired knee/ankle recovery affordable
    while still killing chatter at balance (gate ≈ 1 when the CoM is over
    the feet). The floor is essential — without it the policy manufactures
    CoM swings to unlock chatter (the flailing-limit-cycle exploit of
    capture 20260715_122500: vel_std 0.69-1.34, slew-exceedance 56.9%); at
    floor 0.2 the -10.0 position_rate still contributes -2.0·rate at full
    excursion, enough to keep recovery smooth without making it
    unaffordable. ``gate_dist_std = 0.06 m`` ≈ the old 5.7° tilt-gate width
    (0.65 m torso height × tan(5.7°)); tune up toward 0.08 if small CoM
    corrections feel over-taxed, down if CoM-swing chatter appears.

    Round 3 (Jul 21 2026): raised ``alive`` +1.0 → +2.0 to fix a NEGATIVE
    reward-while-alive budget. Run 2026-07-21_11-54-01 (the round-2 config)
    peaked at eplen 1739 then collapsed to ~565 with ``gain_rate`` the
    dominant penalty (-1.52/step at the peak) and the net per-step
    reward-while-alive at ~-0.96/step. A negative net-while-alive flips the
    survival incentive: with the -200 termination stick, dying before
    ~step 1792 returns MORE than surviving longer (every extra step alive
    costs net reward), so PPO pushed eplen DOWN after the peak and the std
    collapse (0.28) trapped it there. The fix is raising ``alive`` +1.0 →
    +2.0 so the net per-step reward-while-alive is clearly positive (~+0.3
    at the peak) — every step survived is now rewarded, so the gradient
    pushes toward timeout instead of early death, and the freeze exploit (a
    still upright statue collecting ~+2.4/step) is also killed: with a
    positive net per step, surviving long (which only active balance can do
    against pushes) always beats freezing (which topples at the first shove).

    Two round-3 experiments were REVERTED: (1) CoM-gating ``gain_rate`` —
    the gate can't tell "legitimate recovery" from "destructive flailing"
    (both have the CoM off-support), so it removed the anti-flail brake
    exactly during the flailing phase; PPO's entropy pushed ``mean_std`` to
    5.97 (vs 1.30 ungated) and eplen cratered to ~65 (run
    2026-07-22_00-00-48). gain_rate MUST stay ungated — its full-strength
    always-on pressure is the governor that keeps exploration bounded. (2) A
    CoM-on-support balance gate on ``standing_stillness`` — ineffective (a
    frozen balanced statue has the CoM over the feet, so the gate is 1.0) and
    harmful (zeroed the carrot during early learning when the robot is always
    falling, making the reward too sparse — run 2026-07-21_22-27-00 regressed
    to eplen 71 with std 4.19). Both motivated by the same
    net-negative-while-alive budget that the ``alive`` bump alone addresses.

    Round 4 (Jul 22 2026): the alive +4.0 fix exposed the NEXT budget
    break. Runs 2026-07-22_10-55-08 (round-3b config) and 2026-07-22_12-29-28
    (+ the short-lived crouch_height / com_recovery pair) BOTH climbed to a
    peak (eplen 1935 @ iter 2397 / 1315 @ 1475) and then degraded hard
    (eplen ~905 / ~619, reward -21 / -16). The per-term series shows the
    mechanism: eplen kept RISING through the reward peak (1703 -> 1801
    while reward fell +27.7 -> -0.2) because the policy's survival strategy
    at strengthening pushes is continuous kp/kd impedance modulation — and
    the ``gain_rate`` -2.0 quadratic taxed it harder the better the policy
    balanced (gain_rate alone: -1.7/step at the reward peak, -2.9/step at
    collapse; termination stayed ~-0.02: the robot almost never FELL at
    peak, it just couldn't afford to stay alive). A reward tax on the only
    available survival mechanism makes survival unaffordable — the tax has
    to become a hard constraint instead. Fix (user-approved): (1) the
    decoded kp/kd channels are EMA low-passed in the action term
    (``gain_ema_tau_s=0.15``) so gain chatter is PHYSICALLY impossible —
    mirroring how the position slew clamp handles setpoint snapping —
    and ``gain_rate`` is deleted from the reward (RE-ADDED the same day
    in round 5 in a safe anti-noise role — see below); (2) the inert
    ``crouch_height`` / ``com_recovery`` terms (logged +0.0000 / +0.0006
    in run 12-29-28) are removed; (3) ``entropy_coef`` 0.02 -> 0.03 in
    ``BebopPPOPushCfg`` (the config's own documented remedy for "policy
    goes deterministic and stops recovering" — std collapsed to 0.33 by
    iter 1000). With gains filtered, fast recovery must go through the
    (slew-limited, CoM-gate-cheap) position channels — the stepping /
    crouching behavior the pushes are meant to teach. FIRMWARE MUST mirror
    the gain EMA at decode time before the next deploy.

    Round 5 (Jul 22 2026, same day): the round-4 run (2026-07-22_20-49-06)
    failed on a NEW mode — the shared log-std exploded EXPONENTIALLY
    (75k @ iter 1000 → 3.6e12 @ 3000 → 4.7e15 @ 3772, vs bounded ~1.2 in
    every ungated-gain_rate run). Deleting ``gain_rate`` had removed not
    just the chatter tax but the anti-noise brake on the shared log-std:
    with the 16 gain channels EMA-decoupled from physics there was no task
    pressure keeping their exploration noise small, and the entropy bonus
    (simultaneously raised 0.02 → 0.03 — two variables at once, mistake)
    inflated the std unopposed, leaking noise into the position channels
    (a slew-clamped random walk; position_rate -1.71, budget negative).
    The filters shielded physics long enough to peak at eplen 1779 /
    reward +41 (best yet), then the advantage signal drowned. Fix:
    ``gain_rate`` -2.0 RE-ADDED (ungated) in its new, safe role — with
    fast gain modulation physically useless under the EMA, the tax hits
    only noise, not survival — and ``entropy_coef`` reverted to 0.02
    (last-known-good; the EMA already breaks the gain-modulation local
    optimum structurally, no exploration bump needed). Net config vs the
    round-3b breakthrough config: identical rewards + entropy, PLUS the
    gain_ema_tau_s filter — one structural variable.

    Round 6 (Jul 22 2026): round-5 ran stable (std bounded 0.27, early
    gain chatter gone, proper knees-bent stance) but hit the SAME wall as
    every prior config — peak eplen 1768 @ iter 1532 right as pushes
    crossed ~±0.4 m/s, then decay. The tell: ``gain_rate`` CLIMBED
    -1.4 → -2.2 while std FELL, i.e. the policy was deliberately driving
    raw gain rate — fighting the EMA (overshooting raw commands to
    pre-compensate the low-pass) to claw back bandwidth the τ=0.15 s
    filter and the 0.020 rad/tick slew wouldn't pass. Four reward configs
    hitting the same wall at the same push magnitude means the remaining
    bottleneck is PHYSICAL ACTUATION BANDWIDTH, not reward shaping. Fix
    (user-approved, firmware-mirrored): ``gain_ema_tau_s`` 0.15 → 0.08
    (impedance bandwidth 1 → ~2 Hz, stiffen in ~0.08 s; tick-rate ripple
    3.3% → 6.5%, still ~15× snap attenuation) and
    ``max_pos_step_per_tick`` 0.020 → 0.030 (2 → 3 rad/s; a 0.5 rad
    catch-step swing in 0.17 s). No reward changes this round.

    History (Jul 10-17 2026 evolution, preserved for context): the Jul 17
    retarget set a forward-biased pitch band (g_x ∈ [-0.02, +0.05]) and
    balanced via torso pitch — the strategy the Jul 20 retarget reverses.
    The Jul 10-14 design was a *statue* (survive, hold posture, move
    smoothly); the smoothing worked (capture 20260715_041834:
    slew-exceedance 40-55% -> 6.3%) but the robot could not balance: on a
    gantry test it hung at g_x mean +0.065 (12° off the then-band) because
    the movement penalties made every recovery motion unaffordable. The
    Jul 15 fix introduced the balance gate (suppress movement penalties
    during tilt) — the Jul 20 design keeps the gates and changes only the
    balance TARGET (torso pitch → 0°) and the balance ACTUATOR (torso →
    knees/ankles).
    """

    # Survival carrot. Raised +1.0 -> +2.0 Jul 21 2026 (round 3), then +2.0 ->
    # +4.0 Jul 22 2026 (round 3b): run 2026-07-22_02-56-07 (alive +2.0, gain_rate
    # ungated) kept std bounded (max 1.18 — the ungating fix worked) and learned
    # to balance faster than round-2 (eplen 842 @1000 vs 311), but still peaked
    # at 1154 then collapsed to 690 — the net per-step reward-while-alive was
    # STILL NEGATIVE (~-0.647/step at the peak; alive +2.0 closed only a third
    # of the gap from round-2's -0.96). At the peak gain_rate alone (-1.19)
    # nearly cancelled alive (+1.08). Raising alive to +4.0 adds ~+1.08 at the
    # peak (alive fraction ~0.54), flipping the net to ~+0.43/step — clearly
    # positive, so every step survived is rewarded and the gradient pushes
    # toward timeout instead of early death. Also kills the freeze exploit: a
    # frozen upright statue collects ~+4.4/step but topples fast (short
    # collection), while a balancer at +0.43/step × 2000 = +860 dominates. The
    # tradeoff is alive now dominates the shaping (4x stillness), but for a
    # survival task "don't die" IS the primary objective; the secondary terms
    # (pitch penalty, com_over_support, stillness) still shape HOW. gain_rate
    # stays ungated at full -2.0 (see gain_rate). If the policy still dies
    # early at +4.0, the next step is adding an uprightness carrot
    # (upright_pose_exp) rather than raising alive further.
    alive = RewTerm(func=mdp.is_alive, weight=4.0)
    # -200/tick on termination — large stick so the policy treats falls as catastrophic.
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # Joint-limit safety penalties (Jul 24 2026, paper-derived —
    # arXiv:2607.18135, Table III: Joint Position Limit -100 / Joint
    # Velocity Limit -10; their fix for the policy transiently touching
    # limits for single timesteps). Until now the only limit guards were
    # soft_joint_pos_limit_factor=0.9 (a physics constraint, not a
    # learning signal) and termination. These terms make approaching the
    # limits expensive so the policy keeps margin on its own. Expected to
    # log ~0 most of the time: the reset sampler stays inside soft limits
    # and the slew clamp bounds setpoint velocity — verify in the per-term
    # reward logs that they only fire during resets/push transients.
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-100.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=JOINT_NAMES_ALL, preserve_order=True
            )
        },
    )
    joint_vel_limits = RewTerm(
        func=mdp.joint_vel_limits,
        weight=-10.0,
        params={
            "soft_ratio": 0.9,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=JOINT_NAMES_ALL, preserve_order=True
            ),
        },
    )

    # Bounded [0,1] reward exp(-Σv²/σ²) for holding the pose perfectly still,
    # upright-gated so it only pays full when still AND near-upright — the
    # positive "stand stable" carrot; saturates to 0 during a push/recovery
    # transient so it never punishes recovery motion. Jul 20 2026: added the
    # upright gate after push-task run 2026-07-21_03-21-08 showed the
    # un-gated carrot subsidizing a still ~6° torso lean (alive + stillness
    # outweighed the bounded pitch penalty), so the policy froze in a tilted
    # statue instead of correcting to 0°. Jul 21 2026: a CoM-on-support
    # balance gate was TRIED here (round 3) and REVERTED — it was ineffective
    # (a frozen balanced statue has the CoM over the feet, so the gate is 1.0
    # — it doesn't kill the freeze it was meant to) and harmful (it zeroed
    # the carrot during early learning when the robot is always falling,
    # making the reward too sparse — run 2026-07-21_22-27-00 regressed to
    # eplen 71 with std exploding to 4.19). The freeze exploit is instead
    # addressed by raising ``alive`` (+1.0 -> +2.0) so the net per-step
    # reward while alive is positive: with a positive net, surviving long
    # (which only active balance can do) always beats freezing (which
    # topples at the first shove).
    standing_stillness = RewTerm(
        func=stationary_pose_exp,
        weight=1.0,
        params={"upright_gate": True, "tilt_std": 0.12},
    )

    # Flat-top balance band (g_x ∈ [-0.02, +0.05]) with asymmetric Gaussian
    # shoulders and quadratic tails — rewards holding the torso near upright
    # with a slight forward bias, keeps a restoring gradient at large tilt.
    # torso_posture = RewTerm(
    #     func=torso_pitch_asymmetric_reward,
    #     weight=0.5,
    #     params={
    #         "band_gx_min": BALANCE_BAND_GX_MIN,
    #         "band_gx_max": BALANCE_BAND_GX_MAX,
    #         "edge_std_below": BALANCE_EDGE_STD_BELOW,
    #         "edge_std_above": BALANCE_EDGE_STD_ABOVE,
    #         "roll_std": BALANCE_ROLL_STD,
    #         "forward_penalty_gain": BALANCE_FWD_PENALTY_GAIN,
    #         "backward_penalty_gain": BALANCE_BWD_PENALTY_GAIN,
    #     },
    # )

    # Always-on inverted Gaussian on torso TILT (pitch + roll) centered at
    # 0 deg, saturating at ±6 deg tilt magnitude — prices tilt in every
    # state without blocking recovery motion. Re-enabled Jul 20 2026: the
    # knee+ankle CoG strategy pins the torso at 0 deg, and the function was
    # extended to cover roll too (the push task shoves laterally, so a roll
    # lean can be settled into just like a pitch lean — the same
    # settle-off-target exploit this codebase has hit before). Saturation
    # tightened 10 -> 6 deg (Jul 20 2026) after run 2026-07-21_03-21-08
    # settled into a persistent ~6 deg lean: at 10 deg saturation a 6 deg
    # lean only cost ~-1.1/step (below the knee), which the stillness carrot
    # happily paid; at 6 deg saturation that same lean costs the full
    # -1.9/step, making "hold a small lean" clearly worse than "correct to
    # 0 deg". The far-field gradient (beyond 6 deg) now comes from
    # com_over_support + alive, which is by design.
    torso_pitch_penalty = RewTerm(
        func=torso_pitch_zero_penalty,
        weight=-2.0,
        params={"saturation_pitch_deg": 6.0},
    )

    # Superseded by torso_pitch_penalty above — unbounded quadratic on
    # distance from the balance band, stillness-gated.
    # torso_settle = RewTerm(
    #     func=torso_settle_in_band_l2,
    #     weight=-4.0,
    #     params={
    #         "band_gx_min": BALANCE_BAND_GX_MIN,
    #         "band_gx_max": BALANCE_BAND_GX_MAX,
    #         "stillness_std": 1.5,
    #     },
    # )

    # L1 anchor on hip flexion only, balance-gated — keeps hips near straight
    # at steady state, frees them to swing during recovery.
    # hip_flexion_anchor = RewTerm(
    #     func=joint_deviation_l1_balance_gated,
    #     weight=-0.2,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 "hip_flexion_left_joint",
    #                 "hip_flexion_right_joint",
    #             ],
    #             preserve_order=True,
    #         ),
    #         "gate_band_gx_center": BALANCE_BAND_GX_CENTER,
    #         "gate_std": BALANCE_GATE_STD,
    #         "gate_floor": BALANCE_GATE_FLOOR,
    #     },
    # )

    # L2 on joint velocity, CoM-gated — kills residual wobble at balance,
    # frees the legs to move when the CoM heads off the support. Jul 20 2026:
    # switched from the torso-tilt gate to the CoM-distance gate
    # (com_gate=True) — the knee+ankle strategy keeps the torso level during
    # recovery, so a torso-tilt gate stayed closed (full -1.0) during exactly
    # the recovery motion it should relax for; keying on CoM excursion (the
    # actual balance criterion) opens the gate during a level-torso recovery.
    joint_vel = RewTerm(
        func=joint_vel_l2,
        weight=-1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=JOINT_NAMES_ALL, preserve_order=True
            ),
            "balance_gate": True,
            "com_gate": True,
            "foot_body_names": ("foot_left_1", "foot_right_1"),
            "gate_dist_std": BALANCE_GATE_DIST_STD,
            "gate_floor": BALANCE_GATE_FLOOR,
        },
    )

    # L2 on base angular velocity (IMU gyro ω_x² + ω_y²), NOT gated — D-term
    # on the torso that damps the ~3-4 Hz pitch-rate limit cycle. Re-enabled
    # Jul 20 2026: with the torso pinned at 0 deg the residual oscillation
    # shows up as whole-body sway on the ankles (torso pitch/roll RATE), the
    # mode where joint velocities stay small and joint_vel is blind — this
    # term is aligned with a locked torso, not in conflict with it.
    base_ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.30)

    # L2 on tick-to-tick change of the 16 kp/kd channels, NOT gated —
    # gain_rate — RE-ADDED Jul 22 2026 (round 5), one run after being
    # deleted, in a NEW role: anti-noise governor on the shared log-std.
    # Round-4 deleted it because its -2.0 quadratic tax on kp/kd rate
    # conflicted with the kp/kd-modulation survival strategy (see the
    # Round-4 paragraph in the class docstring). The gain_ema_tau_s
    # filter then made fast gain modulation physically useless (~3%
    # ripple), moving the survival strategy onto the position channels —
    # so the tax no longer punishes survival (a slow 0.45 s stiffness
    # ramp costs ~-0.06/step, negligible). But the round-4 run
    # (2026-07-22_20-49-06) immediately showed what gain_rate was REALLY
    # load-bearing for all along: with no penalty on the 16 gain
    # channels' raw rate and the EMA decoupling gain noise from physics,
    # the entropy bonus inflated the SHARED log-std unopposed — std
    # exploded exponentially (75k @ iter 1000 -> 4.7e15 @ 3772), leaking
    # noise into the position channels and drowning the advantage signal
    # (eplen peaked 1779 with reward +41, then degraded). History agrees:
    # round-3 CoM-gating (weak 0.2-floor residual) -> std 5.97; full
    # deletion -> std 4.7e15. With the EMA in place this term is purely
    # an anti-noise brake: it keeps the dead-ish gain channels quiet,
    # which bounds the shared std for ALL channels.
    gain_rate = RewTerm(func=action_gain_rate_l2, weight=-2.0)

    # L2 on tick-to-tick change of the 8 position channels, CoM-gated —
    # kills setpoint chatter at balance (-10.0 full strength), frees the
    # setpoints to move rapidly when the CoM heads off the support. Re-enabled
    # Jul 20 2026: this is the #1 priority anti-chatter term (isolates the
    # position setpoints, which gain_rate does not cover). Quadratic so slow
    # ankle/knee micro-corrections are nearly free (0.005 rad/tick ≈
    # -0.004/tick at gate 1) while slew-limit flipping is taxed; the CoM gate
    # opens to its 0.2 floor during a ±0.7 m/s shove so recovery motion stays
    # 5x cheaper than chatter at steady state. Uses the CoM-distance gate
    # (com_gate=True), NOT the torso-tilt gate — see joint_vel above.
    position_rate = RewTerm(
        func=action_position_rate_l2,
        weight=-10.00,
        params={
            "balance_gate": True,
            "com_gate": True,
            "foot_body_names": ("foot_left_1", "foot_right_1"),
            "gate_dist_std": BALANCE_GATE_DIST_STD,
            "gate_floor": BALANCE_GATE_FLOOR,
        },
    )

    # Bounded [0,1] reward for keeping the torso CoM (base_link xy) over the
    # midpoint between the two feet, AND holding still while doing it — the
    # positive carrot that gives a gradient toward balance. Stillness-gated
    # so the policy can't earn it by swinging the CoM through the target.
    # Non-privileged (FK from joint encoders). Re-enabled Jul 20 2026: this
    # is the "control the CoG with the legs" term — torso_pitch_penalty pins
    # the torso ANGLE but does not price where the mass actually is, so
    # without this term the policy could hold g_x=0 with the CoM parked
    # behind the 23 mm heel margin (a falling pose that earns full pitch
    # credit). com_over_support is what makes the knees/ankles steer the CoM
    # over the feet. The stillness gate still permits small stepping motions
    # (it earns partial credit at moderate motion) — "move around a little
    # to keep balance" is acceptable per the strategy.
    com_over_support = RewTerm(
        func=com_over_support_reward,
        weight=0.4,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "foot_body_names": ("foot_left_1", "foot_right_1"),
            "max_lateral_dist": 0.12,
            "stillness_std": 2.0,
        },
    )

    # crouch_height / com_recovery — DELETED Jul 22 2026 (round 4), one run
    # after being added. Run 2026-07-22_12-29-28 showed both terms INERT:
    # crouch_height logged +0.0000 (the upright gate zeroes it during the
    # tippy short episodes that dominate training, and the 0.06 m Gaussian
    # is too narrow to pay at the 0.65 m standing height), com_recovery
    # logged +0.0006 (potential-based shaping's episode mean is ~0 by
    # construction, so its weight would have to be enormous to matter
    # against the -2.9/step gain_rate tax it was meant to offset). Neither
    # prevented the run's collapse (same arc as the run without them).
    # The structural fix that actually addresses the collapse is the
    # gain_ema_tau_s action filter + gain_rate deletion above; the push
    # curriculum itself is the recovery teacher. Both reward functions
    # remain in bebop_v2_rewards.py (unwired) for future single-variable
    # experiments.

    # L1 anchor over all 8 joints toward the default (zero) pose — general
    # posture regularizer; prevents crouching at the action rail and gives
    # the policy a home pose to return to after recovery. NOT gated (the L1
    # pull is cheap during transients).
    # joint_pos_anchor = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.3,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot", joint_names=JOINT_NAMES_ALL, preserve_order=True
    #         )
    #     },
    # )

    # Σ(q_L + q_R)² over all 4 L/R joint pairs, stillness-gated — the HEAVY
    # symmetry enforcer. Uses the SUM (not difference) because every L/R pair
    # on this robot is sign-mirrored in the URDF: a symmetric stance reads
    # q_L = -q_R, so the residual is (q_L + q_R)². Stillness-gated (not
    # balance-gated) so the posture pull stays active through any held pose,
    # including tilted leans; relaxes only during active recovery motion.
    # See ``bilateral_joint_symmetry_l2`` for the per-pair axis audit.
    # bilateral_symmetry = RewTerm(
    #     func=bilateral_joint_symmetry_l2,
    #     weight=BILATERAL_SYMMETRY_WEIGHT,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "pairs": BILATERAL_SYMMETRY_PAIRS,
    #         "balance_gate": False,
    #         "stillness_std": 1.5,
    #     },
    # )

    # Σ_feet sin²(sole tilt from horizontal), stillness-gated — forces both
    # soles parallel to the ground so the policy can't ride the toe/heel edge
    # of the rigid sim foot (a non-transferable contact cheat that leans on
    # the weak RS02 ankle). Targets the sole *orientation* (FK from encoders
    # + IMU), not the foot joint angle, so the ankle can pick whatever angle
    # makes the sole flat in the hip-flexed stance.
    feet_flat = RewTerm(
        func=feet_flat_orientation_l2,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "foot_body_names": ("foot_left_1", "foot_right_1"),
            "balance_gate": False,
            "stillness_std": 1.5,
        },
    )


@configclass
class TerminationsCfg:
    time_out = TermTerm(func=mdp.time_out, time_out=True)
    # IMU tilt terminations DISABLED — ending at a tilt cliff let the policy
    # farm `alive` up to the boundary without learning to recover. Re-enable
    # (tighten toward 15-20°) once the robot balances off the ground-contact
    # backstop alone.
    # imu_pitch_out_of_bounds = TermTerm(
    #     func=imu_pitch_out_of_bounds,
    #     params={
    #         "pitch_forward_gx_max": PITCH_FALL_LIMIT_GX,
    #         "pitch_back_gx_min": -PITCH_FALL_LIMIT_GX,
    #     },
    # )
    # imu_roll_out_of_bounds = TermTerm(
    #     func=imu_roll_out_of_bounds,
    #     params={"roll_gy_limit": ROLL_FALL_LIMIT_GY},
    # )
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
        # Required the moment privileged foot-contact sensing is re-added.
        self.sim.disable_contact_processing = False

        self.scene.robot = BEBOP_V2_STANDING_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.terrain = terrain_gen.TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
            # Friction reverted to 1.0/1.0 (2026-06-20): the 0.4/0.3 "carpet-
            # matched" friction was tested and made the hardware asymmetry
            # WORSE (hip L-R went from +0.30 to +0.65), proving friction wasn't
            # the root cause. The asymmetry is the policy compensating for real
            # hardware L/R differences (motor friction, mass, alignment), not a
            # contact effect. The bilateral_symmetry + foot_deviation reward
            # terms now enforce the symmetric stance directly. Keep 1.0/1.0
            # friction so the policy can learn to balance without slipping
            # complicating the dynamics — the symmetric-stance enforcement
            # should be done by the reward, not by the contact model.
            #
            # Jul 18 2026: the FRICTION-PAIR floor now lives on the feet, not
            # the terrain — the Jul 2026 polyurethane soles are far grippier
            # than mu=1.0, and at 1.0 the policy could shuffle/skid to recover
            # (impossible on hardware). ``randomize_foot_friction`` (EventCfg)
            # raises the foot material to mu ~1.7-2.1 per env; with combine
            # mode "average" the sole-ground pair resolves to ~(1.7..2.1 +
            # 1.0)/2 = 1.35-1.55, no-slide territory. Lift-and-place footwork
            # becomes the only recovery strategy. The terrain stays at 1.0 so
            # the base_link ground-contact backstop is unchanged.
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


@configclass
class BebopV2StandingFixedGainCfg(BebopV2StandingCfg):
    """Quiet-stand variant with gains STRUCTURALLY pinned on both sides.

    Third design (Jul 11 2026), after two instructive failures:

    1. freeze-in-sim only (run 2026-07-10_04-12-52): physics froze the
       gains at midpoint but firmware kept decoding the 16 inert channels
       over the full clamp range → deployed kp wandered ±13-19 around
       values sim never varied.
    2. live gains + constancy rewards (run 2026-07-11_02-20-09): gain_rate
       -0.80 + gain_anchor -0.50 got the *means* near midpoint but the
       deterministic network still modulates gains with its observations
       (capture 20260711_163155: kp std 1-12 on quiet ticks vs 5-24 on
       active ticks — the variation tracks body motion). A net can only
       emit a constant by ignoring its inputs, and PPO keeps the
       modulation whenever it helps recovery more than the penalty costs.
       Reward pressure CANNOT guarantee constant outputs.

    The lock is therefore structural on BOTH decode paths:

    * sim: ``freeze_gains=True`` — physics always uses the midpoint gains,
      regardless of what the network emits;
    * firmware: ``policy_gain_clamps`` pinned to ±epsilon bands around the
      SAME midpoints (bebop_v2.yaml, Jul 11 2026 — the config loader
      rejects min==max, so e.g. kp 69.5..70.5 for hip flexion). Any raw
      value decodes to within ±0.5 Nm/rad of the trained gain.

    With both sides pinned, the 16 gain channels are irrelevant to
    dynamics everywhere, so no gain SHAPING is needed (no ``gain_anchor``;
    ``gain_rate`` is inherited from the base ``RewardsCfg`` but serves
    only its round-5 anti-noise std-governor role here — harmless, since
    these gain channels are fully decoupled from physics anyway) — the
    position-only stand trains on the simplified
    reward. The channels still exist in the action/obs
    layout (ONNX I/O and the 49-dim obs are unchanged) and both sim and
    firmware echo the raw action into ``last_action``, so the obs
    feedback stays consistent too.
    """

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.freeze_gains = True
        self.actions.joint_pos.kp_fixed = _Kp_MID
        self.actions.joint_pos.kd_fixed = _KD_MID


@configclass
class BebopV2StandingPushCfg(BebopV2StandingCfg):
    """Variable-impedance stand WITH mid-episode pushes (reactive recovery).

    Step-2 follow-up to the clean ``BebopV2StandingCfg`` baseline — validate
    that converges and transfers first, then train this. Two coupled changes:

    1. ``push_robot`` — random-interval ±0.7 m/s fore/aft + lateral root-velocity
       shoves. Non-privileged replacement for the removed ``feet_load_symmetry``
       penalty: a one-foot lean is fragile to a sideways shove, so pushes
       pressure the policy toward a centered, two-foot stance. (Envelope
       widened ±0.4 -> ±0.7 Jul 18 2026 — see PUSH_VELOCITY_RANGE.)
    2. ``push_level`` curriculum — ramps the push envelope from 25% to 100%
       over ~120k control steps (boot ~= +/-0.175 m/s, crossing +/-0.4 at
       ~iter 1600, full +/-0.7 at ~iter 3750 of a 5000-iter run).

    Pair with ``BebopPPOPushCfg`` (higher entropy_coef): pushes enlarge the
    reward landscape, so the actor needs more exploration.
    """

    def __post_init__(self):
        super().__post_init__()

        # interval_range_s picks a fresh wait between pushes per env so the
        # policy sees disturbances at varied phases of its stand. Tightened
        # (4,8) -> (3,6) s (Jul 18 2026): with a 20 s episode that is ~4.4
        # pushes/episode (was ~3.3) — ~33% more recovery reps per rollout.
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(3.0, 6.0),
            params={"velocity_range": dict(PUSH_VELOCITY_RANGE)},
        )

        self.curriculum.push_level = CurrTerm(
            func=push_magnitude_curriculum,
            params={
                "term_name": "push_robot",
                "full_velocity_range": dict(PUSH_VELOCITY_RANGE),
                # Ramp reshaped (Jul 19 2026) after run 2026-07-19_03-46-23
                # collapsed: with start 0.4 / 75k steps over the +/-0.7
                # envelope, pushes at iter 1000 were already +/-0.46 m/s —
                # harder than the +/-0.4 FULL strength the successful run
                # 2026-07-18_17-51-11 only reached at iter 2300 — and the
                # still-learning policy fell 5x more (eplen 908 vs 1677 at
                # iter 1000, termination -0.077 vs -0.015/tick). Start 0.25
                # (boot +/-0.175 ~= the successful run's +/-0.16 boot) and
                # stretch to 120k steps: the ramp crosses +/-0.4 at ~iter
                # 1600, when the policy should already stand solidly, and
                # reaches full +/-0.7 at ~iter 3750 with ~1250 iters left.
                "start_fraction": 0.25,
                "num_curriculum_steps": 120_000,
            },
        )


# Actuator group name -> Robstride motor model (for the response-net swap).
_ACTUATOR_GROUP_TO_MODEL = {
    "hip_flexion": "RS04",
    "hip_abduction": "RS03",
    "knee_flexion": "RS04",
    "foot": "RS02",
}
_MOTOR_STALL_BY_MODEL = {
    "RS04": MOTOR_STALL_TORQUE_RS04,
    "RS03": MOTOR_STALL_TORQUE_RS03,
    "RS02": MOTOR_STALL_TORQUE_RS02,
}
_MOTOR_NOLOAD_BY_MODEL = {
    "RS04": MOTOR_NOLOAD_VEL_RS04,
    "RS03": MOTOR_NOLOAD_VEL_RS03,
    "RS02": MOTOR_NOLOAD_VEL_RS02,
}
# TorchScript nets produced by tools/actuator_net_fit.py (container path,
# same convention as the USD above).
ACTUATOR_NET_DIR = "/workspace/bebop_bot/sim/bebop_training/assets/actuator_nets"


@configclass
class BebopV2StandingPushActNetCfg(BebopV2StandingPushCfg):
    """Push-stand with the hybrid learned torque-response actuator (td-b05f58).

    Single-variable swap vs ``BebopV2StandingPushCfg`` (Jul 24 2026,
    paper-derived — arXiv:2607.18135's actuator-net, adapted for variable
    impedance): each ``DCMotorCfg`` group is replaced by a
    ``RobstrideResponseActuatorCfg`` with IDENTICAL physical parameters
    (datasheet stall / no-load, effort_limit_sim rail, sysid friction +
    armature, midpoint seed gains). Only the command->realized-torque
    mapping changes: analytic-clip-only -> analytic PD + learned response
    net + the same clip as a hard rail.

    Motivation: the sysid logs show realized torque falling far short of
    command at high demand (RS04 cmd 36 Nm -> fb ~14 Nm; RS02 cmd 5.1 ->
    fb ~1.8 Nm) while the DCMotor saturation_effort uses datasheet stall
    (120/60/17) — the sim overestimates torque authority exactly where
    push recovery needs it, a suspect in the +/-0.4-0.5 m/s push wall.

    Nets must exist at ``ACTUATOR_NET_DIR/<MODEL>.pt`` (run
    tools/actuator_net_fit.py first). The nets see desired-torque +
    velocity history only, so the policy I/O contract (49-dim obs /
    24-dim action), the firmware decode, and the ONNX export path are
    all unchanged.
    """

    def __post_init__(self):
        super().__post_init__()
        new_actuators = {}
        for name, old in self.scene.robot.actuators.items():
            model = _ACTUATOR_GROUP_TO_MODEL[name]
            new_actuators[name] = RobstrideResponseActuatorCfg(
                joint_names_expr=old.joint_names_expr,
                saturation_effort=old.saturation_effort,
                effort_limit_sim=old.effort_limit_sim,
                velocity_limit_sim=old.velocity_limit_sim,
                velocity_limit=old.velocity_limit,
                stiffness=old.stiffness,
                damping=old.damping,
                armature=old.armature,
                friction=old.friction,
                network_file=f"{ACTUATOR_NET_DIR}/{model}.pt",
                cmd_scale=_MOTOR_STALL_BY_MODEL[model],
                vel_scale=_MOTOR_NOLOAD_BY_MODEL[model],
            )
        self.scene.robot.actuators = new_actuators
