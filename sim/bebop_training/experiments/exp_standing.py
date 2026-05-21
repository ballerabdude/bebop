"""Minimal "just stand" experiment for the Bebop V2 articulation.

This file is the deliberate anti-thesis of ``exp_flat_balance_v2.py``: it
takes the kitchen-sink training config (variable-impedance MIT-mode
action, wide initial-state randomization, mid-episode pushes, friction
randomization, observation noise, eight reward shaping terms) and
strips every one of them away. The point is to establish a baseline
where every knob is at its boring default, train a policy on it, and
then **add one feature at a time** so each addition's effect on the
final policy is attributable.

What's intentionally kept (because the real robot can't run without them):
  * The IMU sensor (BNO085) is the only thing telling the policy which
    way is up. Without it there is no balance task to learn.
  * Joint encoders feed ``joint_pos`` / ``joint_vel``. Same reason.
  * Fixed PD on each actuator — the policy emits position targets only,
    the underlying Robstride PD loop closes the inner control. Gains
    here mirror the firmware's ``hold_gains`` (the tested "armed but no
    policy active" values from ``bebop_v2.yaml``) so sim and real run
    the same closed-loop dynamics.

What's intentionally OFF in this baseline:
  * Mass / friction / CoM randomization. With the CAD mass now corrected
    to the bench-measured 7 kg torso, "no randomization" means sim
    trains on the *real* mass.
  * Mid-episode pushes (``push_robot`` event).
  * Wide initial-state randomization. Init pose is tight (±3°) so the
    policy never has to learn fall-recovery here.
  * Action delay (``action_delay_steps=0`` on the action term),
    torque penalties, most reward shaping.
  * IMU sample-rate staleness. The real BNO085 emits the AR/VR fused
    rotation vector at 20 Hz, but the sim ``ImuCfg`` runs at the
    physics tick (200 Hz). The observation noise added in v0.3 covers
    Gaussian sensor noise but NOT the staleness — that's a separate
    knob (``update_period=0.05`` on the ImuCfg) we will add when the
    rest of sim-to-real is closer to dialled in.

Change log (this file is being mutated in place to add features one
at a time rather than forking into v1, v2, ... files):
  * v0 — initial four-reward minimum (alive, flat_orientation,
    base_height, joint_pos_limits). Result: policy stands but feet
    micro-oscillate, causing torso to drift horizontally because no
    term penalizes either action-rate jitter OR lateral velocity.
  * v0.1 — added ``action_rate_l2`` (weight -0.05). Directly punishes
    the policy emitting commands that change rapidly between ticks,
    which suppresses the foot oscillation at its source rather than
    via its drift-velocity symptom. Also implicitly caps how large
    the action-distribution std can grow, because wider std produces
    more action-rate variance per step. Result: clean, deterministic
    standing pose — but the policy converged to a heel-balance trick
    (toes lifted, support entirely on the heel) because the foot's
    geometric centre sits forward of the torso CoM. Rotating the
    foot toes-up was the cheapest single-joint way to put the
    support point under the CoM without leaning the torso forward.
  * v0.2 — added ``foot_flat`` reward (weight 1.0, std=0.15). Closes
    the heel-balance loophole by giving up to +1.0/tick when both
    feet's local +z axes align with world up, falling off as
    exp(-(g_x^2 + g_y^2) / std^2). At ~8.6° foot tilt the reward
    drops to 0.37/tick — small enough to crowd out the heel trick
    but tolerant enough to allow the small foot tilt a real ankle
    strategy needs. The policy now has to find option 1 (lean torso
    forward via hip / knee / ankle coordination) to keep both feet
    flat AND the CoM over the support polygon. Result: clean
    feet-flat / slight-forward-lean / bent-knee standing pose,
    mean_reward ≈ 40, foot_flat ≈ 0.985, flat_orientation ≈ -0.0005.
  * v0.3 — added Gaussian observation noise on the four
    sensor-derived observation terms (IMU gyro, projected gravity,
    joint position, joint velocity). The previous-action obs gets
    no noise — that's the policy's own output, not a sensor. Noise
    magnitudes mirror the original kitchen-sink config's tuned
    values for the BNO085 + Robstride encoder hardware:
       * imu_ang_vel:        ±0.3 rad/s   (gyro bias drift envelope)
       * projected_gravity:  ±0.05        (~3° rotation-vector error)
       * joint_pos_rel:      ±0.02 rad    (encoder + backlash)
       * joint_vel_rel:      ±0.5 rad/s   (finite-diff velocity jitter)
    Result: clean stand under noise, but a small left/right rocking
    appears — the policy reacts to spurious projected-gravity-y
    noise through the soft hip-abduction joints (kp=40), and the
    fixed PD can't raise stiffness during steady stance. The fix
    is variable impedance (next).
  * v0.4 — restored variable impedance on the action channel.
    ``JointPositionActionCfg`` (8-dim raw → 8 joint positions)
    becomes ``VariableImpedanceJointActionCfg`` (24-dim raw →
    8 positions + 8 kp + 8 kd per tick). Per-joint kp/kd clamps
    mirror ``POLICY_KP_MIN/MAX`` and ``POLICY_KD_MIN/MAX`` in the
    original kitchen-sink config and the firmware YAML's
    ``policy_gain_clamps``. The slew clamp on the position channel
    is set permissively (``max_pos_step_per_tick=1.0``, equivalent
    to 100 rad/s @ 100 Hz, effectively unbounded) so the soft
    ``action_rate_l2`` penalty stays the only thing shaping the
    per-tick position deltas — that knob will tighten in v0.5.
    Action delay is also off (``action_delay_steps=0``); that's a
    v0.6+ knob.
    Two consequences worth flagging:
       1. Observation dim 30 -> 46 (last_action grows 8 -> 24).
       2. Action dim 8 -> 24. The previous v0.3 checkpoint can't
          warm-start this — the actor's input + output layers have
          different shape. Train from scratch.
    Result: clean stand under noise, lateral rocking eliminated.
    foot_flat ≈ 0.965, alive ≈ 0.999, mean_reward ≈ 36. Confirmed
    that variable impedance lets the policy raise hip-abduction
    stiffness during quiet standing, which was the v0.3 problem.
  * v0.5 — tightened the position-channel slew clamp from 1.0
    rad/tick (effectively unbounded) to 0.020 rad/tick = 2 rad/s
    at the 100 Hz policy tick. This matches the firmware safety
    envelope target. With the clamp this tight the action term
    will sometimes bind — i.e., the policy commands a per-tick
    position delta larger than 0.020 and the clamp truncates it.
    The policy must learn to compensate (either by emitting
    smaller per-tick deltas via a forward-looking action, or by
    relying more on kp/kd modulation when fast position changes
    aren't available). The kp/kd channels are NOT slew-clamped —
    gain modulation can still be instantaneous between ticks.
    Important sim-to-real coupling: the firmware currently
    enforces 0.015 rad/tick (1.5 rad/s). For this policy to
    deploy cleanly, ``defaults.slew.max_pos_step_per_tick`` in
    ``firmware/bebop-linux/config/bebop_v2.yaml`` MUST be raised
    to 0.020 in lockstep with this training run; otherwise the
    firmware will clip more aggressively than the policy expects
    and tracking will lag. Don't deploy this checkpoint to a
    firmware still at 0.015.
  * v0.6 — restored ``base_lin_vel`` (first) and ``velocity_commands``
    (last) in the policy observation vector to match the firmware's
    52-dim contract. v0.0 stripped both terms on the reasoning that
    they were "useless for standing" — base_lin_vel is firmware-zero
    on the real robot (no velocity estimator wired in) and the
    velocity command is hard-coded to (0,0,0) for the stand task —
    but the firmware ``observation.rs`` builder still emits all 52
    dims regardless of task. Deploying the v0.5 policy (obs_dim=46)
    against the firmware (obs_dim=52) latches E-STOP with
    "Inference failed" because the ONNX input layer can't take the
    larger tensor. Adding the two terms back with the same noise
    levels as the kitchen-sink config keeps sim and firmware
    aligned without firmware changes. Observation dim 46 -> 52;
    actor input layer reshapes; previous v0.5 checkpoint cannot
    warm-start, retrain from scratch.
  * v0.6.1 — identified and resolved a sim/firmware mismatch on the
    position scale. v0 through v0.6 trained with sim ``pos_scale=0.5``
    while the firmware decoder used ``SCALE_ACTION=0.8``. Every
    deployed command therefore moved the real joint 1.6x farther
    than the policy expected; combined with an OOD initial pose at
    deployment (hip_abduction at -0.61 rad vs. the +/-0.40 sim
    init envelope), the policy panicked, commanded raw values close
    to +/-1.0, foot joints hit the +/-0.8 rad hard safety limit, and
    the supervisor latched E-STOP.
    Rather than retrain at the firmware's 0.8 value, we aligned the
    firmware to sim's 0.5 (the more task-appropriate value for
    standing -- the policy never needs ±45° of authority per joint,
    ±28° is plenty). Changes elsewhere in the tree:
       * firmware/bebop-linux/src/config.rs::SCALE_ACTION : 0.8 -> 0.5
       * firmware/bebop-linux/config/bebop_v2.yaml ::
         defaults.slew.max_pos_step_per_tick : 0.015 -> 0.020
         (this also brings the firmware slew clamp into alignment
         with the v0.5 sim training, which was the deferred lockstep
         from v0.5)
    With both sides at 0.5 the v0.6 trained ONNX should now decode
    correctly on the real robot. Still need to address the OOD
    initial-pose issue separately (either pose the robot to near-zero
    before engaging RunPolicy, or widen sim init randomization to
    cover the real spawn distribution).
  * v0.7 — recovery / robustness pass. The deployed v0.6 policy
    showed the same failure mode every test: it sits "calm" in a
    slight forward lean (its trained equilibrium), but the slightest
    backward tilt sends raw outputs to the ±1.0 envelope and the
    robot falls. Four reasons, all addressed here together:
       1. Init pose pitch was ±2°. The policy never saw more than
          that during training, so anything past it is OOD. Widened
          to ±0.10 rad (~5.7°) on both pitch AND roll; added a
          modest initial root angular velocity envelope so the policy
          also sees "already-falling" start states. The geometric
          asymmetry (foot center is forward of torso CoM, so
          backward lean is the unstable direction — see v0.1 note)
          means backward starts were the gap; symmetric pitch
          range fills it.
       2. No mid-episode disturbances. The policy converged to "do
          nothing, geometry holds me up", which is brittle. Added a
          ``push_robot`` interval event (4–8 s, ±0.4 m/s linear x,
          ±0.3 m/s linear y, ±0.3 rad/s pitch / roll) so each 20 s
          episode sees 3–5 pushes and the policy must learn an active
          recovery, not just stay-in-place.
       3. ``foot_flat`` std was 0.15 — strict enough that an 8.6°
          foot tilt drops the reward to 0.37. That suppresses the
          natural ankle-strategy recovery from a backward lean
          (lift the heel, pivot on the toes), which IS a foot tilt.
          Widened to 0.25 so a foot tilt up to ~14° still scores
          ~0.37 — loose enough to allow real ankle work during
          recovery, tight enough to still cost more than the
          v0.1 heel-balance trick (which was a much larger toe-up
          rotation).
       4. Reset joint distribution was narrow (±0.05 rad on every
          joint) AND centred on default = 0, so the policy only
          ever saw locked-out legs at reset. The bench-deploy
          spawn pose is closer to a slight crouch (operator stands
          the robot up with hands on the torso, knees a bit bent),
          so the policy needs to recover from there too. Replaced
          the single ``reset_joints`` term with three:
            * ``reset_joints`` — ±0.05 jitter on every joint (as before)
            * ``reset_shin_left_crouch`` — additionally biases the
              left shin into ``(-0.05, +0.45)``, covering both
              straight-knee AND moderately-bent-forward states. The
              joint's mirror convention (see ``shin_symmetry_penalty``
              docstring in ``bebop_v2_rewards.py``) is that +shin_left
              and -shin_right are the same physical motion — knees
              bent forward.
            * ``reset_shin_right_crouch`` — mirror, ``(-0.45, +0.05)``.
          Femur and foot are deliberately NOT biased. The crouch
          requires hip / ankle compensation to keep the CoM over the
          foot and the foot flat; forcing the policy to find that
          compensation from a randomized knee bend is exactly the
          lesson we want it to internalize.
    The widened observation distribution from (1)+(2)+(4) means the
    actor will need more training steps to converge than v0.6, and
    the converged reward will be lower (the policy is now solving a
    harder task: stand AND recover, not just stand). Expect
    mean_reward to drop from v0.6's ~36 to roughly 25–30 at
    convergence — that's the cost of robustness.
    Observation and action layouts are UNCHANGED from v0.6 (52-dim
    obs, 24-dim action). The ONNX export drops into the same firmware
    loader. No firmware changes required for this version.
  * v0.8 — added ``shin_symmetry_penalty`` (weight -0.5). The v0.7
    checkpoint converged to the exact reward-hacking mode the comment
    on ``shin_symmetry_penalty`` in ``bebop_v2_rewards.py`` warned
    about: one knee deeply bent, the other near-straight, with the
    near-straight knee oscillating to micro-balance. All other reward
    terms are pose-symmetry-blind (``foot_flat`` is world-frame,
    ``base_height`` is torso-z, ``flat_orientation`` is torso tilt,
    none of them care which leg is doing the supporting), so PPO
    found the asymmetric attractor and parked there with
    ``action_rate_l2`` symptom ≈ -0.25 (very high jitter) and a
    slowly *rising* entropy curve at 10k iterations (a sign the
    policy hadn't committed because both asymmetric and symmetric
    crouches scored about the same).
    The v0.7 crouch reset bias (independent shin_left / shin_right
    randomization) compounded this — half the time the episode
    started already-asymmetric, so the policy got steady gradient
    signal that asymmetric crouches were valid. With the new
    symmetry penalty (which scores ``(shin_left + shin_right)^2``,
    accounting for the mirrored joint frames so a physically
    symmetric crouch sums to zero), the asymmetric attractor now
    costs reward proportional to how unbalanced the bend is, and
    the policy should commit to a symmetric stance.
    Weight -0.5 is the value used by the original kitchen-sink
    config (``bebop_v2_base_cfg.py``) and is the minimum that
    reliably breaks the asymmetric attractor without dominating the
    other terms. If the converged knees come out STIFFLY symmetric
    (both fully straight, no crouch — i.e. the penalty pushed too
    hard and the policy gave up on bending at all), drop the weight
    to -0.2. If the asymmetric pose persists past ~5k iterations,
    bump to -1.0.
    Expect secondary improvements:
       * ``action_rate_l2`` magnitude should decrease (no more
         one-knee oscillation to micro-balance).
       * Entropy curve should plateau or start descending — once the
         policy commits to a symmetric stance the asymmetric escape
         is no longer free.
       * ``mean_reward`` may drop transiently as the policy gives up
         the asymmetric solution before finding the symmetric one;
         expect a recovery within ~3k iterations.
    Observation and action layouts UNCHANGED. No firmware changes.

Deployment note: as of v0.4 the policy emits the same 24-dim MIT-mode
action vector the firmware ``PolicyRunner`` already expects (8 raw
positions + 8 raw kp + 8 raw kd, decoded with the same affine clamps
as in ``firmware/bebop-linux/config/bebop_v2.yaml``). So an ONNX
export from this experiment should drop into the existing firmware
loader without code changes — barring observation-layout drift
between sim and firmware. The observation builder on the firmware
side ``firmware/bebop-linux/src/observation.rs`` must emit exactly
the 52-dim vector this file consumes (3 base_lin_vel + 3 base_ang_vel
+ 3 projected_gravity + 8 joint_pos + 8 joint_vel + 24 last_action +
3 velocity_commands). If those terms or their order ever drift,
retrain.
"""

import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
import isaaclab.envs.mdp as mdp

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as TermTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ImuCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg

from ..envs.bebop_v2_actions import VariableImpedanceJointActionCfg
from ..envs.bebop_v2_rewards import foot_flat_reward, shin_symmetry_penalty
from ..envs.bebop_v2_terminations import base_link_on_ground


# ---------------------------------------------------------------------------
# Joint order MUST match firmware/bebop-linux/src/observation.rs::JOINT_NAMES
# so an exported ONNX policy from this experiment lines up with the on-robot
# observation builder. Same list as ``bebop_v2_base_cfg.JOINT_NAMES_ALL`` —
# duplicated here on purpose so the file is self-contained.
# ---------------------------------------------------------------------------
JOINT_NAMES_ALL = [
    "hip_abduction_left_joint",
    "hip_abduction_right_joint",
    "femur_left_joint",
    "femur_right_joint",
    "shin_left_joint",
    "shin_right_joint",
    "foot_left_joint",
    "foot_right_joint",
]


# ---------------------------------------------------------------------------
# Per-joint kp / kd clamps for the variable-impedance action term (v0.4).
#
# The policy emits a 24-dim raw action per tick: 8 raw positions, 8 raw kp,
# 8 raw kd. The kp / kd channels go through ``tanh``-style clamping into
# ``[-1, 1]`` then an affine map into the per-joint range below. The 8-tuple
# layout matches ``JOINT_NAMES_ALL`` exactly:
#
#     idx 0,1 = hip_abduction_{left,right}   (Robstride RS04)
#     idx 2,3 = femur_{left,right}           (RS03)
#     idx 4,5 = shin_{left,right}            (RS04)
#     idx 6,7 = foot_{left,right}            (RS02)
#
# These values MUST mirror ``POLICY_KP_MIN/MAX`` and ``POLICY_KD_MIN/MAX`` in
# ``bebop_v2_base_cfg.py`` and ``policy_gain_clamps`` in
# ``firmware/bebop-linux/config/bebop_v2.yaml``. If you change one side,
# change the other two — the policy bakes in this gain envelope and the
# firmware loader will reject any clamp exceeding the motor model's
# encoder ceiling.
#
# The minima are non-zero on purpose: the policy can't fully unload a joint
# (kp -> 0 would mean "the leg goes limp"), only soften it.
# ---------------------------------------------------------------------------
POLICY_KP_MIN = [5.0, 5.0, 20.0, 20.0, 10.0, 10.0, 5.0, 5.0]
POLICY_KP_MAX = [100.0, 100.0, 300.0, 300.0, 250.0, 250.0, 250.0, 250.0]
POLICY_KD_MIN = [0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
POLICY_KD_MAX = [5.0, 5.0, 8.0, 8.0, 8.0, 8.0, 4.5, 4.5]


def _midpoint(lo: list[float], hi: list[float]) -> list[float]:
    return [0.5 * (a + b) for a, b in zip(lo, hi)]


# Midpoints are used as the seeded ImplicitActuator stiffness / damping
# values. The variable-impedance action term overwrites them every tick
# (via write_joint_stiffness_to_sim / write_joint_damping_to_sim), so
# these only matter for the first physics sub-step at episode reset
# before the policy emits its first action.
_KP_MID = _midpoint(POLICY_KP_MIN, POLICY_KP_MAX)
_KD_MID = _midpoint(POLICY_KD_MIN, POLICY_KD_MAX)


# ---------------------------------------------------------------------------
# Articulation: USD asset + per-joint Robstride PD seeded at the midpoint
# of each joint's policy kp / kd clamp range.
#
# Under variable impedance (v0.4) the action term overwrites these gains
# every tick — they only define the actuator state for the first physics
# sub-step after an episode reset, before the policy has emitted its
# first action. Using the midpoint here means a zero-raw-output policy
# would produce the same gains the action term seeds with, so reset
# transients are minimised.
#
# Effort / velocity ceilings mirror the firmware's ``hard_limits`` so
# sim and real share the same saturation behaviour.
# ---------------------------------------------------------------------------
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
        # The USD root carries a built-in +Z translation so feet touch the
        # ground when joint angles are zero. Must match that or the legs
        # spawn below the floor.
        pos=(0.0, 0.0, 0.6539092050794861),
        joint_pos={joint_name: 0.0 for joint_name in JOINT_NAMES_ALL},
        joint_vel={joint_name: 0.0 for joint_name in JOINT_NAMES_ALL},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # Hip abduction (RS04). Policy kp / kd clamps: [5..100] / [0.5..5].
        "hip_abduction": ImplicitActuatorCfg(
            joint_names_expr=[
                "hip_abduction_left_joint",
                "hip_abduction_right_joint",
            ],
            effort_limit_sim=84.0,
            velocity_limit_sim=12.0,
            stiffness=_KP_MID[0],
            damping=_KD_MID[0],
            armature=0.01,
            friction=0.0,
        ),
        # Femur / hip pitch (RS03). Policy clamps: [20..300] / [1..8].
        "femur": ImplicitActuatorCfg(
            joint_names_expr=["femur_left_joint", "femur_right_joint"],
            effort_limit_sim=42.0,
            velocity_limit_sim=12.0,
            stiffness=_KP_MID[2],
            damping=_KD_MID[2],
            armature=0.005,
            friction=0.0,
        ),
        # Shin / knee (RS04). Policy clamps: [10..250] / [1..8].
        "shin": ImplicitActuatorCfg(
            joint_names_expr=["shin_left_joint", "shin_right_joint"],
            effort_limit_sim=84.0,
            velocity_limit_sim=12.0,
            stiffness=_KP_MID[4],
            damping=_KD_MID[4],
            armature=0.01,
            friction=0.0,
        ),
        # Foot / ankle (RS02). Policy clamps: [5..250] / [0.2..4.5].
        "foot": ImplicitActuatorCfg(
            joint_names_expr=["foot_left_joint", "foot_right_joint"],
            effort_limit_sim=17.0,
            velocity_limit_sim=20.0,
            stiffness=_KP_MID[6],
            damping=_KD_MID[6],
            armature=0.003,
            friction=0.0,
        ),
    },
)


# ---------------------------------------------------------------------------
# Actions: MIT-mode variable impedance. 24-dim raw policy output.
#
# Layout (raw):
#   raw[ 0: 8] -> position commands, scaled to ``default + pos_scale * raw``
#   raw[ 8:16] -> kp commands, affine-mapped to per-joint ``[kp_min, kp_max]``
#   raw[16:24] -> kd commands, affine-mapped to per-joint ``[kd_min, kd_max]``
#
# Position-channel hard slew clamp (v0.5):
#   ``max_pos_step_per_tick=0.020`` rad/tick = 2 rad/s @ 100 Hz. This
#   is the safety envelope target. The clamp truncates the absolute
#   per-tick change of the decoded position target, NOT the policy's
#   raw output — so the policy is free to emit any raw value, but the
#   sim only ever advances each joint target by at most 0.020 rad per
#   tick. The kp / kd channels are NOT slew-clamped (variable
#   impedance demands instantaneous between-tick gain changes).
#
#   This MUST mirror ``defaults.slew.max_pos_step_per_tick`` in
#   ``firmware/bebop-linux/config/bebop_v2.yaml`` for any deployed
#   policy. If the firmware is still at 0.015 when you deploy this,
#   the firmware will clip more tightly than sim and the policy will
#   lag behind its own commanded targets. Raise the firmware YAML to
#   0.020 in lockstep, or retrain with the lower value.
#
# Action delay is still off (``action_delay_steps=0``). That's the
# v0.6 knob — adds 2-tick CAN + motor lag to mirror real-robot
# latency.
#
# The kp / kd clamps mirror the firmware YAML's ``policy_gain_clamps``
# block (see POLICY_*_MIN / POLICY_*_MAX constants at the top of this
# file). They are not optional — the firmware loader rejects any clamp
# that exceeds the motor's safe envelope.
# ---------------------------------------------------------------------------
@configclass
class ActionsCfg:
    joint_pos = VariableImpedanceJointActionCfg(
        asset_name="robot",
        joint_names=JOINT_NAMES_ALL,
        # MUST match ``scales::SCALE_ACTION`` in
        # ``firmware/bebop-linux/src/config.rs`` (currently 0.5). The
        # firmware decodes the policy's raw position output as
        # ``target = default + SCALE_ACTION * clamp(raw, -1, 1)``. If
        # this constant ever differs from the firmware value, deployed
        # commands silently mis-scale. 0.5 means raw=±1.0 maps to
        # ±0.5 rad per joint -- enough for standing and slow walking,
        # well inside the hard safety envelope (foot joints cap at
        # ±0.8 rad in firmware). Raise this (sim and firmware
        # together, then retrain) when transitioning to highly dynamic
        # motions that need more per-tick reach.
        pos_scale=0.5,
        use_default_offset=True,
        max_pos_step_per_tick=0.020,
        action_delay_steps=0,
        kp_min=POLICY_KP_MIN,
        kp_max=POLICY_KP_MAX,
        kd_min=POLICY_KD_MIN,
        kd_max=POLICY_KD_MAX,
    )


# ---------------------------------------------------------------------------
# Observations: 52-dim policy input matching the firmware's
# ``observation.rs`` builder contract.
#
# Layout (52-dim, post-v0.6):
#   [ 0: 3] base_lin_vel        — torso linear velocity in body frame
#                                 (firmware-zero on the real robot; sim
#                                 ground-truth with wide noise here as a
#                                 light proxy for "we can't trust this")
#   [ 3: 6] base_ang_vel        — IMU gyro (BNO085 → body-frame rad/s)
#   [ 6: 9] projected_gravity   — IMU rotation vector → body-frame gravity
#   [ 9:17] joint_pos_rel       — encoder positions, relative to default
#   [17:25] joint_vel_rel       — encoder velocities, relative to default
#   [25:49] actions             — previous tick's raw policy output (24-dim:
#                                 8 positions + 8 kp + 8 kd channels)
#   [49:52] velocity_commands   — base velocity command from the command
#                                 manager. Pinned to (0,0,0) for the
#                                 stand task — three constant-zero
#                                 inputs that the firmware still emits.
#
# Per-tick Gaussian noise (v0.3) is injected on the five sensor-derived
# observation terms so the policy learns to be robust to the real
# BNO085 + Robstride encoder noise envelope. ``actions`` and
# ``velocity_commands`` are unnoised — both come from controllable
# upstream sources (the policy itself, and the command manager) rather
# than physical sensors.
#
# The 52-dim layout MUST stay byte-identical to the firmware-side
# observation builder. If the firmware's
# ``firmware/bebop-linux/src/observation.rs`` ever changes the layout
# or order of these terms, this block has to move in lockstep — and
# the policy has to be retrained, because the input layer shape is
# baked into the ONNX.
# ---------------------------------------------------------------------------
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
        )
        base_ang_vel = ObsTerm(
            func=mdp.imu_ang_vel,
            params={"asset_cfg": SceneEntityCfg("imu")},
            noise=UniformNoiseCfg(n_min=-0.3, n_max=0.3),
        )
        projected_gravity = ObsTerm(
            func=mdp.imu_projected_gravity,
            params={"asset_cfg": SceneEntityCfg("imu")},
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=UniformNoiseCfg(n_min=-0.02, n_max=0.02),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5),
        )
        actions = ObsTerm(func=mdp.last_action)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )

    def __post_init__(self):
        self.policy = self.PolicyCfg()


# ---------------------------------------------------------------------------
# Events: reset randomization + mid-episode pushes.
#
# Mass / friction / CoM are still NOT randomized — the whole experiment
# is a clean baseline where sim runs on the bench-measured real values.
# But what IS randomized matters: the policy is exposed to a wider
# initial-state distribution than v0.6 AND to mid-episode disturbances
# (added in v0.7) so the converged behaviour is "stand AND recover from
# a perturbation", not just "stand from near-zero".
#
# Reset bias toward a crouch
# --------------------------
# ``reset_joints`` puts a tight ±0.05 rad jitter on every joint — the
# same small perturbation v0.6 used, just enough to break determinism
# and force the policy to read its observations. On top of that:
#
# ``reset_shin_{left,right}_crouch`` overrides the shin (knee) joints
# with a wider, asymmetric-by-mirror-convention bias so each reset
# samples something between "straight knees" and "moderately bent
# forward". The shin convention on this articulation is that
# ``+shin_left`` and ``-shin_right`` are the same physical knee-forward
# motion (see ``shin_symmetry_penalty`` docstring); the two terms
# below therefore use opposite-signed ranges of equal magnitude.
#
# These run AFTER ``reset_joints`` (IsaacLab event order = config-class
# declaration order) and overwrite the shin values, so the ±0.05 jitter
# on the shins is replaced by the wider crouch range. The other six
# joints retain their ±0.05 jitter.
#
# The femur and foot joints are NOT biased here. A knee bend without
# the matching hip/ankle compensation puts the CoM forward of the
# foot, which the policy has to correct itself — that compensation
# chain (femur, knee, ankle acting together) is the actual balance
# strategy we want the policy to learn, so we don't want to short-
# circuit it by pre-arranging the start pose.
# ---------------------------------------------------------------------------
@configclass
class EventCfg:
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES_ALL),
            "position_range": (-0.05, 0.05),  # ~3° on every joint
            "velocity_range": (-0.1, 0.1),
        },
    )
    reset_shin_left_crouch = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["shin_left_joint"]),
            # (-0.05, +0.45) covers straight knee (~0 rad) up to a
            # moderately-bent-forward stance (~26°). The wider end is
            # close to the v0.2 converged knee-bend angle so the policy
            # can keep its trained pose from this start, while the
            # narrow end keeps backward-compatibility with the v0.6
            # near-zero start.
            "position_range": (-0.05, 0.45),
            "velocity_range": (-0.1, 0.1),
        },
    )
    reset_shin_right_crouch = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["shin_right_joint"]),
            # Mirror of shin_left (see ``shin_symmetry_penalty`` for
            # the sign convention — the right joint's local frame is
            # rotated 180° about X, so +shin_left and -shin_right are
            # the same physical motion).
            "position_range": (-0.45, 0.05),
            "velocity_range": (-0.1, 0.1),
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x":     (0.0, 0.0),
                "y":     (0.0, 0.0),
                "z":     (0.0, 0.0),
                # ±0.10 rad (~5.7°) on both axes. v0.6 was ±0.035
                # (~2°), which the deployed policy showed it couldn't
                # generalise beyond — backward tilts past 2° were OOD
                # and the policy panicked. Symmetric pitch range fills
                # the gap in both directions; this is the single most
                # important v0.7 change for backward-recovery.
                "roll":  (-0.10, 0.10),
                "pitch": (-0.10, 0.10),
                "yaw":   (0.0, 0.0),
            },
            "velocity_range": {
                "x":     (0.0, 0.0),
                "y":     (0.0, 0.0),
                "z":     (0.0, 0.0),
                # Modest initial angular velocity so some episodes
                # start "already falling" — forces the policy to
                # learn dynamic recovery, not just static balance
                # from rest.
                "roll":  (-0.3, 0.3),
                "pitch": (-0.3, 0.3),
                "yaw":   (0.0, 0.0),
            },
        },
    )
    # Mid-episode pushes. The 4–8 s interval means a 20 s episode sees
    # ~3–5 pushes — enough that the policy can't memorize a post-push
    # settling sequence, must learn a generic recovery instead.
    # Magnitudes are tuned for a 7 kg torso: ±0.4 m/s longitudinal is
    # roughly the impulse from a firm shove to the chest, ±0.3 m/s
    # lateral is the binding case because the biped support polygon
    # is narrower in y. The angular components nudge the torso a few
    # degrees per push — enough to require active reaction without
    # straight-up tipping the robot over.
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 8.0),
        params={
            "velocity_range": {
                "x":     (-0.4, 0.4),
                "y":     (-0.3, 0.3),
                "roll":  (-0.3, 0.3),
                "pitch": (-0.3, 0.3),
            },
        },
    )


# ---------------------------------------------------------------------------
# Rewards: seven terms. Every additional shaping term is a knob we have
# to defend later. If standing can't be learned with just these, the
# issue is the dynamics, not the reward landscape.
#
#   alive             — +1 every tick the robot is still upright
#   flat_orientation  — penalise non-vertical torso (gravity off body-z)
#   base_height       — penalise torso CoM far from the standing height
#   joint_pos_limits  — penalise crashing into a joint's hard stop
#   action_rate_l2    — penalise rapid action changes between ticks
#   foot_flat         — reward feet's local +z aligned with world +z
#   shin_symmetry     — penalise asymmetric knee bend (v0.8)
#
# ``action_rate_l2`` (added in v0.1) is the soft analogue of a hard
# slew clamp: it makes large per-tick action deltas expensive, which
# both suppresses foot micro-oscillation and indirectly limits how
# large the action distribution std can grow (a wider std produces
# larger expected ``(a_t - a_{t-1})^2``, which costs reward). At the
# converged smooth-stand pose this term contributes essentially zero
# to the reward — its only job is to penalise the path away from that
# pose.
#
# ``foot_flat`` (added in v0.2) closes the heel-balance loophole the
# policy found in v0.1. Without it the cheapest static equilibrium
# is "rotate the foot toes-up so the heel becomes the contact patch
# directly under the torso CoM" — a single-joint trick that doesn't
# need any active balancing. With it, that trick costs the policy
# the full +1.0/tick foot-flat bonus per foot, forcing it to use a
# coordinated hip + knee + ankle strategy that keeps both feet flat
# AND the CoM over the support polygon (which requires a small
# forward torso lean given this robot's geometry).
#
# ``shin_symmetry`` (added in v0.8) closes a different loophole: the
# v0.7 policy converged to an asymmetric crouch — one knee deeply
# bent, the other near-straight and oscillating — because all the
# other terms are pose-symmetry-blind. ``foot_flat`` only cares
# about world-frame foot orientation, ``base_height`` only about
# torso z, ``flat_orientation`` only about torso tilt; none of them
# distinguish "both knees bent 0.3 rad" from "one knee bent 0.6 rad,
# other straight". The new penalty scores
# ``(shin_left + shin_right)^2`` (the sum, not the difference,
# because the right joint's local frame is rotated 180° about X —
# see the docstring on ``shin_symmetry_penalty`` in
# ``bebop_v2_rewards.py``), which is zero exactly when both knees
# bend forward by the same magnitude. The asymmetric attractor now
# pays a cost proportional to its imbalance, breaking the tie that
# kept PPO sitting there.
#
# Weight tuning notes:
#   * If ``foot_flat`` is too dominant the policy may sacrifice
#     ``flat_orientation`` (let the torso lean significantly) to
#     keep the feet flat. If the torso tilt at convergence exceeds
#     ~10°, drop foot_flat to 0.5 and/or widen its ``std`` (current
#     default 0.25; loosen toward 0.35 to give recovery more room).
#   * If the policy starts using heel-balance again (toes lifted,
#     foot dorsiflexed past ~25°) the v0.7 0.25 std is too loose —
#     drop back to 0.20 or bump foot_flat weight to 1.5–2.0.
#   * If ``shin_symmetry`` is too dominant the policy may collapse
#     to fully-straight knees (the only symmetric pose that
#     completely zeros the penalty), giving up the crouch that
#     ``foot_flat`` + ``base_height`` jointly prefer. Drop weight to
#     -0.2. If asymmetric crouch persists past ~5k iterations,
#     bump to -1.0.
#   * If training entropy collapses early (action std crashes toward
#     zero too fast, policy can't explore, learning stalls), reduce
#     ``action_rate_l2`` magnitude (e.g. -0.05 -> -0.02) or lower the
#     PPO ``entropy_coef`` in tandem so the optimizer keeps pushing
#     on exploration.
# ---------------------------------------------------------------------------
@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-1.0,
        params={
            "target_height": 0.6539092050794861,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    foot_flat = RewTerm(
        func=foot_flat_reward,
        weight=1.0,
        # std bumped from 0.15 (v0.2) to 0.25 (v0.7). The original 0.15
        # made any foot tilt above ~8.6° expensive, which suppressed the
        # ankle-strategy recovery from a backward lean (the natural
        # response is to lift the heel and pivot on the toes — that's a
        # foot tilt). At 0.25 the reward drops to 0.37 at ~14° tilt
        # instead, which is loose enough for real recovery motion but
        # still tight enough to crowd out the v0.1 heel-balance trick
        # (toes-up at ~30°+).
        params={"asset_cfg": SceneEntityCfg("robot"), "std": 0.25},
    )
    # v0.8: breaks the asymmetric-knee attractor the v0.7 policy
    # parked in. Scores ``(shin_left + shin_right)^2`` (sum, not
    # difference — see ``shin_symmetry_penalty`` docstring for the
    # mirrored-frame convention), which is zero when both knees bend
    # forward equally and grows quadratically with imbalance. Weight
    # -0.5 matches the original kitchen-sink config value. A 0.3 rad
    # mismatch (e.g. one knee 0.4, one knee 0.1) costs 0.5 * 0.09 =
    # 0.045/tick — small per-tick but compounded over a 2000-tick
    # episode it's a ~90-point reward gap vs. the symmetric solution,
    # enough to make PPO commit.
    shin_symmetry = RewTerm(
        func=shin_symmetry_penalty,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


# ---------------------------------------------------------------------------
# Terminations: episode ends on timeout (success) or on the torso
# touching the ground (fall). No other termination conditions.
# ---------------------------------------------------------------------------
@configclass
class TerminationsCfg:
    time_out = TermTerm(func=mdp.time_out, time_out=True)
    base_link_ground_contact = TermTerm(
        func=base_link_on_ground,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "ground_height_threshold": 0.30,
        },
    )


# ---------------------------------------------------------------------------
# Commands: required by the rsl_rl runner, but standing means "do
# nothing forever". We keep the term present and pinned to zero so the
# observation layout stays compatible with downstream variants that
# DO take a velocity command (locomotion experiments) — but no
# velocity_commands term appears in ``ObservationsCfg`` here, so the
# command is unused for policy input.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Top-level env cfg.
# ---------------------------------------------------------------------------
@configclass
class BebopV2StandingCfg(ManagerBasedRLEnvCfg):
    decimation = 2          # 200 Hz physics, 100 Hz policy
    episode_length_s = 20.0

    scene = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)
    observations = ObservationsCfg()
    actions = ActionsCfg()
    commands = CommandsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    events = EventCfg()

    def __post_init__(self):
        self.viewer.eye = [2.5, 2.5, 2.5]
        self.viewer.lookat = [0.0, 0.0, 0.0]

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True

        self.scene.robot = BEBOP_V2_STANDING_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Fixed ground friction. The base config randomizes friction in
        # (0.4, 1.2) — we deliberately don't here. Slippery-floor
        # robustness is a feature to be added in a later experiment.
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

        # Body-frame IMU sensor mounted on base_link. Identity offset
        # because the orientation transform between the BNO085 sensor
        # frame and the body frame is handled in firmware (see the
        # ``imu.mount`` block in bebop_v2.yaml) — sim treats the IMU as
        # already-rotated body-frame readings.
        self.scene.imu = ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link",
            update_period=0.0,
            debug_vis=False,
            offset=ImuCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(0.0, 0.0, 0.0, 1.0),
            ),
        )
