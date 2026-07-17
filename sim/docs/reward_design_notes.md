# Bebop V2 Standing Policy — Reward Design Notes

This document explains the reward functions used to train the Bebop V2
standing policy, written for someone who is not a math or ML expert. It
covers what each reward does, why it exists, and the math behind it, using
plain language and concrete numbers wherever possible.

---

## Background: how rewards work in reinforcement learning

Reinforcement learning (RL) is how we train the policy (the neural network
that controls the robot). At every time step (100 times per second, since
the policy runs at 100 Hz), the policy looks at the robot's state and
outputs an action. The simulator applies that action, the robot moves, and
the training system hands the policy a **reward** — a single number that
says "that was good" (positive) or "that was bad" (negative).

The policy's goal is to maximize the total reward it accumulates over an
episode (a 20-second simulation run = 2000 time steps). Over millions of
steps, the policy learns which actions lead to high reward and which lead
to low reward.

A **reward term** is one component of the total reward. The total reward
at each step is the sum of all the terms, each multiplied by a **weight**
that controls how much that term matters relative to the others. For
example:

```
total_reward = (1.0 × alive) + (-2.0 × torso_upright) + (0.5 × stationary_pose) + ...
```

A positive weight means "reward the policy for this" (a carrot). A negative
weight means "penalize the policy for this" (a stick). The weights are
what we tune to shape what the policy learns.

---

## The reward terms

The standing policy uses the following reward terms. Each is described
below in plain language, then with the math.

### 1. `alive` — the survival bonus

**What it does:** Rewards the policy simply for not having fallen over.
Every step the robot is still upright, the policy earns +1.0.

**Why it exists:** This is the most fundamental signal. If the robot falls,
it can't do anything else, so staying alive is the baseline requirement.

**The math:**
```
alive = 1.0 if the robot hasn't terminated, else 0.0
```
Weight: **+1.0** (carrot). Over a full 2000-step episode, surviving the
whole time earns 2000 × 1.0 = 2000 reward from this term alone.

---

### 2. `termination_penalty` — the fall penalty

**What it does:** Penalizes the policy heavily when the robot falls (the
base_link drops below 0.30 m, indicating the torso has hit the ground).

**Why it exists:** A single fall ends the episode and throws away all the
future `alive` reward the policy could have earned. This penalty makes
that cost explicit and sharp, so the policy learns to avoid falls
aggressively rather than just "mostly avoiding" them.

**The math:**
```
termination_penalty = 1.0 if the robot just terminated this step, else 0.0
```
Weight: **-200.0** (stick). This is the largest-magnitude weight in the
entire reward — one fall costs as much as 200 steps of survival. This
makes the policy strongly prefer "barely survive" over "fall while trying
something risky."

---

### 3. `upright_pose` — the "be vertical" carrot

**What it does:** Rewards the policy for keeping the torso upright (gravity
pointing straight down through the robot's body). The more upright, the
more reward. This is a **positive attractor** — it actively *pulls* the
policy toward vertical, rather than just punishing it for being tilted.

**Why it exists:** The prior reward design only had a *penalty* for tilt
(`torso_upright`, below). Penalties have a problem: their gradient goes
flat near the optimum. When the robot is nearly upright, the penalty is
nearly zero, so there's no signal telling the policy "you're almost
there, get a little more upright." The `upright_pose` exp reward solves
this — it has its steepest gradient *near* upright, so the policy is
always being pulled toward perfect vertical.

**The math:**
```
upright_pose = exp(-(g_x² + g_y²) / σ²)
```

Where:
- `g_x` and `g_y` are the horizontal components of **projected gravity** —
  the gravity vector expressed in the robot's body frame. When the robot
  is perfectly upright, gravity points straight down through the body
  z-axis, so `g_x = 0` and `g_y = 0`. When the robot tilts forward by
  angle θ, `g_x ≈ sin(θ)` and `g_y` stays near 0.
- `σ` (sigma) is the **width** of the Gaussian — it controls how quickly
  the reward falls off as the robot tilts. We use `σ = 0.25`, which
  corresponds to about 14° of tilt at the 1/e point (where the reward
  drops to ~37% of its maximum).

The reward is:
- **1.0** when perfectly upright (g_x = g_y = 0)
- **~0.37** when tilted ~14° in any direction
- **~0.05** when tilted ~28°
- **~0** when tilted past ~30° (which is also the fall termination angle)

This is a **Gaussian** (bell curve) centered on upright. The policy earns
maximum reward by being perfectly vertical, and the reward drops off
smoothly in every direction (forward, backward, sideways). It's symmetric
in pitch and roll, so it doesn't bias the policy toward any particular
lean direction.

Weight: **+0.75** (carrot). Per step, when perfectly upright, this pays
0.75. Over 2000 steps that's up to 1500 — a major component of the total
reward.

**Concrete example:** If the robot is tilted forward 7° (g_x ≈ 0.12,
g_y ≈ 0), the reward is `exp(-(0.12² + 0²) / 0.25²) = exp(-0.23) ≈ 0.79`.
So the policy earns 0.79 × 0.75 = 0.59 per step instead of the maximum
0.75. That 0.16 difference per step, over 2000 steps, is 320 reward — a
strong incentive to straighten up.

---

### 4. `torso_upright` — the "don't tilt" stick

**What it does:** Penalizes the robot for being tilted, with the penalty
growing as the tilt increases.

**Why it exists:** This is the complement to `upright_pose`. The exp
reward has its steepest gradient *near* upright (polishing the stand);
this penalty has its steepest gradient *far* from upright (catching a
fall). Together they cover the full range: the carrot pulls you toward
vertical when you're close, the stick pushes you away from a fall when
you're far.

**The math:**
```
torso_upright = g_x² + g_y²
```

This is just the sum of squares of the horizontal gravity components —
the same `g_x` and `g_y` used in `upright_pose`, but without the exp.
When upright, this is 0 (no penalty). When tilted 30°, it's `sin²(30°) ≈
0.25`, a significant penalty.

Weight: **-2.0** (stick). Note this is a **quadratic** penalty — it grows
as the *square* of the tilt, so small tilts are nearly free but large
tilts are expensive. This is intentional: we want to forgive small wobbles
but strongly punish approaches to the fall angle.

---

### 5. `stationary_pose` — the "be still" carrot

**What it does:** Rewards the policy for keeping the joints from moving.
The less the joints move, the more reward. When the robot is perfectly
still, this pays its maximum; when the joints are moving fast (during a
recovery or a fall), it drops toward zero.

**Why it exists:** Without this term, the policy can earn high reward by
swaying in a limit cycle — a slow oscillation that keeps it technically
upright but isn't a "quiet stand." This term makes stillness itself
rewarding, so the policy prefers to lock into a stable pose rather than
constantly move.

**The math:**
```
stationary_pose = exp(-Σ v_i² / σ²)
```

Where:
- `v_i` is the velocity of each joint (in rad/s)
- `Σ v_i²` is the sum of squared velocities across all 8 joints — a
  measure of "how much the joints are moving"
- `σ` (sigma) is the width, set to 0.5

The reward is:
- **1.0** when all joints are motionless (perfect stillness)
- **~0.37** when `Σ v_i² = 0.25` (e.g., one joint moving at 0.5 rad/s,
  or several joints each moving at ~0.2 rad/s)
- **~0** when joints are moving fast (during a fall or recovery)

Weight: **+0.5** (carrot). This is the term that should **dominate** once
the robot is standing — once the policy finds a stable upright pose, the
best thing it can do is stop moving and collect this reward every step.

**Why Gaussian (exp) and not a linear penalty?** Because it's **bounded**
in [0, 1]. A linear penalty (like `-weight × velocity`) would be
unbounded — during a fall, joint velocities spike to huge values, and an
unbounded penalty would overwhelm the survival reward and teach the
policy "falling is catastrophically expensive." The exp form saturates
toward 0 during a fall, so it just stops paying the carrot — it never
becomes punitive. This lets the policy focus on surviving during a fall
and focus on being still once it's recovered.

---

### 6. `bilateral_symmetry` — the "stand straight" stick (balance-gated)

**What it does:** Penalizes the policy when the left and right joints
aren't in the mirrored position — e.g., if the right hip is flexed forward
but the left is flexed back, or the right ankle is cranked positive while
the left is also positive (both feet pointing the same way instead of
mirroring).

**Why it exists:** On hardware, the policy consistently finds an
asymmetric stance — right hip flexed -0.3 to -0.4 rad while the left is
near zero, right ankle cranked +0.3 to +0.4 rad while the left is flat.
This happens because the real robot has left/right physical differences
(different motor friction, mass distribution, joint alignment), and the
policy learns to compensate by shifting its CoM to one side. But the
right ankle position requires ~60-80 N·m of torque to hold, and the RS02
foot motor can only produce 17 N·m — so the ankle gives way and the robot
falls.

This term pushes the policy toward a symmetric stance so it doesn't rely
on the weak ankle motor to hold an extreme position.

**SIGN CONVENTION — the SUM, not the difference (critical):** every L/R
joint pair on this robot is sign-mirrored in the URDF. The right-side
flexion joints (hip_flexion, knee_flexion, foot) use a flipped `-Y` axis;
hip_abduction uses the same `+X` axis but mirrored limits
(`left [-10°, +20°]` vs `right [-20°, +10°]`). The same physical "both
knees flexed the same way" pose therefore reads `q_left = -q_right`, so a
symmetric stance is `q_L + q_R = 0`. The asymmetry residual is the **SUM**.
The pre-Jul-2026 version used the difference `(q_L - q_R)²`, which rewarded
`q_L = q_R` — one leg forward, one leg back — and actively trained the
twisted-hip contortion it was meant to prevent (run 2026-07-09_04-16-40).
The `analyze_capture.py` L/R report prints both `L+R` (correct) and `L-R`
(buggy) so this is auditable per capture.

**The balance gate — the key innovation (Jul 15 2026 redesign):**

The penalty is **multiplied by a balance gate**:

```
gate = max(gate_floor, exp(-((g_x - center)² + g_y²) / gate_std²))
```

- When the robot is **balanced** (in the back-lean band): `gate ≈ 1.0`,
  penalty fires at full strength → policy is pushed toward a symmetric stance
- When the robot is **tilted** (recovering from a tip): `gate → gate_floor`
  (0.2), penalty is relaxed → policy is free to use whatever asymmetric
  catch motion it needs to recover

The floor is essential: without it (gate_floor=0) the penalty vanishes
completely at tilt and the policy can manufacture tilt + chatter to
suppress the symmetry constraint (the flailing exploit of capture
20260715_122500). The gate keeps a -0.4/tick floor at full tilt so the
policy can't escape the constraint by falling.

**Why the balance gate (not the old stillness gate):** the Jul 15 2026
reward redesign moved all movement penalties (joint_vel, position_rate,
hip_flexion_anchor) to balance gates so they relax in lockstep during
recovery. The symmetry term uses the same gate so the entire penalty
suite relaxes together when the robot is tilted — without this, an
ungated or stillness-gated symmetry term would keep fighting the policy
during recovery while the other penalties had already relaxed.

**The full math:**
```
symmetry_error = Σ (q_left + q_right)²    over hip, knee, foot, abduction pairs
gate = max(gate_floor, exp(-((g_x - center)² + g_y²) / gate_std²))
bilateral_symmetry = symmetry_error × gate
```

Weight: **-2.0** (stick, heavily weighted). Capture 20260715_214224 showed
the prior reward had NO active symmetry term (only the diluted -0.3
all-joint anchor), so the policy stood with hip_flexion L+R=-0.82 and
foot L+R=+0.49. At -2.0 the asymmetric stance costs ≈ -1.8/tick (for the
two worst pairs alone), finally exceeding the +1.0 alive reward and making
an asymmetric stand expensive relative to the survival gradient. The
balance gate means the weight can be this strong without fighting recovery.

---

### 7. `foot_deviation` — the "flat feet" stick (stillness-gated)

**What it does:** Penalizes the policy for moving the ankles (foot
joints) away from the neutral (flat-foot) position.

**Why it exists:** This directly addresses the hardware failure mode.
The RS02 foot motor has only 17 N·m of stall torque — the weakest joint
by far (the hip motors have 60-120 N·m). The policy consistently cranks
one ankle to +0.3 to +0.4 rad (17-23°) to compensate for hardware
asymmetry. Holding that position against gravity requires:

```
torque ≈ kp × angle = 180 × 0.4 = 72 N·m
```

But the motor can only produce 17 N·m. In sim this works because the
rigid-foot contact geometry supports the stance (the ground pushes back,
so the motor doesn't need the full torque). On hardware the motor
saturates, the ankle gives way, and the robot falls.

This term makes flat feet the reward optimum so the policy finds stances
that don't rely on extreme ankle torque.

**The stillness gate:** Same as `bilateral_symmetry` — the penalty is
multiplied by `exp(-Σ v_i² / 1.5²)` so it only fires when the robot is
still. During recovery, the policy is free to use the ankle to catch a
fall.

**The math:**
```
deviation = |q_foot_left| + |q_foot_right|
gate = exp(-Σ v_i² / 1.5²)
foot_deviation = deviation × gate
```

Note this uses **absolute value** (L1 norm), not squared (L2 norm). This
means the penalty grows linearly with ankle angle, not quadratically —
small ankle movements are penalized proportionally, and large ones are
penalized proportionally. (The squared form would be too forgiving of
small deviations and too harsh on large ones.)

Weight: **-0.5** (stick).

---

### 8. `feet_straight` — the "don't splay" stick

**What it does:** Penalizes hip abduction (the joints that move the legs
sideways) for deviating from zero.

**Why it exists:** Without this, the policy can adduct/splay the legs
into a wide stance — a sim cheat that's unstable on hardware because it
shifts the CoM laterally. Keeping the hips straight forces a narrow,
stacked stance.

**The math:**
```
feet_straight = |q_abduction_left| + |q_abduction_right|
```

L1 norm (absolute value), same as `foot_deviation`.

Weight: **-1.5** (stick). This is stronger than the foot/symmetry
penalties because hip abduction is a more common cheat and less needed
for legitimate balance.

---

### 9. `base_ang_vel_xy` — the "don't wobble" stick

**What it does:** Penalizes the angular velocity of the torso in the
pitch and roll directions (the "wobble" axes).

**Why it exists:** Damps the ~1 Hz body-sway oscillation (the
underdamped inverted-pendulum mode) that the policy falls into without
this term. Uses the IMU gyro that the policy already observes, so it
creates no sim-to-real gap.

**The math:**
```
base_ang_vel_xy = ω_x² + ω_y²
```

Where `ω_x` and `ω_y` are the pitch and roll angular velocities from the
IMU (in rad/s). Squared (quadratic) so small wobbles are nearly free but
fast wobbles are expensive.

Weight: **-0.10** (stick).

---

### 10. Action regularization terms

These four terms regularize the policy's *actions* (the 24-dim output:
8 joint position targets + 8 kp gains + 8 kd gains) to keep them smooth
and well-behaved.

#### `action_l2` — "don't work too hard"

```
action_l2 = Σ a_i²    over all 24 action channels
```
Weight: **-0.02**. Penalizes large actions. Keeps the policy from
saturating the action space.

#### `gain_l2` — "prefer midpoint gains"

```
gain_l2 = Σ a_i²    over the 16 gain channels (kp + kd, channels 8-24)
```
Weight: **-0.03**. Each gain channel is mapped from [-1, 1] to [min, max],
so `raw = 0` decodes to the midpoint gain. This term gives the
otherwise-flat gain directions a clean optimum at midpoint stiffness.

#### `gain_rate` — "don't chatter the gains"

```
gain_rate = Σ (a_gain_t - a_gain_{t-1})²    over the 16 gain channels
```
Weight: **-0.20**. Penalizes tick-to-tick change in the kp/kd channels.
This is the main anti-chatter term for variable impedance — it stops the
gains from flipping rapidly (e.g., foot kp snapping 250 → 107 → 250
across consecutive ticks, which is the failure seen on hardware).

#### `position_rate` — "move the setpoints smoothly"

```
position_rate = Σ (a_pos_t - a_pos_{t-1})²    over the 8 position channels
```
Weight: **-0.30**. Penalizes tick-to-tick change in the joint position
targets. Kills the setpoint limit cycle that rides the 0.020 rad/tick
slew limiter. Stronger than `gain_rate` because position chattering is
more destabilizing than gain chattering.

---

### 11. `joint_vel` — the "don't move fast" stick

**What it does:** Penalizes high joint velocity (unbounded, unlike the
bounded `stationary_pose`).

**Why it exists:** `stationary_pose` is bounded — it saturates toward 0
when joints are moving fast, so it stops providing gradient. This term
takes over where `stationary_pose` saturates, providing an unbounded
penalty that grows with the *square* of joint velocity. It damps residual
hunts that `stationary_pose` can't reach.

**The math:**
```
joint_vel = Σ v_i²    over all 8 joints
```
Weight: **-0.015** (stick). Small weight because it's meant to be a
gentle damping term, not a primary objective.

---

### 12. `forward_lean` — the "don't lean on your toes" stick

**What it does:** Penalizes the policy for parking the torso pitched
*forward* (CoM over the toes).

**Why it exists:** In sim, the policy can hold a forward lean by riding
the front edge of the flat foot — the rigid foot's contact patch extends
to the toe, so the ground reaction force supports a CoM that has crept
forward. On hardware, the foot is small and the ankle motor is weak, so
that strategy falls. This penalty keeps the resting CoM behind the toe
line.

**The math:**
```
forward_lean = relu(g_x - deadband)²
```

Where:
- `g_x` is the forward component of projected gravity (positive = forward
  lean)
- `deadband = 0.05` (≈ 2.9°) — a small forward excursion costs nothing,
  only *sustained/deep* forward lean is penalized
- `relu(x) = max(0, x)` — the penalty is zero for `g_x < deadband` and
  quadratic above it

This is **asymmetric** — backward lean (g_x < 0) is not penalized by
this term (it's handled by the symmetric `torso_upright` penalty). Only
the forward half is taxed, because forward lean is the hardware-fatal
direction.

Weight: **-3.0** (stick). Strong because this is a safety-critical
constraint — a forward lean that works in sim but falls on hardware is
the worst kind of sim-to-real gap.

---

### 13. `feet_flat` — the "soles parallel to the ground" stick (stillness-gated)

**What it does:** Penalizes each foot whose *sole* is not parallel to the
ground while standing. Replaces `foot_deviation` (section 7) for the
active-balancing reward (Jul 16 2026, user request).

**Why it exists:** `foot_deviation` anchored the foot *joint angle* at
zero, which fights the 8-12° torso back-lean band — under a back lean the
shank tilts with the torso, so a flat sole needs a *nonzero* ankle angle
(q_foot ≈ ±10°). Anchoring the joint at zero meant the policy had to
choose between the posture band and flat feet. `feet_flat` targets the
**result** instead: the sole's world orientation (forward kinematics from
the joint encoders + IMU, non-privileged), so the ankle is free to hold
whatever angle makes the sole flat.

The sim exploits a free ankle: the rigid foot's contact patch lets the
policy ride the toe or heel edge with a tilted sole. That does not
transfer — the real foot is small, the RS02 ankle is the weakest joint
(~17 N·m stall, capped at its 6 N·m continuous rating in sim), and an
edge contact shrinks the support polygon to a line. Flat soles maximize
the contact polygon: hips straight + soles flat = the ankle strategy over
a full contact patch (the foot-side counterpart to `hip_flexion_anchor`).

**The math:**
```
per foot:  u = R(q_foot) @ (0,0,1)      # sole normal in world frame
error = Σ_feet (u_x² + u_y²) = Σ_feet sin²(sole tilt)
gate = exp(-Σ v_i² / 1.5²)              # stillness gate
feet_flat = error × gate
```

- `q_foot` is the foot link's world quaternion from
  `asset.data.body_quat_w`. **Isaac Lab 3.0 returns quaternions in
  `(x, y, z, w)` order** (breaking change from 2.x WXYZ) — the rotation
  uses `isaaclab.utils.math.quat_apply` so the convention stays
  library-owned. The sole normal is the foot's local `+Z` because every
  leg-chain joint origin in the URDF has `rpy="0 0 0"`, so the foot frame
  aligns with base_link FLU at the zero pose.
- The stillness gate (same as `bilateral_symmetry`, σ = 1.5) fires at
  full strength whenever the robot is holding a pose — flat feet are a
  *posture* constraint — and relaxes toward 0 during active motion so
  recovery footwork (toe-off, heel strike, lifting a foot) stays free.

Weight: **-2.0** (stick). A 10° sole tilt on both feet costs
`2·sin²(10°)·2.0 ≈ -0.12`/tick (a firm shaping gradient), 20° ≈ -0.47/tick
— expensive relative to `alive` (+1.0) but below the survival gradient.
Matches the `bilateral_symmetry` weighting precedent for stance-posture
terms.

---

## How the terms work together

The reward is designed as a **layered system**, where each layer addresses
a different failure mode:

1. **Survival layer:** `alive` (+1.0) and `termination_penalty` (-200.0)
   — "don't fall." This is the baseline; everything else is secondary to
   not falling.

2. **Posture layer:** `upright_pose` (+0.75) and `torso_upright` (-2.0)
   — "be vertical." The carrot pulls toward upright, the stick pushes
   away from tilted. Together they cover the full range of tilt.

3. **Stillness layer:** `stationary_pose` (+0.5) and `joint_vel` (-0.015)
   — "be still." The bounded carrot rewards stillness, the unbounded
   stick damps fast motion. This is what makes the stand *quiet* rather
   than a swaying limit cycle.

4. **Symmetry layer (balance-gated):** `bilateral_symmetry` (-2.0) — "stand
   straight." Fires when the robot is balanced, relaxes during recovery.
   Heavily weighted so an asymmetric stance costs more than the alive
   reward earns. This is what makes the stand *hardware-safe* — the
   ankles don't get cranked to positions the weak RS02 motor can't hold.

5. **Anti-cheat layer:** `feet_straight` (-1.5), `base_ang_vel_xy`
   (-0.10), `forward_lean` (-3.0) — "don't find sim-only cheats." These
   close specific loopholes the policy would otherwise exploit (leg
   splay, body wobble, toe-riding).

6. **Action regularization:** `action_l2` (-0.02), `gain_l2` (-0.03),
   `gain_rate` (-0.20), `position_rate` (-0.30) — "be smooth." These keep
   the actions well-behaved and prevent chattering.

The layers are ordered by priority: survival > posture > stillness >
symmetry > anti-cheat > smoothness. The weights are tuned so that a
violation of a higher layer costs more than optimizing a lower layer. For
example, falling (survival) costs -200, while being asymmetric (symmetry
at -2.0 weight) costs maybe -1.8 per step at the capture's mean asymmetry
— so the policy will still accept asymmetry to avoid a fall (a fall costs
-200), which is the right trade-off, but a *sustained* asymmetric stand
is finally expensive enough that the policy prefers to correct it.

---

## The balance gate — why it's the key innovation

The balance gate on `bilateral_symmetry`, `joint_vel`, `position_rate`,
and `hip_flexion_anchor` is the most important design decision in the
current (Jul 15 2026) reward. Here's why:

**The problem:** The robot is physically asymmetric (left/right motor
friction differences, mass distribution, joint alignment). The policy
needs to compensate for this by using asymmetric motions — especially
during recovery, when it might need to step out with one leg or catch
itself on one side. If the symmetry penalty fires during recovery, it
fights the policy's recovery behavior and causes falls.

**The solution:** Multiply the symmetry and movement penalties by a
balance gate:

```
gate = max(gate_floor, exp(-((g_x - center)² + g_y²) / gate_std²))
```

- When the robot is **balanced** (in the back-lean band): gate ≈ 1.0,
  penalty fires → policy is pushed toward a symmetric, flat-foot stance
- When the robot is **tilted** (recovering from a tip): gate → gate_floor
  (0.2), penalty is relaxed → policy is free to use whatever asymmetric
  catch motion it needs

This separates the **goal** (stand balanced, symmetrically) from the
**process** (recover from any perturbation by any means necessary). The
policy can be as asymmetric as it wants while catching a fall — but once
it settles back into the balance band, it gets pushed toward symmetry.

**Why a floor (gate_floor=0.2):** without it the penalty vanishes
completely at tilt and the policy learns to *manufacture* tilt to unlock
chatter — the flailing limit cycle of capture 20260715_122500 (vel_std
0.69-1.34, slew-exceedance 56.9%, g_x swinging ±30°). The floor keeps a
minimum pressure on so the policy can't escape the constraint by
manufacturing tilt.

**The math intuition:** The gate is a Gaussian (bell curve) over how far
the torso tilt is from the back-lean band center. At balance (g_x ≈ -0.17,
g_y ≈ 0), the gate is 1.0 — full penalty. As the robot tilts away, the
gate drops exponentially — the penalty is relaxed. The width
`gate_std = 0.10` means the gate is ~1/e at ~5.7° off the band center. By
the time the robot is in a full recovery (tilted >15°), the gate has
collapsed to the floor (0.2) and the penalty is at 20% strength — relaxed
enough to allow recovery, not zero.

**History note:** the Jul 10-14 2026 reward used a *stillness* gate
(`exp(-Σv²/σ²)` over joint velocities) on the symmetry term instead. The
stillness gate worked for the statue-style stand but broke when the reward
moved to active balancing: the stillness gate suppressed the symmetry
penalty during ANY motion, including the slow limit-cycle sways the policy
uses to chase posture reward. The balance gate ties the suppression to the
balance STATE (am I tilted enough to need recovery?) rather than the
motion (am I moving?), which is the semantically correct question.

---

## Key concepts explained

### Projected gravity (`g_x`, `g_y`, `g_z`)

Projected gravity is the gravity vector expressed in the robot's body
frame. Imagine you're standing on the robot's torso, looking forward:
- `g_x` is gravity pointing forward/backward (positive = leaning
  forward)
- `g_y` is gravity pointing left/right (positive = leaning right)
- `g_z` is gravity pointing down (always near -1.0 when upright, since
  gravity points down)

When the robot is perfectly upright, `g_x = 0`, `g_y = 0`, `g_z = -1.0`.
When tilted forward 30°, `g_x ≈ sin(30°) = 0.5`, `g_z ≈ -cos(30°) =
-0.87`.

The policy observes this from the IMU (the same sensor the firmware
uses), so any reward based on projected gravity is **non-privileged** —
it doesn't use sim-only information, so it creates no sim-to-real gap.

### Gaussian (exp) rewards vs. quadratic penalties

A **Gaussian reward** `exp(-x²/σ²)` is bounded in [0, 1] and has its
steepest gradient near x=0. This makes it ideal as a **carrot** — it
rewards being at the optimum and provides gradient all the way there.

A **quadratic penalty** `x²` is unbounded and has its steepest gradient
far from x=0. This makes it ideal as a **stick** — it punishes being far
from the optimum and doesn't waste gradient near it.

Using both together (carrot near, stick far) gives gradient across the
full range, which is why we have both `upright_pose` (Gaussian carrot
near vertical) and `torso_upright` (quadratic stick away from vertical).

### The L1 vs. L2 norm

- **L1 norm:** `|x|` — the absolute value. Grows linearly. Treats small
  and large deviations proportionally.
- **L2 norm:** `x²` — the square. Grows quadratically. Forgiving of
  small deviations, harsh on large ones.

We use L1 for `feet_straight` and `foot_deviation` (we want proportional
penalty on all deviations) and L2 for `torso_upright` and `bilateral_symmetry`
(we want to be forgiving of small deviations but harsh on large ones).

---

## File locations

- Reward function implementations:
  `sim/bebop_training/envs/bebop_v2_rewards.py`
- Reward term configuration (weights, parameters):
  `sim/bebop_training/experiments/exp_standing.py` → `RewardsCfg` class
- PPO hyperparameters (entropy, learning rate, etc.):
  `sim/bebop_training/agents/rsl_rl_ppo_cfg.py`
- Training analyzer (post-training evaluation):
  `sim/tools/analyze_training.py`
- Hardware capture analyzer (post-deployment evaluation):
  `sim/tools/analyze_capture.py`
