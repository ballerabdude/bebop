# /workspace/bebop_bot/play_bebop.py

"""Play a trained Bebop V2 policy.

Loads a registered Isaac Lab task and a trained rsl_rl checkpoint, then runs
the deterministic policy (no exploration noise) in a small visual env. Useful
for sanity-checking a training run before promoting it to the next curriculum
stage or exporting for deployment.

Example::

    /workspace/isaaclab/isaaclab.sh -p play_bebop.py \\
        --task Isaac-BebopV2-Flat-v0 \\
        --resume logs/rsl_rl/Isaac-BebopV2-Flat-v0/<run>/model_14000.pt

    # Pin the velocity command (e.g. test "walk forward at 0.3 m/s"):
    /workspace/isaaclab/isaaclab.sh -p play_bebop.py \\
        --task Isaac-BebopV2-Locomotion-v0 \\
        --resume logs/rsl_rl/Isaac-BebopV2-Locomotion-v0/<run> \\
        --cmd_lin_vel_x 0.3

Interactive torso pushes
------------------------

While the viewer is focused, you can apply velocity impulses to the robot's
torso to test recovery behaviour. This is the play-time equivalent of grabbing
the chest plate and shoving it. The Isaac Sim viewport must have keyboard
focus — click on it once after the sim window opens.

Key bindings (added on top of the default Isaac Sim viewport bindings):

    Pitch  : I = +pitch_rate (tip forward, nose-down)
             K = -pitch_rate (tip backward, nose-up)
    Roll   : J = -roll_rate  (tip left)
             L = +roll_rate  (tip right)
    Linear : W / S = +/- x   (push forward / backward in body frame)
             A / D = +/- y   (push left / right in body frame)
    Magnitude: +/= raise push magnitude by 1.25x
               -/_ lower push magnitude by 1.25x
               0   print current push magnitudes
    Misc    : R   reset the env (re-randomise pose + zero velocities)
              H   print this help to the console

Push magnitudes are velocity impulses (m/s for linear, rad/s for angular).
Default scaling is set so a single key press lands the robot's torso roughly
where the v0.14 ``push_robot`` random event used to put it (±0.4 m/s linear,
±0.3 rad/s angular). Bump with +/- as needed to find the policy's recovery
envelope.

If running headlessly (``--headless``), the keyboard bindings are skipped and
only the CLI ``--push_*`` flags (which apply a single push on startup) and the
random ``push_robot`` event (controlled by ``--disable_pushes``) are active.
"""

import argparse
import os
import sys

# --- STEP 1: Launch App ---
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trained Bebop policy.")
parser.add_argument(
    "--task",
    type=str,
    required=True,
    help="Registered task name (e.g. Isaac-BebopV2-Flat-v0).",
)
parser.add_argument(
    "--resume",
    type=str,
    required=True,
    help="Path to a checkpoint .pt file (or run directory; latest model_*.pt is used).",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of envs to spawn (default 1 for clean visual play).",
)
parser.add_argument("--seed", type=int, default=None, help="Random seed.")
parser.add_argument(
    "--steps",
    type=int,
    default=10_000_000,
    help="Maximum simulation steps before exiting (default effectively infinite).",
)

# Optional manual command overrides. If any --cmd_* arg is set, the velocity
# command sampler is replaced with a single fixed point so the policy is given
# that exact command for the entire play session.
parser.add_argument("--cmd_lin_vel_x", type=float, default=None, help="Forward velocity command (m/s).")
parser.add_argument("--cmd_lin_vel_y", type=float, default=None, help="Lateral velocity command (m/s).")
parser.add_argument("--cmd_ang_vel_z", type=float, default=None, help="Yaw-rate command (rad/s).")

parser.add_argument(
    "--disable_pushes",
    action="store_true",
    help="Disable random push disturbances during play (clean policy demo).",
)

# Interactive torso-push controls (see module docstring). The two
# `--push_*_scale` flags set the per-press magnitude of each axis; +/-
# in-sim adjust them up/down by 1.25x. Defaults match the v0.7+
# ``push_robot`` event envelope so a single press is roughly the same
# disturbance the policy saw during training.
parser.add_argument(
    "--push_linear_scale",
    type=float,
    default=0.4,
    help="Per-press linear-velocity push magnitude in m/s (W/S/A/D). Default 0.4.",
)
parser.add_argument(
    "--push_angular_scale",
    type=float,
    default=0.3,
    help="Per-press angular-velocity push magnitude in rad/s (I/K/J/L). Default 0.3.",
)
parser.add_argument(
    "--disable_keyboard_push",
    action="store_true",
    help="Skip wiring up the carb keyboard push subscription (CLI-only mode).",
)

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# --- STEP 2: Imports (must come AFTER AppLauncher) ---
import gymnasium as gym
import torch
import isaaclab.envs.mdp as mdp
import bebop_training  # registers Isaac-BebopV2-* tasks

from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


class TorsoPushController:
    """Carb-keyboard-driven torso push controller.

    Subscribes to the Omniverse keyboard input stream and queues velocity
    impulses to apply to the robot root on the next sim tick. Designed to
    let an operator manually probe a trained policy's recovery envelope
    without restarting the run (e.g. "lean it 0.4 m/s forward, see if it
    falls; lean it 0.8 m/s forward, see if it falls").

    Each key press appends an impulse to ``_pending`` (6-dim: x, y, z,
    roll, pitch, yaw). ``apply_pending(env)`` drains the queue, sums the
    impulses, adds them to the current root velocity of every env, and
    writes the result back into the physics sim — same code path used by
    ``isaaclab.envs.mdp.events.push_by_setting_velocity`` so behaviour
    matches the random push event the policy was trained against.

    The reset and help keys ('R' / 'H') are dispatched immediately on
    keypress rather than queued.

    Notes
    -----
    * The Isaac Sim viewport must have keyboard focus for keys to land.
      Click into the viewport once after the window opens.
    * Carb key events fire on both press and release; we only react on
      press to avoid double-impulse per key tap.
    """

    HELP_TEXT = (
        "\n[push] interactive torso push controls:\n"
        "  Pitch  : I = +pitch_rate  K = -pitch_rate\n"
        "  Roll   : L = +roll_rate   J = -roll_rate\n"
        "  Linear : W / S = +/- x    A / D = +/- y\n"
        "  Magnitude:  + raises 1.25x   - lowers 1.25x   0 prints current\n"
        "  R = reset env   H = print this help\n"
    )

    def __init__(
        self,
        env,
        linear_scale: float,
        angular_scale: float,
    ):
        import weakref

        import carb
        import omni

        self._env = env
        self._linear_scale = float(linear_scale)
        self._angular_scale = float(angular_scale)
        self._pending: list[tuple[float, float, float, float, float, float]] = []
        self._reset_requested = False

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *_args, obj=weakref.proxy(self): obj._on_key(event),
        )

        print(self.HELP_TEXT, flush=True)
        print(
            f"[push] linear_scale={self._linear_scale:.3f} m/s  "
            f"angular_scale={self._angular_scale:.3f} rad/s",
            flush=True,
        )

    def __del__(self):
        try:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub)
        except Exception:
            pass

    def _on_key(self, event) -> bool:
        """Carb keyboard callback. Returns True to consume the event."""
        import carb

        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True

        key = event.input
        K = carb.input.KeyboardInput

        # Use a body-frame velocity impulse: (vx, vy, vz, wx, wy, wz).
        # x is forward, y is left, z is up; rotations are roll (about x),
        # pitch (about y), yaw (about z). For Bebop's body frame this
        # matches Isaac Sim's default URDF convention.
        L = self._linear_scale
        A = self._angular_scale
        impulses = {
            K.W: (+L, 0, 0, 0, 0, 0),  # push forward
            K.S: (-L, 0, 0, 0, 0, 0),  # push backward
            K.A: (0, +L, 0, 0, 0, 0),  # push left
            K.D: (0, -L, 0, 0, 0, 0),  # push right
            K.I: (0, 0, 0, 0, +A, 0),  # pitch nose-down  (tip forward)
            K.K: (0, 0, 0, 0, -A, 0),  # pitch nose-up    (tip backward)
            K.L: (0, 0, 0, +A, 0, 0),  # roll right
            K.J: (0, 0, 0, -A, 0, 0),  # roll left
        }
        if key in impulses:
            self._pending.append(impulses[key])
            tag = {
                K.W: "+x (forward)",
                K.S: "-x (backward)",
                K.A: "+y (left)",
                K.D: "-y (right)",
                K.I: "+pitch (tip forward)",
                K.K: "-pitch (tip backward)",
                K.L: "+roll (right)",
                K.J: "-roll (left)",
            }[key]
            print(f"[push] queued impulse: {tag}", flush=True)
            return True

        if key in (K.EQUAL, K.NUMPAD_ADD):
            self._linear_scale *= 1.25
            self._angular_scale *= 1.25
            print(
                f"[push] scaled UP   linear={self._linear_scale:.3f} "
                f"angular={self._angular_scale:.3f}",
                flush=True,
            )
            return True
        if key in (K.MINUS, K.NUMPAD_SUBTRACT):
            self._linear_scale /= 1.25
            self._angular_scale /= 1.25
            print(
                f"[push] scaled DOWN linear={self._linear_scale:.3f} "
                f"angular={self._angular_scale:.3f}",
                flush=True,
            )
            return True
        if key == K.KEY_0:
            print(
                f"[push] linear={self._linear_scale:.3f} m/s   "
                f"angular={self._angular_scale:.3f} rad/s",
                flush=True,
            )
            return True
        if key == K.R:
            self._reset_requested = True
            print("[push] reset queued", flush=True)
            return True
        if key == K.H:
            print(self.HELP_TEXT, flush=True)
            return True
        return True

    def apply_pending(self, env) -> None:
        """Apply any queued impulses + handle reset request."""
        if self._reset_requested:
            try:
                env.reset()
            except Exception as e:
                print(f"[push] reset failed: {e}", flush=True)
            self._reset_requested = False
            self._pending.clear()
            return

        if not self._pending:
            return

        import warp as wp

        unwrapped = getattr(env, "unwrapped", env)
        asset = unwrapped.scene["robot"]
        delta = torch.zeros(6, device=asset.device, dtype=torch.float32)
        for imp in self._pending:
            delta += torch.tensor(imp, device=asset.device, dtype=torch.float32)
        self._pending.clear()

        # Mirror push_by_setting_velocity: read root_vel_w, add delta,
        # write back. delta is added to every env (we only spawn one for
        # play, so this is a single robot in practice).
        root_vel = wp.to_torch(asset.data.root_vel_w).clone()
        root_vel += delta.unsqueeze(0).expand_as(root_vel)
        env_ids = torch.arange(root_vel.shape[0], device=asset.device)
        asset.write_root_velocity_to_sim_index(root_velocity=root_vel, env_ids=env_ids)


def _resolve_checkpoint(path: str) -> str:
    """Accept either a .pt file or a run directory containing model_*.pt files."""
    if os.path.isdir(path):
        ckpts = sorted(
            f for f in os.listdir(path) if f.startswith("model_") and f.endswith(".pt")
        )
        if not ckpts:
            raise FileNotFoundError(f"No model_*.pt found in {path}")
        return os.path.join(path, ckpts[-1])
    return path


def _maybe_override_commands(env_cfg) -> None:
    """If any ``--cmd_*`` arg is set, pin the velocity command to that exact triple."""
    cmd_overrides = (args.cmd_lin_vel_x, args.cmd_lin_vel_y, args.cmd_ang_vel_z)
    if all(v is None for v in cmd_overrides):
        return

    vx = args.cmd_lin_vel_x if args.cmd_lin_vel_x is not None else 0.0
    vy = args.cmd_lin_vel_y if args.cmd_lin_vel_y is not None else 0.0
    wz = args.cmd_ang_vel_z if args.cmd_ang_vel_z is not None else 0.0

    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(vx, vx),
        lin_vel_y=(vy, vy),
        ang_vel_z=(wz, wz),
    )
    print(f"[INFO] Pinned velocity command to ({vx:.2f}, {vy:.2f}, {wz:.2f})")


def main() -> int:
    # 1. Build env + agent configs from the gym registry (mirrors train_bebop.py).
    task_spec = gym.spec(args.task)

    cfg_entry_point = task_spec.kwargs.get("env_cfg_entry_point")
    if not callable(cfg_entry_point):
        raise ValueError(f"Env config entry point {cfg_entry_point} is not callable.")
    env_cfg = cfg_entry_point()

    agent_cfg_entry_point = task_spec.kwargs.get("rsl_rl_cfg_entry_point")
    if not callable(agent_cfg_entry_point):
        raise ValueError(f"Agent config entry point {agent_cfg_entry_point} is not callable.")
    agent_cfg = agent_cfg_entry_point()

    # 2. Apply play-time overrides.
    env_cfg.scene.num_envs = max(1, args.num_envs)
    if args.seed is not None:
        env_cfg.seed = args.seed

    _maybe_override_commands(env_cfg)

    if args.disable_pushes and hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None
        print("[INFO] Disabled push disturbances.")

    # 3. Create env + wrap for rsl_rl.
    print(f"[INFO] Creating environment: {args.task} (num_envs={env_cfg.scene.num_envs})")
    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # 4. Build runner only to load the checkpoint and produce an inference policy.
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=env.device)

    ckpt = _resolve_checkpoint(args.resume)
    print(f"[INFO] Loading checkpoint: {ckpt}")
    runner.load(ckpt)

    # rsl_rl 5.x exposes a deterministic inference callable (mean of the
    # action distribution; no exploration noise).
    policy = runner.get_inference_policy(device=env.device)

    # 5. Play loop.
    # NOTE: rsl_rl 5.x's RslRlVecEnvWrapper returns the obs tensor directly
    # (shape: ``(num_envs, obs_dim)``), not ``(obs, extras)``. The step()
    # return arity varies across rsl_rl versions, so we index defensively.
    obs = env.get_observations()
    if isinstance(obs, tuple):  # older rsl_rl returned (obs, extras)
        obs = obs[0]

    # Optional: wire up the interactive torso push controller. Only enabled
    # when running with a viewport (headless skips it because there is no
    # window to attach a keyboard subscription to) and not explicitly
    # disabled by the user.
    push_ctl: TorsoPushController | None = None
    if not args.disable_keyboard_push and not getattr(args, "headless", False):
        try:
            push_ctl = TorsoPushController(
                env,
                linear_scale=args.push_linear_scale,
                angular_scale=args.push_angular_scale,
            )
        except Exception as e:
            print(
                f"[WARN] Could not attach keyboard push controller "
                f"({type(e).__name__}: {e}). Continuing without it.",
                flush=True,
            )

    print("[INFO] Running policy. Close the viewer or Ctrl-C to stop.")

    step = 0
    try:
        with torch.inference_mode():
            while simulation_app.is_running() and step < args.steps:
                if push_ctl is not None:
                    push_ctl.apply_pending(env)
                actions = policy(obs)
                step_result = env.step(actions)
                obs = step_result[0]
                step += 1
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    env.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n[ERROR] Play crashed: {e}\n")
        raise
    finally:
        simulation_app.close()
