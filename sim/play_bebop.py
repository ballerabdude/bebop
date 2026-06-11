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

Inspecting policy I/O
---------------------

Pass ``--print_obs_actions`` to periodically dump the observation fed *into*
the policy and the action it predicts *out*, for the first robot (env 0).
The vectors are split into named groups read off the env's observation and
action managers (e.g. ``base_ang_vel``, ``projected_gravity``, ``joint_pos``,
... and the action ``pos`` / ``kp`` / ``kd`` channels), so you can sanity-check
what the network sees and emits without a plotting GUI. ``--print_interval``
sets how many sim steps elapse between prints (default 50). Works headless::

    /workspace/isaaclab/isaaclab.sh -p play_bebop.py \\
        --task Isaac-BebopV2-Standing-v0 \\
        --resume logs/rsl_rl/Isaac-BebopV2-Standing-v0/<run> \\
        --print_obs_actions --print_interval 25
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

# Console inspection of the policy I/O for env 0. Prints the observation
# vector going *into* the policy and the action vector coming *out* of it,
# split into named groups (obs terms) and channels (action pos / kp / kd),
# so you can sanity-check what the network sees and emits without a GUI.
parser.add_argument(
    "--print_obs_actions",
    action="store_true",
    help="Periodically print the obs fed in and the action predicted for env 0.",
)
parser.add_argument(
    "--print_interval",
    type=int,
    default=2,
    help="Sim steps between obs/action prints when --print_obs_actions is set (default 50).",
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


def _as_torch(value):
    """Coerce a warp/torch/array obs or action to a CPU torch tensor."""
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu")
    torch_view = getattr(value, "torch", None)
    if isinstance(torch_view, torch.Tensor):
        return torch_view.detach().to("cpu")
    try:
        import warp as wp

        if isinstance(value, wp.array):
            return wp.to_torch(value).detach().to("cpu")
    except Exception:
        pass
    return torch.as_tensor(value).detach().to("cpu")


def _term_dim_length(dim) -> int:
    """Flatten an Isaac Lab obs-term dim spec to a scalar length."""
    if isinstance(dim, int):
        return dim
    length = 1
    try:
        for d in dim:
            length *= int(d)
    except TypeError:
        return int(dim)
    return length


def _extract_policy_obs(obs, env_idx: int = 0) -> torch.Tensor:
    """Return the flat policy observation row for ``env_idx``.

    ``RslRlVecEnvWrapper`` (rsl_rl >= 4) returns a ``TensorDict`` keyed by
    observation group (``"policy"``, ``"critic"``, ...), not a bare tensor.
    Older wrappers returned ``(tensor, extras)`` or the tensor directly.
    """
    if isinstance(obs, tuple):
        obs = obs[0]

    policy_obs = obs
    if isinstance(obs, dict) and "policy" in obs:
        policy_obs = obs["policy"]
    elif hasattr(obs, "keys"):
        try:
            if "policy" in obs.keys():
                policy_obs = obs["policy"]
        except Exception:
            pass

    policy_obs = _as_torch(policy_obs)
    if policy_obs.dim() >= 2:
        return policy_obs[env_idx].reshape(-1)
    return policy_obs.reshape(-1)


def _policy_obs_dim(obs) -> int | None:
    """Best-effort policy observation dimension for startup logging."""
    try:
        return int(_extract_policy_obs(obs).numel())
    except Exception:
        return None


def _build_obs_layout(env) -> list[tuple[str, int, int]]:
    """Return ``[(term_name, start, length), ...]`` for the policy obs group.

    Read straight off the observation manager so the labels track whatever
    obs terms the active task defines (rather than hardcoding the standing
    layout). Falls back to a single ``obs`` span if introspection fails.
    """
    unwrapped = getattr(env, "unwrapped", env)
    try:
        obs_mgr = unwrapped.observation_manager
        names = list(obs_mgr.active_terms["policy"])
        dims = obs_mgr.group_obs_term_dim["policy"]
        layout: list[tuple[str, int, int]] = []
        start = 0
        for name, dim in zip(names, dims):
            length = _term_dim_length(dim)
            layout.append((name, start, length))
            start += length
        return layout
    except Exception:
        return []


def _obs_segments_from_manager(env, env_idx: int = 0) -> list[tuple[str, list[float]]]:
    """Return per-term policy obs values by reading the observation manager.

    Used when the policy group is stored as a dict of tensors rather than a
    single concatenated vector (in that case flat slicing by layout fails).
    """
    unwrapped = getattr(env, "unwrapped", env)
    obs_dict = unwrapped.observation_manager.compute()
    policy = obs_dict.get("policy", obs_dict)
    if not isinstance(policy, dict):
        return []

    segments: list[tuple[str, list[float]]] = []
    for name, value in policy.items():
        row = _as_torch(value)
        if row.dim() >= 2:
            row = row[env_idx]
        segments.append((name, row.reshape(-1).tolist()))
    return segments


def _build_action_layout(env, action_dim: int) -> list[tuple[str, int, int]]:
    """Return ``[(label, start, length), ...]`` for the action vector.

    Splits each action term reported by the action manager. A 24-dim
    variable-impedance term is further split into ``pos`` / ``kp`` / ``kd``
    thirds (the MIT-mode layout) for readability.
    """
    unwrapped = getattr(env, "unwrapped", env)
    layout: list[tuple[str, int, int]] = []
    try:
        act_mgr = unwrapped.action_manager
        names = list(act_mgr.active_terms)
        dims = list(act_mgr.action_term_dim)
    except Exception:
        names, dims = [], []

    if not names:
        return [("action", 0, action_dim)]

    start = 0
    for name, dim in zip(names, dims):
        dim = int(dim)
        if dim % 3 == 0 and dim >= 3:
            third = dim // 3
            layout.append((f"{name}.pos", start, third))
            layout.append((f"{name}.kp", start + third, third))
            layout.append((f"{name}.kd", start + 2 * third, third))
        else:
            layout.append((name, start, dim))
        start += dim
    return layout


def _format_vec(values, per_line: int = 8, indent: int = 6) -> str:
    """Pretty-print a 1-D float sequence, wrapping every ``per_line`` items."""
    pad = " " * indent
    chunks = []
    for i in range(0, len(values), per_line):
        row = "  ".join(f"{v:+8.3f}" for v in values[i : i + per_line])
        chunks.append((pad if i else "") + row)
    return "\n".join(chunks) if chunks else f"{pad}(empty)"


def _print_snapshot(step, obs_segments, action_row, action_layout) -> None:
    """Print a labelled obs/action snapshot for the selected env."""
    label_w = max(
        [len(n) for n, _ in obs_segments]
        + [len(n) for n, _, _ in action_layout]
        + [4]
    )
    lines = [f"\n──── step {step} | env 0 ────", "obs (into policy):"]
    for name, seg in obs_segments:
        lines.append(f"  {name.ljust(label_w)}  {_format_vec(seg, indent=label_w + 4)}")

    lines.append("action (out of policy):")
    for name, lo, ln in action_layout:
        seg = action_row[lo : lo + ln].tolist()
        lines.append(f"  {name.ljust(label_w)}  {_format_vec(seg, indent=label_w + 4)}")
    print("\n".join(lines), flush=True)


def _collect_obs_segments(obs, env, obs_layout, env_idx: int = 0) -> list[tuple[str, list[float]]]:
    """Build labelled obs segments for printing."""
    per_term = _obs_segments_from_manager(env, env_idx)
    if per_term:
        return per_term

    obs_row = _extract_policy_obs(obs, env_idx)
    if obs_layout:
        return [(name, obs_row[lo : lo + ln].tolist()) for name, lo, ln in obs_layout]
    return [("obs", obs_row.tolist())]


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
    # NOTE: recent RslRlVecEnvWrapper returns a TensorDict keyed by obs group
    # (``"policy"``, ``"critic"``, ...). Older wrappers returned a bare tensor
    # or ``(obs, extras)``. The inference policy accepts the TensorDict; our
    # print helper extracts ``obs["policy"]`` before slicing.
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

    # Optional: console inspection of the policy I/O for env 0. Built once
    # off the observation/action managers so the labels track the task's
    # actual obs terms and action layout.
    obs_layout: list[tuple[str, int, int]] = []
    action_layout: list[tuple[str, int, int]] = []
    print_interval = max(1, args.print_interval)
    if args.print_obs_actions:
        obs_layout = _build_obs_layout(env)
        obs_dim = _policy_obs_dim(obs)
        dim_msg = f"obs_dim={obs_dim}" if obs_dim is not None else "obs_dim=?"
        print(
            f"[INFO] Printing env-0 obs/action every {print_interval} step(s). {dim_msg}.",
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

                if args.print_obs_actions and step % print_interval == 0:
                    if not action_layout:
                        action_layout = _build_action_layout(env, actions.shape[-1])
                    _print_snapshot(
                        step,
                        _collect_obs_segments(obs, env, obs_layout),
                        _as_torch(actions)[0],
                        action_layout,
                    )

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
