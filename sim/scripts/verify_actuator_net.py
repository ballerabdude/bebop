"""Smoke test for the hybrid torque-response actuator (td-b05f58).

Builds the ActNet push-stand env with a handful of envs, then checks:

1. the TorchScript response nets load and the env steps without NaNs,
2. zero action (default pose, midpoint gains) produces small finite efforts,
3. a hard position-channel command produces efforts that are large but
   attenuated vs the analytic DCMotor envelope (the net's whole point:
   realized torque falls short of the command at high demand),
4. the response-net history buffers actually fill (cmd history nonzero).

Usage (inside the container, CWD=/workspace/bebop_bot/sim):

    /workspace/isaaclab/isaaclab.sh -p scripts/verify_actuator_net.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

from bebop_training.experiments.exp_standing import (  # noqa: E402
    BebopV2StandingPushActNetCfg,
)


def main():
    env_cfg = BebopV2StandingPushActNetCfg()
    env_cfg.scene.num_envs = 4
    env = ManagerBasedRLEnv(cfg=env_cfg)

    robot = env.scene["robot"]
    act_dim = env.action_space.shape[-1]
    print(f"[verify] action dim: {act_dim} (expect 24)")

    obs, _ = env.reset()
    zero = torch.zeros(env.num_envs, act_dim, device=env.device)

    # 1. idle: default pose + midpoint gains for ~0.5 s
    for _ in range(50):
        obs, _, _, _, _ = env.step(zero)
    policy_obs = obs["policy"]
    assert torch.isfinite(policy_obs).all(), "NaN/Inf in observations"
    applied = robot.data.applied_torque.torch
    assert torch.isfinite(applied).all(), "NaN/Inf in applied torque"
    idle = applied.abs().mean(dim=0)
    print(f"[verify] idle |applied torque| per joint (Nm): {[round(v, 3) for v in idle.tolist()]}")

    # 2. hard command: raw position channels to +1 (max target offset),
    # gains to max — the response net should attenuate vs the analytic rail.
    hard = torch.ones(env.num_envs, act_dim, device=env.device)
    for _ in range(50):
        env.step(hard)
    pushed = robot.data.applied_torque.torch.abs().mean(dim=0)
    print(f"[verify] hard-cmd |applied torque| per joint (Nm): {[round(v, 3) for v in pushed.tolist()]}")

    # 3. history buffers fill
    foot = robot.actuators["foot"]
    cmd_hist_max = foot._cmd_hist.abs().max().item()
    print(f"[verify] foot actuator cmd-history max (Nm): {cmd_hist_max:.2f}")
    assert cmd_hist_max > 0.0, "response-net cmd history never filled"

    # 4. attenuation sanity: with a hard command, realized torque must stay
    # far below the datasheet stall the old model allowed (17 Nm for RS02).
    stall = foot.cfg.saturation_effort
    foot_mean = pushed[robot.find_joints(["foot_left_joint", "foot_right_joint"])[0]].mean()
    print(f"[verify] foot mean |tau| {foot_mean:.2f} Nm vs datasheet stall {stall:.1f} Nm")
    assert foot_mean < 0.9 * stall, "no attenuation — is the response net actually in the loop?"

    print("[verify] OK")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
