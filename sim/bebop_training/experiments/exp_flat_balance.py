# /workspace/bebop_bot/bebop_training/experiments/exp_flat_balance.py

from isaaclab.utils import configclass
from ..envs.bebop_base_cfg import BebopBaseEnvCfg

@configclass
class BebopFlatBalanceCfg(BebopBaseEnvCfg):
    """
    Experiment configuration for flat ground balancing.
    Inherits all Sim-to-Real robustness settings (Noise, Randomization, Pushing)
    from BebopBaseEnvCfg.
    """
    def __post_init__(self):
        super().__post_init__()

        # We do NOT override the push_robot params here.
        # We want the aggressive pushes defined in the Base Config (3.0-6.0s interval, +/- 1.0 m/s)
        # to ensure the policy is robust enough for the real robot.