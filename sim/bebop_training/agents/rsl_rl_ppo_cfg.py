# bebop_lab/agents/rsl_rl_ppo_cfg.py

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class BebopPPOBaseCfg(RslRlOnPolicyRunnerCfg):
    """Base PPO configuration tuned for the Bebop biped on an RTX 5090.

    Uses the Isaac Lab 3.x runner API: `actor`/`critic` model dicts +
    `obs_groups`. The legacy `policy = RslRlPpoActorCriticCfg(...)` field is
    deprecated in rsl_rl >= 4.0.0 and intentionally not set here.
    """

    # General Runner Settings
    num_steps_per_env = 24       # Number of steps to collect per env before updating policy
    max_iterations = 10000        # Total training iterations (approx 1-2 hours on 5090)
    save_interval = 100           # Save checkpoint every 100 iterations
    experiment_name = "bebop_base"

    # Empirical normalization is the deprecated rsl_rl < 4.0.0 way of doing
    # observation normalization. Disable it and use per-model
    # `obs_normalization` (set inside the actor/critic dicts) instead.
    empirical_normalization = False
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}

    # Isaac Lab 3.x runner expects explicit actor/critic model blocks.
    actor = {
        "class_name": "MLPModel",
        "hidden_dims": [512, 256, 128],
        "activation": "elu",
        "obs_normalization": False,
        "distribution_cfg": {
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    }
    critic = {
        "class_name": "MLPModel",
        "hidden_dims": [512, 256, 128],
        "activation": "elu",
        "obs_normalization": False,
        "distribution_cfg": None,
    }

    # PPO Algorithm Hyperparameters
    algorithm = RslRlPpoAlgorithmCfg(
        # Value Function
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        
        # Entropy (Exploration)
        # Empirically tuned across three runs:
        #   * 0.01 -> entropy crashed to -15 by iter 650 (cyan)
        #   * 0.02 -> entropy held at ~-3 to -4 stable (magenta)
        #   * 0.015 (with bumped symmetry penalties) -> on track to
        #     crash again (orange, -11 by iter 1000)
        # The right value scales with the weight of the rest of the
        # reward landscape. With the current symmetry (-4 to -5) and
        # joint_acc (-8e-6) penalties, 0.02 is what holds the entropy
        # in the stable regime. If you add more penalty weight later,
        # bump this up in step (e.g. 0.025 if you double symmetry weights
        # again).
        entropy_coef=0.02,
        
        # Training Updates
        num_learning_epochs=5,   # How many times to reuse the collected data
        
        # Mini Batches:
        # With 4096 envs * 24 steps = 98,304 samples per iteration.
        # 4 mini-batches = 24,576 samples per batch. 
        # The RTX 5090 can handle this easily.
        num_mini_batches=4,
        
        # Learning Rate
        learning_rate=1.0e-3,    # Standard starting point
        schedule="adaptive",     # Lowers LR if updates are too drastic (KL divergence high)
        
        # PPO Math
        gamma=0.99,              # Discount factor (future rewards importance)
        lam=0.95,                # GAE (Generalized Advantage Estimation) lambda
        desired_kl=0.01,         # Target KL divergence for adaptive schedule
        max_grad_norm=1.0,       # Gradient clipping to prevent exploding gradients
    )

@configclass
class BebopPPOLowLRCfg(BebopPPOBaseCfg):
    """
    Variant: Low Learning Rate.
    Use this if the 'Base' config learns to stand but then jitters/explodes later in training.
    """
    experiment_name = "bebop_low_lr"
    algorithm = RslRlPpoAlgorithmCfg(
        learning_rate=1.0e-4, # 10x smaller learning rate
        num_mini_batches=8    # More batches for smoother updates
    )


@configclass
class BebopPPOLocomotionCfg(BebopPPOBaseCfg):
    """Variant tuned for locomotion fine-tuning from a standing checkpoint.

    The standing policy collapses its action std (~0.02) which kills exploration.
    Higher entropy and slightly larger init_std force the actor to keep trying
    new motions long enough to discover walking gaits.
    """

    experiment_name = "bebop_locomotion"

    # Re-initialize the action distribution with more noise so resumed
    # checkpoints can rediscover exploration.
    actor = {
        "class_name": "MLPModel",
        "hidden_dims": [512, 256, 128],
        "activation": "elu",
        "obs_normalization": False,
        "distribution_cfg": {
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    }

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # Strong entropy bonus -> keeps exploration alive while learning to walk.
        entropy_coef=0.04,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.02,
        max_grad_norm=1.0,
    )