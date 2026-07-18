# bebop_lab/agents/rsl_rl_ppo_cfg.py

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class BebopPPOBaseCfg(RslRlOnPolicyRunnerCfg):
    """Base PPO configuration tuned for the Bebop biped on an RTX 5090.

    Uses the Isaac Lab 3.x runner API: `actor`/`critic` model dicts +
    `obs_groups`. The legacy `policy = RslRlPpoActorCriticCfg(...)` field is
    deprecated in rsl_rl >= 4.0.0 and intentionally not set here.
    """

    # General Runner Settings
    num_steps_per_env = 32       # Number of steps to collect per env before updating policy
    max_iterations = 10000        # Total training iterations
    save_interval = 100           # Save checkpoint every 100 iterations
    experiment_name = "bebop_base"

    # Empirical normalization is the deprecated rsl_rl < 4.0.0 way of doing
    # observation normalization. Disable it and use per-model
    # `obs_normalization` (set inside the actor/critic dicts) instead.
    empirical_normalization = False
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}

    # Isaac Lab 3.x runner expects explicit actor/critic model blocks.
    #
    # `std_type` must be ``"scalar"`` or ``"log"`` for the Gaussian head in
    # this repo's bundled rsl_rl (``"per_dim"`` is not supported and crashes
    # at runner init). We use ``"log"`` so the std is ``exp(log_std)`` —
    # mathematically positive by construction. With ``"scalar"`` the std
    # is a directly-learnable parameter that a single bad gradient (on
    # the heels of a value-loss spike) can push negative, causing
    # ``Normal(mean, std)`` to raise ``normal expects all elements of
    # std >= 0.0`` mid-training. MIT-mode 24-dim actions still train fine
    # with a shared log-std; tune ``entropy_coef`` / ``init_std`` if
    # gain channels need more exploration.
    #
    # ``obs_normalization=True`` wraps the policy input with rsl_rl's
    # ``EmpiricalNormalization`` (running mean / std with outlier clip
    # to ±5σ). This is the standard PPO default for mixed-scale inputs
    # — joint_vel is rad/s, projected_gravity is in [-1, 1], and
    # last_action is in [-1, 1] — and more importantly it stops a
    # single transient PhysX outlier (knee contact pop sending one env's
    # joint_vel to a huge value for one tick) from NaN-ing the actor's
    # ``log_std`` gradient and crashing the next rollout with the same
    # ``normal expects std >= 0.0`` error. The normalization stats are
    # saved with the checkpoint, so the ONNX export path picks them up
    # automatically — the deployed firmware sees the same normalized
    # input the policy was trained on.
    actor = {
        "class_name": "MLPModel",
        "hidden_dims": [512, 256, 128],
        "activation": "elu",
        "obs_normalization": True,
        "distribution_cfg": {
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "log",
        },
    }
    # Critic is intentionally LARGER than the actor (asymmetric actor-critic).
    # The critic only exists during training — it never deploys — so growing it
    # is free at inference time on the robot. A higher-capacity value function
    # gives lower-variance advantage estimates, which stabilizes PPO and yields
    # cleaner learning on every action channel, including the under-determined
    # variable-impedance kp/kd channels. The actor stays [512, 256, 128] because
    # it is the network that runs at 100 Hz on bebop-linux and is latency-bound.
    critic = {
        "class_name": "MLPModel",
        "hidden_dims": [1024, 512, 256],
        "activation": "elu",
        "obs_normalization": True,
        "distribution_cfg": None,
    }

    # PPO Algorithm Hyperparameters (RSL-RL defaults)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.016,
        max_grad_norm=1.0,
    )

@configclass
class BebopPPOLowLRCfg(BebopPPOBaseCfg):
    """Variant: Low Learning Rate.

    Use this if the base config learns to stand but then jitters/explodes
    later in training. Only the learning rate and minibatch count are
    overridden; everything else inherits from the base.
    """
    experiment_name = "bebop_low_lr"
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.016,
        max_grad_norm=1.0,
    )


@configclass
class BebopPPOPushCfg(BebopPPOBaseCfg):
    """Variant for the push-recovery stand (``Isaac-BebopV2-Standing-Push-v0``).

    Higher entropy than the base (0.02 vs 0.01, restored Jul 18 2026 for the
    first push-training run): push recovery requires exploring a family of
    recovery motions, not a single quiet pose, and the standing runs showed
    the action std collapsing to ~0.12 within 1k iters at 0.01 — too little
    exploration to discover catch steps under shoves. Raise toward 0.03 if
    the policy goes deterministic and stops recovering; lower back to 0.01
    if the action std refuses to come down.
    """

    experiment_name = "bebop_push"
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.02,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.016,
        max_grad_norm=1.0,
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
    # checkpoints can rediscover exploration. Same ``std_type`` contract as
    # the base cfg — ``"log"`` so std stays positive across any gradient,
    # and ``obs_normalization=True`` to inherit the running stats from
    # the standing checkpoint (rsl_rl restores them automatically).
    actor = {
        "class_name": "MLPModel",
        "hidden_dims": [512, 256, 128],
        "activation": "elu",
        "obs_normalization": True,
        "distribution_cfg": {
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "log",
        },
    }

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.04,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.016,
        max_grad_norm=1.0,
    )