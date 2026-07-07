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

    # PPO Algorithm Hyperparameters
    algorithm = RslRlPpoAlgorithmCfg(
        # Value Function
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        
        # Entropy (Exploration)
        # The right value scales with the total weight of the penalty
        # terms in the reward landscape. If you add penalty weight (e.g.
        # bump the symmetry or deviation penalties), the actor's
        # post-update KL can collapse and entropy crashes to a tiny
        # negative number — symptom of a deterministic policy that
        # ignores observations. Bump this in step when that happens
        # (Locomotion uses 0.04 against its heavier reward landscape).
        #
        # LOWERED 0.01 -> 0.002 for the clean-slate standing reward (alive +
        # termination + torso_upright + feet_straight + action_l2). With most
        # penalty terms stripped, the reward landscape is flat, so a 0.01 bonus
        # overwhelmed the task gradient and inflated the action std without
        # bound: Loss/entropy plateaued ~57 (per-dim std ~2.6, grown from
        # init_std=1.0), meaning the policy sampled near-random clipped actions
        # and never sharpened — alive peaked ~iter 2500 then regressed and the
        # value loss diverged at the end. The fixed-gain variant amplifies this
        # (16 inert kp/kd channels get their std set purely by the entropy bonus
        # vs action_l2, and that noise feeds the last_action obs). 0.002 keeps
        # some exploration but lets std collapse toward a precise stand. If
        # entropy still won't come down, lower further (~0.0005); if the policy
        # goes deterministic too early and ignores the obs, raise it. Re-raise
        # in step when penalty/shaping terms are added back.
        #
        # KEEP THIS LOW (~0.001-0.005). This quiet-stand task REQUIRES the action
        # std to collapse so the policy can emit a precise, still pose — the
        # ``stationary_pose`` reward only pays out at near-zero joint velocity, so
        # a larger entropy bonus keeps injecting action noise, keeps joint_vel up,
        # and the still-pose objective never converges (the documented runaway:
        # Loss/entropy ~57, per-dim std ~2.6). Empirically anything higher than
        # ~0.005 has failed to converge here. So when a harder reward landscape
        # (e.g. the ``leg_posture`` anchor) won't converge, fix it on the REWARD
        # side (soften/curriculum the penalty), NOT by raising entropy.
        #
        # 0.001 -> 0.003 (2026-06-20): the bilateral_symmetry and foot_deviation
        # terms are restored to force a symmetric, flat-foot stance (the hardware
        # foot motor can't hold the asymmetric ankle positions the policy finds).
        # At 0.001 these terms caused std to collapse to ~0.04 before balance was
        # found. 0.003 holds the std floor at ~0.06-0.08, giving the policy
        # enough exploration to find the symmetric stand before locking in. If
        # std still collapses below 0.05 with eplen < 95%, raise to 0.005. If std
        # refuses to come below 0.12 and stationary_pose won't pay out, drop to
        # 0.002. NOTE: a --entropy_coef CLI flag overrides this default.
        entropy_coef=0.003,
        
        # Training Updates
        num_learning_epochs=5,   # How many times to reuse the collected data
        
        # Mini Batches:
        # With 4096 envs * 24 steps = 98,304 samples per iteration.
        # 4 mini-batches = 24,576 samples per batch. 
        # The RTX 5090 can handle this easily.
        num_mini_batches=4,
        
        # Learning Rate
        # 5e-4 (not 1e-3) because the adaptive schedule struggles to
        # outrun a value-loss spike when running with many mini-batches
        # (e.g. --num_envs 8192 --num_mini_batches 12 => 60 updates per
        # iteration). 1e-3 trained fine at the default 4096/4, but the
        # value loss diverged on the 8192/12 config (see git history for
        # the 3380-iter crash with "normal expects std >= 0.0").
        learning_rate=5.0e-4,
        schedule="adaptive",     # Lowers LR if updates are too drastic (KL divergence high)
        
        # PPO Math
        gamma=0.99,              # Discount factor (future rewards importance)
        lam=0.95,                # GAE (Generalized Advantage Estimation) lambda
        desired_kl=0.01,         # Target KL divergence for adaptive schedule
        # 0.5 (not 1.0) caps the worst single-update damage when the
        # critic momentarily learns absurd targets — keeps the actor
        # mean from drifting to ±1e6 in one step on a bad batch.
        max_grad_norm=0.5,
    )

@configclass
class BebopPPOLowLRCfg(BebopPPOBaseCfg):
    """
    Variant: Low Learning Rate.
    Use this if the 'Base' config learns to stand but then jitters/explodes later in training.

    Inherits the full base algorithm block; only the learning rate and
    minibatch count are overridden. The previous version replaced the
    whole ``algorithm`` object with a two-field cfg, which silently
    dropped ``entropy_coef``, ``gamma``, ``lam``, ``clip_param``,
    ``desired_kl``, ``max_grad_norm`` and the LR schedule.
    """
    experiment_name = "bebop_low_lr"
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.02,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class BebopPPOPushCfg(BebopPPOBaseCfg):
    """Variant for the push-recovery stand (``Isaac-BebopV2-Standing-Push-v0``).

    Identical to the base PPO block except for a higher ``entropy_coef``. The
    base was LOWERED to 0.002 for the flat, near-disturbance-free clean-slate
    stand so the std could collapse to a precise stand. Adding mid-episode
    pushes re-inflates the reward landscape (the policy must explore a family of
    recovery motions, not a single quiet pose), so it needs more exploration —
    the base value would let the actor collapse onto the quiet-stand solution
    before it ever discovers how to catch a shove. 0.01 is a middle ground
    between the quiet-stand 0.002 and the locomotion 0.04; raise toward 0.02 if
    the policy goes deterministic and stops recovering, lower if entropy / the
    action std refuses to come down (watch Policy/mean_std — it should still
    settle well below the ~0.5 of the failed run, just higher than the quiet
    stand).
    """

    experiment_name = "bebop_push"
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=0.5,
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
        # Strong entropy bonus -> keeps exploration alive while learning to walk.
        entropy_coef=0.04,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.02,
        max_grad_norm=0.5,
    )