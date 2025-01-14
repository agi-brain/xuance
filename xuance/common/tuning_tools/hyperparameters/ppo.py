from . import Hyperparameter


ppo_hyperparams = [
    Hyperparameter(
        name="representation_hidden_size",  # The choice of representation network structure (for MLP).
        type="list",
        distribution=[[64, ], [128, ], [256, ], [512, ]],
        default=[128, ]
    ),
    Hyperparameter(
        name="actor_hidden_size",  # The choice of policy network structure.
        type="list",
        distribution=[[64, ], [128, ], [256, ], [512, ]],
        default=[256, ]
    ),
    Hyperparameter(
        name="critic_hidden_size",  # The choice of policy network structure.
        type="list",
        distribution=[[64, ], [128, ], [256, ], [512, ]],
        default=[256, ]
    ),
    Hyperparameter(
        name="activation",  # The choice of activation function.
        type="categorical",
        distribution=["relu", "leaky_relu", "tanh", "sigmoid"],
        default="relu"
    ),

    Hyperparameter(
        name="horizon_size",  # The horizon size for an environment.
        type="int",
        distribution=[32, 64, 128, 256, 512],
        log=False,
        default=256
    ),
    Hyperparameter(
        name="n_epochs",  # The number of epochs.
        type="int",
        distribution=[1, 4, 8, 16],
        log=False,
        default=8
    ),
    Hyperparameter(
        name="n_minibatch",  # The number of minibatchs.
        type="int",
        distribution=[1, 4, 8, 16],
        log=False,
        default=8
    ),
    Hyperparameter(
        name="learning_rate",  # The learning rate.
        type="float",
        distribution=(1e-5, 1e-2),
        log=True,
        default=1e-4
    ),

    Hyperparameter(
        name="vf_coef",  # Coefficient factor for value loss.
        type="float",
        distribution=(0.001, 0.5),
        log=False,
        default=0.25
    ),
    Hyperparameter(
        name="ent_coef",  # Coefficient factor for entropy loss.
        type="float",
        distribution=(0.001, 0.5),
        log=False,
        default=0.01
    ),
    Hyperparameter(
        name="target_kl",  # The target KL value. (For PPO-KL)
        type="float",
        distribution=(0.001, 0.5),
        log=False,
        default=0.01
    ),
    Hyperparameter(
        name="kl_coef",  # Coefficient factor for KL divergence. (For PPO-KL)
        type="float",
        distribution=(0.1, 2.0),
        log=False,
        default=1.0
    ),
    Hyperparameter(
        name="clip_range",  # The clip range for ratio. (For PPO-CLIP)
        type="float",
        distribution=(0.0, 1.0),
        log=False,
        default=0.2
    ),
    Hyperparameter(
        name="gamma",  # The discount factor.
        type="float",
        distribution=(0.9, 0.999),
        log=False,
        default=0.99
    ),
    Hyperparameter(
        name="use_gae",  # Whether to use GAE trick.
        type="bool",
        distribution=[True, False],
        log=False,
        default=True
    ),
    Hyperparameter(
        name="gae_lambda",  # The GAE lambda.
        type="float",
        distribution=(0.9, 0.999),
        log=False,
        default=0.95
    ),
    Hyperparameter(
        name="use_advnorm",  # Whether to use advantage normalization trick.
        type="bool",
        distribution=[True, False],
        log=False,
        default=True
    ),

    Hyperparameter(
        name="use_grad_clip",  # Whether to use gradient clip.
        type="bool",
        distribution=[True, False],
        log=False,
        default=True
    ),
    Hyperparameter(
        name="grad_clip_norm",  # Normalization for gradient.
        type="float",
        distribution=(0.1, 1.0),
        log=False,
        default=0.5
    ),
    Hyperparameter(
        name="use_obsnorm",  # Whether to use observation normalization trick.
        type="bool",
        distribution=[True, False],
        log=False,
        default=True
    ),
    Hyperparameter(
        name="obsnorm_range",  # The range of normalized observations.
        type="float",
        distribution=(1, 10),
        log=False,
        default=5
    ),
    Hyperparameter(
        name="use_rewnorm",  # Whether to use reward normalization trick.
        type="bool",
        distribution=[True, False],
        log=False,
        default=True
    ),
    Hyperparameter(
        name="rewnorm_range",  # The range of normalized rewards.
        type="float",
        distribution=(1, 10),
        log=False,
        default=5
    ),
    # Other hyperparameters...
]
