from . import Hyperparameter


mappo_hyperparams = [
    Hyperparameter(
        name="representation_hidden_size",  # The choice of representation network structure (for MLP).
        type="list",
        distribution=[[64, ], [128, ], [256, ], [512, ]],
        default=[64, ]
    ),
    Hyperparameter(
        name="actor_hidden_size",  # The choice of policy network structure.
        type="list",
        distribution=[[64, ], [128, ], [256, ], [512, ]],
        default=[64, ]
    ),
    Hyperparameter(
        name="critic_hidden_size",  # The choice of policy network structure.
        type="list",
        distribution=[[64, ], [128, ], [256, ], [512, ]],
        default=[64, ]
    ),
    Hyperparameter(
        name="activation",  # The choice of activation function.
        type="categorical",
        distribution=["relu", "leaky_relu", "tanh", "sigmoid"],
        default="relu"
    ),

    Hyperparameter(
        name="buffer_size",  # The size of the replay buffer.
        type="int",
        distribution=[32, 64, 128, 256, 512],
        log=False,
        default=32
    ),
    Hyperparameter(
        name="n_epochs",  # The number of epochs.
        type="int",
        distribution=[1, 3, 5, 10],
        log=False,
        default=10
    ),
    Hyperparameter(
        name="n_minibatch",  # The number of minibatchs.
        type="int",
        distribution=[1, 3, 5, 10],
        log=False,
        default=1
    ),
    Hyperparameter(
        name="learning_rate",  # The learning rate.
        type="float",
        distribution=(1e-5, 1e-2),
        log=True,
        default=7e-4
    ),

    Hyperparameter(
        name="vf_coef",  # Coefficient factor for value loss.
        type="float",
        distribution=(0.001, 0.5),
        log=False,
        default=0.5
    ),
    Hyperparameter(
        name="ent_coef",  # Coefficient factor for entropy loss.
        type="float",
        distribution=(0.001, 0.5),
        log=False,
        default=0.01
    ),
    Hyperparameter(
        name="target_kl",  # The target KL value. (For MAPPO-KL)
        type="float",
        distribution=(0.001, 0.5),
        log=False,
        default=0.25
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
        name="use_linear_lr_decay",  # Whether to use linear learning rate decay.
        type="bool",
        distribution=[True, False],
        log=False,
        default=False
    ),
    Hyperparameter(
        name="end_factor_lr_decay",  # The end factor for the decayed learning rate.
        type="float",
        distribution=(0.0, 1.0),
        log=False,
        default=0.5
    ),
    Hyperparameter(
        name="use_global_state",  # Whether to use global state to replace merged observations.
        type="bool",
        distribution=[True, False],
        log=False,
        default=False
    ),
    Hyperparameter(
        name="use_value_clip",  # Limit the value range.
        type="bool",
        distribution=[True, False],
        log=False,
        default=True
    ),
    Hyperparameter(
        name="value_clip_range",  # The value clip range.
        type="float",
        distribution=(0.0, 10.0),
        log=False,
        default=0.2
    ),
    Hyperparameter(
        name="use_value_norm",  # Use running mean and std to normalize rewards.
        type="bool",
        distribution=[True, False],
        log=False,
        default=True
    ),
    Hyperparameter(
        name="use_huber_loss",  # True: use huber loss; False: use MSE loss.
        type="bool",
        distribution=[True, False],
        log=False,
        default=True
    ),
    Hyperparameter(
        name="huber_delta",  # The threshold at which to change between delta-scaled L1 and L2 loss. (For huber loss).
        type="float",
        distribution=(0.0, 20.0),
        log=False,
        default=10.0
    ),
    Hyperparameter(
        name="use_adv_norm",  # Whether to use advantage normalization.
        type="bool",
        distribution=[True, False],
        log=False,
        default=True
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
        distribution=(0.0, 0.999),
        log=False,
        default=0.95
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
        distribution=(0.1, 10.0),
        log=False,
        default=10.0
    ),
    Hyperparameter(
        name="use_parameter_sharing",  # Whether to use parameter sharing for all agents' policies.
        type="bool",
        distribution=[True, False],
        log=False,
        default=True
    ),
    # Other hyperparameters...
]
