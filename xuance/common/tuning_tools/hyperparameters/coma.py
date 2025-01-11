from . import Hyperparameter


coma_hyperparams = [
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
        default=[128, ]
    ),
    Hyperparameter(
        name="critic_hidden_size",  # The choice of policy network structure.
        type="list",
        distribution=[[64, ], [128, ], [256, ], [512, ]],
        default=[128, ]
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
        default=1
    ),
    Hyperparameter(
        name="n_minibatch",  # The number of minibatchs.
        type="int",
        distribution=[1, 3, 5, 10],
        log=False,
        default=1
    ),
    Hyperparameter(
        name="learning_rate_actor",  # The learning rate for actor.
        type="float",
        distribution=(1e-5, 1e-2),
        log=True,
        default=7e-4
    ),
    Hyperparameter(
        name="learning_rate_critic",  # The learning rate for critic.
        type="float",
        distribution=(1e-5, 1e-2),
        log=True,
        default=7e-4
    ),

    Hyperparameter(
        name="start_greedy",  # The start greedy for exploration.
        type="float",
        distribution=(0.1, 1.0),
        log=False,
        default=0.5
    ),
    Hyperparameter(
        name="end_greedy",  # The end greedy for exploration.
        type="float",
        distribution=(0.01, 0.5),  # Note: The start_greedy should be no less than end_greedy.
        log=False,
        default=0.01
    ),
    Hyperparameter(
        name="decay_step_greedy",  # Steps for greedy decay.
        type="int",
        distribution=(1000000, 20000000),
        log=True,
        default=10000000
    ),
    Hyperparameter(
        name="sync_frequency",  # Frequency to update the target network.
        type="int",
        distribution=[50, 100, 500, 1000],
        log=False,
        default=200
    ),

    Hyperparameter(
        name="vf_coef",  # Coefficient factor for value loss.
        type="float",
        distribution=(0.001, 0.5),
        log=False,
        default=0.1
    ),
    Hyperparameter(
        name="ent_coef",  # Coefficient factor for entropy loss.
        type="float",
        distribution=(0.001, 0.5),
        log=False,
        default=0.01
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
        default=0.8
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
