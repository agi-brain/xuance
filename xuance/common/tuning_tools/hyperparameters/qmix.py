from . import Hyperparameter


qmix_hyperparams = [
    Hyperparameter(
        name="representation_hidden_size",  # The choice of representation network structure (for MLP).
        type="list",
        distribution=[[64, ], [128, ], [256, ], [512, ]],
        default=[128, ]
    ),
    Hyperparameter(
        name="q_hidden_size",  # The choice of policy network structure.
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
        name="hidden_dim_mixing_net",  # The size of hidden layers for mixing network.
        type="int",
        distribution=[32, 64, 128, 256, 512, 1024],
        default=128
    ),
    Hyperparameter(
        name="hidden_dim_hyper_net",  # The size of hidden layers for hyper networks.
        type="int",
        distribution=[32, 64, 128, 256, 512, 1024],
        default=128
    ),

    Hyperparameter(
        name="buffer_size",  # The size of replay buffer.
        type="int",
        distribution=(10000, 1000000),
        log=True,
        default=100000
    ),
    Hyperparameter(
        name="batch_size",  # Size of a batch data for training.
        type="int",
        distribution=[32, 64, 128, 256, 512],
        default=256
    ),
    Hyperparameter(
        name="learning_rate",  # The learning rate.
        type="float",
        distribution=(1e-5, 1e-2),
        log=True,
        default=1e-3
    ),
    Hyperparameter(
        name="gamma",  # The discount factor.
        type="float",
        distribution=(0.9, 0.999),
        log=False,
        default=0.99
    ),
    Hyperparameter(
        name="double_q",  # The discount factor.
        type="bool",
        distribution=[True, False],
        log=False,
        default=True
    ),

    Hyperparameter(
        name="start_greedy",  # The start greedy for exploration.
        type="float",
        distribution=(0.1, 1.0),
        log=False,
        default=1.0
    ),
    Hyperparameter(
        name="end_greedy",  # The end greedy for exploration.
        type="float",
        distribution=(0.01, 0.5),  # Note: The start_greedy should be no less than end_greedy.
        log=False,
        default=0.05
    ),
    Hyperparameter(
        name="decay_step_greedy",  # Steps for greedy decay.
        type="int",
        distribution=(1000000, 20000000),
        log=True,
        default=10000000
    ),
    Hyperparameter(
        name="start_training",  # When to start training.
        type="int",
        distribution=(0, 1000000),
        log=True,
        default=1000
    ),
    Hyperparameter(
        name="training_frequency",  # Frequency to train the model when the agent interacts with the environment.
        type="int",
        distribution=[1, 10, 20, 50, 100],
        log=False,
        default=1
    ),
    Hyperparameter(
        name="sync_frequency",  # Frequency to update the target network.
        type="int",
        distribution=[50, 100, 500, 1000],
        log=False,
        default=100
    ),

    Hyperparameter(
        name="use_grad_clip",  # Whether to use gradient clip.
        type="bool",
        distribution=[True, False],
        log=False,
        default=False
    ),
    Hyperparameter(
        name="grad_clip_norm",  # Normalization for gradient.
        type="float",
        distribution=(0.1, 1.0),
        log=False,
        default=0.5
    ),
    Hyperparameter(
        name="use_parameter_sharing",  # Normalization for gradient.
        type="bool",
        distribution=[True, False],
        log=False,
        default=True
    ),
    # Other hyperparameters...
]
