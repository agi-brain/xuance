from . import Hyperparameter


iddpg_hyperparams = [
    Hyperparameter(
        name="actor_hidden_size",  # The choice of actor network structure.
        type="list",
        distribution=[[64, 64], [128, 128], [256, 256], [512, 512]],
        default=[64, 64]
    ),
    Hyperparameter(
        name="critic_hidden_size",  # The choice of critic network structure.
        type="list",
        distribution=[[64, 64], [128, 128], [256, 256], [512, 512]],
        default=[64, 64]
    ),
    Hyperparameter(
        name="activation",  # The choice of activation function.
        type="categorical",
        distribution=["relu", "leaky_relu", "tanh", "sigmoid"],
        default="leaky_relu"
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
        name="learning_rate_actor",  # The learning rate.
        type="float",
        distribution=(1e-5, 1e-2),
        log=True,
        default=1e-2
    ),
    Hyperparameter(
        name="learning_rate_critic",  # The learning rate.
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
        default=0.95
    ),
    Hyperparameter(
        name="tau",  # The discount factor.
        type="float",
        distribution=(0.0001, 0.5),
        log=True,
        default=0.001
    ),

    Hyperparameter(
        name="start_noise",  # The start greedy for exploration.
        type="float",
        distribution=(0.1, 1.0),
        log=False,
        default=1.0
    ),
    Hyperparameter(
        name="end_noise",  # The end greedy for exploration.
        type="float",
        distribution=(0.001, 0.5),  # Note: The start_greedy should be no less than end_greedy.
        log=False,
        default=0.01
    ),
    Hyperparameter(
        name="sigma",  # Random noise for continuous actions.
        type="float",
        distribution=(0.001, 0.5),  # Note: The start_greedy should be no less than end_greedy.
        log=False,
        default=0.1
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
        default=25
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
