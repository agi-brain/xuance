from . import Hyperparameter


drqn_hyperparams = [
    Hyperparameter(
        name="rnn",  # The choice of recurrent neural networks.
        type="categorical",
        distribution=["LSTM", "GRU"],
        default="LSTM"
    ),
    Hyperparameter(
        name="representation_hidden_size",  # The choice of representation network structure (for MLP).
        type="list",
        distribution=[[64, ], [128, ], [256, ], [512, ]],
        default=[128, ]
    ),
    Hyperparameter(
        name="recurrent_hidden_size",  # The hidden size of the recurrent network.
        type="list",
        distribution=[32, 64, 128, 256, 512],
        default=128
    ),
    Hyperparameter(
        name="recurrent_layer_N",  # The number of layers of the recurrent network.
        type="list",
        distribution=[1, 2, 3],
        default=1
    ),
    Hyperparameter(
        name="dropout",  # The probability of an element being zeroed.
        type="float",
        distribution=(0, 1),
        default=0
    ),
    Hyperparameter(
        name="activation",  # The choice of activation function.
        type="categorical",
        distribution=["relu", "leaky_relu", "tanh", "sigmoid"],
        default="relu"
    ),

    Hyperparameter(
        name="buffer_size",  # The size of replay buffer.
        type="int",
        distribution=(10000, 1000000),
        log=True,
        default=500000
    ),
    Hyperparameter(
        name="batch_size",  # Size of a batch data for training.
        type="int",
        distribution=[32, 64, 128],
        default=64
    ),
    Hyperparameter(
        name="learning_rate",  # The learning rate.
        type="float",
        distribution=(1e-5, 1e-2),
        log=True,
        default=1e-4
    ),
    Hyperparameter(
        name="gamma",  # The discount factor.
        type="float",
        distribution=(0.9, 0.999),
        log=False,
        default=0.99
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
        name="sync_frequency",  # Frequency to update the target network.
        type="int",
        distribution=[50, 100, 500, 1000],
        log=False,
        default=100
    ),
    Hyperparameter(
        name="training_frequency",  # Frequency to train the model when the agent interacts with the environment.
        type="int",
        distribution=[1, 10, 20, 50, 100],
        log=False,
        default=1
    ),
    Hyperparameter(
        name="start_training",  # When to start training.
        type="int",
        distribution=(0, 1000000),
        log=True,
        default=1000
    ),
    Hyperparameter(
        name="lookup_length",  # The length of history data.
        type="int",
        distribution=(1, 100),
        log=False,
        default=50
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
        name="use_obsnorm",  # Whether to use observation normalization trick.
        type="bool",
        distribution=[True, False],
        log=False,
        default=False
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
        default=False
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
