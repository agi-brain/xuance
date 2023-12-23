Configs
======================

XuanCe provides a structured way to manage configurations for various DRL scenarios, 
making it easy to experiment with different setups

.. raw:: html

   <br><hr>
   
Basic Configurations
--------------------------

The basic parameter configuration is stored in the "xuance/config/basic.yaml" file, as shown below:

.. code-block:: yaml

    dl_toolbox: "torch"  # The deep learning toolbox. Choices: "torch", "mindspore", "tensorflow"

    project_name: "XuanCe_Benchmark"
    logger: "tensorboard"  # Choices: "tensorboard", "wandb".
    wandb_user_name: "your_user_name"

    parallels: 10
    seed: 2910
    render: True
    render_mode: 'rgb_array' # Choices: 'human', 'rgb_array'.
    test_mode: False
    test_steps: 2000

    device: "cpu"


It should be noted that the value of the `device` variable in the `basic.yaml` file varies depending on the specific deep learning framework, as outlined below:

| - PyTorch: "cpu", "cuda:0";
| - TensorFlow: "cpu"/"CPU", "gpu"/"GPU";
| - MindSpore: "CPU", "GPU", "Ascend", "Davinci"。

.. raw:: html

   <br><hr>
   
Algorithm Configurations for Different Tasks
------------------------------------------------------

As an example, taking the parameter configuration for the DQN algorithm in the Atari environment, 
in addition to the basic parameter configuration, the algorithm-specific parameters are stored in the "xuance/configs/dqn/atari.yaml" file. 

Due to the presence of over 60 different scenarios in the Atari environment, 
where the scenarios are relatively consistent with variations only in tasks, 
a single default parameter configuration file is sufficient.

For environments with significant scene variations, such as the "CarRacing-v2" and "LunarLander" scenarios in the "Box2D" environment, 
the former has a state input of a 96x96x3 RGB image, while the latter consists of an 8-dimensional vector. 
Therefore, the DQN algorithm parameter configurations for these two scenarios are stored in the following two files:

    * xuance/configs/dqn/box2d/CarRacing-v2.yaml
    * xuance/configs/dqn/box2d/LunarLander-v2.yaml

Within the following content, we provid the preset arguments for each implementation that can be run by following the steps in :doc:`Quick Start </documents/usage/basic_usage>`.

.. raw:: html

   <br><hr>

DQN-based Implementations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. tabs::

    .. group-tab:: DQN

        .. tabs::

            .. group-tab:: Classic Control

                .. tabs::

                    .. group-tab:: CartPole-v1

                        .. code-block:: yaml

                            agent: "DQN"
                            env_name: "Classic Control"
                            env_id: "CartPole-v1"
                            vectorize: "Dummy_Gym"
                            policy: "Basic_Q_network"
                            representation: "Basic_MLP"
                            runner: "DRL"

                            representation_hidden_size: [128,]
                            q_hidden_size: [128,]
                            activation: 'ReLU'

                            seed: 1
                            parallels: 10
                            n_size: 10000
                            batch_size: 256
                            learning_rate: 0.001
                            gamma: 0.99

                            start_greedy: 0.5
                            end_greedy: 0.01
                            decay_step_greedy: 10000
                            sync_frequency: 50
                            training_frequency: 1
                            running_steps: 200000  # 200k
                            start_training: 1000

                            use_obsnorm: False
                            use_rewnorm: False
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 20000
                            test_episode: 1
                            log_dir: "./logs/dqn/"
                            model_dir: "./models/dqn/"

                    .. group-tab:: Acrobot-v1

                        .. code-block:: yaml

                            agent: "DQN"
                            env_name: "Classic Control"
                            env_id: "Acrobot-v1"
                            vectorize: "Dummy_Gym"
                            policy: "Basic_Q_network"
                            representation: "Basic_MLP"
                            runner: "DRL"

                            representation_hidden_size: [128,]
                            q_hidden_size: [128,]
                            activation: 'ReLU'

                            seed: 1
                            parallels: 10
                            n_size: 10000
                            batch_size: 256
                            learning_rate: 0.001
                            gamma: 0.99

                            start_greedy: 0.5
                            end_greedy: 0.01
                            decay_step_greedy: 10000
                            sync_frequency: 50
                            training_frequency: 1
                            running_steps: 200000  # 200k
                            start_training: 1000

                            use_obsnorm: False
                            use_rewnorm: False
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 20000
                            test_episode: 1
                            log_dir: "./logs/dqn/"
                            model_dir: "./models/dqn/"

                    .. group-tab:: MountainCar-v0

                        .. code-block:: yaml

                            agent: "DQN"
                            env_name: "Classic Control"
                            env_id: "MountainCar-v0"
                            vectorize: "Dummy_Gym"
                            policy: "Basic_Q_network"
                            representation: "Basic_MLP"
                            runner: "DRL"

                            representation_hidden_size: [256, ]
                            q_hidden_size: [256, ]
                            activation: 'LeakyReLU'

                            seed: 1
                            parallels: 10
                            n_size: 20000
                            batch_size: 256
                            learning_rate: 0.1
                            gamma: 0.99

                            start_greedy: 1.0
                            end_greedy: 0.01
                            sync_frequency: 200
                            training_frequency: 2
                            running_steps: 2000000  # 2M
                            start_training: 1000

                            use_obsnorm: False
                            use_rewnorm: False
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 5
                            log_dir: "./logs/dqn/"
                            model_dir: "./models/dqn/"

            
            .. group-tab:: Box2D

                .. tabs::

                    .. group-tab:: CarRacing-v2

                        .. code-block:: yaml

                            agent: "DQN"
                            env_name: "Box2D"
                            env_id: "CarRacing-v2"
                            vectorize: "Dummy_Gym"
                            policy: "Basic_Q_network"
                            representation: "Basic_CNN"
                            runner: "DRL"

                            # the following three arguments are for "Basic_CNN" representation.
                            filters: [16, 16, 32]  #  [16, 16, 32, 32]
                            kernels: [8, 4, 3]  # [8, 6, 4, 4]
                            strides: [4, 2, 1]  # [2, 2, 2, 2]

                            q_hidden_size: [512,]
                            activation: 'ReLU'

                            seed: 1
                            parallels: 2
                            n_size: 10000
                            batch_size: 32
                            learning_rate: 0.0001
                            gamma: 0.99

                            start_greedy: 0.5
                            end_greedy: 0.01
                            decay_step_greedy: 50000
                            sync_frequency: 500
                            training_frequency: 1
                            running_steps: 2000000
                            start_training: 1000

                            use_obsnorm: False
                            use_rewnorm: False
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 100000
                            test_episode: 1
                            log_dir: "./logs/dqn/"
                            model_dir: "./models/dqn/"

                    .. group-tab:: LunarLander-v2

                        .. code-block:: yaml

                            agent: "DQN"
                            env_name: "Box2D"
                            env_id: "LunarLander-v2"
                            vectorize: "Dummy_Gym"
                            policy: "Basic_Q_network"
                            representation: "Basic_MLP"
                            runner: "DRL"

                            representation_hidden_size: [128,]
                            q_hidden_size: [128,]
                            activation: 'ReLU'

                            seed: 1
                            parallels: 10
                            n_size: 10000
                            batch_size: 256
                            learning_rate: 0.001
                            gamma: 0.99

                            start_greedy: 0.5
                            end_greedy: 0.01
                            decay_step_greedy: 10000
                            sync_frequency: 50
                            training_frequency: 1
                            running_steps: 200000
                            start_training: 1000

                            use_obsnorm: False
                            use_rewnorm: False
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 1
                            log_dir: "./logs/dqn/"
                            model_dir: "./models/dqn/"
            
            .. group-tab:: Atari

                .. code-block:: yaml

                    agent: "DQN"
                    vectorize: "Dummy_Atari"
                    env_name: "Atari"
                    env_id: "ALE/Breakout-v5"
                    obs_type: "grayscale"  # choice for Atari env: ram, rgb, grayscale
                    img_size: [84, 84]  # default is 210 x 160 in gym[Atari]
                    num_stack: 4  # frame stack trick
                    frame_skip: 4  # frame skip trick
                    noop_max: 30  # Do no-op action for a number of steps in [1, noop_max].
                    policy: "Basic_Q_network"
                    representation: "Basic_CNN"

                    # the following three arguments are for "Basic_CNN" representation.
                    filters: [32, 64, 64]  #  [16, 16, 32, 32]
                    kernels: [8, 4, 3]  # [8, 6, 4, 4]
                    strides: [4, 2, 1]  # [2, 2, 2, 2]

                    q_hidden_size: [512, ]
                    activation: "ReLU"

                    seed: 1069
                    parallels: 5
                    n_size: 100000
                    batch_size: 32  # 64
                    learning_rate: 0.0001
                    gamma: 0.99

                    start_greedy: 0.5
                    end_greedy: 0.05
                    decay_step_greedy: 1000000  # 1M
                    sync_frequency: 500
                    training_frequency: 1
                    running_steps: 50000000  # 50M
                    start_training: 10000

                    use_obsnorm: False
                    use_rewnorm: False
                    obsnorm_range: 5
                    rewnorm_range: 5

                    test_steps: 10000
                    eval_interval: 500000
                    test_episode: 3
                    log_dir: "./logs/dqn/"
                    model_dir: "./models/dqn/"

    .. group-tab:: C51

        .. tabs::

            .. group-tab:: Classic Control

                .. tabs::

                    .. group-tab:: CartPole-v1

                        .. code-block:: yaml

                            agent: "C51DQN"
                            env_name: "Classic Control"
                            env_id: "CartPole-v1"
                            vectorize: "Dummy_Gym"
                            policy: "C51_Q_network"
                            representation: "Basic_MLP"
                            runner: "DRL"

                            representation_hidden_size: [128,]
                            q_hidden_size: [128,]
                            activation: 'ReLU'

                            seed: 1
                            parallels: 10
                            n_size: 20000
                            batch_size: 256
                            learning_rate: 0.001
                            gamma: 0.99
                            vmin: 0
                            vmax: 200
                            atom_num: 51

                            start_greedy: 0.5
                            end_greedy: 0.01
                            decay_step_greedy: 10000
                            sync_frequency: 100
                            training_frequency: 1
                            running_steps: 200000
                            start_training: 1000

                            use_obsnorm: False
                            use_rewnorm: False
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 1
                            log_dir: "./logs/c51/"
                            model_dir: "./models/c51/"


                    .. group-tab:: Acrobot-v1

                        .. code-block:: yaml

                            agent: "C51DQN"
                            env_name: "Classic Control"
                            env_id: "Acrobot-v1"
                            vectorize: "Dummy_Gym"
                            policy: "C51_Q_network"
                            representation: "Basic_MLP"
                            runner: "DRL"

                            representation_hidden_size: [128,]
                            q_hidden_size: [128,]
                            activation: 'ReLU'

                            seed: 1
                            parallels: 10
                            n_size: 20000
                            batch_size: 256
                            learning_rate: 0.001
                            gamma: 0.99
                            vmin: 0
                            vmax: 200
                            atom_num: 51

                            start_greedy: 0.5
                            end_greedy: 0.01
                            decay_step_greedy: 10000
                            sync_frequency: 100
                            training_frequency: 1
                            running_steps: 300000
                            start_training: 1000

                            use_obsnorm: False
                            use_rewnorm: False
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 1
                            log_dir: "./logs/c51/"
                            model_dir: "./models/c51/"


                    .. group-tab:: MountainCar-v0

                        .. code-block:: yaml

                            agent: "C51DQN"
                            env_name: "Classic Control"
                            env_id: "MountainCar-v0"
                            vectorize: "Dummy_Gym"
                            policy: "C51_Q_network"
                            representation: "Basic_MLP"
                            runner: "DRL"

                            representation_hidden_size: [128,]
                            q_hidden_size: [128,]
                            activation: 'ReLU'

                            seed: 1
                            parallels: 10
                            n_size: 20000
                            batch_size: 256
                            learning_rate: 0.001
                            gamma: 0.99
                            vmin: 0
                            vmax: 200
                            atom_num: 51

                            start_greedy: 0.5
                            end_greedy: 0.01
                            decay_step_greedy: 10000
                            sync_frequency: 100
                            training_frequency: 1
                            running_steps: 200000
                            start_training: 1000

                            use_obsnorm: False
                            use_rewnorm: False
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 1
                            log_dir: "./logs/c51/"
                            model_dir: "./models/c51/"

            
            .. group-tab:: Box2D

                .. tabs::

                    .. group-tab:: CarRacing-v2

                        .. code-block:: yaml

                            agent: "C51DQN"
                            env_name: "Box2D"
                            env_id: "CarRacing-v2"
                            vectorize: "Dummy_Gym"
                            policy: "C51_Q_network"
                            representation: "Basic_CNN"
                            runner: "DRL"

                            # the following three arguments are for "Basic_CNN" representation.
                            filters: [16, 16, 32]  #  [16, 16, 32, 32]
                            kernels: [8, 4, 3]  # [8, 6, 4, 4]
                            strides: [4, 2, 1]  # [2, 2, 2, 2]

                            q_hidden_size: [512,]
                            activation: 'ReLU'

                            seed: 1
                            parallels: 2
                            n_size: 10000
                            batch_size: 32
                            learning_rate: 0.0001
                            gamma: 0.99
                            vmin: 0
                            vmax: 200
                            atom_num: 51

                            start_greedy: 0.5
                            end_greedy: 0.01
                            decay_step_greedy: 500000
                            sync_frequency: 500
                            training_frequency: 1
                            running_steps: 200000
                            start_training: 1000

                            use_obsnorm: False
                            use_rewnorm: False
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 5000
                            test_episode: 1
                            log_dir: "./logs/c51/"
                            model_dir: "./models/c51/"


                    .. group-tab:: LunarLander-v2

                        .. code-block:: yaml

                            agent: "C51DQN"
                            env_name: "Box2D"
                            env_id: "LunarLander-v2"
                            vectorize: "Dummy_Gym"
                            policy: "C51_Q_network"
                            representation: "Basic_MLP"
                            runner: "DRL"

                            representation_hidden_size: [128,]
                            q_hidden_size: [128,]
                            activation: 'ReLU'

                            seed: 1
                            parallels: 10
                            n_size: 20000
                            batch_size: 256
                            learning_rate: 0.001
                            gamma: 0.99
                            vmin: 0
                            vmax: 200
                            atom_num: 51

                            start_greedy: 0.5
                            end_greedy: 0.01
                            decay_step_greedy: 10000
                            sync_frequency: 100
                            training_frequency: 1
                            running_steps: 200000
                            start_training: 1000

                            use_obsnorm: False
                            use_rewnorm: False
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 1
                            log_dir: "./logs/c51/"
                            model_dir: "./models/c51/"

            
            .. group-tab:: Atari

                .. code-block:: yaml

                    agent: "C51DQN"
                    vectorize: "Dummy_Atari"
                    env_name: "Atari"
                    env_id: "ALE/Breakout-v5"
                    obs_type: "grayscale"  # choice for Atari env: ram, rgb, grayscale
                    img_size: [84, 84]  # default is 210 x 160 in gym[Atari]
                    num_stack: 4  # frame stack trick
                    frame_skip: 4  # frame skip trick
                    noop_max: 30  # Do no-op action for a number of steps in [1, noop_max].
                    policy: "C51_Q_network"
                    representation: "Basic_CNN"
                    runner: "DRL"

                    # the following three arguments are for "Basic_CNN" representation.
                    filters: [32, 64, 64]  #  [16, 16, 32, 32]
                    kernels: [8, 4, 3]  # [8, 6, 4, 4]
                    strides: [4, 2, 1]  # [2, 2, 2, 2]

                    q_hidden_size: [512, ]
                    activation: "ReLU"

                    seed: 1069
                    parallels: 5
                    n_size: 100000
                    batch_size: 32  # 64
                    learning_rate: 0.0001
                    gamma: 0.99
                    vmin: 0
                    vmax: 200
                    atom_num: 51

                    start_greedy: 0.5
                    end_greedy: 0.05
                    decay_step_greedy: 1000000  # 1M
                    sync_frequency: 500
                    training_frequency: 1
                    running_steps: 50000000  # 50M
                    start_training: 10000

                    use_obsnorm: False
                    use_rewnorm: False
                    obsnorm_range: 5
                    rewnorm_range: 5

                    test_steps: 10000
                    eval_interval: 500000
                    test_episode: 3
                    log_dir: "./logs/c51/"
                    model_dir: "./models/c51/"


    .. group-tab:: DoubleDQN

        .. tabs::

            .. group-tab:: Classic Control

                .. tabs::

                    .. group-tab:: CartPole-v1

                        .. code-block:: yaml

                            agent: "DDQN"
                            env_name: "Classic Control"
                            env_id: "CartPole-v1"
                            vectorize: "Dummy_Gym"
                            policy: "Basic_Q_network"
                            representation: "Basic_MLP"
                            runner: "DRL"

                            representation_hidden_size: [128,]
                            q_hidden_size: [128,]
                            activation: 'ReLU'

                            seed: 1
                            parallels: 10
                            n_size: 20000
                            batch_size: 128
                            learning_rate: 0.001
                            gamma: 0.99

                            start_greedy: 0.5
                            end_greedy: 0.01
                            decay_step_greedy: 10000
                            sync_frequency: 100
                            training_frequency: 1
                            running_steps: 300000
                            start_training: 1000

                            use_obsnorm: False
                            use_rewnorm: False
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 1
                            log_dir: "./logs/ddqn/"
                            model_dir: "./models/ddqn/"

                    .. group-tab:: Acrobot-v1

                        .. code-block:: yaml

                            agent: "DDQN"
                            env_name: "Classic Control"
                            env_id: "Acrobot-v1"
                            vectorize: "Dummy_Gym"
                            policy: "Basic_Q_network"
                            representation: "Basic_MLP"
                            runner: "DRL"

                            representation_hidden_size: [128,]
                            q_hidden_size: [128,]
                            activation: 'ReLU'

                            seed: 1
                            parallels: 10
                            n_size: 20000
                            batch_size: 128
                            learning_rate: 0.001
                            gamma: 0.99

                            start_greedy: 0.5
                            end_greedy: 0.01
                            decay_step_greedy: 10000
                            sync_frequency: 100
                            training_frequency: 1
                            running_steps: 300000
                            start_training: 1000

                            use_obsnorm: False
                            use_rewnorm: False
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 1
                            log_dir: "./logs/ddqn/"
                            model_dir: "./models/ddqn/"

                    .. group-tab:: MountainCar-v0

                        .. code-block:: yaml

                            agent: "DDQN"
                            env_name: "Classic Control"
                            env_id: "MountainCar-v0"
                            vectorize: "Dummy_Gym"
                            policy: "Basic_Q_network"
                            representation: "Basic_MLP"
                            runner: "DRL"

                            representation_hidden_size: [128,]
                            q_hidden_size: [128,]
                            activation: 'ReLU'

                            seed: 1
                            parallels: 10
                            n_size: 20000
                            batch_size: 128
                            learning_rate: 0.001
                            gamma: 0.99

                            start_greedy: 0.5
                            end_greedy: 0.01
                            decay_step_greedy: 10000
                            sync_frequency: 100
                            training_frequency: 1
                            running_steps: 300000
                            start_training: 1000

                            use_obsnorm: False
                            use_rewnorm: False
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 1
                            log_dir: "./logs/ddqn/"
                            model_dir: "./models/ddqn/"


            
            .. group-tab:: Box2D

                .. tabs::

                    .. group-tab:: CarRacing-v2

                        .. code-block:: yaml

                            agent: "DDQN"
                            env_name: "Box2D"
                            env_id: "CarRacing-v2"
                            vectorize: "Dummy_Gym"
                            policy: "Basic_Q_network"
                            representation: "Basic_CNN"
                            runner: "DRL"

                            # the following three arguments are for "Basic_CNN" representation.
                            filters: [16, 16, 32]  #  [16, 16, 32, 32]
                            kernels: [8, 4, 3]  # [8, 6, 4, 4]
                            strides: [4, 2, 1]  # [2, 2, 2, 2]

                            q_hidden_size: [512,]
                            activation: 'ReLU'

                            seed: 1
                            parallels: 2
                            n_size: 10000
                            batch_size: 32
                            learning_rate: 0.0001
                            gamma: 0.99

                            start_greedy: 0.5
                            end_greedy: 0.01
                            decay_step_greedy: 50000
                            sync_frequency: 500
                            training_frequency: 1
                            running_steps: 2000000
                            start_training: 1000

                            use_obsnorm: False
                            use_rewnorm: False
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 100000
                            test_episode: 1
                            log_dir: "./logs/ddqn/"
                            model_dir: "./models/ddqn/"


                    .. group-tab:: LunarLander-v2

                        .. code-block:: yaml

                            agent: "DDQN"
                            env_name: "Box2D"
                            env_id: "LunarLander-v2"
                            vectorize: "Dummy_Gym"
                            policy: "Basic_Q_network"
                            representation: "Basic_MLP"
                            runner: "DRL"

                            representation_hidden_size: [128,]
                            q_hidden_size: [128,]
                            activation: 'ReLU'

                            seed: 1
                            parallels: 10
                            n_size: 10000
                            batch_size: 256
                            learning_rate: 0.001
                            gamma: 0.99

                            start_greedy: 0.5
                            end_greedy: 0.01
                            decay_step_greedy: 10000
                            sync_frequency: 50
                            training_frequency: 1
                            running_steps: 300000
                            start_training: 1000

                            use_obsnorm: False
                            use_rewnorm: False
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 1
                            log_dir: "./logs/ddqn/"
                            model_dir: "./models/ddqn/"

            
            .. group-tab:: Atari

                .. code-block:: yaml

                    agent: "DDQN"
                    vectorize: "Dummy_Atari"
                    env_name: "Atari"
                    env_id: "ALE/Breakout-v5"
                    obs_type: "grayscale"  # choice for Atari env: ram, rgb, grayscale
                    img_size: [84, 84]  # default is 210 x 160 in gym[Atari]
                    num_stack: 4  # frame stack trick
                    frame_skip: 4  # frame skip trick
                    noop_max: 30  # Do no-op action for a number of steps in [1, noop_max].
                    policy: "Basic_Q_network"
                    representation: "Basic_CNN"
                    runner: "DRL"

                    # the following three arguments are for "Basic_CNN" representation.
                    filters: [32, 64, 64]  #  [16, 16, 32, 32]
                    kernels: [8, 4, 3]  # [8, 6, 4, 4]
                    strides: [4, 2, 1]  # [2, 2, 2, 2]

                    q_hidden_size: [512, ]
                    activation: "ReLU"

                    seed: 1069
                    parallels: 5
                    n_size: 100000
                    batch_size: 32
                    learning_rate: 0.0001
                    gamma: 0.99

                    start_greedy: 0.5
                    end_greedy: 0.05
                    decay_step_greedy: 1000000  # 1M
                    sync_frequency: 500
                    training_frequency: 1
                    running_steps: 50000000  # 50M
                    start_training: 10000

                    use_obsnorm: False
                    use_rewnorm: False
                    obsnorm_range: 5
                    rewnorm_range: 5

                    test_steps: 10000
                    eval_interval: 500000
                    test_episode: 3
                    log_dir: "./logs/ddqn/"
                    model_dir: "./models/ddqn/"

    .. group-tab:: DuelingDQN

        .. tabs::

            .. group-tab:: Classic Control

                .. tabs::

                    .. group-tab:: CartPole-v1

                        .. code-block:: yaml

                            agent: "Duel_DQN"
                            env_name: "Classic Control"
                            env_id: "CartPole-v1"
                            vectorize: "Dummy_Gym"
                            policy: "Duel_Q_network"
                            representation: "Basic_MLP"
                            runner: "DRL"

                            representation_hidden_size: [128, ]
                            q_hidden_size: [128, ]
                            activation: 'ReLU'

                            seed: 1
                            parallels: 10
                            n_size: 10000
                            batch_size: 256
                            learning_rate: 0.001
                            gamma: 0.99

                            start_greedy: 0.5
                            end_greedy: 0.01
                            decay_step_greedy: 20000
                            sync_frequency: 50
                            training_frequency: 1
                            running_steps: 500000
                            start_training: 1000

                            use_obsnorm: False
                            use_rewnorm: False
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 1
                            log_dir: "./logs/dueldqn/"
                            model_dir: "./models/dueldqn/"


                    .. group-tab:: Acrobot-v1

                        .. code-block:: yaml

                            agent: "Duel_DQN"
                            env_name: "Classic Control"
                            env_id: "Acrobot-v1"
                            vectorize: "Dummy_Gym"
                            policy: "Duel_Q_network"
                            representation: "Basic_MLP"
                            runner: "DRL"

                            representation_hidden_size: [128, ]
                            q_hidden_size: [128, ]
                            activation: 'ReLU'

                            seed: 1
                            parallels: 10
                            n_size: 10000
                            batch_size: 256
                            learning_rate: 0.001
                            gamma: 0.99

                            start_greedy: 0.5
                            end_greedy: 0.01
                            decay_step_greedy: 20000
                            sync_frequency: 50
                            training_frequency: 1
                            running_steps: 500000
                            start_training: 1000

                            use_obsnorm: False
                            use_rewnorm: False
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 1
                            log_dir: "./logs/dueldqn/"
                            model_dir: "./models/dueldqn/"


                    .. group-tab:: MountainCar-v0

                        .. code-block:: yaml

                            agent: "Duel_DQN"
                            env_name: "Classic Control"
                            env_id: "MountainCar-v0"
                            vectorize: "Dummy_Gym"
                            policy: "Duel_Q_network"
                            representation: "Basic_MLP"
                            runner: "DRL"

                            representation_hidden_size: [128, ]
                            q_hidden_size: [128, ]
                            activation: 'ReLU'

                            seed: 1
                            parallels: 10
                            n_size: 10000
                            batch_size: 256
                            learning_rate: 0.0001
                            gamma: 0.99

                            start_greedy: 0.5
                            end_greedy: 0.01
                            decay_step_greedy: 20000
                            sync_frequency: 50
                            training_frequency: 1
                            running_steps: 300000
                            start_training: 1000

                            use_obsnorm: False
                            use_rewnorm: False
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 1
                            log_dir: "./logs/dueldqn/"
                            model_dir: "./models/dueldqn/"

            
            .. group-tab:: Box2D

                .. tabs::

                    .. group-tab:: CarRacing-v2

                        .. code-block:: yaml

                            agent: "Duel_DQN"
                            env_name: "Box2D"
                            env_id: "CarRacing-v2"
                            vectorize: "Dummy_Gym"
                            policy: "Duel_Q_network"
                            representation: "Basic_CNN"
                            runner: "DRL"

                            # the following three arguments are for "Basic_CNN" representation.
                            filters: [16, 16, 32]  #  [16, 16, 32, 32]
                            kernels: [8, 4, 3]  # [8, 6, 4, 4]
                            strides: [4, 2, 1]  # [2, 2, 2, 2]

                            q_hidden_size: [512,]
                            activation: 'ReLU'

                            seed: 1
                            parallels: 2
                            n_size: 10000
                            batch_size: 32
                            learning_rate: 0.0001
                            gamma: 0.99

                            start_greedy: 0.5
                            end_greedy: 0.01
                            decay_step_greedy: 50000
                            sync_frequency: 500
                            training_frequency: 1
                            running_steps: 2000000
                            start_training: 1000

                            use_obsnorm: False
                            use_rewnorm: False
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 100000
                            test_episode: 1
                            log_dir: "./logs/dueldqn/"
                            model_dir: "./models/dueldqn/"


                    .. group-tab:: LunarLander-v2

                        .. code-block:: yaml

                            agent: "Duel_DQN"
                            env_name: "Box2D"
                            env_id: "LunarLander-v2"
                            vectorize: "Dummy_Gym"
                            policy: "Duel_Q_network"
                            representation: "Basic_MLP"
                            runner: "DRL"

                            representation_hidden_size: [128, ]
                            q_hidden_size: [128, ]
                            activation: 'ReLU'

                            seed: 1
                            parallels: 10
                            n_size: 10000
                            batch_size: 256
                            learning_rate: 0.001
                            gamma: 0.99

                            start_greedy: 0.5
                            end_greedy: 0.01
                            decay_step_greedy: 20000
                            sync_frequency: 50
                            training_frequency: 1
                            running_steps: 500000
                            start_training: 1000

                            use_obsnorm: False
                            use_rewnorm: False
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 1
                            log_dir: "./logs/dueldqn/"
                            model_dir: "./models/dueldqn/"

            
            .. group-tab:: Atari

                .. code-block:: yaml

                    agent: "Duel_DQN"
                    vectorize: "Dummy_Atari"
                    env_name: "Atari"
                    env_id: "ALE/Breakout-v5"
                    obs_type: "grayscale"  # choice for Atari env: ram, rgb, grayscale
                    img_size: [84, 84]  # default is 210 x 160 in gym[Atari]
                    num_stack: 4  # frame stack trick
                    frame_skip: 4  # frame skip trick
                    noop_max: 30  # Do no-op action for a number of steps in [1, noop_max].
                    policy: "Duel_Q_network"
                    representation: "Basic_CNN"
                    runner: "DRL"

                    # the following three arguments are for "Basic_CNN" representation.
                    filters: [32, 64, 64]  #  [16, 16, 32, 32]
                    kernels: [8, 4, 3]  # [8, 6, 4, 4]
                    strides: [4, 2, 1]  # [2, 2, 2, 2]

                    q_hidden_size: [512, ]
                    activation: "ReLU"

                    seed: 1069
                    parallels: 5
                    n_size: 100000
                    batch_size: 32  # 64
                    learning_rate: 0.0001
                    gamma: 0.99

                    start_greedy: 0.5
                    end_greedy: 0.05
                    decay_step_greedy: 1000000  # 1M
                    sync_frequency: 500
                    training_frequency: 1
                    running_steps: 50000000  # 50M
                    start_training: 10000

                    use_obsnorm: False
                    use_rewnorm: False
                    obsnorm_range: 5
                    rewnorm_range: 5

                    test_steps: 10000
                    save_model_frequency: 500000
                    test_episode: 1
                    log_dir: "./logs/dueldqn/"
                    model_dir: "./models/dueldqn/"


    .. group-tab:: NoisyDQN

        .. tabs::

            .. group-tab:: Classic Control

                .. tabs::

                    .. group-tab:: CartPole-v1

                        .. code-block:: yaml

                            agent: "NoisyDQN"
                            env_name: "Classic Control"
                            env_id: "CartPole-v1"
                            vectorize: "Dummy_Gym"
                            policy: "Noisy_Q_network"
                            representation: "Basic_MLP"
                            runner: "DRL"

                            representation_hidden_size: [128,]
                            q_hidden_size: [128,]
                            activation: 'ReLU'

                            seed: 1
                            parallels: 10
                            n_size: 20000
                            batch_size: 128
                            learning_rate: 0.001
                            gamma: 0.99

                            start_noise: 0.05
                            end_noise: 0.0
                            decay_step_noise: 50000
                            sync_frequency: 100
                            training_frequency: 2
                            running_steps: 500000
                            start_training: 1000

                            use_obsnorm: False
                            use_rewnorm: False
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 1
                            log_dir: "./logs/noisy_dqn/"
                            model_dir: "./models/noisy_dqn/"


                    .. group-tab:: Acrobot-v1

                        .. code-block:: yaml

                            agent: "NoisyDQN"
                            env_name: "Classic Control"
                            env_id: "Acrobot-v1"
                            vectorize: "Dummy_Gym"
                            policy: "Noisy_Q_network"
                            representation: "Basic_MLP"
                            runner: "DRL"

                            representation_hidden_size: [128,]
                            q_hidden_size: [128,]
                            activation: 'ReLU'

                            seed: 1
                            parallels: 10
                            n_size: 20000
                            batch_size: 128
                            learning_rate: 0.001
                            gamma: 0.99

                            start_noise: 0.05
                            end_noise: 0.0
                            decay_step_noise: 50000
                            sync_frequency: 100
                            training_frequency: 2
                            running_steps: 500000
                            start_training: 1000

                            use_obsnorm: False
                            use_rewnorm: False
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 1
                            log_dir: "./logs/noisy_dqn/"
                            model_dir: "./models/noisy_dqn/"


                    .. group-tab:: MountainCar-v0

                        .. code-block:: yaml

                            agent: "NoisyDQN"
                            env_name: "Classic Control"
                            env_id: "MountainCar-v0"
                            vectorize: "Dummy_Gym"
                            policy: "Noisy_Q_network"
                            representation: "Basic_MLP"
                            runner: "DRL"

                            representation_hidden_size: [128,]
                            q_hidden_size: [128,]
                            activation: 'ReLU'

                            seed: 1
                            parallels: 10
                            n_size: 20000
                            batch_size: 128
                            learning_rate: 0.001
                            gamma: 0.99

                            start_noise: 0.05
                            end_noise: 0.0
                            decay_step_noise: 50000
                            sync_frequency: 100
                            training_frequency: 2
                            running_steps: 500000
                            start_training: 1000

                            use_obsnorm: False
                            use_rewnorm: False
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 1
                            log_dir: "./logs/noisy_dqn/"
                            model_dir: "./models/noisy_dqn/"

            
            .. group-tab:: Box2D

                .. tabs::

                    .. group-tab:: CarRacing-v2

                        .. code-block:: yaml

                            agent: "NoisyDQN"
                            env_name: "Box2D"
                            env_id: "CarRacing-v2"
                            vectorize: "Dummy_Gym"
                            policy: "Noisy_Q_network"
                            representation: "Basic_CNN"
                            runner: "DRL"

                            # the following three arguments are for "Basic_CNN" representation.
                            filters: [16, 16, 32]  #  [16, 16, 32, 32]
                            kernels: [8, 4, 3]  # [8, 6, 4, 4]
                            strides: [4, 2, 1]  # [2, 2, 2, 2]

                            q_hidden_size: [512,]
                            activation: 'ReLU'

                            seed: 1
                            parallels: 2
                            n_size: 10000
                            batch_size: 32
                            learning_rate: 0.0001
                            gamma: 0.99

                            start_noise: 0.05
                            end_noise: 0.0
                            decay_step_noise: 200000
                            sync_frequency: 500
                            training_frequency: 1
                            running_steps: 2000000
                            start_training: 1000

                            use_obsnorm: False
                            use_rewnorm: False
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 100000
                            test_episode: 1
                            log_dir: "./logs/noisy_dqn/"
                            model_dir: "./models/noisy_dqn/"

                    .. group-tab:: LunarLander-v2

                        .. code-block:: yaml

                            agent: "NoisyDQN"
                            env_name: "Box2D"
                            env_id: "LunarLander-v2"
                            vectorize: "Dummy_Gym"
                            policy: "Noisy_Q_network"
                            representation: "Basic_MLP"
                            runner: "DRL"

                            representation_hidden_size: [128,]
                            q_hidden_size: [128,]
                            activation: 'ReLU'

                            seed: 1
                            parallels: 10
                            n_size: 20000
                            batch_size: 128
                            learning_rate: 0.001
                            gamma: 0.99

                            start_noise: 0.05
                            end_noise: 0.0
                            decay_step_noise: 50000
                            sync_frequency: 100
                            training_frequency: 2
                            running_steps: 500000
                            start_training: 1000

                            use_obsnorm: False
                            use_rewnorm: False
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 1
                            log_dir: "./logs/noisy_dqn/"
                            model_dir: "./models/noisy_dqn/"

            
            .. group-tab:: Atari

                .. code-block:: yaml

                    agent: "NoisyDQN"
                    vectorize: "Dummy_Atari"
                    env_name: "Atari"
                    env_id: "ALE/Breakout-v5"
                    obs_type: "grayscale"  # choice for Atari env: ram, rgb, grayscale
                    img_size: [84, 84]  # default is 210 x 160 in gym[Atari]
                    num_stack: 4  # frame stack trick
                    frame_skip: 4  # frame skip trick
                    noop_max: 30  # Do no-op action for a number of steps in [1, noop_max].
                    policy: "Noisy_Q_network"
                    representation: "Basic_CNN"
                    runner: "DRL"

                    # the following three arguments are for "Basic_CNN" representation.
                    filters: [32, 64, 64]  #  [16, 16, 32, 32]
                    kernels: [8, 4, 3]  # [8, 6, 4, 4]
                    strides: [4, 2, 1]  # [2, 2, 2, 2]

                    q_hidden_size: [512, ]
                    activation: "ReLU"

                    seed: 1069
                    parallels: 5
                    n_size: 100000
                    batch_size: 32  # 64
                    learning_rate: 0.0001
                    gamma: 0.99

                    start_noise: 0.05
                    end_noise: 0.0
                    decay_step_greedy: 1000000  # 1M
                    sync_frequency: 500
                    training_frequency: 1
                    running_steps: 50000000  # 50M
                    start_training: 10000

                    use_obsnorm: False
                    use_rewnorm: False
                    obsnorm_range: 5
                    rewnorm_range: 5

                    test_steps: 10000
                    eval_interval: 500000
                    test_episode: 1
                    log_dir: "./logs/noisy_dqn/"
                    model_dir: "./models/noisy_dqn/"


    .. group-tab:: PerDQN

        .. tabs::

            .. group-tab:: Classic Control

                .. tabs::

                    .. group-tab:: CartPole-v1

                        .. code-block:: yaml

                            agent: "PerDQN"
                            env_name: "Classic Control"
                            env_id: "CartPole-v1"
                            vectorize: "Dummy_Gym"
                            policy: "Basic_Q_network"
                            representation: "Basic_MLP"
                            runner: "DRL"

                            representation_hidden_size: [128,]
                            q_hidden_size: [128,]
                            activation: 'ReLU'

                            seed: 1
                            parallels: 10
                            n_size: 20000
                            batch_size: 128
                            learning_rate: 0.001
                            gamma: 0.99

                            start_greedy: 0.5
                            end_greedy: 0.1
                            decay_step_greedy: 20000
                            sync_frequency: 100
                            training_frequency: 4
                            running_steps: 500000
                            start_training: 1000

                            use_obsnorm: False
                            use_rewnorm: False
                            obsnorm_range: 5
                            rewnorm_range: 5

                            PER_alpha: 0.5
                            PER_beta0: 0.4

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 1
                            log_dir: "./logs/perdqn/"
                            model_dir: "./models/perdqn/"

                    .. group-tab:: Acrobot-v1

                        .. code-block:: yaml

                            agent: "PerDQN"
                            env_name: "Classic Control"
                            env_id: "Acrobot-v1"
                            vectorize: "Dummy_Gym"
                            policy: "Basic_Q_network"
                            representation: "Basic_MLP"
                            runner: "DRL"

                            representation_hidden_size: [128,]
                            q_hidden_size: [128,]
                            activation: 'ReLU'

                            seed: 1
                            parallels: 10
                            n_size: 20000
                            batch_size: 128
                            learning_rate: 0.001
                            gamma: 0.99

                            start_greedy: 0.5
                            end_greedy: 0.1
                            decay_step_greedy: 20000
                            sync_frequency: 100
                            training_frequency: 4
                            running_steps: 500000
                            start_training: 1000

                            use_obsnorm: False
                            use_rewnorm: False
                            obsnorm_range: 5
                            rewnorm_range: 5

                            PER_alpha: 0.5
                            PER_beta0: 0.4

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 1
                            log_dir: "./logs/perdqn/"
                            model_dir: "./models/perdqn/"

                    .. group-tab:: MountainCar-v0

                        .. code-block:: yaml

                            agent: "PerDQN"
                            env_name: "Classic Control"
                            env_id: "MountainCar-v0"
                            vectorize: "Dummy_Gym"
                            policy: "Basic_Q_network"
                            representation: "Basic_MLP"
                            runner: "DRL"

                            representation_hidden_size: [128,]
                            q_hidden_size: [128,]
                            activation: 'ReLU'

                            seed: 1
                            parallels: 10
                            n_size: 20000
                            batch_size: 128
                            learning_rate: 0.001
                            gamma: 0.99

                            start_greedy: 0.5
                            end_greedy: 0.1
                            decay_step_greedy: 20000
                            sync_frequency: 100
                            training_frequency: 4
                            running_steps: 500000
                            start_training: 1000

                            use_obsnorm: False
                            use_rewnorm: False
                            obsnorm_range: 5
                            rewnorm_range: 5

                            PER_alpha: 0.5
                            PER_beta0: 0.4

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 1
                            log_dir: "./logs/perdqn/"
                            model_dir: "./models/perdqn/"

            
            .. group-tab:: Box2D

                .. tabs::

                    .. group-tab:: CarRacing-v2

                        .. code-block:: yaml

                            agent: "PerDQN"
                            env_name: "Box2D"
                            env_id: "CarRacing-v2"
                            vectorize: "Dummy_Gym"
                            policy: "Basic_Q_network"
                            representation: "Basic_CNN"
                            runner: "DRL"

                            # the following three arguments are for "Basic_CNN" representation.
                            filters: [16, 16, 32]  #  [16, 16, 32, 32]
                            kernels: [8, 4, 3]  # [8, 6, 4, 4]
                            strides: [4, 2, 1]  # [2, 2, 2, 2]

                            q_hidden_size: [512,]
                            activation: 'ReLU'

                            seed: 1
                            parallels: 2
                            n_size: 10000
                            batch_size: 32
                            learning_rate: 0.0001
                            gamma: 0.99

                            start_greedy: 0.5
                            end_greedy: 0.01
                            decay_step_greedy: 50000
                            sync_frequency: 500
                            training_frequency: 1
                            running_steps: 2000000
                            start_training: 1000

                            use_obsnorm: False
                            use_rewnorm: False
                            obsnorm_range: 5
                            rewnorm_range: 5

                            PER_alpha: 0.5
                            PER_beta0: 0.4

                            test_steps: 10000
                            eval_interval: 100000
                            test_episode: 1
                            log_dir: "./logs/perdqn/"
                            model_dir: "./models/perdqn/"


                    .. group-tab:: LunarLander-v2

                        .. code-block:: yaml

                            agent: "PerDQN"
                            env_name: "Classic Control"
                            env_id: "LunarLander-v2"
                            vectorize: "Dummy_Gym"
                            policy: "Basic_Q_network"
                            representation: "Basic_MLP"
                            runner: "DRL"

                            representation_hidden_size: [128,]
                            q_hidden_size: [128,]
                            activation: 'ReLU'

                            seed: 1
                            parallels: 10
                            n_size: 20000
                            batch_size: 128
                            learning_rate: 0.001
                            gamma: 0.99

                            start_greedy: 0.5
                            end_greedy: 0.1
                            decay_step_greedy: 20000
                            sync_frequency: 100
                            training_frequency: 4
                            running_steps: 500000
                            start_training: 1000

                            use_obsnorm: False
                            use_rewnorm: False
                            obsnorm_range: 5
                            rewnorm_range: 5

                            PER_alpha: 0.5
                            PER_beta0: 0.4

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 1
                            log_dir: "./logs/perdqn/"
                            model_dir: "./models/perdqn/"

            .. group-tab:: Atari

                .. code-block:: yaml

                    agent: "PerDQN"
                    vectorize: "Dummy_Atari"
                    env_name: "Atari"
                    env_id: "ALE/Breakout-v5"
                    obs_type: "grayscale"  # choice for Atari env: ram, rgb, grayscale
                    img_size: [84, 84]  # default is 210 x 160 in gym[Atari]
                    num_stack: 4  # frame stack trick
                    frame_skip: 4  # frame skip trick
                    noop_max: 30  # Do no-op action for a number of steps in [1, noop_max].
                    policy: "Basic_Q_network"
                    representation: "Basic_CNN"
                    runner: "DRL"

                    # the following three arguments are for "Basic_CNN" representation.
                    filters: [32, 64, 64]  #  [16, 16, 32, 32]
                    kernels: [8, 4, 3]  # [8, 6, 4, 4]
                    strides: [4, 2, 1]  # [2, 2, 2, 2]

                    q_hidden_size: [512, ]
                    activation: "ReLU"

                    seed: 1069
                    parallels: 5
                    n_size: 100000
                    batch_size: 32  # 64
                    learning_rate: 0.0001
                    gamma: 0.99

                    start_greedy: 0.5
                    end_greedy: 0.05
                    decay_step_greedy: 1000000  # 1M
                    sync_frequency: 500
                    training_frequency: 1
                    running_steps: 50000000  # 50M
                    start_training: 10000

                    use_obsnorm: False
                    use_rewnorm: False
                    obsnorm_range: 5
                    rewnorm_range: 5

                    PER_alpha: 0.5
                    PER_beta0: 0.4

                    test_steps: 10000
                    eval_interval: 500000
                    test_episode: 1
                    log_dir: "./logs/perdqn/"
                    model_dir: "./models/perdqn/"


    .. group-tab:: QRDQN

        .. tabs::

            .. group-tab:: Classic Control

                .. tabs::

                    .. group-tab:: CartPole-v1

                        .. code-block:: yaml

                            agent: "QRDQN"
                            env_name: "Classic Control"
                            env_id: "CartPole-v1"
                            vectorize: "Dummy_Gym"
                            policy: "QR_Q_network"
                            representation: "Basic_MLP"
                            runner: "DRL"

                            representation_hidden_size: [128,]
                            q_hidden_size: [128,]
                            activation: 'ReLU'

                            seed: 1
                            parallels: 10
                            n_size: 20000
                            batch_size: 256
                            learning_rate: 0.001
                            gamma: 0.99
                            quantile_num: 20

                            start_greedy: 0.25
                            end_greedy: 0.01
                            decay_step_greedy: 30000
                            sync_frequency: 100
                            training_frequency: 1
                            running_steps: 300000
                            start_training: 1000

                            use_obsnorm: False
                            use_rewnorm: False
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 1
                            log_dir: "./logs/qrdqn/"
                            model_dir: "./models/qrdqn/"


                    .. group-tab:: Acrobot-v1

                        .. code-block:: yaml

                            agent: "QRDQN"
                            env_name: "Classic Control"
                            env_id: "Acrobot-v1"
                            vectorize: "Dummy_Gym"
                            policy: "QR_Q_network"
                            representation: "Basic_MLP"
                            runner: "DRL"

                            representation_hidden_size: [128,]
                            q_hidden_size: [128,]
                            activation: 'ReLU'

                            seed: 1
                            parallels: 10
                            n_size: 20000
                            batch_size: 256
                            learning_rate: 0.001
                            gamma: 0.99
                            quantile_num: 20

                            start_greedy: 0.25
                            end_greedy: 0.01
                            decay_step_greedy: 30000
                            sync_frequency: 100
                            training_frequency: 1
                            running_steps: 300000
                            start_training: 1000

                            use_obsnorm: False
                            use_rewnorm: False
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 1
                            log_dir: "./logs/qrdqn/"
                            model_dir: "./models/qrdqn/"

                    .. group-tab:: MountainCar-v0

                        .. code-block:: yaml

                            agent: "QRDQN"
                            env_name: "Classic Control"
                            env_id: "MountainCar-v0"
                            vectorize: "Dummy_Gym"
                            policy: "QR_Q_network"
                            representation: "Basic_MLP"
                            runner: "DRL"

                            representation_hidden_size: [128,]
                            q_hidden_size: [128,]
                            activation: 'ReLU'

                            seed: 1
                            parallels: 10
                            n_size: 20000
                            batch_size: 256
                            learning_rate: 0.001
                            gamma: 0.99
                            quantile_num: 20

                            start_greedy: 0.25
                            end_greedy: 0.01
                            decay_step_greedy: 30000
                            sync_frequency: 100
                            training_frequency: 1
                            running_steps: 300000
                            start_training: 1000

                            use_obsnorm: False
                            use_rewnorm: False
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 1
                            log_dir: "./logs/qrdqn/"
                            model_dir: "./models/qrdqn/"

            .. group-tab:: Box2D

                .. tabs::

                    .. group-tab:: CarRacing-v2

                        .. code-block:: yaml

                            agent: "QRDQN"
                            env_name: "Box2D"
                            env_id: "CarRacing-v2"
                            vectorize: "Dummy_Gym"
                            policy: "QR_Q_network"
                            representation: "Basic_CNN"
                            runner: "DRL"

                            # the following three arguments are for "Basic_CNN" representation.
                            filters: [16, 16, 32]  #  [16, 16, 32, 32]
                            kernels: [8, 4, 3]  # [8, 6, 4, 4]
                            strides: [4, 2, 1]  # [2, 2, 2, 2]

                            q_hidden_size: [512,]
                            activation: 'ReLU'

                            seed: 1
                            parallels: 2
                            n_size: 10000
                            batch_size: 32
                            learning_rate: 0.0001
                            gamma: 0.99
                            quantile_num: 20

                            start_greedy: 0.5
                            end_greedy: 0.01
                            decay_step_greedy: 50000
                            sync_frequency: 500
                            training_frequency: 1
                            running_steps: 2000000
                            start_training: 1000

                            use_obsnorm: False
                            use_rewnorm: False
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 100000
                            test_episode: 1
                            log_dir: "./logs/qrdqn/"
                            model_dir: "./models/qrdqn/"


                    .. group-tab:: LunarLander-v2

                        .. code-block:: yaml

                            agent: "QRDQN"
                            env_name: "Box2D"
                            env_id: "LunarLander-v2"
                            vectorize: "Dummy_Gym"
                            policy: "QR_Q_network"
                            representation: "Basic_MLP"
                            runner: "DRL"

                            representation_hidden_size: [128,]
                            q_hidden_size: [128,]
                            activation: 'ReLU'

                            seed: 1
                            parallels: 10
                            n_size: 10000
                            batch_size: 256
                            learning_rate: 0.001
                            gamma: 0.99
                            quantile_num: 20

                            start_greedy: 0.5
                            end_greedy: 0.01
                            decay_step_greedy: 10000
                            sync_frequency: 50
                            training_frequency: 1
                            running_steps: 200000
                            start_training: 1000

                            use_obsnorm: False
                            use_rewnorm: False
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 1
                            log_dir: "./logs/qrdqn/"
                            model_dir: "./models/qrdqn/"

            
            .. group-tab:: Atari

                .. code-block:: yaml

                    agent: "QRDQN"
                    vectorize: "Dummy_Atari"
                    env_name: "Atari"
                    env_id: "ALE/Breakout-v5"
                    obs_type: "grayscale"  # choice for Atari env: ram, rgb, grayscale
                    img_size: [84, 84]  # default is 210 x 160 in gym[Atari]
                    num_stack: 4  # frame stack trick
                    frame_skip: 4  # frame skip trick
                    noop_max: 30  # Do no-op action for a number of steps in [1, noop_max].
                    policy: "QR_Q_network"
                    representation: "Basic_CNN"
                    runner: "DRL"

                    # the following three arguments are for "Basic_CNN" representation.
                    filters: [32, 64, 64]  #  [16, 16, 32, 32]
                    kernels: [8, 4, 3]  # [8, 6, 4, 4]
                    strides: [4, 2, 1]  # [2, 2, 2, 2]

                    q_hidden_size: [512, ]
                    activation: "ReLU"

                    seed: 1069
                    parallels: 5
                    n_size: 100000
                    batch_size: 32  # 64
                    learning_rate: 0.0001
                    gamma: 0.99
                    quantile_num: 20

                    start_greedy: 0.5
                    end_greedy: 0.05
                    decay_step_greedy: 1000000  # 1M
                    sync_frequency: 500
                    training_frequency: 1
                    running_steps: 50000000  # 50M
                    start_training: 10000

                    use_obsnorm: False
                    use_rewnorm: False
                    obsnorm_range: 5
                    rewnorm_range: 5

                    test_steps: 10000
                    eval_interval: 500000
                    test_episode: 1
                    log_dir: "./logs/qrdqn/"
                    model_dir: "./models/qrdqn/"


.. raw:: html

   <br><hr>

Policy Gradient-based Implementations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. tabs::

    .. group-tab:: PG

        .. tabs::

            .. group-tab:: Classic Control

                .. tabs::

                    .. group-tab:: CartPole-v1

                        .. code-block:: yaml

                            agent: "PG"
                            env_name: "Classic Control"
                            env_id: "CartPole-v1"
                            representation: "Basic_MLP"
                            vectorize: "Dummy_Gym"
                            policy: "Categorical_Actor"
                            runner: "DRL"

                            representation_hidden_size: [128,]
                            actor_hidden_size: [128,]
                            activation: 'ReLU'

                            seed: 1
                            parallels: 10
                            running_steps: 300000
                            n_steps: 128
                            n_epoch: 1
                            n_minibatch: 1
                            learning_rate: 0.0004

                            ent_coef: 0.01
                            clip_grad: 0.5
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.98
                            use_gae: False
                            gae_lambda: 0.95
                            use_advnorm: False

                            use_obsnorm: True
                            use_rewnorm: True
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 1
                            log_dir: "./logs/pg/"
                            model_dir: "./models/pg/"

                    .. group-tab:: Acrobot-v1

                        .. code-block:: yaml

                            agent: "PG"
                            env_name: "Classic Control"
                            env_id: "Acrobot-v1"
                            representation: "Basic_MLP"
                            vectorize: "Dummy_Gym"
                            policy: "Categorical_Actor"
                            runner: "DRL"

                            representation_hidden_size: [128,]
                            actor_hidden_size: [128,]
                            activation: 'ReLU'

                            seed: 1
                            parallels: 10
                            running_steps: 300000
                            n_steps: 500
                            n_epoch: 1
                            n_minibatch: 1
                            learning_rate: 0.0004

                            ent_coef: 0.01
                            clip_grad: 0.5
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.98
                            use_gae: False
                            gae_lambda: 0.95
                            use_advnorm: False

                            use_obsnorm: True
                            use_rewnorm: True
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 1
                            log_dir: "./logs/pg/"
                            model_dir: "./models/pg/"

                    .. group-tab:: Pendulum-v1

                        .. code-block:: yaml

                            agent: "PG"
                            env_name: "Classic Control"
                            env_id: "Pendulum-v1"
                            representation: "Basic_MLP"
                            vectorize: "Dummy_Gym"
                            policy: "Gaussian_Actor"
                            runner: "DRL"

                            representation_hidden_size: [256,]
                            actor_hidden_size: [256,]
                            activation: 'LeakyReLU'

                            seed: 1
                            parallels: 10
                            running_steps: 300000
                            n_steps: 128
                            n_epoch: 1
                            n_minibatch: 1
                            learning_rate: 0.0004

                            ent_coef: 0.01
                            clip_grad: 0.5
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.98
                            use_gae: False
                            gae_lambda: 0.95
                            use_advnorm: False

                            use_obsnorm: True
                            use_rewnorm: True
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 1
                            log_dir: "./logs/pg/"
                            model_dir: "./models/pg/"

                    .. group-tab:: MountainCar-v0

                        .. code-block:: yaml

                            agent: "PG"
                            env_name: "Classic Control"
                            env_id: "MountainCar-v0"
                            representation: "Basic_MLP"
                            vectorize: "Dummy_Gym"
                            policy: "Categorical_Actor"
                            runner: "DRL"

                            representation_hidden_size: [128,]
                            actor_hidden_size: [128,]
                            activation: 'ReLU'

                            seed: 1
                            parallels: 10
                            running_steps: 300000
                            n_steps: 128
                            n_epoch: 3
                            n_minibatch: 1
                            learning_rate: 0.0004

                            ent_coef: 0.01
                            clip_grad: 0.5
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.98
                            use_gae: False
                            gae_lambda: 0.95
                            use_advnorm: False

                            use_obsnorm: True
                            use_rewnorm: True
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 1
                            log_dir: "./logs/pg/"
                            model_dir: "./models/pg/"


            .. group-tab:: Box2D

                .. tabs::

                    .. group-tab:: BipedalWalker-v3

                        .. code-block:: yaml

                            agent: "PG"
                            env_name: "Box2D"
                            env_id: "BipedalWalker-v3"
                            representation: "Basic_MLP"
                            vectorize: "Dummy_Gym"
                            policy: "Gaussian_Actor"
                            runner: "DRL"

                            representation_hidden_size: [128,]
                            actor_hidden_size: [128,]
                            activation: 'ReLU'

                            seed: 1
                            parallels: 10
                            running_steps: 100000
                            n_steps: 1024
                            n_epoch: 3
                            n_minibatch: 1
                            learning_rate: 0.0004

                            ent_coef: 0.01
                            clip_grad: 0.5
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.98
                            use_gae: False
                            gae_lambda: 0.95
                            use_advnorm: False

                            use_obsnorm: True
                            use_rewnorm: True
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 10000
                            test_episode: 1
                            log_dir: "./logs/pg/"
                            model_dir: "./models/pg/"

                    .. group-tab:: LunarLander-v2

                        .. code-block:: yaml

                            agent: "PG"
                            env_name: "Box2D"
                            env_id: "LunarLander-v2"
                            representation: "Basic_MLP"
                            vectorize: "Dummy_Gym"
                            policy: "Categorical_Actor"
                            runner: "DRL"

                            representation_hidden_size: [128,]
                            actor_hidden_size: [128,]
                            activation: 'ReLU'

                            seed: 1
                            parallels: 10
                            running_steps: 300000
                            nsteps: 128
                            nepoch: 3
                            nminibatch: 1
                            learning_rate: 0.0004

                            ent_coef: 0.01
                            clip_grad: 0.5
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.98
                            use_gae: False
                            gae_lambda: 0.95
                            use_advnorm: False

                            use_obsnorm: True
                            use_rewnorm: True
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 1
                            log_dir: "./logs/pg/"
                            model_dir: "./models/pg/"

            .. group-tab:: MuJoCo

                .. code-block:: yaml

                    agent: "PG"
                    env_name: "MuJoCo"
                    env_id: "Ant-v4"
                    vectorize: "Dummy_Gym"
                    policy: "Gaussian_Actor"
                    representation: "Basic_MLP"
                    runner: "DRL"

                    representation_hidden_size: [256, 256]
                    actor_hidden_size: []
                    activation: "LeakyReLU"

                    seed: 1
                    parallels: 16
                    running_steps: 1000000  # 1M
                    n_steps: 256
                    n_epoch: 1
                    n_minibatch: 1
                    learning_rate: 0.0007  # 7e-4

                    ent_coef: 0.0
                    clip_grad: 0.5
                    clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                    gamma: 0.99
                    use_gae: False
                    gae_lambda: 0.95
                    use_advnorm: False

                    use_obsnorm: True
                    use_rewnorm: True
                    obsnorm_range: 5
                    rewnorm_range: 5

                    test_steps: 10000
                    eval_interval: 5000
                    test_episode: 5
                    log_dir: "./logs/pg/"
                    model_dir: "./models/pg/"
 
    .. group-tab:: PPG

        .. tabs::

            .. group-tab:: Classic Control

                .. tabs::

                    .. group-tab:: CartPole-v1

                        .. code-block:: yaml

                            agent: "PPG"
                            env_name: "Classic Control"
                            env_id: "CartPole-v1"
                            vectorize: "Dummy_Gym"
                            representation: "Basic_MLP"
                            policy: "Categorical_PPG"
                            runner: "DRL"

                            representation_hidden_size: [128,]
                            actor_hidden_size: [128,]
                            critic_hidden_size: [128,]
                            activation: "ReLU"

                            seed: 1
                            parallels: 10
                            running_steps: 300000
                            n_steps: 256
                            n_epoch: 1
                            policy_nepoch: 4
                            value_nepoch: 8
                            aux_nepoch: 8
                            n_minibatch: 1
                            learning_rate: 0.0004

                            ent_coef: 0.01
                            clip_range: 0.2
                            kl_beta: 1.0
                            gamma: 0.98
                            use_gae: True
                            gae_lambda: 0.95
                            use_advnorm: True

                            use_obsnorm: True
                            use_rewnorm: True
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 1
                            log_dir: "./logs/ppg/"
                            model_dir: "./models/ppg/"

                    .. group-tab:: Acrobot-v1

                        .. code-block:: yaml

                            agent: "PPG"
                            env_name: "Classic Control"
                            env_id: "Acrobot-v1"
                            vectorize: "Dummy_Gym"
                            representation: "Basic_MLP"
                            policy: "Categorical_PPG"
                            runner: "DRL"

                            representation_hidden_size: [128,]
                            actor_hidden_size: [128,]
                            critic_hidden_size: [128,]
                            activation: "LeakyReLU"

                            seed: 1
                            parallels: 10
                            running_steps: 300000
                            n_steps: 256
                            n_epoch: 1
                            policy_nepoch: 4
                            value_nepoch: 8
                            aux_nepoch: 8
                            n_minibatch: 1
                            learning_rate: 0.001

                            ent_coef: 0.01
                            clip_range: 0.2
                            kl_beta: 1.0
                            gamma: 0.98
                            use_gae: True
                            gae_lambda: 0.95
                            use_advnorm: True

                            use_obsnorm: True
                            use_rewnorm: True
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 1
                            log_dir: "./logs/ppg/"
                            model_dir: "./models/ppg/"


                    .. group-tab:: Pendulum-v1

                        .. code-block:: yaml

                            agent: "PPG"
                            env_name: "Classic Control"
                            env_id: "Pendulum-v1"
                            vectorize: "Dummy_Gym"
                            representation: "Basic_MLP"
                            policy: "Gaussian_PPG"
                            runner: "DRL"

                            representation_hidden_size: [128,]
                            actor_hidden_size: [128,]
                            critic_hidden_size: [128,]
                            activation: "LeakyReLU"

                            seed: 1
                            parallels: 10
                            running_steps: 300000
                            n_steps: 256
                            n_epoch: 1
                            policy_nepoch: 4
                            value_nepoch: 8
                            aux_nepoch: 8
                            n_minibatch: 1
                            learning_rate: 0.001

                            ent_coef: 0.01
                            clip_range: 0.2
                            kl_beta: 1.0
                            gamma: 0.98
                            use_gae: True
                            gae_lambda: 0.95
                            use_advnorm: True

                            use_obsnorm: True
                            use_rewnorm: True
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 1
                            log_dir: "./logs/ppg/"
                            model_dir: "./models/ppg/"


                    .. group-tab:: MountainCar-v0

                        .. code-block:: yaml

                            agent: "PPG"
                            env_name: "Classic Control"
                            env_id: "MountainCar-v0"
                            vectorize: "Dummy_Gym"
                            representation: "Basic_MLP"
                            policy: "Categorical_PPG"
                            runner: "DRL"

                            representation_hidden_size: [128,]
                            actor_hidden_size: [128,]
                            critic_hidden_size: [128,]
                            activation: "LeakyReLU"

                            seed: 1
                            parallels: 10
                            running_steps: 300000
                            n_steps: 256
                            n_epoch: 1
                            policy_nepoch: 4
                            value_nepoch: 8
                            aux_nepoch: 8
                            n_minibatch: 1
                            learning_rate: 0.001

                            ent_coef: 0.01
                            clip_range: 0.2
                            kl_beta: 1.0
                            gamma: 0.98
                            use_gae: True
                            gae_lambda: 0.95
                            use_advnorm: True

                            use_obsnorm: True
                            use_rewnorm: True
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 1
                            log_dir: "./logs/ppg/"
                            model_dir: "./models/ppg/"


            .. group-tab:: Box2D

                .. tabs::

                    .. group-tab:: BipedalWalker-v3

                        .. code-block:: yaml

                            agent: "PPG"
                            env_name: "Box2D"
                            env_id: "BipedalWalker-v3"
                            vectorize: "Dummy_Gym"
                            representation: "Basic_MLP"
                            policy: "Gaussian_PPG"
                            runner: "DRL"

                            representation_hidden_size: [128,]
                            actor_hidden_size: [128,]
                            critic_hidden_size: [128,]
                            activation: "LeakyReLU"

                            seed: 1
                            parallels: 10
                            running_steps: 300000
                            n_steps: 256
                            n_epoch: 1
                            policy_nepoch: 4
                            value_nepoch: 8
                            aux_nepoch: 8
                            n_minibatch: 1
                            learning_rate: 0.001

                            ent_coef: 0.01
                            clip_range: 0.2
                            kl_beta: 1.0
                            gamma: 0.98
                            use_gae: True
                            gae_lambda: 0.95
                            use_advnorm: True

                            use_obsnorm: True
                            use_rewnorm: True
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 1
                            log_dir: "./logs/ppg/"
                            model_dir: "./models/ppg/"


                    .. group-tab:: LunarLander-v2

                        .. code-block:: yaml

                            agent: "PPG"
                            env_name: "Box2D"
                            env_id: "LunarLander-v2"
                            vectorize: "Dummy_Gym"
                            representation: "Basic_MLP"
                            policy: "Categorical_PPG"
                            runner: "DRL"

                            representation_hidden_size: [128,]
                            actor_hidden_size: [128,]
                            critic_hidden_size: [128,]
                            activation: "ReLU"

                            seed: 1
                            parallels: 10
                            running_steps: 300000
                            n_steps: 256
                            n_epoch: 1
                            policy_nepoch: 4
                            value_nepoch: 8
                            aux_nepoch: 8
                            n_minibatch: 1
                            learning_rate: 0.0004

                            ent_coef: 0.01
                            clip_range: 0.2
                            kl_beta: 1.0
                            gamma: 0.98
                            use_gae: True
                            gae_lambda: 0.95
                            use_advnorm: True

                            use_obsnorm: True
                            use_rewnorm: True
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 1
                            log_dir: "./logs/ppg/"
                            model_dir: "./models/ppg/"

            .. group-tab:: MuJoCo

                .. code-block:: yaml

                    agent: "PPG"
                    env_name: "MuJoCo"
                    env_id: "InvertedPendulum-v2"
                    vectorize: "Dummy_Gym"
                    representation: "Basic_MLP"
                    policy: "Gaussian_PPG"
                    runner: "DRL"

                    representation_hidden_size: [256,]
                    actor_hidden_size: [256,]
                    critic_hidden_size: [256,]
                    activation: "LeakyReLU"

                    seed: 1
                    parallels: 16
                    running_steps: 1000000  # 1M
                    n_steps: 256
                    n_minibatch: 4
                    n_epoch: 1
                    policy_nepoch: 2
                    value_nepoch: 4
                    aux_nepoch: 8

                    learning_rate: 0.0007

                    ent_coef: 0.0
                    clip_range: 0.25
                    kl_beta: 2.0
                    gamma: 0.98
                    use_gae: True
                    gae_lambda: 0.95
                    use_advnorm: True

                    use_obsnorm: True
                    use_rewnorm: True
                    obsnorm_range: 5
                    rewnorm_range: 5

                    test_steps: 10000
                    eval_interval: 10000
                    test_episode: 5
                    log_dir: "./logs/ppg/"
                    model_dir: "./models/ppg/"



    .. group-tab:: PPO

        .. tabs::

            .. group-tab:: Classic Control

                .. tabs::

                    .. group-tab:: CartPole-v1

                        .. code-block:: yaml

                            agent: "PPO_Clip"  # Choice: PPO_Clip, PPO_KL
                            env_name: "Classic Control"
                            env_id: "CartPole-v1"
                            vectorize: "Dummy_Gym"
                            representation: "Basic_MLP"
                            policy: "Categorical_AC"
                            runner: "DRL"

                            representation_hidden_size: [128,]
                            actor_hidden_size: [128,]
                            critic_hidden_size: [128,]
                            activation: 'LeakyReLU'

                            seed: 1
                            parallels: 10
                            running_steps: 300000
                            n_steps: 256
                            n_epoch: 8
                            n_minibatch: 8
                            learning_rate: 0.0004

                            use_grad_clip: True

                            vf_coef: 0.25
                            ent_coef: 0.01
                            target_kl: 0.001  # for PPO_KL agent
                            clip_range: 0.2  # for PPO_Clip agent
                            clip_grad_norm: 0.5
                            gamma: 0.98
                            use_gae: True
                            gae_lambda: 0.95
                            use_advnorm: True

                            use_obsnorm: True
                            use_rewnorm: True
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 3
                            log_dir: "./logs/ppo/"
                            model_dir: "./models/ppo/"


                    .. group-tab:: Acrobot-v1

                        .. code-block:: yaml

                            agent: "PPO_Clip"  # Choice: PPO_Clip, PPO_KL
                            env_name: "Classic Control"
                            env_id: "Acrobot-v1"
                            vectorize: "Dummy_Gym"
                            representation: "Basic_MLP"
                            policy: "Categorical_AC"
                            runner: "DRL"

                            representation_hidden_size: [128,]
                            actor_hidden_size: [128,]
                            critic_hidden_size: [128,]
                            activation: 'LeakyReLU'

                            seed: 1
                            parallels: 10
                            running_steps: 300000
                            n_steps: 256
                            n_epoch: 8
                            n_minibatch: 8
                            learning_rate: 0.0004

                            use_grad_clip: True

                            vf_coef: 0.25
                            ent_coef: 0.01
                            target_kl: 0.001  # for PPO_KL agent
                            clip_range: 0.2  # for PPO_Clip agent
                            clip_grad_norm: 0.5
                            gamma: 0.98
                            use_gae: True
                            gae_lambda: 0.95
                            use_advnorm: True

                            use_obsnorm: True
                            use_rewnorm: True
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 3
                            log_dir: "./logs/ppo/"
                            model_dir: "./models/ppo/"


                    .. group-tab:: Pendulum-v1

                        .. code-block:: yaml

                            agent: "PPO_Clip"  # Choice: PPO_Clip, PPO_KL
                            env_name: "Classic Control"
                            env_id: "Pendulum-v1"
                            vectorize: "Dummy_Gym"
                            representation: "Basic_MLP"
                            policy: "Gaussian_AC"
                            runner: "DRL"

                            representation_hidden_size: [128,]
                            actor_hidden_size: [128,]
                            critic_hidden_size: [128,]
                            activation: 'LeakyReLU'

                            seed: 1
                            parallels: 10
                            running_steps: 300000
                            n_steps: 256
                            n_epoch: 8
                            n_minibatch: 8
                            learning_rate: 0.0004

                            use_grad_clip: True

                            vf_coef: 0.25
                            ent_coef: 0.01
                            target_kl: 0.001  # for PPO_KL agent
                            clip_range: 0.2  # for PPO_Clip agent
                            clip_grad_norm: 0.5
                            gamma: 0.98
                            use_gae: True
                            gae_lambda: 0.95
                            use_advnorm: True

                            use_obsnorm: True
                            use_rewnorm: True
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 3
                            log_dir: "./logs/ppo/"
                            model_dir: "./models/ppo/"


                    .. group-tab:: MountainCar-v0

                        .. code-block:: yaml

                            agent: "PPO_Clip"  # Choice: PPO_Clip, PPO_KL
                            env_name: "Classic Control"
                            env_id: "MountainCar-v0"
                            vectorize: "Dummy_Gym"
                            representation: "Basic_MLP"
                            policy: "Categorical_AC"
                            runner: "DRL"

                            representation_hidden_size: [128,]
                            actor_hidden_size: [128,]
                            critic_hidden_size: [128,]
                            activation: 'LeakyReLU'

                            seed: 1
                            parallels: 10
                            running_steps: 300000
                            n_steps: 256
                            n_epoch: 8
                            n_minibatch: 8
                            learning_rate: 0.0004

                            use_grad_clip: True

                            vf_coef: 0.25
                            ent_coef: 0.01
                            target_kl: 0.001  # for PPO_KL agent
                            clip_range: 0.2  # for PPO_Clip agent
                            clip_grad_norm: 0.5
                            gamma: 0.98
                            use_gae: True
                            gae_lambda: 0.95
                            use_advnorm: True

                            use_obsnorm: True
                            use_rewnorm: True
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 3
                            log_dir: "./logs/ppo/"
                            model_dir: "./models/ppo/"


            .. group-tab:: Box2D

                .. tabs::

                    .. group-tab:: BipedalWalker-v3

                        .. code-block:: yaml

                            agent: "PPO_Clip"  # Choice: PPO_Clip, PPO_KL
                            env_name: "Box2D"
                            env_id: "BipedalWalker-v3"
                            vectorize: "Dummy_Gym"
                            representation: "Basic_MLP"
                            policy: "Gaussian_AC"
                            runner: "DRL"

                            representation_hidden_size: [128,]
                            actor_hidden_size: [128,]
                            critic_hidden_size: [128,]
                            activation: 'LeakyReLU'

                            seed: 1
                            parallels: 10
                            running_steps: 300000
                            n_steps: 256
                            n_epoch: 8
                            n_minibatch: 8
                            learning_rate: 0.0004

                            use_grad_clip: True

                            vf_coef: 0.25
                            ent_coef: 0.01
                            target_kl: 0.001  # for PPO_KL agent
                            clip_range: 0.2  # for PPO_Clip agent
                            clip_grad_norm: 0.5
                            gamma: 0.98
                            use_gae: True
                            gae_lambda: 0.95
                            use_advnorm: True

                            use_obsnorm: True
                            use_rewnorm: True
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 3
                            log_dir: "./logs/ppo/"
                            model_dir: "./models/ppo/"

                    .. group-tab:: CarRacing-v2

                        .. code-block:: yaml

                            agent: "PPO_Clip"  # Choice: PPO_Clip, PPO_KL
                            env_name: "Box2D"
                            env_id: "CarRacing-v2"
                            vectorize: "Dummy_Gym"
                            representation: "Basic_CNN"
                            policy: "Categorical_AC"
                            runner: "DRL"

                            # the following three arguments are for "Basic_CNN" representation.
                            filters: [16, 16, 32]  #  [16, 16, 32, 32]
                            kernels: [8, 4, 3]  # [8, 6, 4, 4]
                            strides: [4, 2, 1]  # [2, 2, 2, 2]
                            fc_hidden_sizes: [512, ]  # fully connected layer hidden sizes.
                            actor_hidden_size: []
                            critic_hidden_size: []
                            activation: "ReLU"

                            seed: 1
                            parallels: 2
                            running_steps: 300000
                            n_steps: 256
                            n_epoch: 8
                            n_minibatch: 8
                            learning_rate: 0.0004

                            use_grad_clip: True

                            vf_coef: 0.25
                            ent_coef: 0.01
                            clip_range: 0.2
                            clip_grad_norm: 0.5
                            gamma: 0.99
                            use_gae: True
                            gae_lambda: 0.95  # gae_lambda: Lambda parameter for calculating N-step advantage
                            use_advnorm: True

                            use_obsnorm: False
                            use_rewnorm: False
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 3
                            log_dir: "./logs/ppo/"
                            model_dir: "./models/ppo/"

                    .. group-tab:: LunarLander-v2

                        .. code-block:: yaml

                            agent: "PPO_Clip"  # Choice: PPO_Clip, PPO_KL
                            env_name: "Box2D"
                            env_id: "LunarLander-v2"
                            vectorize: "Dummy_Gym"
                            representation: "Basic_MLP"
                            policy: "Categorical_AC"
                            runner: "DRL"

                            representation_hidden_size: [128,]
                            actor_hidden_size: [128,]
                            critic_hidden_size: [128,]
                            activation: 'LeakyReLU'

                            seed: 1
                            parallels: 10
                            running_steps: 300000
                            n_steps: 256
                            n_epoch: 8
                            n_minibatch: 8
                            learning_rate: 0.0004

                            use_grad_clip: True

                            vf_coef: 0.25
                            ent_coef: 0.01
                            target_kl: 0.001  # for PPO_KL agent
                            clip_range: 0.2  # for PPO_Clip agent
                            clip_grad_norm: 0.5
                            gamma: 0.98
                            use_gae: True
                            gae_lambda: 0.95
                            use_advnorm: True

                            use_obsnorm: True
                            use_rewnorm: True
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 3
                            log_dir: "./logs/ppo/"
                            model_dir: "./models/ppo/"

            .. group-tab:: Atari

                .. code-block:: yaml

                    agent: "PPO_Clip"
                    vectorize: "Dummy_Atari"
                    env_name: "Atari"
                    env_id: "ALE/Breakout-v5"
                    obs_type: "grayscale"  # choice for Atari env: ram, rgb, grayscale
                    img_size: [84, 84]  # default is 210 x 160 in gym[Atari]
                    num_stack: 4  # frame stack trick
                    frame_skip: 4  # frame skip trick
                    noop_max: 30  # Do no-op action for a number of steps in [1, noop_max].
                    representation: "AC_CNN_Atari"  # CNN and FC layers
                    policy: "Categorical_AC"
                    runner: "DRL"

                    # Good HyperParameters for Atari Games, Do not change them.
                    filters: [32, 64, 64]
                    kernels: [8, 4, 3]
                    strides: [4, 2, 1]
                    fc_hidden_sizes: [512, ]  # fully connected layer hidden sizes.
                    actor_hidden_size: []
                    critic_hidden_size: []
                    activation: "ReLU"

                    seed: 1
                    parallels: 8
                    running_steps: 10000000  # 10M
                    n_steps: 128
                    n_epoch: 4
                    n_minibatch: 4
                    learning_rate: 0.00025

                    use_grad_clip: True

                    vf_coef: 0.25
                    ent_coef: 0.01
                    clip_range: 0.2
                    clip_grad_norm: 0.5
                    gamma: 0.99
                    use_gae: True
                    gae_lambda: 0.95  # gae_lambda: Lambda parameter for calculating N-step advantage
                    use_advnorm: True

                    use_obsnorm: False
                    use_rewnorm: False
                    obsnorm_range: 5
                    rewnorm_range: 5

                    test_steps: 10000
                    eval_interval: 100000
                    test_episode: 3
                    log_dir: "./logs/ppo/"
                    model_dir: "./models/ppo/"


            .. group-tab:: MuJoCo

                .. code-block:: yaml

                    agent: "PPO_Clip"  # choice: PPO_Clip, PPO_KL
                    env_name: "MuJoCo"
                    env_id: "Ant-v4"
                    vectorize: "Dummy_Gym"
                    policy: "Gaussian_AC"  # choice: Gaussian_AC for continuous actions, Categorical_AC for discrete actions.
                    representation: "Basic_MLP"
                    runner: "DRL"

                    representation_hidden_size: [256,]
                    actor_hidden_size: [256,]
                    critic_hidden_size: [256,]
                    activation: "LeakyReLU"

                    seed: 79811
                    parallels: 16
                    running_steps: 1000000
                    n_steps: 256
                    n_epoch: 16
                    n_minibatch: 8
                    learning_rate: 0.0004

                    use_grad_clip: True

                    vf_coef: 0.25
                    ent_coef: 0.0
                    target_kl: 0.001  # for PPO_KL agent
                    clip_range: 0.2  # for PPO_Clip agent
                    clip_grad_norm: 0.5
                    gamma: 0.99
                    use_gae: True
                    gae_lambda: 0.95
                    use_advnorm: True

                    use_obsnorm: True
                    use_rewnorm: True
                    obsnorm_range: 5
                    rewnorm_range: 5

                    test_steps: 10000
                    eval_interval: 5000
                    test_episode: 5
                    log_dir: "./logs/ppo/"
                    model_dir: "./models/ppo/"



    .. group-tab:: A2C

        .. tabs::

            .. group-tab:: Classic Control

                .. tabs::

                    .. group-tab:: CartPole-v1

                        .. code-block:: yaml

                            agent: "A2C"
                            env_name: "Classic Control"
                            env_id: "CartPole-v1"
                            vectorize: "Subproc_Gym"
                            policy: "Categorical_AC"
                            representation: "Basic_MLP"
                            runner: "DRL"

                            representation_hidden_size: [256,]
                            actor_hidden_size: [256,]
                            critic_hidden_size: [256,]
                            activation: 'LeakyReLU'

                            seed: 1
                            parallels: 10
                            running_steps: 300000
                            n_steps: 128
                            n_epoch: 1
                            n_minibatch: 1
                            learning_rate: 0.0004

                            vf_coef: 0.25
                            ent_coef: 0.01
                            clip_grad: 0.5
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.98
                            use_gae: True
                            gae_lambda: 0.95
                            use_advnorm: True

                            use_obsnorm: True
                            use_rewnorm: True
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 3
                            log_dir: "./logs/a2c/"
                            model_dir: "./models/a2c/"


                    .. group-tab:: Acrobot-v1

                        .. code-block:: yaml

                            agent: "A2C"
                            env_name: "Classic Control"
                            env_id: "Acrobot-v1"
                            vectorize: "Dummy_Gym"
                            policy: "Categorical_AC"
                            representation: "Basic_MLP"
                            runner: "DRL"

                            representation_hidden_size: [256,]
                            actor_hidden_size: [256,]
                            critic_hidden_size: [256,]
                            activation: 'LeakyReLU'

                            seed: 1
                            parallels: 10
                            running_steps: 300000
                            n_steps: 128
                            n_epoch: 1
                            n_minibatch: 1
                            learning_rate: 0.0004

                            vf_coef: 0.25
                            ent_coef: 0.01
                            clip_grad: 0.5
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.98
                            use_gae: True
                            gae_lambda: 0.95
                            use_advnorm: True

                            use_obsnorm: True
                            use_rewnorm: True
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 3
                            log_dir: "./logs/a2c/"
                            model_dir: "./models/a2c/"


                    .. group-tab:: Pendulum-v1

                        .. code-block:: yaml

                            agent: "A2C"
                            env_name: "Classic Control"
                            env_id: "Pendulum-v1"
                            vectorize: "Dummy_Gym"
                            policy: "Gaussian_AC"
                            representation: "Basic_MLP"
                            runner: "DRL"

                            representation_hidden_size: [256,]
                            actor_hidden_size: [256,]
                            critic_hidden_size: [256,]
                            activation: 'LeakyReLU'

                            seed: 1
                            parallels: 10
                            running_steps: 300000
                            n_steps: 64
                            n_epoch: 1
                            n_minibatch: 1
                            learning_rate: 0.0004

                            vf_coef: 0.25
                            ent_coef: 0.01
                            clip_grad: 0.5
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.98
                            use_gae: True
                            gae_lambda: 0.95
                            use_advnorm: True

                            use_obsnorm: True
                            use_rewnorm: True
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 3
                            log_dir: "./logs/a2c/"
                            model_dir: "./models/a2c/"


                    .. group-tab:: MountainCar-v0

                        .. code-block:: yaml

                            agent: "A2C"
                            env_name: "Classic Control"
                            env_id: "MountainCar-v0"
                            vectorize: "Dummy_Gym"
                            policy: "Categorical_AC"
                            representation: "Basic_MLP"
                            runner: "DRL"

                            representation_hidden_size: [256,]
                            actor_hidden_size: [256,]
                            critic_hidden_size: [256,]
                            activation: 'LeakyReLU'

                            seed: 1
                            parallels: 10
                            running_steps: 300000
                            n_steps: 128
                            n_epoch: 1
                            n_minibatch: 1
                            learning_rate: 0.0004

                            vf_coef: 0.25
                            ent_coef: 0.01
                            clip_grad: 0.5
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.98
                            use_gae: True
                            gae_lambda: 0.95
                            use_advnorm: True

                            use_obsnorm: True
                            use_rewnorm: True
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 3
                            log_dir: "./logs/a2c/"
                            model_dir: "./models/a2c/"


            .. group-tab:: Box2D

                .. tabs::

                    .. group-tab:: BipedalWalker-v3

                        .. code-block:: yaml

                            agent: "A2C"
                            env_name: "Box2D"
                            env_id: "BipedalWalker-v3"
                            vectorize: "Dummy_Gym"
                            policy: "Gaussian_AC"
                            representation: "Basic_MLP"
                            runner: "DRL"

                            representation_hidden_size: [64,]
                            actor_hidden_size: [64,]
                            critic_hidden_size: [64,]
                            activation: 'LeakyReLU'

                            seed: 1
                            parallels: 10
                            running_steps: 1000000
                            n_steps: 128
                            n_epoch: 5
                            learning_rate: 0.0004

                            vf_coef: 0.25
                            ent_coef: 0.01
                            clip_grad: 0.5
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.98
                            use_gae: True
                            gae_lambda: 0.95
                            use_advnorm: True

                            use_obsnorm: True
                            use_rewnorm: True
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 5
                            log_dir: "./logs/a2c/"
                            model_dir: "./models/a2c/"


                    .. group-tab:: LunarLander-v2

                        .. code-block:: yaml

                            agent: "A2C"
                            env_name: "Box2D"
                            env_id: "LunarLander-v2"
                            vectorize: "Dummy_Gym"
                            policy: "Categorical_AC"
                            representation: "Basic_MLP"
                            runner: "DRL"

                            representation_hidden_size: [256,]
                            actor_hidden_size: [256,]
                            critic_hidden_size: [256,]
                            activation: 'LeakyReLU'

                            seed: 1
                            parallels: 10
                            running_steps: 300000
                            n_steps: 128
                            n_epoch: 1
                            learning_rate: 0.0004

                            vf_coef: 0.25
                            ent_coef: 0.01
                            clip_grad: 0.5
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.98
                            use_gae: True
                            gae_lambda: 0.95
                            use_advnorm: True

                            use_obsnorm: True
                            use_rewnorm: True
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 3
                            log_dir: "./logs/a2c/"
                            model_dir: "./models/a2c/"


            .. group-tab:: Atari

                .. code-block:: yaml

                    agent: "A2C"
                    vectorize: "Dummy_Atari"
                    env_name: "Atari"
                    env_id: "ALE/Breakout-v5"
                    obs_type: "grayscale"  # choice for Atari env: ram, rgb, grayscale
                    img_size: [84, 84]  # default is 210 x 160 in gym[Atari]
                    num_stack: 4  # frame stack trick
                    frame_skip: 4  # frame skip trick
                    noop_max: 30  # Do no-op action for a number of steps in [1, noop_max].
                    policy: "Categorical_AC"
                    representation: "Basic_CNN"
                    runner: "DRL"

                    # the following three arguments are for "Basic_CNN" representation.
                    filters: [32, 32, 64, 64]
                    kernels: [8, 4, 4, 4]
                    strides: [4, 2, 2, 2]
                    actor_hidden_size: [128, 128]
                    critic_hidden_size: [128, 128]
                    activation: "LeakyReLU"

                    seed: 1
                    parallels: 5
                    running_steps: 10000000  # 10M
                    n_steps: 256  #
                    n_epoch: 4
                    n_minibatch: 8
                    learning_rate: 0.0007

                    vf_coef: 0.25
                    ent_coef: 0.01
                    clip_grad: 0.2
                    clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                    gamma: 0.99
                    use_gae: True
                    gae_lambda: 0.95
                    use_advnorm: True

                    use_obsnorm: False
                    use_rewnorm: False
                    obsnorm_range: 5
                    rewnorm_range: 5

                    test_steps: 10000
                    eval_interval: 100000
                    test_episode: 3
                    log_dir: "./logs/a2c/"
                    model_dir: "./models/a2c/"


            .. group-tab:: MuJoCo

                .. code-block:: yaml

                    agent: "A2C"
                    env_name: "MuJoCo"
                    env_id: "Ant-v4"
                    vectorize: "Dummy_Gym"
                    policy: "Gaussian_AC"
                    representation: "Basic_MLP"
                    runner: "DRL"

                    representation_hidden_size: [256,]
                    actor_hidden_size: [256,]
                    critic_hidden_size: [256,]
                    activation: "LeakyReLU"

                    seed: 6782
                    parallels: 16
                    running_steps: 1000000  # 1M
                    n_steps: 16
                    n_epoch: 1
                    n_minibatch: 1
                    learning_rate: 0.0007

                    vf_coef: 0.25
                    ent_coef: 0.0
                    clip_grad: 0.5
                    clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                    gamma: 0.99
                    use_gae: True
                    gae_lambda: 0.95
                    use_advnorm: True

                    use_obsnorm: True
                    use_rewnorm: True
                    obsnorm_range: 5
                    rewnorm_range: 5

                    test_steps: 10000
                    eval_interval: 5000
                    test_episode: 5
                    log_dir: "./logs/a2c/"
                    model_dir: "./models/a2c/"



    .. group-tab:: SAC

        .. tabs::

            .. group-tab:: Classic Control

                .. tabs::

                    .. group-tab:: CartPole-v1

                        .. code-block:: yaml

                            agent: "SACDIS"
                            env_name: "Classic Control"
                            env_id: "CartPole-v1"
                            vectorize: "Dummy_Gym"
                            policy: "Discrete_SAC"
                            representation: "Basic_MLP"
                            runner: "DRL"

                            representation_hidden_size: [256,]
                            actor_hidden_size: [128,128,]
                            critic_hidden_size: [128,128,]
                            activation: "ReLU"

                            seed: 1
                            parallels: 16
                            n_size: 20000
                            batch_size: 256
                            actor_learning_rate: 0.001
                            critic_learning_rate: 0.01
                            gamma: 0.98
                            tau: 0.005

                            start_noise: 0.25
                            end_noise: 0.05
                            training_frequency: 2
                            running_steps: 500000
                            start_training: 2000
                            action_type: "DISCRETE"

                            use_obsnorm: False
                            use_rewnorm: False
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 1
                            log_dir: "./logs/sac/"
                            model_dir: "./models/sac/"


                    .. group-tab:: Acrobot-v1

                        .. code-block:: yaml

                            agent: "SACDIS"
                            env_name: "Classic Control"
                            env_id: "Acrobot-v1"
                            vectorize: "Dummy_Gym"
                            policy: "Discrete_SAC"
                            representation: "Basic_MLP"
                            runner: "DRL"

                            representation_hidden_size: [256,]
                            actor_hidden_size: [128,128,]
                            critic_hidden_size: [128,128,]
                            activation: "ReLU"

                            seed: 1
                            parallels: 16
                            n_size: 20000
                            batch_size: 256
                            actor_learning_rate: 0.001
                            critic_learning_rate: 0.01
                            gamma: 0.98
                            tau: 0.005

                            start_noise: 0.25
                            end_noise: 0.05
                            training_frequency: 2
                            running_steps: 500000
                            start_training: 2000
                            action_type: "DISCRETE"

                            use_obsnorm: False
                            use_rewnorm: False
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 1
                            log_dir: "./logs/sac/"
                            model_dir: "./models/sac/"


                    .. group-tab:: Pendulum-v1

                        .. code-block:: yaml

                            agent: "SAC"
                            env_name: "Classic Control"
                            env_id: "Pendulum-v1"
                            vectorize: "Dummy_Gym"
                            policy: "Gaussian_SAC"
                            representation: "Basic_MLP"
                            runner: "DRL"

                            representation_hidden_size: [256,]
                            actor_hidden_size: [256,]
                            critic_hidden_size: [256,]
                            activation: "ReLU"

                            seed: 1
                            parallels: 16
                            n_size: 20000
                            batch_size: 256
                            actor_learning_rate: 0.001
                            critic_learning_rate: 0.001
                            gamma: 0.98
                            tau: 0.005

                            start_noise: 0.25
                            end_noise: 0.05
                            training_frequency: 2
                            running_steps: 500000
                            start_training: 2000

                            use_obsnorm: False
                            use_rewnorm: False
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 1
                            log_dir: "./logs/sac/"
                            model_dir: "./models/sac/"


                    .. group-tab:: MountainCar-v0

                        .. code-block:: yaml

                            agent: "SACDIS"
                            env_name: "Classic Control"
                            env_id: "MountainCar-v0"
                            vectorize: "Dummy_Gym"
                            policy: "Discrete_SAC"
                            representation: "Basic_MLP"
                            runner: "DRL"

                            representation_hidden_size: [256,]
                            actor_hidden_size: [128,128,]
                            critic_hidden_size: [128,128,]
                            activation: "ReLU"

                            seed: 1
                            parallels: 16
                            n_size: 20000
                            batch_size: 256
                            actor_learning_rate: 0.001
                            critic_learning_rate: 0.01
                            gamma: 0.98
                            tau: 0.005

                            start_noise: 0.25
                            end_noise: 0.05
                            training_frequency: 2
                            running_steps: 500000
                            start_training: 2000
                            action_type: "DISCRETE"

                            use_obsnorm: False
                            use_rewnorm: False
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 1
                            log_dir: "./logs/sac/"
                            model_dir: "./models/sac/"


            .. group-tab:: Box2D

                .. tabs::

                    .. group-tab:: BipedalWalker-v3

                        .. code-block:: yaml

                            agent: "SAC"
                            env_name: "Box2D"
                            env_id: "BipedalWalker-v3"
                            vectorize: "Dummy_Gym"
                            policy: "Gaussian_SAC"
                            representation: "Basic_MLP"
                            runner: "DRL"

                            representation_hidden_size: [256,]
                            actor_hidden_size: [256,]
                            critic_hidden_size: [256,]
                            activation: "ReLU"

                            seed: 1
                            parallels: 16
                            n_size: 20000
                            batch_size: 256
                            actor_learning_rate: 0.001
                            critic_learning_rate: 0.001
                            gamma: 0.98
                            tau: 0.005

                            start_noise: 0.25
                            end_noise: 0.05
                            training_frequency: 2
                            running_steps: 500000
                            start_training: 2000

                            use_obsnorm: False
                            use_rewnorm: False
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 1
                            log_dir: "./logs/sac/"
                            model_dir: "./models/sac/"


                    .. group-tab:: LunarLander-v2

                        .. code-block:: yaml

                            agent: "SACDIS"
                            env_name: "Box2D"
                            env_id: "LunarLander-v2"
                            vectorize: "Dummy_Gym"
                            policy: "Discrete_SAC"
                            representation: "Basic_MLP"
                            runner: "DRL"

                            representation_hidden_size: [256,]
                            actor_hidden_size: [128,128,]
                            critic_hidden_size: [128,128,]
                            activation: "ReLU"

                            seed: 1
                            parallels: 16
                            n_size: 20000
                            batch_size: 256
                            actor_learning_rate: 0.001
                            critic_learning_rate: 0.01
                            gamma: 0.98
                            tau: 0.005

                            start_noise: 0.25
                            end_noise: 0.05
                            training_frequency: 2
                            running_steps: 500000
                            start_training: 2000
                            action_type: "DISCRETE"

                            use_obsnorm: False
                            use_rewnorm: False
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 1
                            log_dir: "./logs/sac/"
                            model_dir: "./models/sac/"


            .. group-tab:: Atari

                .. code-block:: yaml

                    agent: "SACDIS"
                    vectorize: "Dummy_Atari"
                    env_name: "Atari"
                    env_id: "ALE/Breakout-v5"
                    obs_type: "grayscale"  # choice for Atari env: ram, rgb, grayscale
                    img_size: [84, 84]  # default is 210 x 160 in gym[Atari]
                    num_stack: 4  # frame stack trick
                    frame_skip: 4  # frame skip trick
                    noop_max: 30  # Do no-op action for a number of steps in [1, noop_max].
                    representation: "Basic_CNN"
                    policy: "Discrete_SAC"
                    runner: "DRL"

                    filters: [32, 32, 64, 64]
                    kernels: [8, 4, 4, 4]
                    strides: [4, 2, 2, 2]
                    actor_hidden_size: [128, 128]
                    critic_hidden_size: [128, 128]
                    activation: "LeakyReLU"

                    seed: 1069
                    parallels: 5
                    n_size: 100000
                    batch_size: 32  # 64
                    actor_learning_rate: 0.001
                    critic_learning_rate: 0.001
                    gamma: 0.99
                    alpha: 0.01
                    tau: 0.005
                    learning_rate: 0.0007

                    start_noise: 0.25
                    end_noise: 0.05
                    training_frequency: 1
                    running_steps: 50000000  # 50M
                    start_training: 10000

                    use_obsnorm: False
                    use_rewnorm: False
                    obsnorm_range: 5
                    rewnorm_range: 5

                    test_steps: 10000
                    eval_interval: 500000
                    test_episode: 1
                    log_dir: "./logs/sac/"
                    model_dir: "./models/sac/"


            .. group-tab:: MuJoCo

                .. code-block:: yaml

                    agent: "SAC"
                    env_name: "MuJoCo"
                    env_id: "Ant-v4"
                    vectorize: "Dummy_Gym"
                    policy: "Gaussian_SAC"
                    representation: "Basic_Identical"
                    runner: "DRL"

                    representation_hidden_size:
                    actor_hidden_size: [256, 256]
                    critic_hidden_size: [256, 256]
                    activation: "LeakyReLU"

                    seed: 1
                    parallels: 4
                    n_size: 250000
                    batch_size: 256
                    actor_learning_rate: 0.001
                    critic_learning_rate: 0.001
                    gamma: 0.99
                    alpha: 0.2
                    tau: 0.005
                    learning_rate: 0.0003

                    start_noise: 0.25
                    end_noise: 0.01
                    training_frequency: 1
                    running_steps: 1000000
                    start_training: 10000

                    use_obsnorm: False
                    use_rewnorm: False
                    obsnorm_range: 5
                    rewnorm_range: 5

                    test_steps: 10000
                    eval_interval: 5000
                    test_episode: 5
                    log_dir: "./logs/sac/"
                    model_dir: "./models/sac/"



    .. group-tab:: DDPG

        .. tabs::

            .. group-tab:: Classic Control

                .. tabs::

                    .. group-tab:: Pendulum-v1

                        .. code-block:: yaml

                            agent: "DDPG"
                            env_name: "Classic Control"
                            env_id: "Pendulum-v1"
                            vectorize: "Dummy_Gym"
                            policy: "DDPG_Policy"
                            representation: "Basic_Identical"
                            runner: "DRL"

                            actor_hidden_size: [256,]
                            critic_hidden_size: [256,]
                            activation: "ReLU"

                            seed: 1
                            parallels: 10
                            n_size: 200000
                            batch_size: 256
                            actor_learning_rate: 0.001
                            critic_learning_rate: 0.001
                            gamma: 0.98
                            tau: 0.005
                            learning_rate: 0.0007

                            start_noise: 0.1
                            end_noise: 0.1
                            training_frequency: 1
                            running_steps: 500000
                            start_training: 1000

                            use_obsnorm: False
                            use_rewnorm: False
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 3
                            log_dir: "./logs/ddpg/"
                            model_dir: "./models/ddpg/"

            .. group-tab:: Box2D

                .. tabs::

                    .. group-tab:: BipedalWalker-v3

                        .. code-block:: yaml

                            agent: "DDPG"
                            env_name: "Box2D"
                            env_id: "BipedalWalker-v3"
                            vectorize: "Dummy_Gym"
                            policy: "DDPG_Policy"
                            representation: "Basic_Identical"
                            runner: "DRL"

                            actor_hidden_size: [256,]
                            critic_hidden_size: [256,]
                            activation: "ReLU"

                            seed: 1
                            parallels: 10
                            n_size: 200000
                            batch_size: 256
                            actor_learning_rate: 0.001
                            critic_learning_rate: 0.001
                            gamma: 0.98
                            tau: 0.005
                            learning_rate: 0.0007

                            start_noise: 0.1
                            end_noise: 0.1
                            training_frequency: 1
                            running_steps: 500000
                            start_training: 1000

                            use_obsnorm: False
                            use_rewnorm: False
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 1
                            log_dir: "./logs/ddpg/"
                            model_dir: "./models/ddpg/"

            .. group-tab:: MuJoCo

                .. code-block:: yaml

                    agent: "DDPG"
                    env_name: "MuJoCo"
                    env_id: "Ant-v4"
                    vectorize: "Dummy_Gym"
                    policy: "DDPG_Policy"
                    representation: "Basic_Identical"
                    runner: "DRL"

                    representation_hidden_size:  # If you choose Basic_Identical representation, then ignore this value
                    actor_hidden_size: [400, 300]
                    critic_hidden_size: [400, 300]
                    activation: "LeakyReLU"

                    seed: 19089
                    parallels: 4  # number of environments
                    n_size: 50000  # replay buffer size
                    batch_size: 100
                    actor_learning_rate: 0.001
                    critic_learning_rate: 0.001
                    gamma: 0.99
                    tau: 0.005

                    start_noise: 0.5
                    end_noise: 0.1
                    training_frequency: 1
                    running_steps: 1000000  # 1M
                    start_training: 10000

                    use_obsnorm: False
                    use_rewnorm: False
                    obsnorm_range: 5
                    rewnorm_range: 5

                    test_steps: 10000
                    eval_interval: 5000
                    test_episode: 5
                    log_dir: "./logs/ddpg/"
                    model_dir: "./models/ddpg/"



    .. group-tab:: TD3

        .. tabs::

            .. group-tab:: Classic Control

                .. tabs::

                    .. group-tab:: Pendulum-v1

                        .. code-block:: yaml

                            agent: "TD3"
                            env_name: "Classic Control"
                            env_id: "Pendulum-v1"
                            vectorize: "Dummy_Gym"
                            representation: "Basic_Identical"
                            policy: "TD3_Policy"
                            runner: "DRL"

                            actor_hidden_size: [256, ]
                            critic_hidden_size: [256, ]
                            activation: "LeakyReLU"

                            seed: 1
                            parallels: 10
                            n_size: 20000
                            batch_size: 256
                            actor_learning_rate: 0.0005
                            critic_learning_rate: 0.001
                            gamma: 0.98
                            tau: 0.005
                            actor_update_delay: 3

                            start_noise: 0.25
                            end_noise: 0.05
                            training_frequency: 2
                            running_steps: 500000
                            start_training: 2000

                            use_obsnorm: False
                            use_rewnorm: False
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 1
                            log_dir: "./logs/td3/"
                            model_dir: "./models/td3/"

            .. group-tab:: Box2D

                .. tabs::

                    .. group-tab:: BipedalWalker-v3

                        .. code-block:: yaml

                            agent: "TD3"
                            env_name: "Box2D"
                            env_id: "BipedalWalker-v3"
                            vectorize: "Dummy_Gym"
                            representation: "Basic_Identical"
                            policy: "TD3_Policy"
                            runner: "DRL"

                            actor_hidden_size: [256, ]
                            critic_hidden_size: [256, ]
                            activation: "LeakyReLU"

                            seed: 1
                            parallels: 10
                            n_size: 20000
                            batch_size: 256
                            actor_learning_rate: 0.0005
                            critic_learning_rate: 0.001
                            gamma: 0.98
                            tau: 0.005
                            actor_update_delay: 3

                            start_noise: 0.25
                            end_noise: 0.05
                            training_frequency: 2
                            running_steps: 500000
                            start_training: 2000

                            use_obsnorm: False
                            use_rewnorm: False
                            obsnorm_range: 5
                            rewnorm_range: 5

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 1
                            log_dir: "./logs/td3/"
                            model_dir: "./models/td3/"

            .. group-tab:: MuJoCo

                .. code-block:: yaml

                    agent: "TD3"
                    env_name: "MuJoCo"
                    env_id: "Ant-v4"
                    vectorize: "Dummy_Gym"
                    policy: "TD3_Policy"
                    representation: "Basic_Identical"
                    runner: "DRL"

                    representation_hidden_size:  # If you choose Basic_Identical representation, then ignore this value
                    actor_hidden_size: [400, 300]
                    critic_hidden_size: [400, 300]
                    activation: "LeakyReLU"

                    seed: 6782
                    parallels: 4  # number of environments
                    n_size: 50000
                    batch_size: 100
                    actor_learning_rate: 0.001
                    actor_update_delay: 2
                    critic_learning_rate: 0.001
                    gamma: 0.99
                    tau: 0.005

                    start_noise: 0.5
                    end_noise: 0.1
                    training_frequency: 1
                    running_steps: 1000000
                    start_training: 10000

                    use_obsnorm: False
                    use_rewnorm: False
                    obsnorm_range: 5
                    rewnorm_range: 5

                    test_steps: 10000
                    eval_interval: 5000
                    test_episode: 5
                    log_dir: "./logs/td3/"
                    model_dir: "./models/td3/"


.. raw:: html

   <br><hr>

MARL Implementations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. tabs::

    .. group-tab:: IQL

        .. tabs::

            .. group-tab:: MPE

                .. tabs::

                    .. group-tab:: simple_spread_v3

                        .. code-block:: yaml

                            agent: "IQL"  # the learning algorithms_marl
                            env_name: "mpe"
                            env_id: "simple_spread_v3"
                            continuous_action: False
                            policy: "Basic_Q_network_marl"
                            representation: "Basic_MLP"
                            vectorize: "Dummy_Pettingzoo"
                            runner: "Pettingzoo_Runner"

                            use_recurrent: False
                            rnn:
                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            activation: "ReLU"

                            seed: 1
                            parallels: 16
                            buffer_size: 100000
                            batch_size: 256
                            learning_rate: 0.001
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 2500000
                            start_training: 1000  # start training after n episodes
                            running_steps: 10000000  # 10M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 100

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 100000
                            test_episode: 5
                            log_dir: "./logs/iql/"
                            model_dir: "./models/iql/"


            .. group-tab:: Magent2

                .. tabs::

                    .. group-tab:: adversarial_pursuit_v4

                        .. code-block:: yaml

                            agent: "IQL"  # the learning algorithms_marl
                            env_name: "MAgent2"
                            env_id: "adversarial_pursuit_v4"
                            minimap_mode: False
                            max_cycles: 500
                            extra_features: False
                            map_size: 45
                            render_mode: "rgb_array"
                            policy: "Basic_Q_network_marl"
                            representation: "Basic_MLP"
                            vectorize: "Dummy_MAgent"
                            runner: "MAgent_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: False
                            rnn:
                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            activation: "ReLU"

                            seed: 1
                            parallels: 10
                            buffer_size: 20000
                            batch_size: 256
                            learning_rate: 0.001
                            gamma: 0.95  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 2500000
                            start_training: 1000  # start training after n episodes
                            running_steps: 10000000  # 10M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 100

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 100000
                            test_episode: 5
                            log_dir: "./logs/iql/"
                            model_dir: "./models/iql/"

            .. group-tab:: SC2

                .. tabs::

                    .. group-tab:: 1c3s5z

                        .. code-block:: yaml

                            agent: "IQL"  # the learning algorithms_marl
                            env_name: "StarCraft2"
                            env_id: "1c3s5z"
                            fps: 15
                            policy: "Basic_Q_network_marl"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 50000
                            start_training: 1000  # start training after n episodes
                            running_steps: 2000000  # 2M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 20000
                            test_episode: 16
                            log_dir: "./logs/iql/"
                            model_dir: "./models/iql/"

                    .. group-tab:: 2m_vs_1z

                        .. code-block:: yaml

                            agent: "IQL"  # the learning algorithms_marl
                            global_state: False
                            env_name: "StarCraft2"
                            env_id: "2m_vs_1z"
                            fps: 15
                            policy: "Basic_Q_network_marl"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 50000
                            start_training: 1000  # start training after n episodes
                            running_steps: 1000000  # 1M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 10000
                            test_episode: 16
                            log_dir: "./logs/iql/"
                            model_dir: "./models/iql/"


                    .. group-tab:: 2s3z

                        .. code-block:: yaml

                            agent: "IQL"  # the learning algorithms_marl
                            env_name: "StarCraft2"
                            env_id: "2s3z"
                            fps: 15
                            policy: "Basic_Q_network_marl"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 50000
                            start_training: 1000  # start training after n episodes
                            running_steps: 2000000  # 2M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 20000
                            test_episode: 16
                            log_dir: "./logs/iql/"
                            model_dir: "./models/iql/"

                    .. group-tab:: 3m

                        .. code-block:: yaml

                            agent: "IQL"  # the learning algorithms_marl
                            env_name: "StarCraft2"
                            env_id: "3m"
                            fps: 15
                            policy: "Basic_Q_network_marl"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 50000
                            start_training: 1000  # start training after n episodes
                            running_steps: 1000000  # 1M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 10000
                            test_episode: 16
                            log_dir: "./logs/iql/"
                            model_dir: "./models/iql/"


                    .. group-tab:: 5m_vs_6m

                        .. code-block:: yaml

                            agent: "IQL"  # the learning algorithms_marl
                            env_name: "StarCraft2"
                            env_id: "5m_vs_6m"
                            fps: 15
                            policy: "Basic_Q_network_marl"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 50000
                            start_training: 1000  # start training after n episodes
                            running_steps: 10000000  # 10M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 100000
                            test_episode: 16
                            log_dir: "./logs/iql/"
                            model_dir: "./models/iql/"


                    .. group-tab:: 8m

                        .. code-block:: yaml

                            agent: "IQL"  # the learning algorithms_marl
                            env_name: "StarCraft2"
                            env_id: "8m"
                            fps: 15
                            policy: "Basic_Q_network_marl"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 50000
                            start_training: 1000  # start training after n episodes
                            running_steps: 1000000  # 1M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 10000
                            test_episode: 16
                            log_dir: "./logs/iql/"
                            model_dir: "./models/iql/"


                    .. group-tab:: 8m_vd_9m

                        .. code-block:: yaml

                            agent: "IQL"  # the learning algorithms_marl
                            env_name: "StarCraft2"
                            env_id: "8m_vs_9m"
                            fps: 15
                            policy: "Basic_Q_network_marl"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 50000
                            start_training: 1000  # start training after n episodes
                            running_steps: 10000000  # 10M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 100000
                            test_episode: 16
                            log_dir: "./logs/iql/"
                            model_dir: "./models/iql/"


                    .. group-tab:: 25m

                        .. code-block:: yaml

                            agent: "IQL"  # the learning algorithms_marl
                            env_name: "StarCraft2"
                            env_id: "25m"
                            fps: 15
                            policy: "Basic_Q_network_marl"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 50000
                            start_training: 1000  # start training after n episodes
                            running_steps: 5000000  # 5M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 50000
                            test_episode: 16
                            log_dir: "./logs/iql/"
                            model_dir: "./models/iql/"


                    .. group-tab:: corridor

                        .. code-block:: yaml

                            agent: "IQL"  # the learning algorithms_marl
                            env_name: "StarCraft2"
                            env_id: "corridor"
                            fps: 15
                            policy: "Basic_Q_network_marl"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 50000
                            start_training: 1000  # start training after n episodes
                            running_steps: 10000000  # 10M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 100000
                            test_episode: 16
                            log_dir: "./logs/iql/"
                            model_dir: "./models/iql/"

                    
                    .. group-tab:: MMM2

                        .. code-block:: yaml

                            agent: "IQL"  # the learning algorithms_marl
                            env_name: "StarCraft2"
                            env_id: "MMM2"
                            fps: 15
                            policy: "Basic_Q_network_marl"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 50000
                            start_training: 1000  # start training after n episodes
                            running_steps: 10000000  # 10M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 100000
                            test_episode: 16
                            log_dir: "./logs/iql/"
                            model_dir: "./models/iql/"


            .. group-tab:: Football

                .. tabs::

                    .. group-tab:: 3v1

                        .. code-block:: yaml

                            agent: "IQL"  # the learning algorithms_marl
                            global_state: False
                            # environment settings
                            env_name: "Football"
                            scenario: "academy_3_vs_1_with_keeper"
                            use_stacked_frames: False  # Whether to use stacked_frames
                            num_agent: 3
                            num_adversary: 0
                            obs_type: "simple115v2"  # representation used to build the observation, choices: ["simple115v2", "extracted", "pixels_gray", "pixels"]
                            rewards_type: "scoring,checkpoints"  # comma separated list of rewards to be added
                            smm_width: 96  # width of super minimap
                            smm_height: 72  # height of super minimap
                            fps: 15
                            policy: "Basic_Q_network_marl"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_Football"
                            runner: "Football_Runner"

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [128, ]
                            recurrent_hidden_size: 128
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [128, ]
                            q_hidden_size: [128, ]  # the units for each hidden layer
                            activation: "ReLU"

                            seed: 1
                            parallels: 50
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 1000000
                            start_training: 1000  # start training after n episodes
                            running_steps: 25000000  # 25M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 250000
                            test_episode: 50
                            log_dir: "./logs/iql/"
                            model_dir: "./models/iql/"
                            videos_dir: "./videos/iql/"

    
    .. group-tab:: VDN

        .. tabs::

            .. group-tab:: MPE

                .. tabs::

                    .. group-tab:: simple_spread_v3

                        .. code-block:: yaml

                            agent: "VDN"  # the learning algorithms_marl
                            env_name: "mpe"
                            env_id: "simple_spread_v3"
                            continuous_action: False
                            policy: "Mixing_Q_network"
                            representation: "Basic_MLP"
                            vectorize: "Dummy_Pettingzoo"
                            runner: "Pettingzoo_Runner"
                            on_policy: False

                            use_recurrent: False
                            rnn:
                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            activation: "ReLU"

                            seed: 1
                            parallels: 16
                            buffer_size: 100000
                            batch_size: 256
                            learning_rate: 0.001
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 2500000
                            start_training: 1000  # start training after n episodes
                            running_steps: 10000000  # 10M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 100

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 100000
                            test_episode: 5
                            log_dir: "./logs/vdn/"
                            model_dir: "./models/vdn/"


            .. group-tab:: SC2

                .. tabs::

                    .. group-tab:: 1c3s5z

                        .. code-block:: yaml

                            agent: "VDN"  # the learning algorithms_marl
                            global_state: False
                            env_name: "StarCraft2"
                            env_id: "1c3s5z"
                            fps: 15
                            policy: "Mixing_Q_network"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 50000
                            start_training: 1000  # start training after n episodes
                            running_steps: 2000000  # 2M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 20000
                            test_episode: 16
                            log_dir: "./logs/vdn/"
                            model_dir: "./models/vdn/"

                    .. group-tab:: 2m_vs_1z

                        .. code-block:: yaml

                            agent: "VDN"  # the learning algorithms_marl
                            global_state: False
                            env_name: "StarCraft2"
                            env_id: "2m_vs_1z"
                            fps: 15
                            policy: "Mixing_Q_network"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 50000
                            start_training: 1000  # start training after n episodes
                            running_steps: 1000000  # 1M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 10000
                            test_episode: 16
                            log_dir: "./logs/vdn/"
                            model_dir: "./models/vdn/"


                    .. group-tab:: 2s3z

                        .. code-block:: yaml

                            agent: "VDN"  # the learning algorithms_marl
                            global_state: False
                            env_name: "StarCraft2"
                            env_id: "2s3z"
                            fps: 15
                            policy: "Mixing_Q_network"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 50000
                            start_training: 1000  # start training after n episodes
                            running_steps: 2000000  # 2M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 20000
                            test_episode: 16
                            log_dir: "./logs/vdn/"
                            model_dir: "./models/vdn/"


                    .. group-tab:: 3m

                        .. code-block:: yaml

                            agent: "VDN"  # the learning algorithms_marl
                            global_state: False
                            env_name: "StarCraft2"
                            env_id: "3m"
                            fps: 15
                            policy: "Mixing_Q_network"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 50000
                            start_training: 1000  # start training after n episodes
                            running_steps: 1000000  # 1M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 10000
                            test_episode: 16
                            log_dir: "./logs/vdn/"
                            model_dir: "./models/vdn/"


                    .. group-tab:: 5m_vs_6m

                        .. code-block:: yaml

                            agent: "VDN"  # the learning algorithms_marl
                            global_state: False
                            env_name: "StarCraft2"
                            env_id: "5m_vs_6m"
                            fps: 15
                            policy: "Mixing_Q_network"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 50000
                            start_training: 1000  # start training after n episodes
                            running_steps: 10000000  # 10M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 100000
                            test_episode: 16
                            log_dir: "./logs/vdn/"
                            model_dir: "./models/vdn/"


                    .. group-tab:: 8m

                        .. code-block:: yaml

                            agent: "VDN"  # the learning algorithms_marl
                            global_state: False
                            env_name: "StarCraft2"
                            env_id: "8m"
                            fps: 15
                            policy: "Mixing_Q_network"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 50000
                            start_training: 1000  # start training after n episodes
                            running_steps: 1000000  # 1M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 10000
                            test_episode: 16
                            log_dir: "./logs/vdn/"
                            model_dir: "./models/vdn/"


                    .. group-tab:: 8m_vd_9m

                        .. code-block:: yaml

                            agent: "VDN"  # the learning algorithms_marl
                            global_state: False
                            env_name: "StarCraft2"
                            env_id: "8m_vs_9m"
                            fps: 15
                            policy: "Mixing_Q_network"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 50000
                            start_training: 1000  # start training after n episodes
                            running_steps: 10000000  # 10M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 100000
                            test_episode: 16
                            log_dir: "./logs/vdn/"
                            model_dir: "./models/vdn/"


                    .. group-tab:: 25m

                        .. code-block:: yaml

                            agent: "VDN"  # the learning algorithms_marl
                            global_state: False
                            env_name: "StarCraft2"
                            env_id: "25m"
                            fps: 15
                            policy: "Mixing_Q_network"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 50000
                            start_training: 1000  # start training after n episodes
                            running_steps: 5000000  # 5M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 50000
                            test_episode: 16
                            log_dir: "./logs/vdn/"
                            model_dir: "./models/vdn/"


                    .. group-tab:: corridor

                        .. code-block:: yaml

                            agent: "VDN"  # the learning algorithms_marl
                            global_state: False
                            env_name: "StarCraft2"
                            env_id: "corridor"
                            fps: 15
                            policy: "Mixing_Q_network"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 50000
                            start_training: 1000  # start training after n episodes
                            running_steps: 10000000  # 10M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 100000
                            test_episode: 16
                            log_dir: "./logs/vdn/"
                            model_dir: "./models/vdn/"

                    
                    .. group-tab:: MMM2

                        .. code-block:: yaml

                            agent: "VDN"  # the learning algorithms_marl
                            global_state: False
                            env_name: "StarCraft2"
                            env_id: "MMM2"
                            fps: 15
                            policy: "Mixing_Q_network"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 50000
                            start_training: 1000  # start training after n episodes
                            running_steps: 10000000  # 1M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 100000
                            test_episode: 16
                            log_dir: "./logs/vdn/"
                            model_dir: "./models/vdn/"


            .. group-tab:: Football

                .. tabs::

                    .. group-tab:: 3v1

                        .. code-block:: yaml

                            agent: "VDN"  # the learning algorithms_marl
                            global_state: False
                            # environment settings
                            env_name: "Football"
                            scenario: "academy_3_vs_1_with_keeper"
                            use_stacked_frames: False  # Whether to use stacked_frames
                            num_agent: 3
                            num_adversary: 0
                            obs_type: "simple115v2"  # representation used to build the observation, choices: ["simple115v2", "extracted", "pixels_gray", "pixels"]
                            rewards_type: "scoring,checkpoints"  # comma separated list of rewards to be added
                            smm_width: 96  # width of super minimap
                            smm_height: 72  # height of super minimap
                            fps: 15
                            policy: "Mixing_Q_network"
                            representation: "Basic_RNN"
                            vectorize: "Dummy_Football"
                            runner: "Football_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [128, ]
                            recurrent_hidden_size: 128
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [128, ]
                            q_hidden_size: [128, ]  # the units for each hidden layer
                            activation: "ReLU"

                            seed: 1
                            parallels: 50
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 1000000
                            start_training: 1000  # start training after n episodes
                            running_steps: 25000000  # 25M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 250000
                            test_episode: 50
                            log_dir: "./logs/vdn/"
                            model_dir: "./models/vdn/"
                            videos_dir: "./videos/vdn/"

    
    .. group-tab:: QMIX

        .. tabs::

            .. group-tab:: MPE

                .. tabs::

                    .. group-tab:: simple_spread_v3

                        .. code-block:: yaml

                            agent: "QMIX"  # the learning algorithms_marl
                            env_name: "mpe"
                            env_id: "simple_spread_v3"
                            continuous_action: False
                            policy: "Mixing_Q_network"
                            representation: "Basic_MLP"
                            vectorize: "Dummy_Pettingzoo"
                            runner: "Pettingzoo_Runner"

                            use_recurrent: False
                            rnn:
                            representation_hidden_size: [64, ]
                            q_hidden_size: [128, ]  # the units for each hidden layer
                            activation: "ReLU"

                            hidden_dim_mixing_net: 64  # hidden units of mixing network
                            hidden_dim_hyper_net: 64  # hidden units of hyper network

                            seed: 1
                            parallels: 16
                            buffer_size: 100000
                            batch_size: 256
                            learning_rate: 0.001
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 2500000
                            start_training: 1000  # start training after n episodes
                            running_steps: 10000000  # 10M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 100

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 100000
                            test_episode: 5
                            log_dir: "./logs/qmix/"
                            model_dir: "./models/qmix/"


            .. group-tab:: SC2

                .. tabs::

                    .. group-tab:: 1c3s5z

                        .. code-block:: yaml

                            agent: "QMIX"  # the learning algorithms_marl
                            global_state: True
                            env_name: "StarCraft2"
                            env_id: "1c3s5z"
                            fps: 15
                            policy: "Mixing_Q_network"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            activation: "ReLU"

                            hidden_dim_mixing_net: 32  # hidden units of mixing network
                            hidden_dim_hyper_net: 32  # hidden units of hyper network

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 50000
                            start_training: 1000  # start training after n episodes
                            running_steps: 2000000  # 2M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 20000
                            test_episode: 16
                            log_dir: "./logs/qmix/"
                            model_dir: "./models/qmix/"

                    .. group-tab:: 2m_vs_1z

                        .. code-block:: yaml

                            agent: "QMIX"  # the learning algorithms_marl
                            global_state: True
                            env_name: "StarCraft2"
                            env_id: "2m_vs_1z"
                            fps: 15
                            policy: "Mixing_Q_network"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            activation: "ReLU"

                            hidden_dim_mixing_net: 32  # hidden units of mixing network
                            hidden_dim_hyper_net: 32  # hidden units of hyper network

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 50000
                            start_training: 1000  # start training after n episodes
                            running_steps: 1000000  # 1M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 10000
                            test_episode: 16
                            log_dir: "./logs/qmix/"
                            model_dir: "./models/qmix/"

                    .. group-tab:: 2s3z

                        .. code-block:: yaml

                            agent: "QMIX"  # the learning algorithms_marl
                            global_state: True
                            env_name: "StarCraft2"
                            env_id: "2s3z"
                            fps: 15
                            policy: "Mixing_Q_network"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            activation: "ReLU"

                            hidden_dim_mixing_net: 32  # hidden units of mixing network
                            hidden_dim_hyper_net: 32  # hidden units of hyper network

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 50000
                            start_training: 1000  # start training after n episodes
                            running_steps: 2000000  # 2M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 20000
                            test_episode: 16
                            log_dir: "./logs/qmix/"
                            model_dir: "./models/qmix/"

                    .. group-tab:: 3m

                        .. code-block:: yaml

                            agent: "QMIX"  # the learning algorithms_marl
                            global_state: True
                            env_name: "StarCraft2"
                            env_id: "3m"
                            fps: 15
                            policy: "Mixing_Q_network"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            activation: "ReLU"

                            hidden_dim_mixing_net: 32  # hidden units of mixing network
                            hidden_dim_hyper_net: 32  # hidden units of hyper network

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 50000
                            start_training: 1000  # start training after n episodes
                            running_steps: 1000000  # 1M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 10000
                            test_episode: 16
                            log_dir: "./logs/qmix/"
                            model_dir: "./models/qmix/"

                    .. group-tab:: 5m_vs_6m

                        .. code-block:: yaml

                            agent: "QMIX"  # the learning algorithms_marl
                            global_state: True
                            env_name: "StarCraft2"
                            env_id: "5m_vs_6m"
                            fps: 15
                            policy: "Mixing_Q_network"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            activation: "ReLU"

                            hidden_dim_mixing_net: 32  # hidden units of mixing network
                            hidden_dim_hyper_net: 32  # hidden units of hyper network

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 5000000
                            start_training: 1000  # start training after n episodes
                            running_steps: 10000000  # 10M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 100000
                            test_episode: 16
                            log_dir: "./logs/qmix/"
                            model_dir: "./models/qmix/"

                    .. group-tab:: 8m

                        .. code-block:: yaml

                            agent: "QMIX"  # the learning algorithms_marl
                            global_state: True
                            env_name: "StarCraft2"
                            env_id: "8m"
                            fps: 15
                            policy: "Mixing_Q_network"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            activation: "ReLU"

                            hidden_dim_mixing_net: 32  # hidden units of mixing network
                            hidden_dim_hyper_net: 32  # hidden units of hyper network

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 500000
                            start_training: 1000  # start training after n episodes
                            running_steps: 1000000  # 1M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 10000
                            test_episode: 16
                            log_dir: "./logs/qmix/"
                            model_dir: "./models/qmix/"

                    .. group-tab:: 8m_vd_9m

                        .. code-block:: yaml

                            agent: "QMIX"  # the learning algorithms_marl
                            global_state: True
                            env_name: "StarCraft2"
                            env_id: "8m_vs_9m"
                            fps: 15
                            policy: "Mixing_Q_network"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            activation: "ReLU"

                            hidden_dim_mixing_net: 32  # hidden units of mixing network
                            hidden_dim_hyper_net: 32  # hidden units of hyper network

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 5000000
                            start_training: 1000  # start training after n episodes
                            running_steps: 10000000  # 10M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 100000
                            test_episode: 16
                            log_dir: "./logs/qmix/"
                            model_dir: "./models/qmix/"

                    .. group-tab:: 25m

                        .. code-block:: yaml

                            agent: "QMIX"  # the learning algorithms_marl
                            global_state: True
                            env_name: "StarCraft2"
                            env_id: "25m"
                            fps: 15
                            policy: "Mixing_Q_network"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            activation: "ReLU"

                            hidden_dim_mixing_net: 32  # hidden units of mixing network
                            hidden_dim_hyper_net: 32  # hidden units of hyper network

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 1000000
                            start_training: 1000  # start training after n episodes
                            running_steps: 5000000  # 5M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 50000
                            test_episode: 16
                            log_dir: "./logs/qmix/"
                            model_dir: "./models/qmix/"

                    .. group-tab:: corridor

                        .. code-block:: yaml

                            agent: "QMIX"  # the learning algorithms_marl
                            global_state: True
                            env_name: "StarCraft2"
                            env_id: "corridor"
                            fps: 15
                            policy: "Mixing_Q_network"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            activation: "ReLU"

                            hidden_dim_mixing_net: 32  # hidden units of mixing network
                            hidden_dim_hyper_net: 32  # hidden units of hyper network

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 1000000
                            start_training: 1000  # start training after n episodes
                            running_steps: 10000000  # 10M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 100000
                            test_episode: 16
                            log_dir: "./logs/qmix/"
                            model_dir: "./models/qmix/"

                    .. group-tab:: MMM2

                        .. code-block:: yaml

                            agent: "QMIX"  # the learning algorithms_marl
                            global_state: True
                            env_name: "StarCraft2"
                            env_id: "MMM2"
                            fps: 15
                            policy: "Mixing_Q_network"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            activation: "ReLU"

                            hidden_dim_mixing_net: 32  # hidden units of mixing network
                            hidden_dim_hyper_net: 32  # hidden units of hyper network

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 1000000
                            start_training: 1000  # start training after n episodes
                            running_steps: 10000000  # 10M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 100000
                            test_episode: 16
                            log_dir: "./logs/qmix/"
                            model_dir: "./models/qmix/"

            .. group-tab:: Football

                .. tabs::

                    .. group-tab:: 3v1

                        .. code-block:: yaml

                            agent: "QMIX"  # the learning algorithms_marl
                            global_state: True
                            # environment settings
                            env_name: "Football"
                            scenario: "academy_3_vs_1_with_keeper"
                            use_stacked_frames: False  # Whether to use stacked_frames
                            num_agent: 3
                            num_adversary: 0
                            obs_type: "simple115v2"  # representation used to build the observation, choices: ["simple115v2", "extracted", "pixels_gray", "pixels"]
                            rewards_type: "scoring,checkpoints"  # comma separated list of rewards to be added
                            smm_width: 96  # width of super minimap
                            smm_height: 72  # height of super minimap
                            fps: 15
                            policy: "Mixing_Q_network"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_Football"
                            runner: "Football_Runner"

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [128, ]
                            recurrent_hidden_size: 128
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [128, ]
                            q_hidden_size: [128, ]  # the units for each hidden layer
                            activation: "ReLU"

                            hidden_dim_mixing_net: 128  # hidden units of mixing network
                            hidden_dim_hyper_net: 128  # hidden units of hyper network

                            seed: 1
                            parallels: 50
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 50000
                            start_training: 1000  # start training after n episodes
                            running_steps: 25000000  # 25M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 250000
                            test_episode: 50
                            log_dir: "./logs/qmix/"
                            model_dir: "./models/qmix/"
                            videos_dir: "./videos/qmix/"


    .. group-tab:: WQMIX

        .. tabs::

            .. group-tab:: MPE

                .. tabs::

                    .. group-tab:: simple_spread_v3

                        .. code-block:: yaml

                            agent: "OWQMIX"  # choice: CWQMIX, OWQMIX
                            env_name: "mpe"
                            env_id: "simple_spread_v3"
                            continuous_action: False
                            policy: "Weighted_Mixing_Q_network"
                            representation: "Basic_MLP"
                            vectorize: "Dummy_Pettingzoo"
                            runner: "Pettingzoo_Runner"
                            on_policy: False

                            use_recurrent: False
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [ 64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [128, ]  # for Basic_MLP representation
                            q_hidden_size: [128, ]  # the units for each hidden layer
                            activation: "ReLU"
                            alpha: 0.1

                            hidden_dim_mixing_net: 32  # hidden units of mixing network
                            hidden_dim_hyper_net: 64  # hidden units of hyper network
                            hidden_dim_ff_mix_net: 256  # hidden units of mixing network

                            seed: 1
                            parallels: 16
                            buffer_size: 100000
                            batch_size: 256
                            learning_rate: 0.001
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 5000000
                            start_training: 1000  # start training after n episodes
                            running_steps: 10000000  # 10M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 100000
                            test_episode: 5
                            log_dir: "./logs/wqmix/"
                            model_dir: "./models/wqmix/"

            .. group-tab:: SC2

                .. tabs::

                    .. group-tab:: 1c3s5z

                        .. code-block:: yaml

                            agent: "OWQMIX"  # choice: CWQMIX, OWQMIX
                            env_name: "StarCraft2"
                            env_id: "1c3s5z"
                            fps: 15
                            policy: "Weighted_Mixing_Q_network"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            activation: "ReLU"
                            alpha: 0.1

                            hidden_dim_mixing_net: 32  # hidden units of mixing network
                            hidden_dim_hyper_net: 64  # hidden units of hyper network

                            hidden_dim_ff_mix_net: 256  # hidden units of mixing network

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 50000
                            start_training: 1000  # start training after n episodes
                            running_steps: 2000000  # 2M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 20000
                            test_episode: 16
                            log_dir: "./logs/wqmix/"
                            model_dir: "./models/wqmix/"

                    .. group-tab:: 2m_vs_1z

                        .. code-block:: yaml

                            agent: "OWQMIX"  # choice: CWQMIX, OWQMIX
                            env_name: "StarCraft2"
                            env_id: "2m_vs_1z"
                            fps: 15
                            policy: "Weighted_Mixing_Q_network"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            activation: "ReLU"
                            alpha: 0.1

                            hidden_dim_mixing_net: 32  # hidden units of mixing network
                            hidden_dim_hyper_net: 64  # hidden units of hyper network

                            hidden_dim_ff_mix_net: 256  # hidden units of mixing network

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 50000
                            start_training: 1000  # start training after n episodes
                            running_steps: 1000000  # 1M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 10000
                            test_episode: 16
                            log_dir: "./logs/wqmix/"
                            model_dir: "./models/wqmix/"

                    .. group-tab:: 2s3z

                        .. code-block:: yaml

                            agent: "OWQMIX"  # choice: CWQMIX, OWQMIX
                            env_name: "StarCraft2"
                            env_id: "2s3z"
                            fps: 15
                            policy: "Weighted_Mixing_Q_network"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            activation: "ReLU"
                            alpha: 0.1

                            hidden_dim_mixing_net: 32  # hidden units of mixing network
                            hidden_dim_hyper_net: 64  # hidden units of hyper network

                            hidden_dim_ff_mix_net: 256  # hidden units of mixing network

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 50000
                            start_training: 1000  # start training after n episodes
                            running_steps: 2000000  # 2M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 20000
                            test_episode: 16
                            log_dir: "./logs/wqmix/"
                            model_dir: "./models/wqmix/"

                    .. group-tab:: 3m

                        .. code-block:: yaml

                            agent: "OWQMIX"  # choice: CWQMIX, OWQMIX
                            env_name: "StarCraft2"
                            env_id: "3m"
                            fps: 15
                            policy: "Weighted_Mixing_Q_network"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            activation: "ReLU"
                            alpha: 0.1

                            hidden_dim_mixing_net: 32  # hidden units of mixing network
                            hidden_dim_hyper_net: 64  # hidden units of hyper network

                            hidden_dim_ff_mix_net: 256  # hidden units of mixing network

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 50000
                            start_training: 1000  # start training after n episodes
                            running_steps: 1000000  # 1M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 10000
                            test_episode: 16
                            log_dir: "./logs/wqmix/"
                            model_dir: "./models/wqmix/"

                    .. group-tab:: 5m_vs_6m

                        .. code-block:: yaml

                            agent: "OWQMIX"  # choice: CWQMIX, OWQMIX
                            env_name: "StarCraft2"
                            env_id: "5m_vs_6m"
                            fps: 15
                            policy: "Weighted_Mixing_Q_network"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            activation: "ReLU"
                            alpha: 0.1

                            hidden_dim_mixing_net: 32  # hidden units of mixing network
                            hidden_dim_hyper_net: 64  # hidden units of hyper network

                            hidden_dim_ff_mix_net: 256  # hidden units of mixing network

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 1000000
                            start_training: 1000  # start training after n episodes
                            running_steps: 10000000  # 10M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 100000
                            test_episode: 16
                            log_dir: "./logs/wqmix/"
                            model_dir: "./models/wqmix/"

                    .. group-tab:: 8m

                        .. code-block:: yaml

                            agent: "OWQMIX"  # choice: CWQMIX, OWQMIX
                            env_name: "StarCraft2"
                            env_id: "8m"
                            fps: 15
                            policy: "Weighted_Mixing_Q_network"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            activation: "ReLU"
                            alpha: 0.1

                            hidden_dim_mixing_net: 32  # hidden units of mixing network
                            hidden_dim_hyper_net: 64  # hidden units of hyper network

                            hidden_dim_ff_mix_net: 256  # hidden units of mixing network

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 50000
                            start_training: 1000  # start training after n episodes
                            running_steps: 1000000  # 1M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 10000
                            test_episode: 16
                            log_dir: "./logs/wqmix/"
                            model_dir: "./models/wqmix/"

                    .. group-tab:: 8m_vd_9m

                        .. code-block:: yaml

                            agent: "OWQMIX"  # choice: CWQMIX, OWQMIX
                            env_name: "StarCraft2"
                            env_id: "8m_vs_9m"
                            fps: 15
                            policy: "Weighted_Mixing_Q_network"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            activation: "ReLU"
                            alpha: 0.1

                            hidden_dim_mixing_net: 32  # hidden units of mixing network
                            hidden_dim_hyper_net: 64  # hidden units of hyper network

                            hidden_dim_ff_mix_net: 256  # hidden units of mixing network

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 1000000
                            start_training: 1000  # start training after n episodes
                            running_steps: 10000000  # 10M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 100000
                            test_episode: 16
                            log_dir: "./logs/wqmix/"
                            model_dir: "./models/wqmix/"

                    .. group-tab:: 25m

                        .. code-block:: yaml

                            agent: "OWQMIX"  # choice: CWQMIX, OWQMIX
                            env_name: "StarCraft2"
                            env_id: "25m"
                            fps: 15
                            policy: "Weighted_Mixing_Q_network"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            activation: "ReLU"
                            alpha: 0.1

                            hidden_dim_mixing_net: 32  # hidden units of mixing network
                            hidden_dim_hyper_net: 64  # hidden units of hyper network

                            hidden_dim_ff_mix_net: 256  # hidden units of mixing network

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 1000000
                            start_training: 1000  # start training after n episodes
                            running_steps: 5000000  # 5M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 50000
                            test_episode: 16
                            log_dir: "./logs/wqmix/"
                            model_dir: "./models/wqmix/"

                    .. group-tab:: corridor

                        .. code-block:: yaml

                            agent: "OWQMIX"  # choice: CWQMIX, OWQMIX
                            env_name: "StarCraft2"
                            env_id: "corridor"
                            fps: 15
                            policy: "Weighted_Mixing_Q_network"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            activation: "ReLU"
                            alpha: 0.1

                            hidden_dim_mixing_net: 32  # hidden units of mixing network
                            hidden_dim_hyper_net: 64  # hidden units of hyper network

                            hidden_dim_ff_mix_net: 256  # hidden units of mixing network

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 1000000
                            start_training: 1000  # start training after n episodes
                            running_steps: 10000000  # 10M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 100000
                            test_episode: 16
                            log_dir: "./logs/wqmix/"
                            model_dir: "./models/wqmix/"

                    .. group-tab:: MMM2

                        .. code-block:: yaml

                            agent: "OWQMIX"  # choice: CWQMIX, OWQMIX
                            env_name: "StarCraft2"
                            env_id: "3m"
                            fps: 15
                            policy: "Weighted_Mixing_Q_network"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            activation: "ReLU"
                            alpha: 0.1

                            hidden_dim_mixing_net: 32  # hidden units of mixing network
                            hidden_dim_hyper_net: 64  # hidden units of hyper network

                            hidden_dim_ff_mix_net: 256  # hidden units of mixing network

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 1000000
                            start_training: 1000  # start training after n episodes
                            running_steps: 10000000  # 10M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 100000
                            test_episode: 16
                            log_dir: "./logs/wqmix/"
                            model_dir: "./models/wqmix/"


    .. group-tab:: QTRAN

        .. tabs::

            .. group-tab:: MPE

                .. tabs::

                    .. group-tab:: simple_spread_v3

                        .. code-block:: yaml

                            agent: "QTRAN_base"  # Options: QTRAN_base, QTRAN_alt
                            env_name: "mpe"
                            env_id: "simple_spread_v3"
                            continuous_action: False
                            policy: "Qtran_Mixing_Q_network"
                            representation: "Basic_MLP"
                            vectorize: "Dummy_Pettingzoo"
                            runner: "Pettingzoo_Runner"

                            use_recurrent: False
                            rnn:
                            representation_hidden_size: [256, ]
                            q_hidden_size: [256, ]  # the units for each hidden layer
                            activation: "ReLU"

                            hidden_dim_mixing_net: 64  # hidden units of mixing network
                            hidden_dim_hyper_net: 64  # hidden units of hyper network
                            qtran_net_hidden_dim: 64
                            lambda_opt: 1.0
                            lambda_nopt: 1.0

                            seed: 1
                            parallels: 16
                            buffer_size: 100000
                            batch_size: 256
                            learning_rate: 0.001
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 2500000
                            start_training: 1000  # start training after n episodes
                            running_steps: 10000000  # 10M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 100000
                            test_episode: 5
                            log_dir: "./logs/qtran/"
                            model_dir: "./models/qtran/"


    .. group-tab:: DCG

        .. tabs::

            .. group-tab:: MPE

                .. tabs::

                    .. group-tab:: simple_spread_v3

                        .. code-block:: yaml

                            agent: "DCG"  # Options: DCG, DCG_S
                            env_name: "mpe"
                            env_id: "simple_spread_v3"
                            continuous_action: False
                            policy: "DCG_Policy"
                            representation: "Basic_MLP"
                            vectorize: "Dummy_Pettingzoo"
                            runner: "Pettingzoo_Runner"
                            on_policy: False

                            use_recurrent: False
                            rnn:
                            representation_hidden_size: [32, ]
                            q_hidden_size: [128, ]  # the units for each hidden layer
                            hidden_utility_dim: 256  # hidden units of the utility function
                            hidden_payoff_dim: 256  # hidden units of the payoff function
                            bias_net: "Basic_MLP"
                            hidden_bias_dim: [256, ]  # hidden units of the bias network with global states as input
                            activation: "ReLU"

                            low_rank_payoff: False  # low-rank approximation of payoff function
                            payoff_rank: 5  # the rank K in the paper
                            graph_type: "FULL"  # specific type of the coordination graph
                            n_msg_iterations: 1  # number of iterations for message passing during belief propagation
                            msg_normalized: True  # Message normalization during greedy action selection (Kok and Vlassis, 2006)

                            seed: 1
                            parallels: 16
                            buffer_size: 100000
                            batch_size: 256
                            learning_rate: 0.001
                            gamma: 0.95  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 2500000
                            start_training: 1000  # start training after n episodes
                            running_steps: 10000000  # 10M
                            train_per_step: False  # True: train model per step; False: train model per episode.training_frequency: 1
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 100000
                            test_episode: 5
                            log_dir: "./logs/dcg/"
                            model_dir: "./models/dcg/"

            .. group-tab:: SC2

                .. tabs::

                    .. group-tab:: 1c3s5z

                        .. code-block:: yaml

                            agent: "DCG"  # Options: DCG, DCG_S
                            env_name: "StarCraft2"
                            env_id: "1c3s5z"
                            fps: 15
                            policy: "DCG_Policy"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            hidden_utility_dim: 64  # hidden units of the utility function
                            hidden_payoff_dim: 64  # hidden units of the payoff function
                            bias_net: "Basic_MLP"
                            hidden_bias_dim: [64, ]  # hidden units of the bias network with global states as input
                            activation: "ReLU"

                            low_rank_payoff: False  # low-rank approximation of payoff function
                            payoff_rank: 5  # the rank K in the paper
                            graph_type: "FULL"  # specific type of the coordination graph
                            n_msg_iterations: 8  # number of iterations for message passing during belief propagation
                            msg_normalized: True  # Message normalization during greedy action selection (Kok and Vlassis, 2006)

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 50000
                            start_training: 1000  # start training after n episodes
                            running_steps: 2000000  # 2M
                            train_per_step: False  # True: train model per step; False: train model per episode.training_frequency: 1
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 20000
                            test_episode: 16
                            log_dir: "./logs/dcg/"
                            model_dir: "./models/dcg/"

                    .. group-tab:: 2m_vs_1z

                        .. code-block:: yaml

                            agent: "DCG"  # Options: DCG, DCG_S
                            env_name: "StarCraft2"
                            env_id: "2m_vs_1z"
                            fps: 15
                            policy: "DCG_Policy"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            hidden_utility_dim: 64  # hidden units of the utility function
                            hidden_payoff_dim: 64  # hidden units of the payoff function
                            bias_net: "Basic_MLP"
                            hidden_bias_dim: [64, ]  # hidden units of the bias network with global states as input
                            activation: "ReLU"

                            low_rank_payoff: False  # low-rank approximation of payoff function
                            payoff_rank: 5  # the rank K in the paper
                            graph_type: "FULL"  # specific type of the coordination graph
                            n_msg_iterations: 8  # number of iterations for message passing during belief propagation
                            msg_normalized: True  # Message normalization during greedy action selection (Kok and Vlassis, 2006)

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 50000
                            start_training: 1000  # start training after n episodes
                            running_steps: 1000000  # 1M
                            train_per_step: False  # True: train model per step; False: train model per episode.training_frequency: 1
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 10000
                            test_episode: 16
                            log_dir: "./logs/dcg/"
                            model_dir: "./models/dcg/"

                    .. group-tab:: 2s3z

                        .. code-block:: yaml

                            agent: "DCG"  # Options: DCG, DCG_S
                            env_name: "StarCraft2"
                            env_id: "2s3z"
                            fps: 15
                            policy: "DCG_Policy"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            hidden_utility_dim: 64  # hidden units of the utility function
                            hidden_payoff_dim: 64  # hidden units of the payoff function
                            bias_net: "Basic_MLP"
                            hidden_bias_dim: [64, ]  # hidden units of the bias network with global states as input
                            activation: "ReLU"

                            low_rank_payoff: False  # low-rank approximation of payoff function
                            payoff_rank: 5  # the rank K in the paper
                            graph_type: "FULL"  # specific type of the coordination graph
                            n_msg_iterations: 8  # number of iterations for message passing during belief propagation
                            msg_normalized: True  # Message normalization during greedy action selection (Kok and Vlassis, 2006)

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 50000
                            start_training: 1000  # start training after n episodes
                            running_steps: 2000000  # 2M
                            train_per_step: False  # True: train model per step; False: train model per episode.training_frequency: 1
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 20000
                            test_episode: 16
                            log_dir: "./logs/dcg/"
                            model_dir: "./models/dcg/"

                    .. group-tab:: 3m

                        .. code-block:: yaml

                            agent: "DCG"  # Options: DCG, DCG_S
                            env_name: "StarCraft2"
                            env_id: "3m"
                            fps: 15
                            policy: "DCG_Policy"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            hidden_utility_dim: 64  # hidden units of the utility function
                            hidden_payoff_dim: 64  # hidden units of the payoff function
                            bias_net: "Basic_MLP"
                            hidden_bias_dim: [64, ]  # hidden units of the bias network with global states as input
                            activation: "ReLU"

                            low_rank_payoff: False  # low-rank approximation of payoff function
                            payoff_rank: 5  # the rank K in the paper
                            graph_type: "FULL"  # specific type of the coordination graph
                            n_msg_iterations: 8  # number of iterations for message passing during belief propagation
                            msg_normalized: True  # Message normalization during greedy action selection (Kok and Vlassis, 2006)

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 50000
                            start_training: 1000  # start training after n episodes
                            running_steps: 1000000  # 1M
                            train_per_step: False  # True: train model per step; False: train model per episode.training_frequency: 1
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 10000
                            test_episode: 16
                            log_dir: "./logs/dcg/"
                            model_dir: "./models/dcg/"

                    .. group-tab:: 5m_vs_6m

                        .. code-block:: yaml

                            agent: "DCG"  # Options: DCG, DCG_S
                            env_name: "StarCraft2"
                            env_id: "5m_vs_6m"
                            fps: 15
                            policy: "DCG_Policy"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            hidden_utility_dim: 64  # hidden units of the utility function
                            hidden_payoff_dim: 64  # hidden units of the payoff function
                            bias_net: "Basic_MLP"
                            hidden_bias_dim: [64, ]  # hidden units of the bias network with global states as input
                            activation: "ReLU"

                            low_rank_payoff: False  # low-rank approximation of payoff function
                            payoff_rank: 5  # the rank K in the paper
                            graph_type: "FULL"  # specific type of the coordination graph
                            n_msg_iterations: 8  # number of iterations for message passing during belief propagation
                            msg_normalized: True  # Message normalization during greedy action selection (Kok and Vlassis, 2006)

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 50000
                            start_training: 1000  # start training after n episodes
                            running_steps: 10000000  # 10M
                            train_per_step: False  # True: train model per step; False: train model per episode.training_frequency: 1
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 100000
                            test_episode: 16
                            log_dir: "./logs/dcg/"
                            model_dir: "./models/dcg/"

                    .. group-tab:: 8m

                        .. code-block:: yaml

                            agent: "DCG"  # Options: DCG, DCG_S
                            env_name: "StarCraft2"
                            env_id: "8m"
                            fps: 15
                            policy: "DCG_Policy"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            hidden_utility_dim: 64  # hidden units of the utility function
                            hidden_payoff_dim: 64  # hidden units of the payoff function
                            bias_net: "Basic_MLP"
                            hidden_bias_dim: [64, ]  # hidden units of the bias network with global states as input
                            activation: "ReLU"

                            low_rank_payoff: False  # low-rank approximation of payoff function
                            payoff_rank: 5  # the rank K in the paper
                            graph_type: "FULL"  # specific type of the coordination graph
                            n_msg_iterations: 8  # number of iterations for message passing during belief propagation
                            msg_normalized: True  # Message normalization during greedy action selection (Kok and Vlassis, 2006)

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 50000
                            start_training: 1000  # start training after n episodes
                            running_steps: 1000000  # 1M
                            train_per_step: False  # True: train model per step; False: train model per episode.training_frequency: 1
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 10000
                            test_episode: 16
                            log_dir: "./logs/dcg/"
                            model_dir: "./models/dcg/"

                    .. group-tab:: 8m_vd_9m

                        .. code-block:: yaml

                            agent: "DCG"  # Options: DCG, DCG_S
                            env_name: "StarCraft2"
                            env_id: "8m_vs_9m"
                            fps: 15
                            policy: "DCG_Policy"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            hidden_utility_dim: 64  # hidden units of the utility function
                            hidden_payoff_dim: 64  # hidden units of the payoff function
                            bias_net: "Basic_MLP"
                            hidden_bias_dim: [64, ]  # hidden units of the bias network with global states as input
                            activation: "ReLU"

                            low_rank_payoff: False  # low-rank approximation of payoff function
                            payoff_rank: 5  # the rank K in the paper
                            graph_type: "FULL"  # specific type of the coordination graph
                            n_msg_iterations: 8  # number of iterations for message passing during belief propagation
                            msg_normalized: True  # Message normalization during greedy action selection (Kok and Vlassis, 2006)

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 50000
                            start_training: 1000  # start training after n episodes
                            running_steps: 10000000  # 10M
                            train_per_step: False  # True: train model per step; False: train model per episode.training_frequency: 1
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 100000
                            test_episode: 16
                            log_dir: "./logs/dcg/"
                            model_dir: "./models/dcg/"

                    .. group-tab:: 25m

                        .. code-block:: yaml

                            agent: "DCG"  # Options: DCG, DCG_S
                            env_name: "StarCraft2"
                            env_id: "25m"
                            fps: 15
                            policy: "DCG_Policy"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            hidden_utility_dim: 64  # hidden units of the utility function
                            hidden_payoff_dim: 64  # hidden units of the payoff function
                            bias_net: "Basic_MLP"
                            hidden_bias_dim: [64, ]  # hidden units of the bias network with global states as input
                            activation: "ReLU"

                            low_rank_payoff: False  # low-rank approximation of payoff function
                            payoff_rank: 5  # the rank K in the paper
                            graph_type: "FULL"  # specific type of the coordination graph
                            n_msg_iterations: 8  # number of iterations for message passing during belief propagation
                            msg_normalized: True  # Message normalization during greedy action selection (Kok and Vlassis, 2006)

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 50000
                            start_training: 1000  # start training after n episodes
                            running_steps: 5000000  # 5M
                            train_per_step: False  # True: train model per step; False: train model per episode.training_frequency: 1
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 50000
                            test_episode: 16
                            log_dir: "./logs/dcg/"
                            model_dir: "./models/dcg/"

                    .. group-tab:: corridor

                        .. code-block:: yaml

                            agent: "DCG"  # Options: DCG, DCG_S
                            env_name: "StarCraft2"
                            env_id: "corridor"
                            fps: 15
                            policy: "DCG_Policy"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            hidden_utility_dim: 64  # hidden units of the utility function
                            hidden_payoff_dim: 64  # hidden units of the payoff function
                            bias_net: "Basic_MLP"
                            hidden_bias_dim: [64, ]  # hidden units of the bias network with global states as input
                            activation: "ReLU"

                            low_rank_payoff: False  # low-rank approximation of payoff function
                            payoff_rank: 5  # the rank K in the paper
                            graph_type: "FULL"  # specific type of the coordination graph
                            n_msg_iterations: 8  # number of iterations for message passing during belief propagation
                            msg_normalized: True  # Message normalization during greedy action selection (Kok and Vlassis, 2006)

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 50000
                            start_training: 1000  # start training after n episodes
                            running_steps: 10000000  # 10M
                            train_per_step: False  # True: train model per step; False: train model per episode.training_frequency: 1
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 100000
                            test_episode: 16
                            log_dir: "./logs/dcg/"
                            model_dir: "./models/dcg/"

                    .. group-tab:: MMM2

                        .. code-block:: yaml

                            agent: "DCG"  # Options: DCG, DCG_S
                            env_name: "StarCraft2"
                            env_id: "MMM2"
                            fps: 15
                            policy: "DCG_Policy"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0

                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            hidden_utility_dim: 64  # hidden units of the utility function
                            hidden_payoff_dim: 64  # hidden units of the payoff function
                            bias_net: "Basic_MLP"
                            hidden_bias_dim: [64, ]  # hidden units of the bias network with global states as input
                            activation: "ReLU"

                            low_rank_payoff: False  # low-rank approximation of payoff function
                            payoff_rank: 5  # the rank K in the paper
                            graph_type: "FULL"  # specific type of the coordination graph
                            n_msg_iterations: 8  # number of iterations for message passing during belief propagation
                            msg_normalized: True  # Message normalization during greedy action selection (Kok and Vlassis, 2006)

                            seed: 1
                            parallels: 8
                            buffer_size: 5000
                            batch_size: 32
                            learning_rate: 0.0007
                            gamma: 0.99  # discount factor
                            double_q: True  # use double q learning

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 50000
                            start_training: 1000  # start training after n episodes
                            running_steps: 10000000  # 10M
                            train_per_step: False  # True: train model per step; False: train model per episode.training_frequency: 1
                            training_frequency: 1
                            sync_frequency: 200

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            eval_interval: 100000
                            test_episode: 16
                            log_dir: "./logs/dcg/"
                            model_dir: "./models/dcg/"

    .. group-tab:: IDDPG

        .. tabs::

            .. group-tab:: MPE

                .. tabs::

                    .. group-tab:: simple_adversary_v3

                        .. code-block:: yaml

                            agent: "IDDPG"  # the learning algorithms_marl
                            env_name: "mpe"
                            env_id: "simple_adversary_v3"
                            continuous_action: True
                            policy: "Independent_DDPG_Policy"
                            representation: "Basic_Identical"
                            vectorize: "Dummy_Pettingzoo"
                            runner: "Pettingzoo_Runner"

                            representation_hidden_size: [64, ]  # the units for each hidden layer
                            actor_hidden_size: [64, 64]
                            critic_hidden_size: [64, 64]
                            activation: 'LeakyReLU'
                            activation_action: 'sigmoid'

                            seed: 1
                            parallels: 16
                            buffer_size: 100000
                            batch_size: 256
                            lr_a: 0.01  # learning rate for actor
                            lr_c: 0.001  # learning rate for critic
                            gamma: 0.95  # discount factor
                            tau: 0.001  # soft update for target networks

                            start_noise: 1.0
                            end_noise: 0.01
                            sigma: 0.1  # random noise for continuous actions
                            start_training: 1000  # start training after n episodes
                            running_steps: 10000000
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1

                            use_grad_clip: True
                            grad_clip_norm: 0.5

                            eval_interval: 100000
                            test_episode: 5
                            log_dir: "./logs/iddpg/"
                            model_dir: "./models/iddpg/"

                    .. group-tab:: simple_push_v3

                        .. code-block:: yaml

                            agent: "IDDPG"  # the learning algorithms_marl
                            env_name: "mpe"
                            env_id: "simple_push_v3"
                            continuous_action: True
                            policy: "Independent_DDPG_Policy"
                            representation: "Basic_Identical"
                            vectorize: "Dummy_Pettingzoo"
                            runner: "Pettingzoo_Runner"

                            representation_hidden_size: [64, ]  # the units for each hidden layer
                            actor_hidden_size: [64, 64]
                            critic_hidden_size: [64, 64]
                            activation: 'LeakyReLU'
                            activation_action: 'sigmoid'

                            seed: 1
                            parallels: 16
                            buffer_size: 100000
                            batch_size: 256
                            lr_a: 0.01  # learning rate for actor
                            lr_c: 0.001  # learning rate for critic
                            gamma: 0.95  # discount factor
                            tau: 0.001  # soft update for target networks

                            start_noise: 1.0
                            end_noise: 0.01
                            sigma: 0.1  # random noise for continuous actions
                            start_training: 1000  # start training after n episodes
                            running_steps: 10000000
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1

                            use_grad_clip: True
                            grad_clip_norm: 0.5

                            eval_interval: 100000
                            test_episode: 5
                            log_dir: "./logs/iddpg/"
                            model_dir: "./models/iddpg/"

                    .. group-tab:: simple_spread_v3

                        .. code-block:: yaml

                            agent: "IDDPG"  # the learning algorithms_marl
                            env_name: "mpe"
                            env_id: "simple_spread_v3"
                            continuous_action: True
                            policy: "Independent_DDPG_Policy"
                            representation: "Basic_Identical"
                            vectorize: "Dummy_Pettingzoo"
                            runner: "Pettingzoo_Runner"

                            representation_hidden_size: []  # the units for each hidden layer
                            actor_hidden_size: [64, 64]
                            critic_hidden_size: [64, 64]
                            activation: 'LeakyReLU'
                            activation_action: 'sigmoid'

                            seed: 1
                            parallels: 16
                            buffer_size: 100000
                            batch_size: 256
                            lr_a: 0.01  # learning rate for actor
                            lr_c: 0.001  # learning rate for critic
                            gamma: 0.95  # discount factor
                            tau: 0.001  # soft update for target networks

                            start_noise: 1.0
                            end_noise: 0.01
                            sigma: 0.1  # random noise for continuous actions
                            start_training: 1000  # start training after n episodes
                            running_steps: 10000000
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1

                            use_grad_clip: True
                            grad_clip_norm: 0.5

                            eval_interval: 100000
                            test_episode: 5
                            log_dir: "./logs/iddpg/"
                            model_dir: "./models/iddpg/"

    .. group-tab:: MADDPG

        .. tabs::

            .. group-tab:: MPE

                .. tabs::

                    .. group-tab:: simple_adversary_v3

                        .. code-block:: yaml

                            agent: "MADDPG"  # the learning algorithms_marl
                            env_name: "mpe"
                            env_id: "simple_adversary_v3"
                            continuous_action: True
                            policy: "MADDPG_Policy"
                            representation: "Basic_Identical"
                            vectorize: "Dummy_Pettingzoo"
                            runner: "Pettingzoo_Runner"
                            on_policy: False

                            representation_hidden_size: []  # the units for each hidden layer
                            actor_hidden_size: [64, ]
                            critic_hidden_size: [64, ]
                            activation: 'LeakyReLU'
                            activation_action: 'sigmoid'

                            seed: 1
                            parallels: 16
                            buffer_size: 100000
                            batch_size: 256
                            lr_a: 0.01  # learning rate for actor
                            lr_c: 0.001  # learning rate for critic
                            gamma: 0.95  # discount factor
                            tau: 0.001  # soft update for target networks

                            start_noise: 1.0
                            end_noise: 0.01
                            sigma: 0.1
                            start_training: 1000  # start training after n episodes
                            running_steps: 10000000
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1

                            use_grad_clip: True
                            grad_clip_norm: 0.5

                            eval_interval: 100000
                            test_episode: 5
                            log_dir: "./logs/maddpg/"
                            model_dir: "./models/maddpg/"

                    .. group-tab:: simple_push_v3

                        .. code-block:: yaml

                            agent: "MADDPG"  # the learning algorithms_marl
                            env_name: "mpe"
                            env_id: "simple_push_v3"
                            continuous_action: True
                            policy: "MADDPG_Policy"
                            representation: "Basic_Identical"
                            vectorize: "Dummy_Pettingzoo"
                            runner: "Pettingzoo_Runner"

                            representation_hidden_size: []  # the units for each hidden layer
                            actor_hidden_size: [64, 64]
                            critic_hidden_size: [64, 64]
                            activation: 'LeakyReLU'
                            activation_action: 'sigmoid'

                            seed: 1
                            parallels: 16
                            buffer_size: 100000
                            batch_size: 256
                            lr_a: 0.01  # learning rate for actor
                            lr_c: 0.001  # learning rate for critic
                            gamma: 0.95  # discount factor
                            tau: 0.001  # soft update for target networks

                            start_noise: 1.0
                            end_noise: 0.01
                            sigma: 0.1  # random noise for continuous actions
                            start_training: 1000  # start training after n episodes
                            running_steps: 10000000
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1

                            use_grad_clip: True
                            grad_clip_norm: 0.5

                            eval_interval: 100000
                            test_episode: 5
                            log_dir: "./logs/maddpg/"
                            model_dir: "./models/maddpg/"

                    .. group-tab:: simple_spread_v3

                        .. code-block:: yaml

                            agent: "MADDPG"  # the learning algorithms_marl
                            env_name: "mpe"
                            env_id: "simple_spread_v3"
                            continuous_action: True
                            policy: "MADDPG_Policy"
                            representation: "Basic_Identical"
                            vectorize: "Dummy_Pettingzoo"
                            runner: "Pettingzoo_Runner"

                            representation_hidden_size: []  # the units for each hidden layer
                            actor_hidden_size: [64, 64]
                            critic_hidden_size: [64, 64]
                            activation: 'LeakyReLU'
                            activation_action: 'sigmoid'

                            seed: 1
                            parallels: 16
                            buffer_size: 100000
                            batch_size: 256
                            lr_a: 0.01  # learning rate for actor
                            lr_c: 0.001  # learning rate for critic
                            gamma: 0.95  # discount factor
                            tau: 0.001  # soft update for target networks

                            #start_noise: 1.0
                            #end_noise: 0.01
                            sigma: 0.1
                            start_training: 1000  # start training after n episodes
                            running_steps: 10000000
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1

                            use_grad_clip: True
                            grad_clip_norm: 0.5

                            eval_interval: 100000
                            test_episode: 5
                            log_dir: "./logs/maddpg/"
                            model_dir: "./models/maddpg/"


    .. group-tab:: ISAC

        .. tabs::

            .. group-tab:: MPE

                .. tabs::

                    .. group-tab:: simple_adversary_v3

                        .. code-block:: yaml

                            agent: "ISAC"  # the learning algorithms_marl
                            env_name: "mpe"
                            env_id: "simple_adversary_v3"
                            continuous_action: True
                            policy: "Gaussian_ISAC_Policy"
                            representation: "Basic_Identical"
                            vectorize: "Dummy_Pettingzoo"
                            runner: "Pettingzoo_Runner"

                            representation_hidden_size: [64, ]  # the units for each hidden layer
                            actor_hidden_size: [64, 64]
                            critic_hidden_size: [64, 64]
                            activation: 'LeakyReLU'
                            activation_action: 'sigmoid'

                            seed: 1
                            parallels: 16
                            buffer_size: 100000
                            batch_size: 256
                            lr_a: 0.01  # learning rate for actor
                            lr_c: 0.001  # learning rate for critic
                            gamma: 0.95  # discount factor
                            tau: 0.001  # soft update for target networks
                            alpha: 0.01

                            start_noise: 1.0
                            end_noise: 0.01
                            sigma: 0.1  # random noise for continuous actions
                            start_training: 1000  # start training after n episodes
                            running_steps: 10000000
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1

                            use_grad_clip: True
                            grad_clip_norm: 0.5

                            eval_interval: 100000
                            test_episode: 5
                            log_dir: "./logs/isac/"
                            model_dir: "./models/isac/"

                    .. group-tab:: simple_push_v3

                        .. code-block:: yaml

                            agent: "ISAC"  # the learning algorithms_marl
                            env_name: "mpe"
                            env_id: "simple_push_v3"
                            continuous_action: True
                            policy: "Gaussian_ISAC_Policy"
                            representation: "Basic_Identical"
                            vectorize: "Dummy_Pettingzoo"
                            runner: "Pettingzoo_Runner"

                            representation_hidden_size: [64, ]  # the units for each hidden layer
                            actor_hidden_size: [64, 64]
                            critic_hidden_size: [64, 64]
                            activation: 'LeakyReLU'
                            activation_action: 'sigmoid'

                            seed: 1
                            parallels: 16
                            buffer_size: 100000
                            batch_size: 256
                            lr_a: 0.01  # learning rate for actor
                            lr_c: 0.001  # learning rate for critic
                            gamma: 0.95  # discount factor
                            tau: 0.001  # soft update for target networks
                            alpha: 0.01

                            start_noise: 1.0
                            end_noise: 0.01
                            sigma: 0.1  # random noise for continuous actions
                            start_training: 1000  # start training after n episodes
                            running_steps: 10000000
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1

                            use_grad_clip: True
                            grad_clip_norm: 0.5

                            eval_interval: 100000
                            test_episode: 5
                            log_dir: "./logs/isac/"
                            model_dir: "./models/isac/"

                    .. group-tab:: simple_spread_v3

                        .. code-block:: yaml

                            agent: "ISAC"  # the learning algorithms_marl
                            env_name: "mpe"
                            env_id: "simple_spread_v3"
                            continuous_action: True
                            policy: "Gaussian_ISAC_Policy"
                            representation: "Basic_Identical"
                            vectorize: "Dummy_Pettingzoo"
                            runner: "Pettingzoo_Runner"

                            representation_hidden_size: [64, ]  # the units for each hidden layer
                            actor_hidden_size: [64, 64]
                            critic_hidden_size: [64, 64]
                            activation: 'LeakyReLU'
                            activation_action: 'sigmoid'

                            seed: 1
                            parallels: 16
                            buffer_size: 100000
                            batch_size: 256
                            lr_a: 0.01  # learning rate for actor
                            lr_c: 0.001  # learning rate for critic
                            gamma: 0.95  # discount factor
                            tau: 0.001  # soft update for target networks
                            alpha: 0.01

                            start_noise: 1.0
                            end_noise: 0.01
                            sigma: 0.1  # random noise for continuous actions
                            start_training: 1000  # start training after n episodes
                            running_steps: 10000000
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1

                            use_grad_clip: True
                            grad_clip_norm: 0.5

                            eval_interval: 100000
                            test_episode: 5
                            log_dir: "./logs/isac/"
                            model_dir: "./models/isac/"

    .. group-tab:: MASAC

        .. tabs::

            .. group-tab:: MPE

                .. tabs::

                    .. group-tab:: simple_adversary_v3

                        .. code-block:: yaml

                            agent: "MASAC"  # the learning algorithms_marl
                            env_name: "mpe"
                            env_id: "simple_adversary_v3"
                            continuous_action: True
                            policy: "Gaussian_MASAC_Policy"
                            representation: "Basic_Identical"
                            vectorize: "Dummy_Pettingzoo"
                            runner: "Pettingzoo_Runner"

                            representation_hidden_size: [64, ]  # the units for each hidden layer
                            actor_hidden_size: [64, 64]
                            critic_hidden_size: [64, 64]
                            activation: 'LeakyReLU'
                            activation_action: 'sigmoid'

                            seed: 1
                            parallels: 16
                            buffer_size: 100000
                            batch_size: 256
                            lr_a: 0.01  # learning rate for actor
                            lr_c: 0.001  # learning rate for critic
                            gamma: 0.95  # discount factor
                            tau: 0.001  # soft update for target networks
                            alpha: 0.01

                            start_noise: 1.0
                            end_noise: 0.01
                            sigma: 0.1  # random noise for continuous actions
                            start_training: 1000  # start training after n episodes
                            running_steps: 10000000
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1

                            use_grad_clip: True
                            grad_clip_norm: 0.5

                            eval_interval: 100000
                            test_episode: 5
                            log_dir: "./logs/masac/"
                            model_dir: "./models/masac/"


                    .. group-tab:: simple_push_v3

                        .. code-block:: yaml

                            agent: "MASAC"  # the learning algorithms_marl
                            env_name: "mpe"
                            env_id: "simple_push_v3"
                            continuous_action: True
                            policy: "Gaussian_MASAC_Policy"
                            representation: "Basic_Identical"
                            vectorize: "Dummy_Pettingzoo"
                            runner: "Pettingzoo_Runner"

                            representation_hidden_size: [64, ]  # the units for each hidden layer
                            actor_hidden_size: [64, 64]
                            critic_hidden_size: [64, 64]
                            activation: 'LeakyReLU'
                            activation_action: 'sigmoid'

                            seed: 1
                            parallels: 16
                            buffer_size: 100000
                            batch_size: 256
                            lr_a: 0.01  # learning rate for actor
                            lr_c: 0.001  # learning rate for critic
                            gamma: 0.95  # discount factor
                            tau: 0.001  # soft update for target networks
                            alpha: 0.01

                            start_noise: 1.0
                            end_noise: 0.01
                            sigma: 0.1  # random noise for continuous actions
                            start_training: 1000  # start training after n episodes
                            running_steps: 10000000
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1

                            use_grad_clip: True
                            grad_clip_norm: 0.5

                            eval_interval: 100000
                            test_episode: 5
                            log_dir: "./logs/masac/"
                            model_dir: "./models/masac/"


                    .. group-tab:: simple_spread_v3

                        .. code-block:: yaml

                            agent: "MASAC"  # the learning algorithms_marl
                            env_name: "mpe"
                            env_id: "simple_spread_v3"
                            continuous_action: True
                            policy: "Gaussian_MASAC_Policy"
                            representation: "Basic_Identical"
                            vectorize: "Dummy_Pettingzoo"
                            runner: "Pettingzoo_Runner"

                            representation_hidden_size: [64, ]  # the units for each hidden layer
                            actor_hidden_size: [64, 64]
                            critic_hidden_size: [64, 64]
                            activation: 'LeakyReLU'
                            activation_action: 'sigmoid'

                            seed: 1
                            parallels: 16
                            buffer_size: 100000
                            batch_size: 256
                            lr_a: 0.01  # learning rate for actor
                            lr_c: 0.001  # learning rate for critic
                            gamma: 0.95  # discount factor
                            tau: 0.001  # soft update for target networks
                            alpha: 0.01

                            start_noise: 1.0
                            end_noise: 0.01
                            sigma: 0.1  # random noise for continuous actions
                            start_training: 1000  # start training after n episodes
                            running_steps: 10000000
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1

                            use_grad_clip: True
                            grad_clip_norm: 0.5

                            eval_interval: 100000
                            test_episode: 5
                            log_dir: "./logs/masac/"
                            model_dir: "./models/masac/"



    .. group-tab:: IPPO

        .. tabs::

            .. group-tab:: MPE

                .. tabs::

                    .. group-tab:: simple_spread_v3

                        .. code-block:: yaml

                            agent: "IPPO"
                            env_name: "mpe"
                            env_id: "simple_spread_v3"
                            continuous_action: False
                            policy: "Categorical_MAAC_Policy"
                            representation: "Basic_MLP"
                            vectorize: "Dummy_Pettingzoo"
                            runner: "Pettingzoo_Runner"

                            use_recurrent: False
                            rnn:
                            representation_hidden_size: [64, ]  # the units for each hidden layer
                            gain: 0.01

                            actor_hidden_size: [64, ]
                            critic_hidden_size: [256, ]
                            activation: "ReLU"

                            seed: 1
                            parallels: 128
                            n_size: 3200
                            n_epoch: 10
                            n_minibatch: 1
                            learning_rate: 0.0007
                            weight_decay: 0

                            vf_coef: 0.5
                            ent_coef: 0.01
                            target_kl: 0.25  # for MAPPO_KL learner
                            clip_range: 0.2  # ratio clip range, for MAPPO_Clip learner
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.95  # discount factor

                            # tricks
                            use_linear_lr_decay: False  # if use linear learning rate decay
                            end_factor_lr_decay: 0.5
                            use_global_state: False  # if use global state to replace merged observations
                            use_grad_norm: True  # gradient normalization
                            max_grad_norm: 10.0
                            use_value_clip: True  # limit the value range
                            value_clip_range: 0.2
                            use_value_norm: True  # use running mean and std to normalize rewards.
                            use_huber_loss: True  # True: use huber loss; False: use MSE loss.
                            huber_delta: 10.0
                            use_advnorm: True
                            use_gae: True
                            gae_lambda: 0.95

                            start_training: 1
                            running_steps: 10000000
                            train_per_step: True
                            training_frequency: 1

                            test_steps: 10000
                            eval_interval: 100000
                            test_episode: 5
                            log_dir: "./logs/ippo/"
                            model_dir: "./models/ippo/"


            .. group-tab:: SC2

                .. tabs::

                    .. group-tab:: 1c3s5z

                        .. code-block:: yaml

                            agent: "IPPO"
                            env_name: "StarCraft2"
                            env_id: "1c3s5z"
                            fps: 15
                            policy: "Categorical_MAAC_Policy"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, 64, 64]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0
                            normalize: "LayerNorm"
                            initialize: "orthogonal"
                            gain: 0.01

                            actor_hidden_size: []
                            critic_hidden_size: []
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            n_size: 128
                            n_epoch: 15
                            n_minibatch: 1
                            learning_rate: 0.0007  # 7e-4
                            weight_decay: 0

                            vf_coef: 1.0
                            ent_coef: 0.01
                            target_kl: 0.25
                            clip_range: 0.2
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.99  # discount factor

                            # tricks
                            use_linear_lr_decay: False  # if use linear learning rate decay
                            end_factor_lr_decay: 0.5
                            use_global_state: False  # if use global state to replace joint observations
                            use_grad_norm: True  # gradient normalization
                            max_grad_norm: 10.0
                            use_value_clip: True  # limit the value range
                            value_clip_range: 0.2
                            use_value_norm: True  # use running mean and std to normalize rewards.
                            use_huber_loss: True  # True: use huber loss; False: use MSE loss.
                            huber_delta: 10.0
                            use_advnorm: True  # use advantage normalization.
                            use_gae: True  # use GAE trick to calculate returns.
                            gae_lambda: 0.95

                            start_training: 1
                            running_steps: 2000000
                            training_frequency: 1

                            eval_interval: 20000
                            test_episode: 16
                            log_dir: "./logs/ippo/"
                            model_dir: "./models/ippo/"


                    .. group-tab:: 2m_vs_1z

                        .. code-block:: yaml

                            agent: "IPPO"
                            env_name: "StarCraft2"
                            env_id: "2m_vs_1z"
                            fps: 15
                            policy: "Categorical_MAAC_Policy"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: True

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, 64, 64]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0
                            normalize: "LayerNorm"
                            initialize: "orthogonal"
                            gain: 0.01

                            actor_hidden_size: []
                            critic_hidden_size: []
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            n_size: 128
                            n_epoch: 15
                            n_minibatch: 1
                            learning_rate: 0.0007  # 7e-4
                            weight_decay: 0

                            vf_coef: 1.0
                            ent_coef: 0.01
                            target_kl: 0.25
                            clip_range: 0.2
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.99  # discount factor

                            # tricks
                            use_linear_lr_decay: False  # if use linear learning rate decay
                            end_factor_lr_decay: 0.5
                            use_global_state: False  # if use global state to replace merged observations
                            use_grad_norm: True  # gradient normalization
                            max_grad_norm: 10.0
                            use_value_clip: True  # limit the value range
                            value_clip_range: 0.2
                            use_value_norm: True  # use running mean and std to normalize rewards.
                            use_huber_loss: True  # True: use huber loss; False: use MSE loss.
                            huber_delta: 10.0
                            use_advnorm: True  # use advantage normalization.
                            use_gae: True  # use GAE trick to calculate returns.
                            gae_lambda: 0.95

                            start_training: 1
                            running_steps: 1000000
                            train_per_step: True
                            training_frequency: 1

                            test_steps: 10000
                            eval_interval: 10000
                            test_episode: 16
                            log_dir: "./logs/ippo/"
                            model_dir: "./models/ippo/"


                    .. group-tab:: 2s3z

                        .. code-block:: yaml

                            agent: "IPPO"
                            env_name: "StarCraft2"
                            env_id: "2s3z"
                            fps: 15
                            policy: "Categorical_MAAC_Policy"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, 64, 64]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0
                            normalize: "LayerNorm"
                            initialize: "orthogonal"
                            gain: 0.01

                            actor_hidden_size: []
                            critic_hidden_size: []
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            n_size: 128
                            n_epoch: 15
                            n_minibatch: 1
                            learning_rate: 0.0007  # 7e-4
                            weight_decay: 0

                            vf_coef: 1.0
                            ent_coef: 0.01
                            target_kl: 0.25
                            clip_range: 0.2
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.99  # discount factor

                            # tricks
                            use_linear_lr_decay: False  # if use linear learning rate decay
                            end_factor_lr_decay: 0.5
                            use_global_state: False  # if use global state to replace joint observations
                            use_grad_norm: True  # gradient normalization
                            max_grad_norm: 10.0
                            use_value_clip: True  # limit the value range
                            value_clip_range: 0.2
                            use_value_norm: True  # use running mean and std to normalize rewards.
                            use_huber_loss: True  # True: use huber loss; False: use MSE loss.
                            huber_delta: 10.0
                            use_advnorm: True  # use advantage normalization.
                            use_gae: True  # use GAE trick to calculate returns.
                            gae_lambda: 0.95

                            start_training: 1
                            running_steps: 2000000
                            training_frequency: 1

                            eval_interval: 20000
                            test_episode: 16
                            log_dir: "./logs/ippo/"
                            model_dir: "./models/ippo/"


                    .. group-tab:: 3m

                        .. code-block:: yaml

                            agent: "IPPO"
                            env_name: "StarCraft2"
                            env_id: "3m"
                            fps: 15
                            policy: "Categorical_MAAC_Policy"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, 64, 64]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0
                            normalize: "LayerNorm"
                            initialize: "orthogonal"
                            gain: 0.01

                            actor_hidden_size: []
                            critic_hidden_size: []
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            n_size: 128
                            n_epoch: 15
                            n_minibatch: 1
                            learning_rate: 0.0007  # 7e-4
                            weight_decay: 0

                            vf_coef: 1.0
                            ent_coef: 0.01
                            target_kl: 0.25
                            clip_range: 0.2
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.99  # discount factor

                            # tricks
                            use_linear_lr_decay: False  # if use linear learning rate decay
                            end_factor_lr_decay: 0.5
                            use_global_state: False  # if use global state to replace joint observations
                            use_grad_norm: True  # gradient normalization
                            max_grad_norm: 10.0
                            use_value_clip: True  # limit the value range
                            value_clip_range: 0.2
                            use_value_norm: True  # use running mean and std to normalize rewards.
                            use_huber_loss: True  # True: use huber loss; False: use MSE loss.
                            huber_delta: 10.0
                            use_advnorm: True  # use advantage normalization.
                            use_gae: True  # use GAE trick to calculate returns.
                            gae_lambda: 0.95

                            start_training: 1
                            running_steps: 1000000
                            training_frequency: 1

                            eval_interval: 10000
                            test_episode: 16
                            log_dir: "./logs/ippo/"
                            model_dir: "./models/ippo/"


                    .. group-tab:: 5m_vs_6m

                        .. code-block:: yaml

                            agent: "IPPO"
                            env_name: "StarCraft2"
                            env_id: "5m_vs_6m"
                            fps: 15
                            policy: "Categorical_MAAC_Policy"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: True

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, 64, 64]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0
                            normalize: "LayerNorm"
                            initialize: "orthogonal"
                            gain: 0.01

                            actor_hidden_size: []
                            critic_hidden_size: []
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            n_size: 128
                            n_epoch: 10
                            n_minibatch: 1
                            learning_rate: 0.0007  # 7e-4
                            weight_decay: 0

                            vf_coef: 1.0
                            ent_coef: 0.01
                            target_kl: 0.25
                            clip_range: 0.05
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.99  # discount factor

                            # tricks
                            use_linear_lr_decay: False  # if use linear learning rate decay
                            end_factor_lr_decay: 0.5
                            use_global_state: False  # if use global state to replace joint observations
                            use_grad_norm: True  # gradient normalization
                            max_grad_norm: 10.0
                            use_value_clip: True  # limit the value range
                            value_clip_range: 0.05
                            use_value_norm: True  # use running mean and std to normalize rewards.
                            use_huber_loss: True  # True: use huber loss; False: use MSE loss.
                            huber_delta: 10.0
                            use_advnorm: True  # use advantage normalization.
                            use_gae: True  # use GAE trick to calculate returns.
                            gae_lambda: 0.95

                            start_training: 1
                            running_steps: 10000000  # 10M
                            training_frequency: 1

                            eval_interval: 100000
                            test_episode: 16
                            log_dir: "./logs/ippo/"
                            model_dir: "./models/ippo/"


                    .. group-tab:: 8m

                        .. code-block:: yaml

                            agent: "IPPO"
                            env_name: "StarCraft2"
                            env_id: "8m"
                            fps: 15
                            policy: "Categorical_MAAC_Policy"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: True

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, 64, 64]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0
                            normalize: "LayerNorm"
                            initialize: "orthogonal"
                            gain: 0.01

                            actor_hidden_size: []
                            critic_hidden_size: []
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            n_size: 128
                            n_epoch: 15
                            n_minibatch: 1
                            learning_rate: 0.0007  # 7e-4
                            weight_decay: 0

                            vf_coef: 1.0
                            ent_coef: 0.01
                            target_kl: 0.25
                            clip_range: 0.2
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.99  # discount factor

                            # tricks
                            use_linear_lr_decay: False  # if use linear learning rate decay
                            end_factor_lr_decay: 0.5
                            use_global_state: False  # if use global state to replace joint observations
                            use_grad_norm: True  # gradient normalization
                            max_grad_norm: 10.0
                            use_value_clip: True  # limit the value range
                            value_clip_range: 0.2
                            use_value_norm: True  # use running mean and std to normalize rewards.
                            use_huber_loss: True  # True: use huber loss; False: use MSE loss.
                            huber_delta: 10.0
                            use_advnorm: True  # use advantage normalization.
                            use_gae: True  # use GAE trick to calculate returns.
                            gae_lambda: 0.95

                            start_training: 1
                            running_steps: 1000000  # 1M
                            training_frequency: 1

                            eval_interval: 10000
                            test_episode: 16
                            log_dir: "./logs/ippo/"
                            model_dir: "./models/ippo/"


                    .. group-tab:: 8m_vd_9m

                        .. code-block:: yaml

                            agent: "IPPO"
                            env_name: "StarCraft2"
                            env_id: "8m_vs_9m"
                            fps: 15
                            policy: "Categorical_MAAC_Policy"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: True

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, 64, 64]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0
                            normalize: "LayerNorm"
                            initialize: "orthogonal"
                            gain: 0.01

                            actor_hidden_size: []
                            critic_hidden_size: []
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            n_size: 128
                            n_epoch: 15
                            n_minibatch: 1
                            learning_rate: 0.0007  # 7e-4
                            weight_decay: 0

                            vf_coef: 1.0
                            ent_coef: 0.01
                            target_kl: 0.25
                            clip_range: 0.05
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.99  # discount factor

                            # tricks
                            use_linear_lr_decay: False  # if use linear learning rate decay
                            end_factor_lr_decay: 0.5
                            use_global_state: False  # if use global state to replace joint observations
                            use_grad_norm: True  # gradient normalization
                            max_grad_norm: 10.0
                            use_value_clip: True  # limit the value range
                            value_clip_range: 0.05
                            use_value_norm: True  # use running mean and std to normalize rewards.
                            use_huber_loss: True  # True: use huber loss; False: use MSE loss.
                            huber_delta: 10.0
                            use_advnorm: True  # use advantage normalization.
                            use_gae: True  # use GAE trick to calculate returns.
                            gae_lambda: 0.95

                            start_training: 1
                            running_steps: 10000000  # 10M
                            training_frequency: 1

                            eval_interval: 100000
                            test_episode: 16
                            log_dir: "./logs/ippo/"
                            model_dir: "./models/ippo/"


                    .. group-tab:: 25m

                        .. code-block:: yaml

                            agent: "IPPO"
                            env_name: "StarCraft2"
                            env_id: "25m"
                            fps: 15
                            policy: "Categorical_MAAC_Policy"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: True

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, 64, 64]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0
                            normalize: "LayerNorm"
                            initialize: "orthogonal"
                            gain: 0.01

                            actor_hidden_size: []
                            critic_hidden_size: []
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            n_size: 128
                            n_epoch: 10
                            n_minibatch: 1
                            learning_rate: 0.0007  # 7e-4
                            weight_decay: 0

                            vf_coef: 1.0
                            ent_coef: 0.01
                            target_kl: 0.25
                            clip_range: 0.2
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.99  # discount factor

                            # tricks
                            use_linear_lr_decay: False  # if use linear learning rate decay
                            end_factor_lr_decay: 0.5
                            use_global_state: False  # if use global state to replace joint observations
                            use_grad_norm: True  # gradient normalization
                            max_grad_norm: 10.0
                            use_value_clip: True  # limit the value range
                            value_clip_range: 0.2
                            use_value_norm: True  # use running mean and std to normalize rewards.
                            use_huber_loss: True  # True: use huber loss; False: use MSE loss.
                            huber_delta: 10.0
                            use_advnorm: True  # use advantage normalization.
                            use_gae: True  # use GAE trick to calculate returns.
                            gae_lambda: 0.95

                            start_training: 1
                            running_steps: 5000000  # 5M
                            training_frequency: 1

                            eval_interval: 50000
                            test_episode: 16
                            log_dir: "./logs/ippo/"
                            model_dir: "./models/ippo/"


                    .. group-tab:: corridor

                        .. code-block:: yaml

                            agent: "IPPO"
                            env_name: "StarCraft2"
                            env_id: "corridor"
                            fps: 15
                            policy: "Categorical_MAAC_Policy"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: True

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, 64, 64]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0
                            normalize: "LayerNorm"
                            initialize: "orthogonal"
                            gain: 0.01

                            actor_hidden_size: []
                            critic_hidden_size: []
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            n_size: 128
                            n_epoch: 5
                            n_minibatch: 1
                            learning_rate: 0.0007  # 7e-4
                            weight_decay: 0

                            vf_coef: 1.0
                            ent_coef: 0.01
                            target_kl: 0.25
                            clip_range: 0.2
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.99  # discount factor

                            # tricks
                            use_linear_lr_decay: False  # if use linear learning rate decay
                            end_factor_lr_decay: 0.5
                            use_global_state: False  # if use global state to replace joint observations
                            use_grad_norm: True  # gradient normalization
                            max_grad_norm: 10.0
                            use_value_clip: True  # limit the value range
                            value_clip_range: 0.2
                            use_value_norm: True  # use running mean and std to normalize rewards.
                            use_huber_loss: True  # True: use huber loss; False: use MSE loss.
                            huber_delta: 10.0
                            use_advnorm: True  # use advantage normalization.
                            use_gae: True  # use GAE trick to calculate returns.
                            gae_lambda: 0.95

                            start_training: 1
                            running_steps: 10000000  # 10M
                            training_frequency: 1

                            eval_interval: 100000
                            test_episode: 16
                            log_dir: "./logs/ippo/"
                            model_dir: "./models/ippo/"

                    
                    .. group-tab:: MMM2

                        .. code-block:: yaml

                            agent: "IPPO"
                            env_name: "StarCraft2"
                            env_id: "MMM2"
                            fps: 15
                            policy: "Categorical_MAAC_Policy"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: True

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, 64, 64]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0
                            normalize: "LayerNorm"
                            initialize: "orthogonal"
                            gain: 1.0

                            actor_hidden_size: []
                            critic_hidden_size: []
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            n_size: 128
                            n_epoch: 5
                            n_minibatch: 2
                            learning_rate: 0.0007  # 7e-4
                            weight_decay: 0

                            vf_coef: 1.0
                            ent_coef: 0.01
                            target_kl: 0.25
                            clip_range: 0.2
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.99  # discount factor

                            # tricks
                            use_linear_lr_decay: False  # if use linear learning rate decay
                            end_factor_lr_decay: 0.5
                            use_global_state: False  # if use global state to replace joint observations
                            use_grad_norm: True  # gradient normalization
                            max_grad_norm: 10.0
                            use_value_clip: True  # limit the value range
                            value_clip_range: 0.2
                            use_value_norm: True  # use running mean and std to normalize rewards.
                            use_huber_loss: True  # True: use huber loss; False: use MSE loss.
                            huber_delta: 10.0
                            use_advnorm: True  # use advantage normalization.
                            use_gae: True  # use GAE trick to calculate returns.
                            gae_lambda: 0.95

                            start_training: 1
                            running_steps: 10000000  # 10M
                            training_frequency: 1

                            eval_interval: 100000
                            test_episode: 16
                            log_dir: "./logs/ippo/"
                            model_dir: "./models/ippo/"


            .. group-tab:: Football

                .. tabs::

                    .. group-tab:: 3v1

                        .. code-block:: yaml

                            agent: "IPPO"  # the learning algorithms_marl
                            global_state: True
                            # environment settings
                            env_name: "Football"
                            scenario: "academy_3_vs_1_with_keeper"
                            use_stacked_frames: False  # Whether to use stacked_frames
                            num_agent: 3
                            num_adversary: 0
                            obs_type: "simple115v2"  # representation used to build the observation, choices: ["simple115v2", "extracted", "pixels_gray", "pixels"]
                            rewards_type: "scoring,checkpoints"  # comma separated list of rewards to be added
                            smm_width: 96  # width of super minimap
                            smm_height: 72  # height of super minimap
                            fps: 15
                            policy: "Categorical_MAAC_Policy"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_Football"
                            runner: "Football_Runner"

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0
                            normalize: "LayerNorm"
                            initialize: "orthogonal"
                            gain: 0.01

                            actor_hidden_size: [64, 64]
                            critic_hidden_size: [64, 64]
                            activation: "ReLU"

                            seed: 1
                            parallels: 50
                            n_size: 400
                            n_epoch: 15
                            n_minibatch: 2
                            learning_rate: 5.0e-4
                            weight_decay: 0

                            vf_coef: 1.0
                            ent_coef: 0.01
                            clip_range: 0.2
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.99  # discount factor

                            # tricks
                            use_linear_lr_decay: False  # if use linear learning rate decay
                            end_factor_lr_decay: 0.5
                            use_global_state: False  # if use global state to calculate values
                            use_grad_norm: True  # gradient normalization
                            max_grad_norm: 10.0
                            use_value_clip: True  # limit the value range
                            value_clip_range: 0.2
                            use_value_norm: True  # use running mean and std to normalize rewards.
                            use_huber_loss: True  # True: use huber loss; False: use MSE loss.
                            huber_delta: 10.0
                            use_advnorm: True  # use advantage normalization.
                            use_gae: True  # use GAE trick to calculate returns.
                            gae_lambda: 0.95

                            start_training: 1
                            running_steps: 25000000
                            training_frequency: 1

                            eval_interval: 200000
                            test_episode: 50
                            log_dir: "./logs/ippo/"
                            model_dir: "./models/ippo/"
                            videos_dir: "./videos/ippo/"


    .. group-tab:: MAPPO

        .. tabs::

            .. group-tab:: MPE

                .. tabs::

                    .. group-tab:: simple_adversary_v3

                        .. code-block:: yaml

                            agent: "MAPPO"
                            env_name: "mpe"
                            env_id: "simple_adversary_v3"
                            continuous_action: True
                            policy: "Gaussian_MAAC_Policy"
                            representation: "Basic_MLP"
                            vectorize: "Dummy_Pettingzoo"
                            runner: "Pettingzoo_Runner"

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: False
                            rnn:
                            representation_hidden_size: [64, ]  # the units for each hidden layer
                            gain: 0.01

                            actor_hidden_size: [64, ]
                            critic_hidden_size: [256, ]
                            activation: "ReLU"

                            seed: 1
                            parallels: 128
                            n_size: 3200
                            n_epoch: 10
                            n_minibatch: 1
                            learning_rate: 0.0007
                            weight_decay: 0

                            vf_coef: 0.5
                            ent_coef: 0.01
                            target_kl: 0.25  # for MAPPO_KL learner
                            clip_range: 0.2  # ratio clip range, for MAPPO_Clip learner
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.95  # discount factor

                            # tricks
                            use_linear_lr_decay: False  # if use linear learning rate decay
                            end_factor_lr_decay: 0.5
                            use_global_state: False  # if use global state to replace merged observations
                            use_grad_norm: True  # gradient normalization
                            max_grad_norm: 10.0
                            use_value_clip: True  # limit the value range
                            value_clip_range: 0.2
                            use_value_norm: True  # use running mean and std to normalize rewards.
                            use_huber_loss: True  # True: use huber loss; False: use MSE loss.
                            huber_delta: 10.0
                            use_advnorm: True
                            use_gae: True
                            gae_lambda: 0.95

                            start_training: 1
                            running_steps: 10000000
                            train_per_step: True
                            training_frequency: 1

                            test_steps: 10000
                            eval_interval: 100000
                            test_episode: 5
                            log_dir: "./logs/mappo/"
                            model_dir: "./models/mappo/"


                    .. group-tab:: simple_push_v3

                        .. code-block:: yaml

                            agent: "MAPPO"
                            env_name: "mpe"
                            env_id: "simple_push_v3"
                            continuous_action: True
                            policy: "Gaussian_MAAC_Policy"
                            representation: "Basic_MLP"
                            vectorize: "Dummy_Pettingzoo"
                            runner: "Pettingzoo_Runner"

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: False
                            rnn:
                            representation_hidden_size: [64, ]  # the units for each hidden layer
                            gain: 0.01

                            actor_hidden_size: [64, ]
                            critic_hidden_size: [256, ]
                            activation: "ReLU"

                            seed: 1
                            parallels: 128
                            n_size: 3200
                            n_epoch: 10
                            n_minibatch: 1
                            learning_rate: 0.0007
                            weight_decay: 0

                            vf_coef: 0.5
                            ent_coef: 0.01
                            target_kl: 0.25  # for MAPPO_KL learner
                            clip_range: 0.2  # ratio clip range, for MAPPO_Clip learner
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.95  # discount factor

                            # tricks
                            use_linear_lr_decay: False  # if use linear learning rate decay
                            end_factor_lr_decay: 0.5
                            use_global_state: False  # if use global state to replace merged observations
                            use_grad_norm: True  # gradient normalization
                            max_grad_norm: 10.0
                            use_value_clip: True  # limit the value range
                            value_clip_range: 0.2
                            use_value_norm: True  # use running mean and std to normalize rewards.
                            use_huber_loss: True  # True: use huber loss; False: use MSE loss.
                            huber_delta: 10.0
                            use_advnorm: True
                            use_gae: True
                            gae_lambda: 0.95

                            start_training: 1
                            running_steps: 10000000
                            train_per_step: True
                            training_frequency: 1

                            test_steps: 10000
                            eval_interval: 100000
                            test_episode: 5
                            log_dir: "./logs/mappo/"
                            model_dir: "./models/mappo/"


                    .. group-tab:: simple_spread_v3

                        .. code-block:: yaml

                            agent: "MAPPO"
                            env_name: "mpe"
                            env_id: "simple_spread_v3"
                            continuous_action: True
                            policy: "Gaussian_MAAC_Policy"
                            representation: "Basic_MLP"
                            vectorize: "Dummy_Pettingzoo"
                            runner: "Pettingzoo_Runner"

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: False
                            rnn:
                            representation_hidden_size: [64, ]  # the units for each hidden layer
                            gain: 0.01

                            actor_hidden_size: [64, 64]
                            critic_hidden_size: [64, 64]
                            activation: "ReLU"

                            seed: 1
                            parallels: 8  # 128
                            n_size: 3200
                            n_epoch: 10
                            n_minibatch: 1
                            learning_rate: 0.0007
                            weight_decay: 0

                            vf_coef: 0.5
                            ent_coef: 0.01
                            target_kl: 0.25  # for MAPPO_KL learner
                            clip_range: 0.2  # ratio clip range, for MAPPO_Clip learner
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.95  # discount factor

                            # tricks
                            use_linear_lr_decay: False  # if use linear learning rate decay
                            end_factor_lr_decay: 0.5
                            use_global_state: False  # if use global state to replace merged observations
                            use_grad_norm: True  # gradient normalization
                            max_grad_norm: 10.0
                            use_value_clip: True  # limit the value range
                            value_clip_range: 0.2
                            use_value_norm: True  # use running mean and std to normalize rewards.
                            use_huber_loss: True  # True: use huber loss; False: use MSE loss.
                            huber_delta: 10.0
                            use_advnorm: True
                            use_gae: True
                            gae_lambda: 0.95

                            start_training: 1
                            running_steps: 10000000
                            train_per_step: True
                            training_frequency: 1

                            test_steps: 10000
                            eval_interval: 100000
                            test_episode: 5
                            log_dir: "./logs/mappo/"
                            model_dir: "./models/mappo/"


            .. group-tab:: SC2

                .. tabs::

                    .. group-tab:: 1c3s5z

                        .. code-block:: yaml

                            agent: "MAPPO"
                            env_name: "StarCraft2"
                            env_id: "1c3s5z"
                            fps: 15
                            policy: "Categorical_MAAC_Policy"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, 64, 64]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0
                            normalize: "LayerNorm"
                            initialize: "orthogonal"
                            gain: 0.01

                            actor_hidden_size: []
                            critic_hidden_size: []
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            n_size: 128
                            n_epoch: 15
                            n_minibatch: 1
                            learning_rate: 0.0007  # 7e-4
                            weight_decay: 0

                            vf_coef: 1.0
                            ent_coef: 0.01
                            target_kl: 0.25
                            clip_range: 0.2
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.99  # discount factor

                            # tricks
                            use_linear_lr_decay: False  # if use linear learning rate decay
                            end_factor_lr_decay: 0.5
                            use_global_state: True  # if use global state to calculate values
                            use_grad_norm: True  # gradient normalization
                            max_grad_norm: 10.0
                            use_value_clip: True  # limit the value range
                            value_clip_range: 0.2
                            use_value_norm: True  # use running mean and std to normalize rewards.
                            use_huber_loss: True  # True: use huber loss; False: use MSE loss.
                            huber_delta: 10.0
                            use_advnorm: True  # use advantage normalization.
                            use_gae: True  # use GAE trick to calculate returns.
                            gae_lambda: 0.95

                            start_training: 1
                            running_steps: 2000000
                            training_frequency: 1

                            eval_interval: 20000
                            test_episode: 16
                            log_dir: "./logs/mappo/"
                            model_dir: "./models/mappo/"


                    .. group-tab:: 2m_vs_1z

                        .. code-block:: yaml

                            agent: "MAPPO"
                            env_name: "StarCraft2"
                            env_id: "2m_vs_1z"
                            fps: 15
                            policy: "Categorical_MAAC_Policy"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: True

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, 64, 64]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0
                            normalize: "LayerNorm"
                            initialize: "orthogonal"
                            gain: 0.01

                            actor_hidden_size: []
                            critic_hidden_size: []
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            n_size: 128
                            n_epoch: 15
                            n_minibatch: 1
                            learning_rate: 0.0007  # 7e-4
                            weight_decay: 0

                            vf_coef: 1.0
                            ent_coef: 0.01
                            target_kl: 0.25
                            clip_range: 0.2
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.99  # discount factor

                            # tricks
                            use_linear_lr_decay: False  # if use linear learning rate decay
                            end_factor_lr_decay: 0.5
                            use_global_state: True  # if use global state to replace merged observations
                            use_grad_norm: True  # gradient normalization
                            max_grad_norm: 10.0
                            use_value_clip: True  # limit the value range
                            value_clip_range: 0.2
                            use_value_norm: True  # use running mean and std to normalize rewards.
                            use_huber_loss: True  # True: use huber loss; False: use MSE loss.
                            huber_delta: 10.0
                            use_advnorm: True  # use advantage normalization.
                            use_gae: True  # use GAE trick to calculate returns.
                            gae_lambda: 0.95

                            start_training: 1
                            running_steps: 1000000
                            train_per_step: True
                            training_frequency: 1

                            test_steps: 10000
                            eval_interval: 10000
                            test_episode: 16
                            log_dir: "./logs/mappo/"
                            model_dir: "./models/mappo/"


                    .. group-tab:: 2s3z

                        .. code-block:: yaml

                            agent: "MAPPO"
                            env_name: "StarCraft2"
                            env_id: "2s3z"
                            fps: 15
                            policy: "Categorical_MAAC_Policy"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, 64, 64]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0
                            normalize: "LayerNorm"
                            initialize: "orthogonal"
                            gain: 0.01

                            actor_hidden_size: []
                            critic_hidden_size: []
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            n_size: 128
                            n_epoch: 15
                            n_minibatch: 1
                            learning_rate: 0.0007  # 7e-4
                            weight_decay: 0

                            vf_coef: 1.0
                            ent_coef: 0.01
                            target_kl: 0.25
                            clip_range: 0.2
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.99  # discount factor

                            # tricks
                            use_linear_lr_decay: False  # if use linear learning rate decay
                            end_factor_lr_decay: 0.5
                            use_global_state: True  # if use global state to calculate values
                            use_grad_norm: True  # gradient normalization
                            max_grad_norm: 10.0
                            use_value_clip: True  # limit the value range
                            value_clip_range: 0.2
                            use_value_norm: True  # use running mean and std to normalize rewards.
                            use_huber_loss: True  # True: use huber loss; False: use MSE loss.
                            huber_delta: 10.0
                            use_advnorm: True  # use advantage normalization.
                            use_gae: True  # use GAE trick to calculate returns.
                            gae_lambda: 0.95

                            start_training: 1
                            running_steps: 2000000
                            training_frequency: 1

                            eval_interval: 20000
                            test_episode: 16
                            log_dir: "./logs/mappo/"
                            model_dir: "./models/mappo/"


                    .. group-tab:: 3m

                        .. code-block:: yaml

                            agent: "MAPPO"
                            env_name: "StarCraft2"
                            env_id: "3m"
                            fps: 15
                            policy: "Categorical_MAAC_Policy"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, 64, 64]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0
                            normalize: "LayerNorm"
                            initialize: "orthogonal"
                            gain: 0.01

                            actor_hidden_size: []
                            critic_hidden_size: []
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            n_size: 128
                            n_epoch: 15
                            n_minibatch: 1
                            learning_rate: 0.0007  # 7e-4
                            weight_decay: 0

                            vf_coef: 1.0
                            ent_coef: 0.01
                            target_kl: 0.25
                            clip_range: 0.2
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.99  # discount factor

                            # tricks
                            use_linear_lr_decay: False  # if use linear learning rate decay
                            end_factor_lr_decay: 0.5
                            use_global_state: True  # if use global state to calculate values
                            use_grad_norm: True  # gradient normalization
                            max_grad_norm: 10.0
                            use_value_clip: True  # limit the value range
                            value_clip_range: 0.2
                            use_value_norm: True  # use running mean and std to normalize rewards.
                            use_huber_loss: True  # True: use huber loss; False: use MSE loss.
                            huber_delta: 10.0
                            use_advnorm: True  # use advantage normalization.
                            use_gae: True  # use GAE trick to calculate returns.
                            gae_lambda: 0.95

                            start_training: 1
                            running_steps: 1000000
                            training_frequency: 1

                            eval_interval: 10000
                            test_episode: 16
                            log_dir: "./logs/mappo/"
                            model_dir: "./models/mappo/"


                    .. group-tab:: 5m_vs_6m

                        .. code-block:: yaml

                            agent: "MAPPO"
                            env_name: "StarCraft2"
                            env_id: "5m_vs_6m"
                            fps: 15
                            policy: "Categorical_MAAC_Policy"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: True

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0
                            normalize: "LayerNorm"
                            initialize: "orthogonal"
                            gain: 0.01

                            actor_hidden_size: [64, 64]
                            critic_hidden_size: [64, 64]
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            n_size: 128
                            n_epoch: 10
                            n_minibatch: 1
                            learning_rate: 0.0007  # 7e-4
                            weight_decay: 0

                            vf_coef: 1.0
                            ent_coef: 0.01
                            target_kl: 0.25
                            clip_range: 0.05
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.99  # discount factor

                            # tricks
                            use_linear_lr_decay: False  # if use linear learning rate decay
                            end_factor_lr_decay: 0.5
                            use_global_state: True  # if use global state to calculate values
                            use_grad_norm: True  # gradient normalization
                            max_grad_norm: 10.0
                            use_value_clip: True  # limit the value range
                            value_clip_range: 0.05
                            use_value_norm: True  # use running mean and std to normalize rewards.
                            use_huber_loss: True  # True: use huber loss; False: use MSE loss.
                            huber_delta: 10.0
                            use_advnorm: True  # use advantage normalization.
                            use_gae: True  # use GAE trick to calculate returns.
                            gae_lambda: 0.95

                            start_training: 1
                            running_steps: 10000000  # 10M
                            training_frequency: 1

                            eval_interval: 100000
                            test_episode: 16
                            log_dir: "./logs/mappo/"
                            model_dir: "./models/mappo/"


                    .. group-tab:: 8m

                        .. code-block:: yaml

                            agent: "MAPPO"
                            env_name: "StarCraft2"
                            env_id: "8m"
                            fps: 15
                            policy: "Categorical_MAAC_Policy"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: True

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, 64, 64]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0
                            normalize: "LayerNorm"
                            initialize: "orthogonal"
                            gain: 0.01

                            actor_hidden_size: []
                            critic_hidden_size: []
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            n_size: 128
                            n_epoch: 15
                            n_minibatch: 1
                            learning_rate: 0.0007  # 7e-4
                            weight_decay: 0

                            vf_coef: 1.0
                            ent_coef: 0.01
                            target_kl: 0.25
                            clip_range: 0.2
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.99  # discount factor

                            # tricks
                            use_linear_lr_decay: False  # if use linear learning rate decay
                            end_factor_lr_decay: 0.5
                            use_global_state: True  # if use global state to calculate values
                            use_grad_norm: True  # gradient normalization
                            max_grad_norm: 10.0
                            use_value_clip: True  # limit the value range
                            value_clip_range: 0.2
                            use_value_norm: True  # use running mean and std to normalize rewards.
                            use_huber_loss: True  # True: use huber loss; False: use MSE loss.
                            huber_delta: 10.0
                            use_advnorm: True  # use advantage normalization.
                            use_gae: True  # use GAE trick to calculate returns.
                            gae_lambda: 0.95

                            start_training: 1
                            running_steps: 1000000  # 1M
                            training_frequency: 1

                            eval_interval: 10000
                            test_episode: 16
                            log_dir: "./logs/mappo/"
                            model_dir: "./models/mappo/"


                    .. group-tab:: 8m_vd_9m

                        .. code-block:: yaml

                            agent: "MAPPO"
                            env_name: "StarCraft2"
                            env_id: "8m_vs_9m"
                            fps: 15
                            policy: "Categorical_MAAC_Policy"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: True

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64,]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0
                            normalize: "LayerNorm"
                            initialize: "orthogonal"
                            gain: 0.01

                            actor_hidden_size: [64, 64]
                            critic_hidden_size: [64, 64]
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            n_size: 128
                            n_epoch: 15
                            n_minibatch: 1
                            learning_rate: 0.0007  # 7e-4
                            weight_decay: 0

                            vf_coef: 1.0
                            ent_coef: 0.01
                            target_kl: 0.25
                            clip_range: 0.05
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.99  # discount factor

                            # tricks
                            use_linear_lr_decay: False  # if use linear learning rate decay
                            end_factor_lr_decay: 0.5
                            use_global_state: True  # if use global state to calculate values
                            use_grad_norm: True  # gradient normalization
                            max_grad_norm: 10.0
                            use_value_clip: True  # limit the value range
                            value_clip_range: 0.05
                            use_value_norm: True  # use running mean and std to normalize rewards.
                            use_huber_loss: True  # True: use huber loss; False: use MSE loss.
                            huber_delta: 10.0
                            use_advnorm: True  # use advantage normalization.
                            use_gae: True  # use GAE trick to calculate returns.
                            gae_lambda: 0.95

                            start_training: 1
                            running_steps: 10000000  # 10M
                            training_frequency: 1

                            eval_interval: 100000
                            test_episode: 16
                            log_dir: "./logs/mappo/"
                            model_dir: "./models/mappo/"


                    .. group-tab:: 25m

                        .. code-block:: yaml

                            agent: "MAPPO"
                            env_name: "StarCraft2"
                            env_id: "25m"
                            fps: 15
                            policy: "Categorical_MAAC_Policy"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: True

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, 64, 64]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0
                            normalize: "LayerNorm"
                            initialize: "orthogonal"
                            gain: 0.01

                            actor_hidden_size: []
                            critic_hidden_size: []
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            n_size: 128
                            n_epoch: 10
                            n_minibatch: 1
                            learning_rate: 0.0007  # 7e-4
                            weight_decay: 0

                            vf_coef: 1.0
                            ent_coef: 0.01
                            target_kl: 0.25
                            clip_range: 0.2
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.99  # discount factor

                            # tricks
                            use_linear_lr_decay: False  # if use linear learning rate decay
                            end_factor_lr_decay: 0.5
                            use_global_state: True  # if use global state to calculate values
                            use_grad_norm: True  # gradient normalization
                            max_grad_norm: 10.0
                            use_value_clip: True  # limit the value range
                            value_clip_range: 0.2
                            use_value_norm: True  # use running mean and std to normalize rewards.
                            use_huber_loss: True  # True: use huber loss; False: use MSE loss.
                            huber_delta: 10.0
                            use_advnorm: True  # use advantage normalization.
                            use_gae: True  # use GAE trick to calculate returns.
                            gae_lambda: 0.95

                            start_training: 1
                            running_steps: 5000000  # 5M
                            training_frequency: 1

                            eval_interval: 50000
                            test_episode: 16
                            log_dir: "./logs/mappo/"
                            model_dir: "./models/mappo/"


                    .. group-tab:: corridor

                        .. code-block:: yaml

                            agent: "MAPPO"
                            env_name: "StarCraft2"
                            env_id: "corridor"
                            fps: 15
                            policy: "Categorical_MAAC_Policy"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: True

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, 64, 64]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0
                            normalize: "LayerNorm"
                            initialize: "orthogonal"
                            gain: 0.01

                            actor_hidden_size: []
                            critic_hidden_size: []
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            n_size: 128
                            n_epoch: 5
                            n_minibatch: 1
                            learning_rate: 0.0007  # 7e-4
                            weight_decay: 0

                            vf_coef: 1.0
                            ent_coef: 0.01
                            target_kl: 0.25
                            clip_range: 0.2
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.99  # discount factor

                            # tricks
                            use_linear_lr_decay: False  # if use linear learning rate decay
                            end_factor_lr_decay: 0.5
                            use_global_state: True  # if use global state to calculate values
                            use_grad_norm: True  # gradient normalization
                            max_grad_norm: 10.0
                            use_value_clip: True  # limit the value range
                            value_clip_range: 0.2
                            use_value_norm: True  # use running mean and std to normalize rewards.
                            use_huber_loss: True  # True: use huber loss; False: use MSE loss.
                            huber_delta: 10.0
                            use_advnorm: True  # use advantage normalization.
                            use_gae: True  # use GAE trick to calculate returns.
                            gae_lambda: 0.95

                            start_training: 1
                            running_steps: 10000000  # 10M
                            training_frequency: 1

                            eval_interval: 100000
                            test_episode: 16
                            log_dir: "./logs/mappo/"
                            model_dir: "./models/mappo/"

                    
                    .. group-tab:: MMM2

                        .. code-block:: yaml

                            agent: "MAPPO"
                            env_name: "StarCraft2"
                            env_id: "MMM2"
                            fps: 15
                            policy: "Categorical_MAAC_Policy"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: True

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, 64, 64]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0
                            normalize: "LayerNorm"
                            initialize: "orthogonal"
                            gain: 1.0

                            actor_hidden_size: []
                            critic_hidden_size: []
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            n_size: 128
                            n_epoch: 5
                            n_minibatch: 2
                            learning_rate: 0.0007  # 7e-4
                            weight_decay: 0

                            vf_coef: 1.0
                            ent_coef: 0.01
                            target_kl: 0.25
                            clip_range: 0.2
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.99  # discount factor

                            # tricks
                            use_linear_lr_decay: False  # if use linear learning rate decay
                            end_factor_lr_decay: 0.5
                            use_global_state: True  # if use global state to calculate values
                            use_grad_norm: True  # gradient normalization
                            max_grad_norm: 10.0
                            use_value_clip: True  # limit the value range
                            value_clip_range: 0.2
                            use_value_norm: True  # use running mean and std to normalize rewards.
                            use_huber_loss: True  # True: use huber loss; False: use MSE loss.
                            huber_delta: 10.0
                            use_advnorm: True  # use advantage normalization.
                            use_gae: True  # use GAE trick to calculate returns.
                            gae_lambda: 0.95

                            start_training: 1
                            running_steps: 10000000  # 10M
                            training_frequency: 1

                            eval_interval: 100000
                            test_episode: 16
                            log_dir: "./logs/mappo/"
                            model_dir: "./models/mappo/"


            .. group-tab:: Football

                .. tabs::

                    .. group-tab:: 1v1

                        .. code-block:: yaml

                            agent: "MAPPO"  # the learning algorithms_marl
                            global_state: True
                            # environment settings
                            env_name: "Football"
                            scenario: "1_vs_1_easy"
                            use_stacked_frames: False  # Whether to use stacked_frames
                            num_agent: 1
                            num_adversary: 0
                            obs_type: "simple115v2"  # representation used to build the observation, choices: ["simple115v2", "extracted", "pixels_gray", "pixels"]
                            rewards_type: "scoring,checkpoints"  # comma separated list of rewards to be added
                            smm_width: 96  # width of super minimap
                            smm_height: 72  # height of super minimap
                            fps: 15
                            policy: "Categorical_MAAC_Policy"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_Football"
                            runner: "Football_Runner"

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0
                            normalize: "LayerNorm"
                            initialize: "orthogonal"
                            gain: 0.01

                            actor_hidden_size: [64, 64]
                            critic_hidden_size: [64, 64]
                            activation: "ReLU"

                            seed: 1
                            parallels: 50
                            n_size: 100
                            n_epoch: 10
                            n_minibatch: 1
                            learning_rate: 5.0e-4
                            weight_decay: 0

                            vf_coef: 1.0
                            ent_coef: 0.01
                            clip_range: 0.2
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.99  # discount factor

                            # tricks
                            use_linear_lr_decay: False  # if use linear learning rate decay
                            end_factor_lr_decay: 0.5
                            use_global_state: True  # if use global state to calculate values
                            use_grad_norm: True  # gradient normalization
                            max_grad_norm: 10.0
                            use_value_clip: True  # limit the value range
                            value_clip_range: 0.2
                            use_value_norm: True  # use running mean and std to normalize rewards.
                            use_huber_loss: True  # True: use huber loss; False: use MSE loss.
                            huber_delta: 10.0
                            use_advnorm: True  # use advantage normalization.
                            use_gae: True  # use GAE trick to calculate returns.
                            gae_lambda: 0.95

                            start_training: 1
                            running_steps: 25000000
                            training_frequency: 1

                            eval_interval: 200000
                            test_episode: 50
                            log_dir: "./logs/mappo/"
                            model_dir: "./models/mappo/"
                            videos_dir: "./videos/mappo/"

                    
                    .. group-tab:: 3v1

                        .. code-block:: yaml

                            agent: "MAPPO"  # the learning algorithms_marl
                            global_state: True
                            # environment settings
                            env_name: "Football"
                            scenario: "academy_3_vs_1_with_keeper"
                            use_stacked_frames: False  # Whether to use stacked_frames
                            num_agent: 3
                            num_adversary: 0
                            obs_type: "simple115v2"  # representation used to build the observation, choices: ["simple115v2", "extracted", "pixels_gray", "pixels"]
                            rewards_type: "scoring,checkpoints"  # comma separated list of rewards to be added
                            smm_width: 96  # width of super minimap
                            smm_height: 72  # height of super minimap
                            fps: 15
                            policy: "Categorical_MAAC_Policy"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_Football"
                            runner: "Football_Runner"

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0
                            normalize: "LayerNorm"
                            initialize: "orthogonal"
                            gain: 0.01

                            actor_hidden_size: [64, 64]
                            critic_hidden_size: [64, 64]
                            activation: "ReLU"

                            seed: 1
                            parallels: 50
                            n_size: 400
                            n_epoch: 15
                            n_minibatch: 2
                            learning_rate: 5.0e-4
                            weight_decay: 0

                            vf_coef: 1.0
                            ent_coef: 0.01
                            clip_range: 0.2
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.99  # discount factor

                            # tricks
                            use_linear_lr_decay: False  # if use linear learning rate decay
                            end_factor_lr_decay: 0.5
                            use_global_state: True  # if use global state to calculate values
                            use_grad_norm: True  # gradient normalization
                            max_grad_norm: 10.0
                            use_value_clip: True  # limit the value range
                            value_clip_range: 0.2
                            use_value_norm: True  # use running mean and std to normalize rewards.
                            use_huber_loss: True  # True: use huber loss; False: use MSE loss.
                            huber_delta: 10.0
                            use_advnorm: True  # use advantage normalization.
                            use_gae: True  # use GAE trick to calculate returns.
                            gae_lambda: 0.95

                            start_training: 1
                            running_steps: 25000000
                            training_frequency: 1

                            eval_interval: 200000
                            test_episode: 50
                            log_dir: "./logs/mappo/"
                            model_dir: "./models/mappo/"
                            videos_dir: "./videos/mappo/"


    .. group-tab:: MATD3

        .. tabs::

            .. group-tab:: MPE

                .. tabs::

                    .. group-tab:: simple_adversary_v3

                        .. code-block:: yaml

                            agent: "MATD3"  # the learning algorithms_marl
                            env_name: "mpe"
                            env_id: "simple_adversary_v3"
                            continuous_action: True
                            policy: "MATD3_Policy"
                            representation: "Basic_Identical"
                            vectorize: "Dummy_Pettingzoo"
                            runner: "Pettingzoo_Runner"

                            representation_hidden_size: [64, ]  # the units for each hidden layer
                            actor_hidden_size: [64, 64]
                            critic_hidden_size: [64, 64]
                            activation: 'LeakyReLU'
                            activation_action: 'sigmoid'

                            seed: 1
                            parallels: 16
                            buffer_size: 100000
                            batch_size: 256
                            lr_a: 0.01  # learning rate for actor
                            lr_c: 0.001  # learning rate for critic
                            gamma: 0.95  # discount factor
                            tau: 0.001  # soft update for target networks

                            start_noise: 1.0
                            end_noise: 0.01
                            sigma: 0.1
                            start_training: 1000  # start training after n episodes
                            running_steps: 10000000
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1

                            use_grad_clip: True
                            grad_clip_norm: 0.5

                            eval_interval: 100000
                            test_episode: 5
                            log_dir: "./logs/matd3/"
                            model_dir: "./models/matd3/"


                    .. group-tab:: simple_push_v3

                        .. code-block:: yaml

                            agent: "MATD3"  # the learning algorithms_marl
                            env_name: "mpe"
                            env_id: "simple_push_v3"
                            continuous_action: True
                            policy: "MATD3_Policy"
                            representation: "Basic_Identical"
                            vectorize: "Dummy_Pettingzoo"
                            runner: "Pettingzoo_Runner"

                            representation_hidden_size: [64, ]  # the units for each hidden layer
                            actor_hidden_size: [64, 64]
                            critic_hidden_size: [64, 64]
                            activation: 'LeakyReLU'
                            activation_action: 'sigmoid'

                            seed: 1
                            parallels: 16
                            buffer_size: 100000
                            batch_size: 256
                            lr_a: 0.01  # learning rate for actor
                            lr_c: 0.001  # learning rate for critic
                            gamma: 0.95  # discount factor
                            tau: 0.001  # soft update for target networks

                            start_noise: 1.0
                            end_noise: 0.01
                            sigma: 0.1
                            start_training: 1000  # start training after n episodes
                            running_steps: 10000000
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1

                            use_grad_clip: True
                            grad_clip_norm: 0.5

                            eval_interval: 100000
                            test_episode: 5
                            log_dir: "./logs/matd3/"
                            model_dir: "./models/matd3/"


                    .. group-tab:: simple_spread_v3

                        .. code-block:: yaml

                            agent: "MATD3"  # the learning algorithms_marl
                            env_name: "mpe"
                            env_id: "simple_spread_v3"
                            continuous_action: True
                            policy: "MATD3_Policy"
                            representation: "Basic_Identical"
                            vectorize: "Dummy_Pettingzoo"
                            runner: "Pettingzoo_Runner"

                            representation_hidden_size: [64, ]  # the units for each hidden layer
                            actor_hidden_size: [64, 64]
                            critic_hidden_size: [64, 64]
                            activation: 'LeakyReLU'
                            activation_action: 'sigmoid'

                            seed: 1
                            parallels: 16
                            buffer_size: 100000
                            batch_size: 256
                            lr_a: 0.01  # learning rate for actor
                            lr_c: 0.001  # learning rate for critic
                            gamma: 0.95  # discount factor
                            tau: 0.001  # soft update for target networks

                            start_noise: 1.0
                            end_noise: 0.01
                            sigma: 0.1
                            start_training: 1000  # start training after n episodes
                            running_steps: 10000000
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1

                            use_grad_clip: True
                            grad_clip_norm: 0.5

                            eval_interval: 100000
                            test_episode: 5
                            log_dir: "./logs/matd3/"
                            model_dir: "./models/matd3/"


    .. group-tab:: VDAC

        .. tabs::

            .. group-tab:: MPE

                .. tabs::

                    .. group-tab:: simple_spread_v3

                        .. code-block:: yaml

                            agent: "VDAC"
                            env_name: "mpe"
                            env_id: "simple_spread_v3"
                            continuous_action: False
                            policy: "Categorical_MAAC_Policy_Share"
                            representation: "Basic_MLP"
                            vectorize: "Dummy_Pettingzoo"
                            runner: "Pettingzoo_Runner"

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: False
                            rnn:
                            representation_hidden_size: [64, ]  # the units for each hidden layer
                            gain: 0.01

                            actor_hidden_size: [64, ]
                            critic_hidden_size: [64, ]
                            activation: "ReLU"

                            mixer: "VDN"  # choices: VDN (sum), QMIX (monotonic)
                            hidden_dim_mixing_net: 32  # hidden units of mixing network
                            hidden_dim_hyper_net: 32  # hidden units of hyper network

                            seed: 1
                            parallels: 128
                            n_size: 3200
                            n_epoch: 10
                            n_minibatch: 1
                            learning_rate: 0.0007
                            weight_decay: 0

                            vf_coef: 0.5
                            ent_coef: 0.01
                            target_kl: 0.25  # for MAPPO_KL learner
                            clip_range: 0.2  # ratio clip range, for MAPPO_Clip learner
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.95  # discount factor

                            # tricks
                            use_linear_lr_decay: False  # if use linear learning rate decay
                            end_factor_lr_decay: 0.5
                            use_global_state: False  # if use global state to replace merged observations
                            use_grad_norm: True  # gradient normalization
                            max_grad_norm: 10.0
                            use_value_clip: True  # limit the value range
                            value_clip_range: 0.2
                            use_value_norm: True  # use running mean and std to normalize rewards.
                            use_huber_loss: True  # True: use huber loss; False: use MSE loss.
                            huber_delta: 10.0
                            use_advnorm: True
                            use_gae: True
                            gae_lambda: 0.95

                            start_training: 1
                            running_steps: 10000000
                            train_per_step: True
                            training_frequency: 1

                            test_steps: 10000
                            eval_interval: 100000
                            test_episode: 5
                            log_dir: "./logs/vdac/"
                            model_dir: "./models/vdac/"


            .. group-tab:: SC2

                .. tabs::

                    .. group-tab:: 1c3s5z

                        .. code-block:: yaml

                            agent: "VDAC"
                            env_name: "StarCraft2"
                            env_id: "1c3s5z"
                            fps: 15
                            policy: "Categorical_MAAC_Policy_Share"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0
                            normalize: "LayerNorm"
                            initialize: "orthogonal"
                            gain: 0.01

                            actor_hidden_size: []
                            critic_hidden_size: []
                            activation: "ReLU"

                            mixer: "QMIX"  # choices: VDN (sum), QMIX (monotonic)
                            hidden_dim_mixing_net: 32  # hidden units of mixing network
                            hidden_dim_hyper_net: 64  # hidden units of hyper network

                            seed: 1
                            parallels: 8
                            n_size: 8
                            n_epoch: 1
                            n_minibatch: 1
                            learning_rate: 0.0007  # 7e-4
                            weight_decay: 0

                            vf_coef: 1.0
                            ent_coef: 0.01
                            target_kl: 0.25
                            clip_range: 0.2
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.99  # discount factor

                            # tricks
                            use_linear_lr_decay: False  # if use linear learning rate decay
                            end_factor_lr_decay: 0.5
                            use_grad_norm: True  # gradient normalization
                            max_grad_norm: 10.0
                            use_value_clip: True  # limit the value range
                            value_clip_range: 0.2
                            use_value_norm: True  # use running mean and std to normalize rewards.
                            use_huber_loss: True  # True: use huber loss; False: use MSE loss.
                            huber_delta: 10.0
                            use_advnorm: True  # use advantage normalization.
                            use_gae: True  # use GAE trick to calculate returns.
                            gae_lambda: 0.95

                            start_training: 1
                            running_steps: 2000000
                            train_per_step: True
                            training_frequency: 1

                            test_steps: 10000
                            eval_interval: 10000
                            test_episode: 16
                            log_dir: "./logs/vdac/"
                            model_dir: "./models/vdac/"


                    .. group-tab:: 2m_vs_1z

                        .. code-block:: yaml

                            agent: "VDAC"
                            env_name: "StarCraft2"
                            env_id: "2m_vs_1z"
                            fps: 15
                            policy: "Categorical_MAAC_Policy_Share"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0
                            normalize: "LayerNorm"
                            initialize: "orthogonal"
                            gain: 0.01

                            actor_hidden_size: []
                            critic_hidden_size: []
                            activation: "ReLU"

                            mixer: "QMIX"  # choices: VDN (sum), QMIX (monotonic)
                            hidden_dim_mixing_net: 32  # hidden units of mixing network
                            hidden_dim_hyper_net: 64  # hidden units of hyper network

                            seed: 1
                            parallels: 8
                            n_size: 8
                            n_epoch: 1
                            n_minibatch: 1
                            learning_rate: 0.0007  # 7e-4
                            weight_decay: 0

                            vf_coef: 1.0
                            ent_coef: 0.01
                            target_kl: 0.25
                            clip_range: 0.2
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.99  # discount factor

                            # tricks
                            use_linear_lr_decay: False  # if use linear learning rate decay
                            end_factor_lr_decay: 0.5
                            use_grad_norm: True  # gradient normalization
                            max_grad_norm: 10.0
                            use_value_clip: True  # limit the value range
                            value_clip_range: 0.2
                            use_value_norm: True  # use running mean and std to normalize rewards.
                            use_huber_loss: True  # True: use huber loss; False: use MSE loss.
                            huber_delta: 10.0
                            use_advnorm: True  # use advantage normalization.
                            use_gae: True  # use GAE trick to calculate returns.
                            gae_lambda: 0.95

                            start_training: 1
                            running_steps: 1000000
                            train_per_step: True
                            training_frequency: 1

                            test_steps: 10000
                            eval_interval: 5000
                            test_episode: 16
                            log_dir: "./logs/vdac/"
                            model_dir: "./models/vdac/"


                    .. group-tab:: 2s3z

                        .. code-block:: yaml

                            agent: "VDAC"
                            env_name: "StarCraft2"
                            env_id: "2s3z"
                            fps: 15
                            policy: "Categorical_MAAC_Policy_Share"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0
                            normalize: "LayerNorm"
                            initialize: "orthogonal"
                            gain: 0.01

                            actor_hidden_size: []
                            critic_hidden_size: []
                            activation: "ReLU"

                            mixer: "QMIX"  # choices: VDN (sum), QMIX (monotonic)
                            hidden_dim_mixing_net: 32  # hidden units of mixing network
                            hidden_dim_hyper_net: 64  # hidden units of hyper network

                            seed: 1
                            parallels: 8
                            n_size: 8
                            n_epoch: 1
                            n_minibatch: 1
                            learning_rate: 0.0007  # 7e-4
                            weight_decay: 0

                            vf_coef: 1.0
                            ent_coef: 0.01
                            target_kl: 0.25
                            clip_range: 0.2
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.99  # discount factor

                            # tricks
                            use_linear_lr_decay: False  # if use linear learning rate decay
                            end_factor_lr_decay: 0.5
                            use_grad_norm: True  # gradient normalization
                            max_grad_norm: 10.0
                            use_value_clip: True  # limit the value range
                            value_clip_range: 0.2
                            use_value_norm: True  # use running mean and std to normalize rewards.
                            use_huber_loss: True  # True: use huber loss; False: use MSE loss.
                            huber_delta: 10.0
                            use_advnorm: True  # use advantage normalization.
                            use_gae: True  # use GAE trick to calculate returns.
                            gae_lambda: 0.95

                            start_training: 1
                            running_steps: 2000000
                            train_per_step: True
                            training_frequency: 1

                            test_steps: 10000
                            eval_interval: 10000
                            test_episode: 16
                            log_dir: "./logs/vdac/"
                            model_dir: "./models/vdac/"


                    .. group-tab:: 3m

                        .. code-block:: yaml

                            agent: "VDAC"
                            env_name: "StarCraft2"
                            env_id: "3m"
                            fps: 15
                            policy: "Categorical_MAAC_Policy_Share"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64,]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0
                            normalize: "LayerNorm"
                            initialize: "orthogonal"
                            gain: 0.01

                            actor_hidden_size: []
                            critic_hidden_size: []
                            activation: "ReLU"

                            mixer: "QMIX"  # choices: Independent, VDN (sum), QMIX (monotonic)
                            hidden_dim_mixing_net: 32  # hidden units of mixing network
                            hidden_dim_hyper_net: 64  # hidden units of hyper network

                            seed: 1
                            parallels: 8
                            n_size: 8
                            n_epoch: 1
                            n_minibatch: 1
                            learning_rate: 0.0007  # 7e-4
                            weight_decay: 0

                            vf_coef: 1.0
                            ent_coef: 0.0
                            clip_range: 0.2
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.99  # discount factor

                            # tricks
                            use_linear_lr_decay: False  # if use linear learning rate decay
                            end_factor_lr_decay: 0.5
                            use_grad_norm: True  # gradient normalization
                            max_grad_norm: 10.0
                            use_value_norm: False  # use running mean and std to normalize rewards.
                            use_advnorm: False  # use advantage normalization.
                            use_gae: True  # use GAE trick to calculate returns.
                            gae_lambda: 0.95

                            start_training: 1
                            running_steps: 1000000
                            training_frequency: 1

                            eval_interval: 5000
                            test_episode: 16
                            log_dir: "./logs/vdac/"
                            model_dir: "./models/vdac/"


                    .. group-tab:: 5m_vs_6m

                        .. code-block:: yaml

                            agent: "VDAC"
                            env_name: "StarCraft2"
                            env_id: "5m_vs_6m"
                            fps: 15
                            policy: "Categorical_MAAC_Policy_Share"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: True

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0
                            normalize: "LayerNorm"
                            initialize: "orthogonal"
                            gain: 0.01

                            actor_hidden_size: []
                            critic_hidden_size: []
                            activation: "ReLU"

                            mixer: "QMIX"  # choices: VDN (sum), QMIX (monotonic)
                            hidden_dim_mixing_net: 32  # hidden units of mixing network
                            hidden_dim_hyper_net: 64  # hidden units of hyper network

                            seed: 1
                            parallels: 8
                            n_size: 8
                            n_epoch: 1
                            n_minibatch: 1
                            learning_rate: 0.0007  # 7e-4
                            weight_decay: 0

                            vf_coef: 1.0
                            ent_coef: 0.01
                            target_kl: 0.25
                            clip_range: 0.05
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.99  # discount factor

                            # tricks
                            use_linear_lr_decay: False  # if use linear learning rate decay
                            end_factor_lr_decay: 0.5
                            use_global_state: False  # if use global state to replace joint observations
                            use_grad_norm: True  # gradient normalization
                            max_grad_norm: 10.0
                            use_value_clip: True  # limit the value range
                            value_clip_range: 0.05
                            use_value_norm: True  # use running mean and std to normalize rewards.
                            use_huber_loss: True  # True: use huber loss; False: use MSE loss.
                            huber_delta: 10.0
                            use_advnorm: True  # use advantage normalization.
                            use_gae: True  # use GAE trick to calculate returns.
                            gae_lambda: 0.95

                            start_training: 1
                            running_steps: 10000000  # 10M
                            training_frequency: 1

                            eval_interval: 50000
                            test_episode: 16
                            log_dir: "./logs/vdac/"
                            model_dir: "./models/vdac/"


                    .. group-tab:: 8m

                        .. code-block:: yaml

                            agent: "VDAC"
                            env_name: "StarCraft2"
                            env_id: "8m"
                            fps: 15
                            policy: "Categorical_MAAC_Policy_Share"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: True

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0
                            normalize: "LayerNorm"
                            initialize: "orthogonal"
                            gain: 0.01

                            actor_hidden_size: []
                            critic_hidden_size: []
                            activation: "ReLU"

                            mixer: "QMIX"  # choices: VDN (sum), QMIX (monotonic)
                            hidden_dim_mixing_net: 32  # hidden units of mixing network
                            hidden_dim_hyper_net: 64  # hidden units of hyper network

                            seed: 1
                            parallels: 8
                            n_size: 8
                            n_epoch: 1
                            n_minibatch: 1
                            learning_rate: 0.0007  # 7e-4
                            weight_decay: 0

                            vf_coef: 1.0
                            ent_coef: 0.01
                            target_kl: 0.25
                            clip_range: 0.2
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.99  # discount factor

                            # tricks
                            use_linear_lr_decay: False  # if use linear learning rate decay
                            end_factor_lr_decay: 0.5
                            use_global_state: False  # if use global state to replace joint observations
                            use_grad_norm: True  # gradient normalization
                            max_grad_norm: 10.0
                            use_value_clip: True  # limit the value range
                            value_clip_range: 0.2
                            use_value_norm: True  # use running mean and std to normalize rewards.
                            use_huber_loss: True  # True: use huber loss; False: use MSE loss.
                            huber_delta: 10.0
                            use_advnorm: True  # use advantage normalization.
                            use_gae: True  # use GAE trick to calculate returns.
                            gae_lambda: 0.95

                            start_training: 1
                            running_steps: 1000000  # 1M
                            training_frequency: 1

                            eval_interval: 5000
                            test_episode: 16
                            log_dir: "./logs/vdac/"
                            model_dir: "./models/vdac/"


                    .. group-tab:: 8m_vd_9m

                        .. code-block:: yaml

                            agent: "VDAC"
                            env_name: "StarCraft2"
                            env_id: "8m_vs_9m"
                            fps: 15
                            policy: "Categorical_MAAC_Policy_Share"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: True

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0
                            normalize: "LayerNorm"
                            initialize: "orthogonal"
                            gain: 0.01

                            actor_hidden_size: []
                            critic_hidden_size: []
                            activation: "ReLU"

                            mixer: "QMIX"  # choices: VDN (sum), QMIX (monotonic)
                            hidden_dim_mixing_net: 32  # hidden units of mixing network
                            hidden_dim_hyper_net: 64  # hidden units of hyper network

                            seed: 1
                            parallels: 8
                            n_size: 8
                            n_epoch: 1
                            n_minibatch: 1
                            learning_rate: 0.0007  # 7e-4
                            weight_decay: 0

                            vf_coef: 1.0
                            ent_coef: 0.01
                            target_kl: 0.25
                            clip_range: 0.05
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.99  # discount factor

                            # tricks
                            use_linear_lr_decay: False  # if use linear learning rate decay
                            end_factor_lr_decay: 0.5
                            use_global_state: False  # if use global state to replace joint observations
                            use_grad_norm: True  # gradient normalization
                            max_grad_norm: 10.0
                            use_value_clip: True  # limit the value range
                            value_clip_range: 0.05
                            use_value_norm: True  # use running mean and std to normalize rewards.
                            use_huber_loss: True  # True: use huber loss; False: use MSE loss.
                            huber_delta: 10.0
                            use_advnorm: True  # use advantage normalization.
                            use_gae: True  # use GAE trick to calculate returns.
                            gae_lambda: 0.95

                            start_training: 1
                            running_steps: 10000000  # 10M
                            training_frequency: 1

                            eval_interval: 50000
                            test_episode: 16
                            log_dir: "./logs/vdac/"
                            model_dir: "./models/vdac/"


                    .. group-tab:: 25m

                        .. code-block:: yaml

                            agent: "VDAC"
                            env_name: "StarCraft2"
                            env_id: "25m"
                            fps: 15
                            policy: "Categorical_MAAC_Policy_Share"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: True

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0
                            normalize: "LayerNorm"
                            initialize: "orthogonal"
                            gain: 0.01

                            actor_hidden_size: []
                            critic_hidden_size: []
                            activation: "ReLU"

                            mixer: "QMIX"  # choices: VDN (sum), QMIX (monotonic)
                            hidden_dim_mixing_net: 32  # hidden units of mixing network
                            hidden_dim_hyper_net: 64  # hidden units of hyper network

                            seed: 1
                            parallels: 8
                            n_size: 8
                            n_epoch: 1
                            n_minibatch: 1
                            learning_rate: 0.0007  # 7e-4
                            weight_decay: 0

                            vf_coef: 1.0
                            ent_coef: 0.01
                            target_kl: 0.25
                            clip_range: 0.2
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.99  # discount factor

                            # tricks
                            use_linear_lr_decay: False  # if use linear learning rate decay
                            end_factor_lr_decay: 0.5
                            use_global_state: False  # if use global state to replace joint observations
                            use_grad_norm: True  # gradient normalization
                            max_grad_norm: 10.0
                            use_value_clip: True  # limit the value range
                            value_clip_range: 0.2
                            use_value_norm: True  # use running mean and std to normalize rewards.
                            use_huber_loss: True  # True: use huber loss; False: use MSE loss.
                            huber_delta: 10.0
                            use_advnorm: True  # use advantage normalization.
                            use_gae: True  # use GAE trick to calculate returns.
                            gae_lambda: 0.95

                            start_training: 1
                            running_steps: 5000000  # 5M
                            training_frequency: 1

                            eval_interval: 25000
                            test_episode: 16
                            log_dir: "./logs/vdac/"
                            model_dir: "./models/vdac/"


                    .. group-tab:: corridor

                        .. code-block:: yaml

                            agent: "VDAC"
                            env_name: "StarCraft2"
                            env_id: "corridor"
                            fps: 15
                            policy: "Categorical_MAAC_Policy_Share"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: True

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0
                            normalize: "LayerNorm"
                            initialize: "orthogonal"
                            gain: 0.01

                            actor_hidden_size: []
                            critic_hidden_size: []
                            activation: "ReLU"

                            mixer: "QMIX"  # choices: VDN (sum), QMIX (monotonic)
                            hidden_dim_mixing_net: 32  # hidden units of mixing network
                            hidden_dim_hyper_net: 64  # hidden units of hyper network

                            seed: 1
                            parallels: 8
                            n_size: 8
                            n_epoch: 1
                            n_minibatch: 1
                            learning_rate: 0.0007  # 7e-4
                            weight_decay: 0

                            vf_coef: 1.0
                            ent_coef: 0.01
                            target_kl: 0.25
                            clip_range: 0.2
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.99  # discount factor

                            # tricks
                            use_linear_lr_decay: False  # if use linear learning rate decay
                            end_factor_lr_decay: 0.5
                            use_global_state: False  # if use global state to replace joint observations
                            use_grad_norm: True  # gradient normalization
                            max_grad_norm: 10.0
                            use_value_clip: True  # limit the value range
                            value_clip_range: 0.2
                            use_value_norm: True  # use running mean and std to normalize rewards.
                            use_huber_loss: True  # True: use huber loss; False: use MSE loss.
                            huber_delta: 10.0
                            use_advnorm: True  # use advantage normalization.
                            use_gae: True  # use GAE trick to calculate returns.
                            gae_lambda: 0.95

                            start_training: 1
                            running_steps: 10000000  # 10M
                            training_frequency: 1

                            eval_interval: 50000
                            test_episode: 16
                            log_dir: "./logs/vdac/"
                            model_dir: "./models/vdac/"

                    
                    .. group-tab:: MMM2

                        .. code-block:: yaml

                            agent: "VDAC"
                            env_name: "StarCraft2"
                            env_id: "MMM2"
                            fps: 15
                            policy: "Categorical_MAAC_Policy_Share"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"
                            on_policy: True

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0
                            normalize: "LayerNorm"
                            initialize: "orthogonal"
                            gain: 1.0

                            actor_hidden_size: []
                            critic_hidden_size: []
                            activation: "ReLU"

                            mixer: "QMIX"  # choices: VDN (sum), QMIX (monotonic)
                            hidden_dim_mixing_net: 32  # hidden units of mixing network
                            hidden_dim_hyper_net: 64  # hidden units of hyper network

                            seed: 1
                            parallels: 8
                            n_size: 8
                            n_epoch: 1
                            n_minibatch: 2
                            learning_rate: 0.0007  # 7e-4
                            weight_decay: 0

                            vf_coef: 1.0
                            ent_coef: 0.01
                            target_kl: 0.25
                            clip_range: 0.2
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.99  # discount factor

                            # tricks
                            use_linear_lr_decay: False  # if use linear learning rate decay
                            end_factor_lr_decay: 0.5
                            use_global_state: False  # if use global state to replace joint observations
                            use_grad_norm: True  # gradient normalization
                            max_grad_norm: 10.0
                            use_value_clip: True  # limit the value range
                            value_clip_range: 0.2
                            use_value_norm: True  # use running mean and std to normalize rewards.
                            use_huber_loss: True  # True: use huber loss; False: use MSE loss.
                            huber_delta: 10.0
                            use_advnorm: True  # use advantage normalization.
                            use_gae: True  # use GAE trick to calculate returns.
                            gae_lambda: 0.95

                            start_training: 1
                            running_steps: 10000000  # 10M
                            training_frequency: 1

                            eval_interval: 50000
                            test_episode: 16
                            log_dir: "./logs/vdac/"
                            model_dir: "./models/vdac/"



    .. group-tab:: COMA

        .. tabs::

            .. group-tab:: MPE

                .. tabs::

                    .. group-tab:: simple_spread_v3

                        .. code-block:: yaml

                            agent: "COMA"  # the learning algorithms_marl
                            env_name: "mpe"
                            env_id: "simple_spread_v3"
                            continuous_action: False
                            policy: "Categorical_COMA_Policy"
                            representation: "Basic_MLP"
                            representation_critic: "Basic_MLP"
                            vectorize: "Dummy_Pettingzoo"
                            runner: "Pettingzoo_Runner"

                            use_recurrent: False
                            rnn:
                            representation_hidden_size: [128, ]
                            gain: 0.01

                            actor_hidden_size: [128, ]
                            critic_hidden_size: [128, ]
                            activation: "ReLU"

                            seed: 1
                            parallels: 10
                            n_size: 250
                            n_epoch: 10
                            n_minibatch: 1
                            learning_rate_actor: 0.0007
                            learning_rate_critic: 0.0007

                            clip_grad: 10
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.95  # discount factor
                            td_lambda: 0.1

                            start_greedy: 0.5
                            end_greedy: 0.01
                            decay_step_greedy: 2500000
                            sync_frequency: 200

                            use_global_state: True  # if use global state to replace merged observations
                            use_advnorm: True
                            use_gae: True
                            gae_lambda: 0.95

                            start_training: 1
                            running_steps: 10000000
                            train_per_step: True
                            training_frequency: 1

                            test_steps: 10000
                            eval_interval: 100000
                            test_episode: 5
                            log_dir: "./logs/coma/"
                            model_dir: "./models/coma/"


            .. group-tab:: SC2

                .. tabs::

                    .. group-tab:: 1c3s5z

                        .. code-block:: yaml

                            agent: "COMA"  # the learning algorithms_marl
                            env_name: "StarCraft2"
                            env_id: "1c3s5z"
                            fps: 15
                            policy: "Categorical_COMA_Policy"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"

                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0
                            normalize: "LayerNorm"
                            initialize: "orthogonal"
                            gain: 0.01

                            actor_hidden_size: [64, ]
                            critic_hidden_size: [128, 128]
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            n_size: 8
                            n_epoch: 1
                            n_minibatch: 1
                            learning_rate_actor: 0.0007
                            learning_rate_critic: 0.0007

                            clip_grad: 10
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.95  # discount factor
                            td_lambda: 0.8

                            start_greedy: 0.5
                            end_greedy: 0.01
                            decay_step_greedy: 2500000
                            sync_frequency: 200

                            use_global_state: True  # if use global state to replace merged observations
                            use_advnorm: False
                            use_gae:
                            gae_lambda: 0.95

                            start_training: 1
                            running_steps: 2000000
                            train_per_step: True
                            training_frequency: 1

                            test_steps: 10000
                            eval_interval: 20000
                            test_episode: 16
                            log_dir: "./logs/coma/"
                            model_dir: "./models/coma/"


                    .. group-tab:: 2m_vs_1z

                        .. code-block:: yaml

                            agent: "COMA"  # the learning algorithms_marl
                            env_name: "StarCraft2"
                            env_id: "2m_vs_1z"
                            fps: 15
                            policy: "Categorical_COMA_Policy"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"

                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0
                            normalize: "LayerNorm"
                            initialize: "orthogonal"
                            gain: 0.01

                            actor_hidden_size: [64, ]
                            critic_hidden_size: [128, 128]
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            n_size: 8
                            n_epoch: 1
                            n_minibatch: 1
                            learning_rate_actor: 0.0007
                            learning_rate_critic: 0.0007

                            clip_grad: 10
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.95  # discount factor
                            td_lambda: 0.8

                            start_greedy: 0.5
                            end_greedy: 0.01
                            decay_step_greedy: 2500000
                            sync_frequency: 200

                            use_global_state: True  # if use global state to replace merged observations
                            use_advnorm: False
                            use_gae:
                            gae_lambda: 0.95

                            start_training: 1
                            running_steps: 1000000
                            train_per_step: True
                            training_frequency: 1

                            test_steps: 10000
                            eval_interval: 10000
                            test_episode: 16
                            log_dir: "./logs/coma/"
                            model_dir: "./models/coma/"


                    .. group-tab:: 2s3z

                        .. code-block:: yaml

                            agent: "COMA"  # the learning algorithms_marl
                            env_name: "StarCraft2"
                            env_id: "2s3z"
                            fps: 15
                            policy: "Categorical_COMA_Policy"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"

                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0
                            normalize: "LayerNorm"
                            initialize: "orthogonal"
                            gain: 0.01

                            actor_hidden_size: [64, ]
                            critic_hidden_size: [128, 128]
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            n_size: 8
                            n_epoch: 1
                            n_minibatch: 1
                            learning_rate_actor: 0.0007
                            learning_rate_critic: 0.0007

                            clip_grad: 10
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.95  # discount factor
                            td_lambda: 0.8

                            start_greedy: 0.5
                            end_greedy: 0.01
                            decay_step_greedy: 2500000
                            sync_frequency: 200

                            use_global_state: True  # if use global state to replace merged observations
                            use_advnorm: False
                            use_gae:
                            gae_lambda: 0.95

                            start_training: 1
                            running_steps: 2000000
                            train_per_step: True
                            training_frequency: 1

                            test_steps: 10000
                            eval_interval: 20000
                            test_episode: 16
                            log_dir: "./logs/coma/"
                            model_dir: "./models/coma/"


                    .. group-tab:: 3m

                        .. code-block:: yaml

                            agent: "COMA"  # the learning algorithms_marl
                            env_name: "StarCraft2"
                            env_id: "3m"
                            fps: 15
                            policy: "Categorical_COMA_Policy"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"

                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0
                            normalize: "LayerNorm"
                            initialize: "orthogonal"
                            gain: 0.01

                            actor_hidden_size: [64, ]
                            critic_hidden_size: [128, 128]
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            n_size: 8
                            n_epoch: 1
                            n_minibatch: 1
                            learning_rate_actor: 0.0007
                            learning_rate_critic: 0.0007

                            clip_grad: 10
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.95  # discount factor
                            td_lambda: 0.8

                            start_greedy: 0.5
                            end_greedy: 0.01
                            decay_step_greedy: 100000
                            sync_frequency: 200

                            use_global_state: True  # if use global state to replace merged observations
                            use_advnorm: False
                            use_gae:
                            gae_lambda: 0.95

                            start_training: 1
                            running_steps: 1000000
                            train_per_step: True
                            training_frequency: 1

                            test_steps: 10000
                            eval_interval: 10000
                            test_episode: 16
                            log_dir: "./logs/coma/"
                            model_dir: "./models/coma/"


                    .. group-tab:: 5m_vs_6m

                        .. code-block:: yaml

                            agent: "COMA"  # the learning algorithms_marl
                            env_name: "StarCraft2"
                            env_id: "5m_vs_6m"
                            fps: 15
                            policy: "Categorical_COMA_Policy"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"

                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0
                            normalize: "LayerNorm"
                            initialize: "orthogonal"
                            gain: 0.01

                            actor_hidden_size: [64, ]
                            critic_hidden_size: [128, 128]
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            n_size: 8
                            n_epoch: 1
                            n_minibatch: 1
                            learning_rate_actor: 0.0007
                            learning_rate_critic: 0.0007

                            clip_grad: 10
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.95  # discount factor
                            td_lambda: 0.8

                            start_greedy: 0.5
                            end_greedy: 0.01
                            decay_step_greedy: 2000000
                            sync_frequency: 200

                            use_global_state: True  # if use global state to replace merged observations
                            use_advnorm: False
                            use_gae:
                            gae_lambda: 0.95

                            start_training: 1
                            running_steps: 10000000
                            train_per_step: True
                            training_frequency: 1

                            test_steps: 10000
                            eval_interval: 100000
                            test_episode: 16
                            log_dir: "./logs/coma/"
                            model_dir: "./models/coma/"


                    .. group-tab:: 8m

                        .. code-block:: yaml

                            agent: "COMA"  # the learning algorithms_marl
                            env_name: "StarCraft2"
                            env_id: "8m"
                            fps: 15
                            policy: "Categorical_COMA_Policy"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"

                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0
                            normalize: "LayerNorm"
                            initialize: "orthogonal"
                            gain: 0.01

                            actor_hidden_size: [64, ]
                            critic_hidden_size: [128, 128]
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            n_size: 8
                            n_epoch: 1
                            n_minibatch: 1
                            learning_rate_actor: 0.0007
                            learning_rate_critic: 0.0007

                            clip_grad: 10
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.95  # discount factor
                            td_lambda: 0.8

                            start_greedy: 0.5
                            end_greedy: 0.01
                            decay_step_greedy: 2500000
                            sync_frequency: 200

                            use_global_state: True  # if use global state to replace merged observations
                            use_advnorm: False
                            use_gae:
                            gae_lambda: 0.95

                            start_training: 1
                            running_steps: 1000000
                            train_per_step: True
                            training_frequency: 1

                            test_steps: 10000
                            eval_interval: 10000
                            test_episode: 16
                            log_dir: "./logs/coma/"
                            model_dir: "./models/coma/"


                    .. group-tab:: 8m_vd_9m

                        .. code-block:: yaml

                            agent: "COMA"  # the learning algorithms_marl
                            env_name: "StarCraft2"
                            env_id: "8m_vs_9m"
                            fps: 15
                            policy: "Categorical_COMA_Policy"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"

                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0
                            normalize: "LayerNorm"
                            initialize: "orthogonal"
                            gain: 0.01

                            actor_hidden_size: [64, ]
                            critic_hidden_size: [128, 128]
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            n_size: 8
                            n_epoch: 1
                            n_minibatch: 1
                            learning_rate_actor: 0.0007
                            learning_rate_critic: 0.0007

                            clip_grad: 10
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.95  # discount factor
                            td_lambda: 0.8

                            start_greedy: 0.5
                            end_greedy: 0.01
                            decay_step_greedy: 2500000
                            sync_frequency: 200

                            use_global_state: True  # if use global state to replace merged observations
                            use_advnorm: False
                            use_gae:
                            gae_lambda: 0.95

                            start_training: 1
                            running_steps: 10000000
                            train_per_step: True
                            training_frequency: 1

                            test_steps: 10000
                            eval_interval: 100000
                            test_episode: 16
                            log_dir: "./logs/coma/"
                            model_dir: "./models/coma/"


                    .. group-tab:: 25m

                        .. code-block:: yaml

                            agent: "COMA"  # the learning algorithms_marl
                            env_name: "StarCraft2"
                            env_id: "25m"
                            fps: 15
                            policy: "Categorical_COMA_Policy"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"

                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0
                            normalize: "LayerNorm"
                            initialize: "orthogonal"
                            gain: 0.01

                            actor_hidden_size: [64, ]
                            critic_hidden_size: [128, 128]
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            n_size: 8
                            n_epoch: 1
                            n_minibatch: 1
                            learning_rate_actor: 0.0007
                            learning_rate_critic: 0.0007

                            clip_grad: 10
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.95  # discount factor
                            td_lambda: 0.8

                            start_greedy: 0.5
                            end_greedy: 0.01
                            decay_step_greedy: 1000000
                            sync_frequency: 200

                            use_global_state: True  # if use global state to replace merged observations
                            use_advnorm: False
                            use_gae:
                            gae_lambda: 0.95

                            start_training: 1
                            running_steps: 5000000
                            train_per_step: True
                            training_frequency: 1

                            test_steps: 10000
                            eval_interval: 50000
                            test_episode: 16
                            log_dir: "./logs/coma/"
                            model_dir: "./models/coma/"


                    .. group-tab:: corridor

                        .. code-block:: yaml

                            agent: "COMA"  # the learning algorithms_marl
                            env_name: "StarCraft2"
                            env_id: "corridor"
                            fps: 15
                            policy: "Categorical_COMA_Policy"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"

                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0
                            normalize: "LayerNorm"
                            initialize: "orthogonal"
                            gain: 0.01

                            actor_hidden_size: [64, ]
                            critic_hidden_size: [128, 128]
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            n_size: 8
                            n_epoch: 1
                            n_minibatch: 1
                            learning_rate_actor: 0.0007
                            learning_rate_critic: 0.0007

                            clip_grad: 10
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.95  # discount factor
                            td_lambda: 0.8

                            start_greedy: 0.5
                            end_greedy: 0.01
                            decay_step_greedy: 1000000
                            sync_frequency: 200

                            use_global_state: True  # if use global state to replace merged observations
                            use_advnorm: False
                            use_gae:
                            gae_lambda: 0.95

                            start_training: 1
                            running_steps: 10000000
                            train_per_step: True
                            training_frequency: 1

                            test_steps: 10000
                            eval_interval: 100000
                            test_episode: 16
                            log_dir: "./logs/coma/"
                            model_dir: "./models/coma/"

                    
                    .. group-tab:: MMM2

                        .. code-block:: yaml

                            agent: "COMA"  # the learning algorithms_marl
                            env_name: "StarCraft2"
                            env_id: "MMM2"
                            fps: 15
                            policy: "Categorical_COMA_Policy"
                            representation: "Basic_RNN"
                            vectorize: "Subproc_StarCraft2"
                            runner: "StarCraft2_Runner"

                            use_recurrent: True
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1
                            dropout: 0
                            normalize: "LayerNorm"
                            initialize: "orthogonal"
                            gain: 0.01

                            actor_hidden_size: [64, ]
                            critic_hidden_size: [128, 128]
                            activation: "ReLU"

                            seed: 1
                            parallels: 8
                            n_size: 8
                            n_epoch: 1
                            n_minibatch: 1
                            learning_rate_actor: 0.0007
                            learning_rate_critic: 0.0007

                            clip_grad: 10
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.95  # discount factor
                            td_lambda: 0.8

                            start_greedy: 0.5
                            end_greedy: 0.01
                            decay_step_greedy: 1000000
                            sync_frequency: 200

                            use_global_state: True  # if use global state to replace merged observations
                            use_advnorm: False
                            use_gae:
                            gae_lambda: 0.95

                            start_training: 1
                            running_steps: 10000000
                            train_per_step: True
                            training_frequency: 1

                            test_steps: 10000
                            eval_interval: 100000
                            test_episode: 16
                            log_dir: "./logs/coma/"
                            model_dir: "./models/coma/"


    .. group-tab:: MFQ

        .. tabs::

            .. group-tab:: MPE

                .. tabs::

                    .. group-tab:: simple_spread_v3

                        .. code-block:: yaml

                            agent: "MFQ"  # the learning algorithms_marl
                            env_name: "mpe"
                            env_id: "simple_spread_v3"
                            continuous_action: False
                            policy: "MF_Q_network"
                            representation: "Basic_Identical"
                            vectorize: "Dummy_Pettingzoo"
                            runner: "Pettingzoo_Runner"

                            use_recurrent: False
                            rnn:
                            representation_hidden_size: [64, ]
                            q_hidden_size: [64, ]  # the units for each hidden layer
                            activation: "ReLU"

                            seed: 1
                            parallels: 16
                            buffer_size: 200000
                            batch_size: 256
                            learning_rate: 0.001
                            gamma: 0.95  # discount factor
                            double_q: True  # use double q learning
                            temperature: 0.1  # softmax for policy

                            start_greedy: 1.0
                            end_greedy: 0.05
                            decay_step_greedy: 2500000
                            start_training: 1000  # start training after n episodes
                            running_steps: 10000000  # 10M
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 100

                            use_grad_clip: False
                            grad_clip_norm: 0.5

                            n_tests: 5
                            test_period: 100

                            eval_interval: 100000
                            test_episode: 5
                            log_dir: "./logs/mfq/"
                            model_dir: "./models/mfq/"


            .. group-tab:: Magent2

                .. tabs::

                    .. group-tab:: adversarial_pursuit_v4

                        .. code-block:: yaml

                            agent: "MFQ"  # the learning algorithms_marl
                            env_name: "MAgent2"
                            env_id: "adversarial_pursuit_v4"
                            minimap_mode: False
                            max_cycles: 500
                            extra_features: False
                            map_size: 45
                            render_mode: "rgb_array"
                            policy: "MF_Q_network"
                            representation: "Basic_MLP"
                            vectorize: "Dummy_MAgent"
                            runner: "MAgent_Runner"
                            on_policy: False

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: False
                            rnn: "GRU"
                            recurrent_layer_N: 1
                            fc_hidden_sizes: [64, ]
                            recurrent_hidden_size: 64
                            N_recurrent_layers: 1

                            representation_hidden_size: [512, ]  # the units for each hidden layer
                            q_hidden_size: [512, ]
                            activation: "ReLU"

                            seed: 1
                            parallels: 10
                            buffer_size: 2000
                            batch_size: 256
                            learning_rate: 0.001
                            gamma: 0.95  # discount factor
                            temperature: 0.1  # softmax for policy

                            start_greedy: 0.0
                            end_greedy: 0.95
                            decay_step_greedy: 5000
                            start_training: 1000  # start training after n episodes
                            running_steps: 1000000
                            train_per_step: False  # True: train model per step; False: train model per episode.
                            training_frequency: 1
                            sync_frequency: 200

                            n_tests: 5
                            test_episodes: 10
                            eval_interval: 10000
                            test_episode: 5
                            log_dir: "./logs/mfq/"
                            model_dir: "./models/mfq/"


    .. group-tab:: MFAC

        .. tabs::

            .. group-tab:: MPE

                .. tabs::

                    .. group-tab:: simple_spread_v3

                        .. code-block:: yaml

                            agent: "MFAC"  # the learning algorithms_marl
                            env_name: "mpe"
                            env_id: "simple_spread_v3"
                            continuous_action: False
                            policy: "Categorical_MFAC_Policy"
                            representation: "Basic_Identical"
                            vectorize: "Dummy_Pettingzoo"
                            runner: "Pettingzoo_Runner"

                            # recurrent settings for Basic_RNN representation
                            use_recurrent: False
                            rnn:
                            representation_hidden_size: [64, ]  # the units for each hidden layer
                            gain: 0.01

                            actor_hidden_size: [128, ]
                            critic_hidden_size: [128, ]
                            activation: 'LeakyReLU'
                            activation_action: 'sigmoid'

                            seed: 1
                            parallels: 128
                            buffer_size: 3200
                            n_epoch: 10
                            n_minibatch: 1
                            learning_rate: 0.01  # learning rate
                            weight_decay: 0

                            vf_coef: 0.5
                            ent_coef: 0.01
                            target_kl: 0.25  # for MAPPO_KL learner
                            clip_range: 0.2  # ratio clip range, for MAPPO_Clip learner
                            clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm()
                            gamma: 0.95  # discount factor
                            tau: 0.005

                            # tricks
                            use_linear_lr_decay: False  # if use linear learning rate decay
                            end_factor_lr_decay: 0.5
                            use_global_state: False  # if use global state to replace merged observations
                            use_grad_norm: True  # gradient normalization
                            max_grad_norm: 10.0
                            use_value_clip: True  # limit the value range
                            value_clip_range: 0.2
                            use_value_norm: True  # use running mean and std to normalize rewards.
                            use_huber_loss: True  # True: use huber loss; False: use MSE loss.
                            huber_delta: 10.0
                            use_advnorm: True
                            use_gae: True
                            gae_lambda: 0.95

                            start_training: 1000  # start training after n episodes
                            running_steps: 10000000
                            train_per_step: True
                            training_frequency: 1

                            test_steps: 10000
                            eval_interval: 100000
                            test_episode: 5
                            log_dir: "./logs/mfac/"
                            model_dir: "./models/mfac/"


.. raw:: html

   <br><hr>
   
Customized Configurations
--------------------------
Users can also choose not to use the default parameters provided by XuanCe,
or in cases where XuanCe does not include the user's specific task, they can customize their own .yaml parameter configuration file in the same manner. 

However, during the process of obtaining the runner, it is necessary to specify the location where the parameter file is stored, as shown below:

.. code-block:: python

    import xuance as xp
    runner = xp.get_runner(method='dqn', 
                           env='classic_control',
                           env_id='CartPole-v1', 
                           config_path="xxx/xxx.yaml",
                           is_test=False)
    runner.run()
