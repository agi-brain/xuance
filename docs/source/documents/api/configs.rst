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

**DQN-based Algorithms:**

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

                    .. group-tab:: Acrobot-v1

                        .. code-block:: yaml

                    .. group-tab:: MountainCar-v0

                        .. code-block:: yaml
            
            .. group-tab:: Box2D

                .. tabs::

                    .. group-tab:: CarRacing-v2

                        .. code-block:: yaml

                    .. group-tab:: LunarLander-v2

                        .. code-block:: yaml
            
            .. group-tab:: Atari

                .. code-block:: yaml

    .. group-tab:: NoisyDQN

        .. tabs::

            .. group-tab:: Classic Control

                .. tabs::

                    .. group-tab:: CartPole-v1

                        .. code-block:: yaml

                    .. group-tab:: Acrobot-v1

                        .. code-block:: yaml

                    .. group-tab:: MountainCar-v0

                        .. code-block:: yaml
            
            .. group-tab:: Box2D

                .. tabs::

                    .. group-tab:: CarRacing-v2

                        .. code-block:: yaml

                    .. group-tab:: LunarLander-v2

                        .. code-block:: yaml
            
            .. group-tab:: Atari

                .. code-block:: yaml

    .. group-tab:: PerDQN

        .. tabs::

            .. group-tab:: Classic Control

                .. tabs::

                    .. group-tab:: CartPole-v1

                        .. code-block:: yaml

                    .. group-tab:: Acrobot-v1

                        .. code-block:: yaml

                    .. group-tab:: MountainCar-v0

                        .. code-block:: yaml
            
            .. group-tab:: Box2D

                .. tabs::

                    .. group-tab:: CarRacing-v2

                        .. code-block:: yaml

                    .. group-tab:: LunarLander-v2

                        .. code-block:: yaml
            
            .. group-tab:: Atari

                .. code-block:: yaml

    .. group-tab:: QRDQN

        .. tabs::

            .. group-tab:: Classic Control

                .. tabs::

                    .. group-tab:: CartPole-v1

                        .. code-block:: yaml

                    .. group-tab:: Acrobot-v1

                        .. code-block:: yaml

                    .. group-tab:: MountainCar-v0

                        .. code-block:: yaml
            
            .. group-tab:: Box2D

                .. tabs::

                    .. group-tab:: CarRacing-v2

                        .. code-block:: yaml

                    .. group-tab:: LunarLander-v2

                        .. code-block:: yaml
            
            .. group-tab:: Atari

                .. code-block:: yaml

.. raw:: html

   <br><hr>

**Policy Gradient-based Algorithms:**

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

                    .. group-tab:: Acrobot-v1

                        .. code-block:: yaml

                    .. group-tab:: Pendulum-v1

                        .. code-block:: yaml

                    .. group-tab:: MountainCar-v0

                        .. code-block:: yaml

            .. group-tab:: Box2D

                .. tabs::

                    .. group-tab:: BipedalWalker-v3

                        .. code-block:: yaml

                    .. group-tab:: CarRacing-v2

                        .. code-block:: yaml

                    .. group-tab:: LunarLander-v2

                        .. code-block:: yaml

            .. group-tab:: Atari

                .. code-block:: yaml

            .. group-tab:: MuJoCo

                .. code-block:: yaml


    .. group-tab:: A2C

        .. tabs::

            .. group-tab:: Classic Control

                .. tabs::

                    .. group-tab:: CartPole-v1

                        .. code-block:: yaml

                    .. group-tab:: Acrobot-v1

                        .. code-block:: yaml

                    .. group-tab:: Pendulum-v1

                        .. code-block:: yaml

                    .. group-tab:: MountainCar-v0

                        .. code-block:: yaml

            .. group-tab:: Box2D

                .. tabs::

                    .. group-tab:: BipedalWalker-v3

                        .. code-block:: yaml

                    .. group-tab:: CarRacing-v2

                        .. code-block:: yaml

                    .. group-tab:: LunarLander-v2

                        .. code-block:: yaml

            .. group-tab:: Atari

                .. code-block:: yaml

            .. group-tab:: MuJoCo

                .. code-block:: yaml


    .. group-tab:: SAC

        .. tabs::

            .. group-tab:: Classic Control

                .. tabs::

                    .. group-tab:: CartPole-v1

                        .. code-block:: yaml

                    .. group-tab:: Acrobot-v1

                        .. code-block:: yaml

                    .. group-tab:: Pendulum-v1

                        .. code-block:: yaml

                    .. group-tab:: MountainCar-v0

                        .. code-block:: yaml

            .. group-tab:: Box2D

                .. tabs::

                    .. group-tab:: BipedalWalker-v3

                        .. code-block:: yaml

                    .. group-tab:: CarRacing-v2

                        .. code-block:: yaml

                    .. group-tab:: LunarLander-v2

                        .. code-block:: yaml

            .. group-tab:: Atari

                .. code-block:: yaml

            .. group-tab:: MuJoCo

                .. code-block:: yaml


    .. group-tab:: DDPG

        .. tabs::

            .. group-tab:: Classic Control

                .. tabs::

                    .. group-tab:: Pendulum-v1

                        .. code-block:: yaml

            .. group-tab:: Box2D

                .. tabs::

                    .. group-tab:: BipedalWalker-v3

                        .. code-block:: yaml

            .. group-tab:: MuJoCo

                .. code-block:: yaml


    .. group-tab:: TD3

        .. tabs::

            .. group-tab:: Classic Control

                .. tabs::

                    .. group-tab:: Pendulum-v1

                        .. code-block:: yaml

            .. group-tab:: Box2D

                .. tabs::

                    .. group-tab:: BipedalWalker-v3

                        .. code-block:: yaml

            .. group-tab:: MuJoCo

                .. code-block:: yaml

.. raw:: html

   <br><hr>

**MARL-based Algorithms:**

.. tabs::

    .. group-tab:: IQL

        .. tabs::

            .. group-tab:: MPE

                .. tabs::

                    .. group-tab:: simple_spread_v3

                        .. code-block:: yaml

            .. group-tab:: Magent2

                .. tabs::

                    .. group-tab:: adversarial_pursuit_v4

                        .. code-block:: yaml

            .. group-tab:: SC2

                .. tabs::

                    .. group-tab:: 1c3s5z

                        .. code-block:: yaml

                    .. group-tab:: 2m_vs_1z

                        .. code-block:: yaml

                    .. group-tab:: 2s3z

                        .. code-block:: yaml

                    .. group-tab:: 3m

                        .. code-block:: yaml

                    .. group-tab:: 5m_vs_6m

                        .. code-block:: yaml

                    .. group-tab:: 8m

                        .. code-block:: yaml

                    .. group-tab:: 8m_vd_9m

                        .. code-block:: yaml

                    .. group-tab:: 25m

                        .. code-block:: yaml

                    .. group-tab:: corridor

                        .. code-block:: yaml
                    
                    .. group-tab:: MMM2

                        .. code-block:: yaml

            .. group-tab:: Football

                .. tabs::

                    .. group-tab:: 3v1

                        .. code-block:: yaml
    
    .. group-tab:: VDN

        .. tabs::

            .. group-tab:: MPE

                .. tabs::

                    .. group-tab:: simple_spread_v3

                        .. code-block:: yaml

            .. group-tab:: SC2

                .. tabs::

                    .. group-tab:: 1c3s5z

                        .. code-block:: yaml

                    .. group-tab:: 2m_vs_1z

                        .. code-block:: yaml

                    .. group-tab:: 2s3z

                        .. code-block:: yaml

                    .. group-tab:: 3m

                        .. code-block:: yaml

                    .. group-tab:: 5m_vs_6m

                        .. code-block:: yaml

                    .. group-tab:: 8m

                        .. code-block:: yaml

                    .. group-tab:: 8m_vd_9m

                        .. code-block:: yaml

                    .. group-tab:: 25m

                        .. code-block:: yaml

                    .. group-tab:: corridor

                        .. code-block:: yaml
                    
                    .. group-tab:: MMM2

                        .. code-block:: yaml

            .. group-tab:: Football

                .. tabs::

                    .. group-tab:: 3v1

                        .. code-block:: yaml
    
    .. group-tab:: QMIX

        .. tabs::

            .. group-tab:: MPE

                .. tabs::

                    .. group-tab:: simple_spread_v3

                        .. code-block:: yaml

            .. group-tab:: SC2

                .. tabs::

                    .. group-tab:: 1c3s5z

                        .. code-block:: yaml

                    .. group-tab:: 2m_vs_1z

                        .. code-block:: yaml

                    .. group-tab:: 2s3z

                        .. code-block:: yaml

                    .. group-tab:: 3m

                        .. code-block:: yaml

                    .. group-tab:: 5m_vs_6m

                        .. code-block:: yaml

                    .. group-tab:: 8m

                        .. code-block:: yaml

                    .. group-tab:: 8m_vd_9m

                        .. code-block:: yaml

                    .. group-tab:: 25m

                        .. code-block:: yaml

                    .. group-tab:: corridor

                        .. code-block:: yaml
                    
                    .. group-tab:: MMM2

                        .. code-block:: yaml

            .. group-tab:: Football

                .. tabs::

                    .. group-tab:: 3v1

                        .. code-block:: yaml

    .. group-tab:: WQMIX

        .. tabs::

            .. group-tab:: MPE

                .. tabs::

                    .. group-tab:: simple_spread_v3

                        .. code-block:: yaml

            .. group-tab:: SC2

                .. tabs::

                    .. group-tab:: 1c3s5z

                        .. code-block:: yaml

                    .. group-tab:: 2m_vs_1z

                        .. code-block:: yaml

                    .. group-tab:: 2s3z

                        .. code-block:: yaml

                    .. group-tab:: 3m

                        .. code-block:: yaml

                    .. group-tab:: 5m_vs_6m

                        .. code-block:: yaml

                    .. group-tab:: 8m

                        .. code-block:: yaml

                    .. group-tab:: 8m_vd_9m

                        .. code-block:: yaml

                    .. group-tab:: 25m

                        .. code-block:: yaml

                    .. group-tab:: corridor

                        .. code-block:: yaml
                    
                    .. group-tab:: MMM2

                        .. code-block:: yaml

    .. group-tab:: QTRAN

        .. tabs::

            .. group-tab:: MPE

                .. tabs::

                    .. group-tab:: simple_spread_v3

                        .. code-block:: yaml

    .. group-tab:: DCG

        .. tabs::

            .. group-tab:: MPE

                .. tabs::

                    .. group-tab:: simple_spread_v3

                        .. code-block:: yaml

            .. group-tab:: SC2

                .. tabs::

                    .. group-tab:: 1c3s5z

                        .. code-block:: yaml

                    .. group-tab:: 2m_vs_1z

                        .. code-block:: yaml

                    .. group-tab:: 2s3z

                        .. code-block:: yaml

                    .. group-tab:: 3m

                        .. code-block:: yaml

                    .. group-tab:: 5m_vs_6m

                        .. code-block:: yaml

                    .. group-tab:: 8m

                        .. code-block:: yaml

                    .. group-tab:: 8m_vd_9m

                        .. code-block:: yaml

                    .. group-tab:: 25m

                        .. code-block:: yaml

                    .. group-tab:: corridor

                        .. code-block:: yaml
                    
                    .. group-tab:: MMM2

                        .. code-block:: yaml

    .. group-tab:: IDDPG

        .. tabs::

            .. group-tab:: MPE

                .. tabs::

                    .. group-tab:: simple_adversary_v3

                        .. code-block:: yaml

                    .. group-tab:: simple_push_v3

                        .. code-block:: yaml

                    .. group-tab:: simple_spread_v3

                        .. code-block:: yaml

    .. group-tab:: MADDPG

        .. tabs::

            .. group-tab:: MPE

                .. tabs::

                    .. group-tab:: simple_adversary_v3

                        .. code-block:: yaml

                    .. group-tab:: simple_push_v3

                        .. code-block:: yaml

                    .. group-tab:: simple_spread_v3

                        .. code-block:: yaml

    .. group-tab:: ISAC

        .. tabs::

            .. group-tab:: MPE

                .. tabs::

                    .. group-tab:: simple_adversary_v3

                        .. code-block:: yaml

                    .. group-tab:: simple_push_v3

                        .. code-block:: yaml

                    .. group-tab:: simple_spread_v3

                        .. code-block:: yaml

    .. group-tab:: MASAC

        .. tabs::

            .. group-tab:: MPE

                .. tabs::

                    .. group-tab:: simple_adversary_v3

                        .. code-block:: yaml

                    .. group-tab:: simple_push_v3

                        .. code-block:: yaml

                    .. group-tab:: simple_spread_v3

                        .. code-block:: yaml

    .. group-tab:: IPPO

        .. tabs::

            .. group-tab:: MPE

                .. tabs::

                    .. group-tab:: simple_spread_v3

                        .. code-block:: yaml

            .. group-tab:: SC2

                .. tabs::

                    .. group-tab:: 1c3s5z

                        .. code-block:: yaml

                    .. group-tab:: 2m_vs_1z

                        .. code-block:: yaml

                    .. group-tab:: 2s3z

                        .. code-block:: yaml

                    .. group-tab:: 3m

                        .. code-block:: yaml

                    .. group-tab:: 5m_vs_6m

                        .. code-block:: yaml

                    .. group-tab:: 8m

                        .. code-block:: yaml

                    .. group-tab:: 8m_vd_9m

                        .. code-block:: yaml

                    .. group-tab:: 25m

                        .. code-block:: yaml

                    .. group-tab:: corridor

                        .. code-block:: yaml
                    
                    .. group-tab:: MMM2

                        .. code-block:: yaml

            .. group-tab:: Football

                .. tabs::

                    .. group-tab:: 3v1

                        .. code-block:: yaml

    .. group-tab:: MAPPO

        .. tabs::

            .. group-tab:: MPE

                .. tabs::

                    .. group-tab:: simple_adversary_v3

                        .. code-block:: yaml

                    .. group-tab:: simple_push_v3

                        .. code-block:: yaml

                    .. group-tab:: simple_spread_v3

                        .. code-block:: yaml

            .. group-tab:: SC2

                .. tabs::

                    .. group-tab:: 1c3s5z

                        .. code-block:: yaml

                    .. group-tab:: 2m_vs_1z

                        .. code-block:: yaml

                    .. group-tab:: 2s3z

                        .. code-block:: yaml

                    .. group-tab:: 3m

                        .. code-block:: yaml

                    .. group-tab:: 5m_vs_6m

                        .. code-block:: yaml

                    .. group-tab:: 8m

                        .. code-block:: yaml

                    .. group-tab:: 8m_vd_9m

                        .. code-block:: yaml

                    .. group-tab:: 25m

                        .. code-block:: yaml

                    .. group-tab:: corridor

                        .. code-block:: yaml
                    
                    .. group-tab:: MMM2

                        .. code-block:: yaml

            .. group-tab:: Football

                .. tabs::

                    .. group-tab:: 1v1

                        .. code-block:: yaml
                    
                    .. group-tab:: 3v1

                        .. code-block:: yaml

    .. group-tab:: MATD3

        .. tabs::

            .. group-tab:: MPE

                .. tabs::

                    .. group-tab:: simple_adversary_v3

                        .. code-block:: yaml

                    .. group-tab:: simple_push_v3

                        .. code-block:: yaml

                    .. group-tab:: simple_spread_v3

                        .. code-block:: yaml

    .. group-tab:: VDAC

        .. tabs::

            .. group-tab:: MPE

                .. tabs::

                    .. group-tab:: simple_spread_v3

                        .. code-block:: yaml

            .. group-tab:: SC2

                .. tabs::

                    .. group-tab:: 1c3s5z

                        .. code-block:: yaml

                    .. group-tab:: 2m_vs_1z

                        .. code-block:: yaml

                    .. group-tab:: 2s3z

                        .. code-block:: yaml

                    .. group-tab:: 3m

                        .. code-block:: yaml

                    .. group-tab:: 5m_vs_6m

                        .. code-block:: yaml

                    .. group-tab:: 8m

                        .. code-block:: yaml

                    .. group-tab:: 8m_vd_9m

                        .. code-block:: yaml

                    .. group-tab:: 25m

                        .. code-block:: yaml

                    .. group-tab:: corridor

                        .. code-block:: yaml
                    
                    .. group-tab:: MMM2

                        .. code-block:: yaml


    .. group-tab:: COMA

        .. tabs::

            .. group-tab:: MPE

                .. tabs::

                    .. group-tab:: simple_spread_v3

                        .. code-block:: yaml

            .. group-tab:: SC2

                .. tabs::

                    .. group-tab:: 1c3s5z

                        .. code-block:: yaml

                    .. group-tab:: 2m_vs_1z

                        .. code-block:: yaml

                    .. group-tab:: 2s3z

                        .. code-block:: yaml

                    .. group-tab:: 3m

                        .. code-block:: yaml

                    .. group-tab:: 5m_vs_6m

                        .. code-block:: yaml

                    .. group-tab:: 8m

                        .. code-block:: yaml

                    .. group-tab:: 8m_vd_9m

                        .. code-block:: yaml

                    .. group-tab:: 25m

                        .. code-block:: yaml

                    .. group-tab:: corridor

                        .. code-block:: yaml
                    
                    .. group-tab:: MMM2

                        .. code-block:: yaml

    .. group-tab:: MFQ

        .. tabs::

            .. group-tab:: MPE

                .. tabs::

                    .. group-tab:: simple_spread_v3

                        .. code-block:: yaml

            .. group-tab:: Magent2

                .. tabs::

                    .. group-tab:: adversarial_pursuit_v4

                        .. code-block:: yaml

    .. group-tab:: MFAC

        .. tabs::

            .. group-tab:: MPE

                .. tabs::

                    .. group-tab:: simple_spread_v3

                        .. code-block:: yaml

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
