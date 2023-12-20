Configs
======================



.. raw:: html

   <br><hr>
   
Basic Configurations
--------------------------

The basic parameter configuration is stored in the "xuance/config/basic.yaml" file, as shown below:

.. code-block:: yaml

    dl_toolbox: "torch"  # Values: "torch", "mindspore", "tensorlayer"

    project_name: "XuanCe_Benchmark"
    logger: "tensorboard"  # Values: tensorboard, wandb.
    wandb_user_name: "papers_liu"

    parallels: 10
    seed: 2910
    render: True
    render_mode: 'rgb_array' # Values: 'human', 'rgb_array'.
    test_mode: False
    test_steps: 2000

    device: "cuda:0"


It should be noted that the value of the `device` variable in the `basic.yaml` file varies depending on the specific deep learning framework, as outlined below:

| - PyTorch: "cpu", "cuda:0";
| - TensorFlow: "cpu"/"CPU", "gpu"/"GPU";
| - MindSpore: "CPU", "GPU", "Ascend", "Davinci"。

.. raw:: html

   <br><hr>
   
Algorithm Configurations
--------------------------

As an example, taking the parameter configuration for the DQN algorithm in the Atari environment, 
in addition to the basic parameter configuration, the algorithm-specific parameters are stored in the "xuance/configs/dqn/atari.yaml" file, with the following content:

.. raw:: html

    <center>
        <select id="env-mujoco" onchange="showMujocoEnv(this)"></select>
        <br>
        <div id="vis-mujoco"></div>
        <br>
    </center>

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

Due to the presence of over 60 different scenarios in the Atari environment, 
where the scenarios are relatively consistent with variations only in tasks, 
a single default parameter configuration file is sufficient.

For environments with significant scene variations, such as the "CarRacing-v2" and "LunarLander" scenarios in the "Box2D" environment, 
the former has a state input of a 96x96x3 RGB image, while the latter consists of an 8-dimensional vector. 
Therefore, the DQN algorithm parameter configurations for these two scenarios are stored in the following two files:

    * xuance/configs/dqn/box2d/CarRacing-v2.yaml
    * xuance/configs/dqn/box2d/LunarLander-v2.yaml

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
