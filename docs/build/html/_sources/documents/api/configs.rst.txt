Configs
======================

Configs模块用于存放有关算法、环境、系统配置等参数，玄策提供的算法案例，其参数均存于该模块中。

为了便于训练和调参，玄策给出了默认的参数配置。
默认参数配置分为基础参数配置和算法参数配置，二者中的参数名称可重复，
但是算法参数配置的优先级高于基础参数配置，即重复的参数名已算法参数配置文件中的值为主。

针对不同的算法、不同的环境，强化学习示例需要用到的参数种类各不相同，因此需要根据实际需求配置必要的参数。
玄策中的所有算法均有相应的示例，每个示例给出了该算法运行所必须的参数。若用户在实现自己的新任务时需要用到其它参数，
可直接在参数配置文件中添加。

.. raw:: html

   <br><hr>
   
基础参数配置
--------------------------
基础参数配置存于xuanpolicy/config/basic.yaml文件中，示例如下：

.. code-block:: yaml

    dl_toolbox: "torch"  # Values: "torch", "mindspore", "tensorlayer"

    project_name: "XuanPolicy_Benchmark"
    logger: "tensorboard"  # Values: tensorboard, wandb.
    wandb_user_name: "papers_liu"

    parallels: 10
    seed: 2910
    render: True
    render_mode: 'rgb_array' # Values: 'human', 'rgb_array'.
    test_mode: False
    test_steps: 2000

    device: "cuda:0"


需要注意的是， `basic.yaml` 文件中的 ``device`` 变量取值根据不同的深度学习框架有所差异，分别如下：

| - PyTorch: "cpu", "cuda:0";
| - TensorFlow: "cpu"/"CPU", "gpu"/"GPU";
| - MindSpore: "CPU", "GPU", "Ascend", "Davinci"。

.. raw:: html

   <br><hr>
   
算法参数配置
--------------------------

以DQN算法在Atari环境中的参数配置为例，除了基础参数配置外，其算法参数配置存放于 xuanpolicy/configs/dqn/atari.yaml
文件中，内容如下：

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

由于Atari环境中一共超过60个不同场景，场景比较统一，只是任务不同，因此只需要一个默认的参数配置文件即可。

针对场景差异较大的环境，如 ``Box2D`` 环境中的 ``CarRacing-v2`` 和 ``LunarLander`` 场景，
前者的状态输入是96*96*3的RGB图像，后者则是一个8维向量。因此，针对这两个场景的DQN算法参数配置分别存于以下两个文件中：

    * xuanpolicy/configs/dqn/box2d/CarRacing-v2.yaml
    * xuanpolicy/configs/dqn/box2d/LunarLander-v2.yaml

.. raw:: html

   <br><hr>
   
自定义参数配置
--------------------------
用户也可以选择不适用玄策提供的默认参数，或者玄策中不包含用户的任务时，可用同样的方式自定义.yaml参数配置文件。
但是在获取runner的过程中，需指定参数文件的存放位置，示例如下：

.. code-block:: python3

    import xuanpolicy as xp
    runner = xp.get_runner(method='dqn', 
                           env='classic_control',
                           env_id='CartPole-v1', 
                           config_path="xxx/xxx.yaml",
                           is_test=False)
    runner.run()
