New Environment
=========================================================

In XuanCe, users have the flexibility to create and run their own customized environments in addition to utilizing the provided ones.


.. raw:: html

   <br><hr>

Step 1: Create a New Environment
-------------------------------------------------------------

First, you need to prepare an original environment, i.e., an Markov decision process.
Then define a new environment based on the basic class ``RawEnvironment`` of XuanCe.

Here is an example:

.. code-block:: python

    import numpy as np
    from gym.spaces import Box
    from xuance.environment import RawEnvironment

    class MyNewEnv(RawEnvironment):
        def __init__(self, env_config):
            super(MyNewEnv, self).__init__()
            self.env_id = env_config.env_id  # The environment id.
            self.observation_space = Box(-np.inf, np.inf, shape=[18, ])  # Define observation space.
            self.action_space = Box(-np.inf, np.inf, shape=[5, ])  # Define action space. In this example, the action space is continuous.
            self.max_episode_steps = 32  # The max episode length.
            self._current_step = 0  # The count of steps of current episode.

        def reset(self, **kwargs):  # Reset your environment.
            self._current_step = 0
            return self.observation_space.sample(), {}

        def step(self, action):  # Run a step with an action.
            self._current_step += 1
            observation = self.observation_space.sample()
            rewards = np.random.random()
            terminated = False
            truncated = False if self._current_step < self.max_episode_steps else True
            info = {}
            return observation, rewards, terminated, truncated, info

        def render(self, *args, **kwargs):  # Render your environment and return an image if the render_mode is "rgb_array".
            return np.ones([64, 64, 64])

        def close(self):  # Close your environment.
            return


Step 2: Create the Config File and Read the Configurations
-------------------------------------------------------------

Then, you need to create a YAML file by following the step 1 in :doc:`Further Usage <further_usage>`.

Here is an example of configurations for DDPG algorithm, named "ddpg_new_env.yaml".

.. code-block:: python

    dl_toolbox: "torch"  # The deep learning toolbox. Choices: "torch", "mindspore", "tensorlayer"
    project_name: "XuanCe_Benchmark"
    logger: "tensorboard"  # Choices: tensorboard, wandb.
    wandb_user_name: "your_user_name"
    render: True
    render_mode: 'rgb_array' # Choices: 'human', 'rgb_array'.
    fps: 50
    test_mode: False
    device: "cuda:0"

    agent: "DDPG"
    env_name: "MyNewEnv"
    env_id: "new-v1"
    vectorize: "DummyVecEnv"
    policy: "DDPG_Policy"
    representation: "Basic_Identical"
    runner: "DRL"

    representation_hidden_size:  # If you choose Basic_Identical representation, then ignore this value
    actor_hidden_size: [400, 300]
    critic_hidden_size: [400, 300]
    activation: "leaky_relu"
    activation_action: 'tanh'

    seed: 19089
    parallels: 4  # number of environments
    buffer_size: 200000  # replay buffer size
    batch_size: 100
    learning_rate_actor: 0.001
    learning_rate_critic: 0.001
    gamma: 0.99
    tau: 0.005

    start_noise: 0.5
    end_noise: 0.1
    training_frequency: 1
    running_steps: 1000000  # 1M
    start_training: 10000

    use_grad_clip: False  # gradient normalization
    grad_clip_norm: 0.5
    use_obsnorm: False
    use_rewnorm: False
    obsnorm_range: 5
    rewnorm_range: 5

    test_steps: 10000
    eval_interval: 5000
    test_episode: 5

    log_dir: "./logs/ddpg/"
    model_dir: "./models/ddpg/"

Then, read the configurations:

.. code-block:: python

    import argparse
    from xuance.common import get_configs
    configs_dict = get_configs(file_dir="ddpg_new_env.yaml")
    configs = argparse.Namespace(**configs_dict)


Step 3: Add the Environment to the Registry
-------------------------------------------------------------

After defining a new class of environment, you need to add it to the ``REGISTRY_ENV``.

.. code-block:: python

    from xuance.environment import REGISTRY_ENV
    REGISTRY_ENV[configs.env_name] = MyNewEnv


Step 4: Make Your Environment and Run it with XuanCe
-------------------------------------------------------------

You can now make your environment and run it directly with XuanCe's algorithms.

Here is the example of DDPG algorithm:

.. code-block:: python

    from xuance.environment import make_envs
    from xuance.torch.agents import DDPG_Agent

    envs = make_envs(configs)  # Make parallel environments.
    Agent = DDPG_Agent(config=configs, envs=envs)  # Create a DDPG agent from XuanCe.
    Agent.train(configs.running_steps // configs.parallels)  # Train the model for numerous steps.
    Agent.save_model("final_train_model.pth")  # Save the model to model_dir.
    Agent.finish()  # Finish the training.
