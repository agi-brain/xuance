New Multi-Agent Environment
=========================================================

In XuanCe, users also have the flexibility to create and run their own customized environments with multiple agents in addition to utilizing the provided ones.


.. raw:: html

   <br><hr>

Step 1: Create a New Multi-Agent Environment
-------------------------------------------------------------

First, you need to prepare an original environment, i.e., an Partial Observed Markov decision process (POMDP).
Then define a new environment based on the basic class ``RawMultiAgentEnv`` of XuanCe.

Here is an example:

.. code-block:: python

    import numpy as np
    from gym.spaces import Box
    from xuance.environment import RawMultiAgentEnv

    class MyNewMultiAgentEnv(RawMultiAgentEnv):
        def __init__(self, env_config):
            super(MyNewMultiAgentEnv, self).__init__()
            self.env_id = env_config.env_id
            self.num_agents = 3
            self.agents = [f"agent_{i}" for i in range(self.num_agents)]
            self.state_space = Box(-np.inf, np.inf, shape=[8, ])
            self.observation_space = {agent: Box(-np.inf, np.inf, shape=[8, ]) for agent in self.agents}
            self.action_space = {agent: Box(-np.inf, np.inf, shape=[2, ]) for agent in self.agents}
            self.max_episode_steps = 25
            self._current_step = 0

        def get_env_info(self):
            return {'state_space': self.state_space,
                    'observation_space': self.observation_space,
                    'action_space': self.action_space,
                    'agents': self.agents,
                    'num_agents': self.num_agents,
                    'max_episode_steps': self.max_episode_steps}

        def avail_actions(self):
            return None

        def agent_mask(self):
            """Returns boolean mask variables indicating which agents are currently alive."""
            return {agent: True for agent in self.agents}

        def state(self):
            """Returns the global state of the environment."""
            return self.state_space.sample()

        def reset(self):
            observation = {agent: self.observation_space[agent].sample() for agent in self.agents}
            info = {}
            self._current_step = 0
            return observation, info

        def step(self, action_dict):
            self._current_step += 1
            observation = {agent: self.observation_space[agent].sample() for agent in self.agents}
            rewards = {agent: np.random.random() for agent in self.agents}
            terminated = {agent: False for agent in self.agents}
            truncated = False if self._current_step < self.max_episode_steps else True
            info = {}
            return observation, rewards, terminated, truncated, info

        def render(self, *args, **kwargs):
            return np.ones([64, 64, 64])

        def close(self):
            return


Step 2: Create the Config File and Read the Configurations
-------------------------------------------------------------

Then, you need to create a YAML file by following the step 1 in :doc:`Further Usage <further_usage>`.

Here is an example of configurations for DDPG algorithm, named "ippo_new_configs.yaml".

.. code-block:: python

    dl_toolbox: "torch"  # The deep learning toolbox. Choices: "torch", "mindspore", "tensorlayer"
    project_name: "XuanCe_Benchmark"
    logger: "tensorboard"  # Choices: tensorboard, wandb.
    wandb_user_name: "your_user_name"
    render: True
    render_mode: 'rgb_array' # Choices: 'human', 'rgb_array'.
    test_mode: False
    device: "cuda:0"

    agent: "IPPO"
    env_name: "MyNewMultiAgentEnv"
    env_id: "new_env_id"
    fps: 50
    continuous_action: True
    policy: "Gaussian_MAAC_Policy"
    representation: "Basic_MLP"
    vectorize: "DummyVecMultiAgentEnv"

    # recurrent settings for Basic_RNN representation.
    use_rnn: False  # If to use recurrent neural network as representation. (The representation should be "Basic_RNN").
    rnn: "GRU"  # The type of recurrent layer.
    fc_hidden_sizes: [64, 64, 64]  # The hidden size of feed forward layer in RNN representation.
    recurrent_hidden_size: 64  # The hidden size of the recurrent layer.
    N_recurrent_layers: 1  # The number of recurrent layer.
    dropout: 0  # dropout should be a number in range [0, 1], the probability of an element being zeroed.
    normalize: "LayerNorm"  # Layer normalization.
    initialize: "orthogonal"  # Network initializer.
    gain: 0.01  # Gain value for network initialization.

    # recurrent settings for Basic_RNN representation.
    representation_hidden_size: [64, ]  # A list of hidden units for each layer of Basic_MLP representation networks.
    actor_hidden_size: [64, ]  # A list of hidden units for each layer of actor network.
    critic_hidden_size: [64, ]  # A list of hidden units for each layer of critic network.
    activation: "relu"  # The activation function of each hidden layer.
    activation_action: "sigmoid"  # The activation function for the last layer of the actor.
    use_parameter_sharing: True  # If to use parameter sharing for all agents' policies.
    use_actions_mask: False  # If to use actions mask for unavailable actions.

    seed: 1  # Random seed.
    parallels: 16  # The number of environments to run in parallel.
    buffer_size: 3200  # Number of the transitions (use_rnn is False), or the episodes (use_rnn is True) in replay buffer.
    n_epochs: 10  # Number of epochs to train.
    n_minibatch: 1 # Number of minibatch to sample and train.  batch_size = buffer_size // n_minibatch.
    learning_rate: 0.0007  # Learning rate.
    weight_decay: 0  # The steps to decay the greedy epsilon.

    vf_coef: 0.5  # Coefficient factor for critic loss.
    ent_coef: 0.01  # Coefficient factor for entropy loss.
    target_kl: 0.25  # For MAPPO_KL learner.
    clip_range: 0.2  # The clip range for ratio in MAPPO_Clip learner.
    gamma: 0.99  # Discount factor.

    # tricks
    use_linear_lr_decay: False  # If to use linear learning rate decay.
    end_factor_lr_decay: 0.5  # The end factor for learning rate scheduler.
    use_global_state: False  # If to use global state to replace merged observations.
    use_value_clip: True  # Limit the value range.
    value_clip_range: 0.2  # The value clip range.
    use_value_norm: True  # Use running mean and std to normalize rewards.
    use_huber_loss: True  # True: use huber loss; False: use MSE loss.
    huber_delta: 10.0  # The threshold at which to change between delta-scaled L1 and L2 loss. (For huber loss).
    use_advnorm: True  # If to use advantage normalization.
    use_gae: True  # Use GAE trick.
    gae_lambda: 0.95  # The GAE lambda.
    use_grad_clip: True  # Gradient normalization.
    grad_clip_norm: 10.0  # The max norm of the gradient.
    clip_type: 1  # Gradient clip for Mindspore: 0: ms.ops.clip_by_value; 1: ms.nn.ClipByNorm().

    running_steps: 10000000  # The total running steps.
    eval_interval: 100000  # The interval between every two trainings.
    test_episode: 5  # The episodes to test in each test period.

    log_dir: "./logs/ippo/"
    model_dir: "./models/ippo/"


Then, read the configurations:

.. code-block:: python

    import argparse
    from xuance.common import get_configs
    configs_dict = get_configs(file_dir="ippo_new_configs.yaml")
    configs = argparse.Namespace(**configs_dict)


Step 3: Add the Environment to the Registry
-------------------------------------------------------------

After defining a new class of environment, you need to add it to the ``REGISTRY_MULTI_AGENT_ENV``.

.. code-block:: python

    from xuance.environment import REGISTRY_MULTI_AGENT_ENV
    REGISTRY_MULTI_AGENT_ENV[configs.env_name] = MyNewMultiAgentEnv


Step 4: Make Your Environment and Run it with XuanCe
-------------------------------------------------------------

You can now make your environment and run it directly with XuanCe's algorithms.

Here is the example of IPPO algorithm:

.. code-block:: python

    from xuance.environment import make_envs
    from xuance.torch.agents import IPPO_Agents

    envs = make_envs(configs)  # Make parallel environments.
    Agent = IPPO_Agents(config=configs, envs=envs)  # Create a DDPG agent from XuanCe.
    Agent.train(configs.running_steps // configs.parallels)  # Train the model for numerous steps.
    Agent.save_model("final_train_model.pth")  # Save the model to model_dir.
    Agent.finish()  # Finish the training.
